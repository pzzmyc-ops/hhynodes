import os
import random
import time
import sqlite3
import numpy as np
import torch
from PIL import Image
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def read_txt_file(file_path):
    encodings = ['utf-8', 'gbk', 'big5', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""

def find_image(txt_path, extensions):
    base_name = os.path.splitext(txt_path)[0]
    for ext in extensions.split(','):
        ext = ext.strip().lower()
        candidates = [
            f"{base_name}.{ext}",
            f"{base_name}_image.{ext}",
            f"{base_name}-img.{ext}",
            os.path.join(os.path.dirname(txt_path), "images", f"{os.path.basename(base_name)}.{ext}")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None

def process_file_batch(file_batch_data):
    """多线程处理单个文件批次 - 只处理有配对图片的文本文件"""
    folder_path, txt_path, mod_time, image_extensions = file_batch_data
    try:
        # 首先检查是否有配对的图片
        image_path = find_image(txt_path, image_extensions)
        if not image_path:
            # 没有配对图片，跳过这个文件
            return None
        
        # 有配对图片，读取文本内容
        content = read_txt_file(txt_path)
        if not content.strip():
            # 文本内容为空，也跳过
            return None
            
        return (folder_path, txt_path, content, image_path, mod_time)
    except Exception as e:
        print(f"⚠️ 处理文件失败 {txt_path}: {str(e)}")
        return None

def background_database_builder(db_path, folder_path, image_extensions, progress_queue, stop_event, generate_missing_txt=False):
    import sys
    try:
        print("🔧 后台数据库构建线程已启动")
        sys.stdout.flush()
        
        # 为后台线程创建独立的数据库连接，使用连接池优化
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60.0)
        cursor = conn.cursor()
        
        # 后台线程使用高性能数据库设置
        cursor.execute('PRAGMA journal_mode = WAL')  # WAL模式支持并发读写
        cursor.execute('PRAGMA synchronous = OFF')  # 关闭同步，最大化性能
        cursor.execute('PRAGMA cache_size = -2048000')  # 2GB缓存用于后台构建
        cursor.execute('PRAGMA temp_store = MEMORY')
        cursor.execute('PRAGMA mmap_size = 4294967296')  # 4GB内存映射
        cursor.execute('PRAGMA page_size = 65536')  # 64KB页面大小，适合大量写入
        cursor.execute('PRAGMA locking_mode = NORMAL')  # 正常锁模式
        cursor.execute('PRAGMA busy_timeout = 60000')  # 60秒超时
        cursor.execute('PRAGMA threads = 16')  # 增加SQLite内部线程数
        cursor.execute('PRAGMA optimize')  # 启用查询优化器
        
        folder_path = os.path.normpath(folder_path)
        start_time = time.time()
        progress_queue.put({"type": "scan_start", "folder": folder_path})
        scan_start = time.time()
        txt_files = []
        file_mod_times = {}
        
        # 若启用生成缺失标注，先为没有同名txt的图片创建txt，内容为图片文件名
        if generate_missing_txt:
            try:
                created_count = 0
                total_images = 0
                ext_list = [ext.strip().lower() for ext in image_extensions.split(',') if ext.strip()]
                for root, _, files in os.walk(folder_path):
                    if stop_event.is_set():
                        conn.close()
                        return
                    for f in files:
                        lf = f.lower()
                        if any(lf.endswith(f'.{ext}') for ext in ext_list):
                            total_images += 1
                            image_path = os.path.join(root, f)
                            base, _ = os.path.splitext(image_path)
                            txt_path = f"{base}.txt"
                            if not os.path.exists(txt_path):
                                try:
                                    with open(txt_path, 'w', encoding='utf-8') as tf:
                                        tf.write(os.path.basename(base))
                                    created_count += 1
                                except Exception as e:
                                    print(f"⚠️ 创建标注文件失败 [{txt_path}]: {str(e)}")
                progress_queue.put({
                    "type": "txt_generation",
                    "created": created_count,
                    "total_images": total_images
                })
            except Exception as e:
                print(f"⚠️ 生成缺失标注时出错: {str(e)}")
        
        # 文件扫描阶段
        for root, _, files in os.walk(folder_path):
            if stop_event.is_set():
                conn.close()
                return
            for f in files:
                if f.lower().endswith(".txt"):
                    txt_path = os.path.join(root, f)
                    mod_time = os.path.getmtime(txt_path)
                    txt_files.append(txt_path)
                    file_mod_times[txt_path] = mod_time
        
        scan_time = time.time() - scan_start
        progress_queue.put({
            "type": "scan_complete", 
            "count": len(txt_files), 
            "time": scan_time
        })
        
        if not txt_files or stop_event.is_set():
            conn.close()
            return
        
        # 数据库查询阶段 - 使用更大的批次大小
        progress_queue.put({"type": "query_start"})
        query_start = time.time()
        db_files = {}
        query_batch_size = 50000  # 增加批次大小
        
        for i in range(0, len(txt_files), query_batch_size):
            if stop_event.is_set():
                conn.close()
                return
            batch_files = txt_files[i:i + query_batch_size]
            placeholders = ','.join(['?'] * len(batch_files))
            cursor.execute(f"""
                SELECT txt_path, last_modified 
                FROM file_cache 
                WHERE txt_path IN ({placeholders})
            """, batch_files)
            batch_results = dict(cursor.fetchall())
            db_files.update(batch_results)
            
            if len(txt_files) > query_batch_size:
                progress_queue.put({
                    "type": "query_progress",
                    "current": min(i + query_batch_size, len(txt_files)),
                    "total": len(txt_files)
                })
        
        query_time = time.time() - query_start
        progress_queue.put({"type": "query_complete", "time": query_time})
        
        # 分析需要处理的文件
        files_to_process = []
        for txt_path in txt_files:
            if stop_event.is_set():
                conn.close()
                return
            current_mod_time = file_mod_times[txt_path]
            db_mod_time = db_files.get(txt_path)
            if db_mod_time is None or db_mod_time != current_mod_time:
                files_to_process.append((txt_path, current_mod_time))
        
        existing_files = len(txt_files) - len(files_to_process)
        progress_queue.put({
            "type": "analysis_complete",
            "existing": existing_files,
            "new_or_modified": len(files_to_process)
        })
        
        if not files_to_process or stop_event.is_set():
            conn.close()
            return
        
        # 多线程文件处理阶段
        progress_queue.put({"type": "process_start", "total": len(files_to_process)})
        process_start = time.time()
        processed_count = 0
        skipped_count = 0  # 跳过的文件数量
        
        # 动态确定线程数，基于CPU核心数和文件数量
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(max(cpu_count * 2, 8), 32)  # 最少8个，最多32个线程
        file_processing_batch_size = 100  # 文件处理批次大小
        db_batch_size = 1000  # 数据库批次大小
        
        print(f"🚀 使用 {max_workers} 个线程并行处理文件（只处理图文配对文件）")
        
        # 准备文件处理数据
        file_batch_data = [
            (folder_path, txt_path, mod_time, image_extensions)
            for txt_path, mod_time in files_to_process
        ]
        
        batch_data = []
        last_progress_time = process_start
        last_progress_count = 0  # 记录上次打印进度时的处理数量
        
        # 使用线程池并行处理文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 分批提交任务，避免内存占用过高
            for i in range(0, len(file_batch_data), file_processing_batch_size):
                if stop_event.is_set():
                    break
                
                batch_files = file_batch_data[i:i + file_processing_batch_size]
                future_to_file = {
                    executor.submit(process_file_batch, file_data): file_data 
                    for file_data in batch_files
                }
                
                # 收集结果
                for future in as_completed(future_to_file):
                    if stop_event.is_set():
                        break
                    
                    result = future.result()
                    if result:
                        batch_data.append(result)
                        processed_count += 1
                    else:
                        skipped_count += 1
                    
                    # 当积累足够数据或处理完一批时，写入数据库
                    if len(batch_data) >= db_batch_size or (processed_count + skipped_count) % file_processing_batch_size == 0:
                        if batch_data:
                            # 使用事务快速写入
                            cursor.execute('BEGIN IMMEDIATE')
                            try:
                                cursor.executemany('''
                                    INSERT OR REPLACE INTO file_cache 
                                    (folder_path, txt_path, content, image_path, last_modified)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', batch_data)
                                cursor.execute('COMMIT')
                            except Exception as e:
                                cursor.execute('ROLLBACK')
                                raise e
                            
                            batch_data = []
                        
                        # 优化进度更新：减少打印频率
                        current_time = time.time()
                        elapsed_since_last = current_time - last_progress_time
                        total_elapsed = current_time - process_start
                        total_processed_files = processed_count + skipped_count
                        current_progress_percent = total_processed_files * 100 // len(files_to_process)
                        last_progress_percent = (last_progress_count) * 100 // len(files_to_process)
                        
                        # 只在以下情况下打印进度：
                        should_print_progress = (
                            elapsed_since_last >= 15.0 or  # 至少15秒间隔
                            total_processed_files % 50000 == 0 or  # 每5万个文件
                            current_progress_percent >= last_progress_percent + 5  # 每5%进度变化
                        )
                        
                        if should_print_progress:
                            files_processed_since_last = total_processed_files - last_progress_count
                            speed = files_processed_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
                            
                            cursor.execute("SELECT COUNT(*) FROM file_cache")
                            current_total_files = cursor.fetchone()[0]
                            
                            progress_queue.put({
                                "type": "process_progress",
                                "processed": processed_count,
                                "skipped": skipped_count,
                                "total": len(files_to_process),
                                "speed": speed,
                                "elapsed": total_elapsed,
                                "db_total": current_total_files,
                                "progress_percent": current_progress_percent
                            })
                            last_progress_time = current_time
                            last_progress_count = total_processed_files
        
        # 处理剩余数据
        if batch_data and not stop_event.is_set():
            cursor.execute('BEGIN IMMEDIATE')
            try:
                cursor.executemany('''
                    INSERT OR REPLACE INTO file_cache 
                    (folder_path, txt_path, content, image_path, last_modified)
                    VALUES (?, ?, ?, ?, ?)
                ''', batch_data)
                cursor.execute('COMMIT')
            except Exception as e:
                cursor.execute('ROLLBACK')
                raise e
        
        # 优化数据库
        cursor.execute('PRAGMA optimize')
        cursor.execute('VACUUM')  # 清理数据库文件
        
        total_time = time.time() - start_time
        process_time = time.time() - process_start
        avg_speed = (processed_count + skipped_count) / process_time if process_time > 0 else 0
        
        cursor.execute("SELECT COUNT(*) FROM file_cache")
        final_total = cursor.fetchone()[0]
        
        progress_queue.put({
            "type": "complete",
            "processed": processed_count,
            "skipped": skipped_count,
            "avg_speed": avg_speed,
            "total_time": total_time,
            "db_total": final_total
        })
        
        conn.close()
        
    except Exception as e:
        progress_queue.put({"type": "error", "message": str(e)})

class TextWithImageReader:
    def __init__(self):
        self.history = []
        self.blank_image = self.create_blank_image()
        
        # 根据操作系统选择不同的数据库文件名
        import platform
        system = platform.system().lower()
        if system == "linux":
            db_filename = "text_image_reader_cache_linux.db"
        else:
            db_filename = "text_image_reader_cache.db"
        
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_filename)
        self.conn = None
        self.cursor = None
        self.last_folder = ""
        self.background_process = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.progress_thread = None
        self.is_building = False
        # 智能搜索结果缓存
        self.last_search_conditions = None
        self.last_search_results = None
        # 连接池锁
        self.db_lock = threading.RLock()
        
        # 顺序读取相关属性
        self.sequential_mode = False
        self.sequential_results = []
        self.sequential_index = 0
        self.sequential_folder = ""
        self.sequential_conditions = None
        
        print(f"🔧 初始化数据库: {os.path.basename(self.db_path)}")
        self.initialize_database()
    
    def initialize_database(self):
        # 主线程数据库连接，优化为高性能读取
        with self.db_lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60.0)
            self.cursor = self.conn.cursor()
            
            # 主线程使用读优化配置
            self.cursor.execute('PRAGMA journal_mode = WAL')  # WAL模式支持并发读写
            self.cursor.execute('PRAGMA synchronous = NORMAL')  # 平衡性能和安全性
            self.cursor.execute('PRAGMA cache_size = -3072000')  # 3GB缓存用于主线程读取
            self.cursor.execute('PRAGMA temp_store = MEMORY')  # 临时表全部内存
            self.cursor.execute('PRAGMA mmap_size = 6442450944')  # 6GB内存映射
            self.cursor.execute('PRAGMA page_size = 4096')  # 标准页面大小
            self.cursor.execute('PRAGMA locking_mode = NORMAL')  # 正常锁模式，允许并发
            self.cursor.execute('PRAGMA busy_timeout = 60000')  # 60秒超时
            self.cursor.execute('PRAGMA count_changes = OFF')  # 关闭计数
            self.cursor.execute('PRAGMA auto_vacuum = NONE')  # 关闭自动清理
            self.cursor.execute('PRAGMA threads = 16')  # 主线程使用更多线程
            self.cursor.execute('PRAGMA optimize')  # 启用查询优化器
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_cache (
                    id INTEGER PRIMARY KEY,
                    folder_path TEXT NOT NULL,
                    txt_path TEXT UNIQUE NOT NULL,
                    content TEXT,
                    image_path TEXT,
                    last_modified REAL NOT NULL
                )
            ''')
            
            # 高性能索引策略 - 添加更多复合索引
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_folder_fast ON file_cache(folder_path) WHERE folder_path IS NOT NULL')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_folder_content_fast ON file_cache(folder_path, content) WHERE content IS NOT NULL')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_fast ON file_cache(content) WHERE content IS NOT NULL')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_txt_path_fast ON file_cache(txt_path)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_modified ON file_cache(last_modified)')
            
            self.conn.commit()
    
    def create_blank_image(self):
        blank = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        blank = blank.convert("RGB")
        blank_array = np.array(blank).astype(np.float32) / 255.0
        return torch.from_numpy(blank_array)[None,]
    
    def start_background_building(self, folder_path, image_extensions, generate_missing_txt=False):
        self.stop_background_building()
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.background_process = threading.Thread(
            target=background_database_builder,
            args=(self.db_path, folder_path, image_extensions, self.progress_queue, self.stop_event, generate_missing_txt)
        )
        self.background_process.daemon = True
        self.background_process.start()
        self.progress_thread = threading.Thread(target=self.monitor_progress)
        self.progress_thread.daemon = True
        self.progress_thread.start()
        self.is_building = True
        print(f"🚀 已启动后台数据库构建线程 (数据库: {os.path.basename(self.db_path)})")
    
    def stop_background_building(self):
        if self.background_process and self.background_process.is_alive():
            print("⏹️ 停止后台数据库构建线程...")
            self.stop_event.set()
            self.background_process.join(timeout=3)
            print("✅ 后台线程已停止")
        self.is_building = False
        self.background_process = None
    
    def monitor_progress(self):
        import sys
        print("🔧 进度监控线程已启动")
        sys.stdout.flush()
        while self.is_building:
            try:
                progress = self.progress_queue.get(timeout=1)
                if progress["type"] == "scan_start":
                    print(f"📁 后台扫描文件夹: {progress['folder']}")
                    sys.stdout.flush()
                elif progress["type"] == "scan_complete":
                    print(f"📊 输入文件夹共找到 {progress['count']} 个文本文件 (耗时: {progress['time']:.2f}秒)")
                    sys.stdout.flush()
                elif progress["type"] == "query_start":
                    print("🔍 检查数据库中的文件状态...")
                    sys.stdout.flush()
                elif progress["type"] == "query_progress":
                    print(f"🔍 已检查 {progress['current']}/{progress['total']} 个文件...")
                    sys.stdout.flush()
                elif progress["type"] == "query_complete":
                    print(f"🔍 数据库查询完成 (耗时: {progress['time']:.2f}秒)")
                    sys.stdout.flush()
                elif progress["type"] == "txt_generation":
                    print(f"📝 已为 {progress.get('total_images', 0)} 张图片生成缺失标注中的 {progress.get('created', 0)} 个")
                    sys.stdout.flush()
                elif progress["type"] == "analysis_complete":
                    print(f"📊 文件分析:")
                    print(f"   - 已存在且未修改: {progress['existing']} 个文件")
                    print(f"   - 新增或已修改: {progress['new_or_modified']} 个文件")
                    sys.stdout.flush()
                elif progress["type"] == "process_start":
                    print(f"🔄 需要处理 {progress['total']} 个文件")
                    sys.stdout.flush()
                elif progress["type"] == "process_progress":
                    progress_percent = progress.get('progress_percent', 0)
                    processed = progress['processed']
                    skipped = progress.get('skipped', 0)
                    total = progress['total']
                    speed = progress['speed']
                    elapsed = progress['elapsed']
                    db_total = progress['db_total']
                    
                    # 计算预计剩余时间
                    total_processed = processed + skipped
                    if speed > 0:
                        remaining_files = total - total_processed
                        eta_seconds = remaining_files / speed
                        eta_minutes = eta_seconds / 60
                        if eta_minutes < 1:
                            eta_str = f"{eta_seconds:.0f}秒"
                        elif eta_minutes < 60:
                            eta_str = f"{eta_minutes:.1f}分钟"
                        else:
                            eta_hours = eta_minutes / 60
                            eta_str = f"{eta_hours:.1f}小时"
                    else:
                        eta_str = "计算中..."
                    
                    print(f"⏳ 数据库构建进度: {progress_percent}% ({total_processed:,}/{total:,}) | "
                          f"图文对: {processed:,} | 跳过: {skipped:,} | "
                          f"速度: {speed:.0f}文件/秒 | 已用时: {elapsed/60:.1f}分钟 | "
                          f"预计剩余: {eta_str} | 数据库总计: {db_total:,}")
                    sys.stdout.flush()
                elif progress["type"] == "complete":
                    processed = progress['processed']
                    skipped = progress.get('skipped', 0)
                    total_files = processed + skipped
                    print(f"✅ 后台构建完成！共处理了 {total_files:,} 个文件")
                    print(f"📊 统计结果: 图文对 {processed:,} 个，跳过 {skipped:,} 个（无配对图片或内容为空）")
                    print(f"📈 处理速度: {progress['avg_speed']:.0f}文件/秒, 总耗时: {progress['total_time']:.2f}秒")
                    print(f"📊 数据库中共有 {progress['db_total']:,} 个图文对缓存")
                    sys.stdout.flush()
                    self.is_building = False
                    # 构建完成后清空搜索缓存，确保用户能获取到最新数据
                    self.clear_search_cache("数据库构建完成")
                elif progress["type"] == "error":
                    print(f"❌ 后台构建出错: {progress['message']}")
                    sys.stdout.flush()
                    self.is_building = False
                    # 构建出错时也清空搜索缓存
                    self.clear_search_cache("数据库构建出错")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 进度监控出错: {str(e)}")
                sys.stdout.flush()
                break
        print("🔧 进度监控线程已结束")
        sys.stdout.flush()
    
    def ensure_database_connection(self):
        """确保数据库连接可用，如果连接断开则重新连接"""
        try:
            with self.db_lock:
                # 测试连接是否可用
                self.cursor.execute("SELECT 1")
                return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError, AttributeError):
            print("🔧 数据库连接已断开，正在重新连接...")
            try:
                if self.conn:
                    self.conn.close()
            except:
                pass
            self.initialize_database()
            print("✅ 数据库连接已恢复")
            return True
        except Exception as e:
            print(f"❌ 数据库连接检查失败: {str(e)}")
            return False

    def fast_query(self, folder_path, keywords, negative_keywords, strict_match, search_entire_database, history_paths):
        """高速查询方法 - 使用线程安全的数据库访问"""
        # 确保数据库连接可用
        if not self.ensure_database_connection():
            print("❌ 无法建立数据库连接")
            return []
            
        max_retries = 5  # 增加重试次数
        retry_delay = 0.05  # 减少初始延迟
        
        for attempt in range(max_retries):
            try:
                with self.db_lock:  # 使用锁保护数据库访问
                    query_params = []
                    where_clauses = []
                    
                    # 文件夹条件
                    if not search_entire_database and folder_path:
                        where_clauses.append("folder_path = ?")
                        query_params.append(folder_path)
                    
                    # 历史记录过滤
                    if history_paths:
                        placeholders = ','.join(['?'] * len(history_paths))
                        where_clauses.append(f"txt_path NOT IN ({placeholders})")
                        query_params.extend(history_paths)
                    
                    # 关键词匹配
                    keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
                    if keyword_list:
                        if strict_match:
                            # 严格匹配逻辑
                            keyword_conditions = []
                            for keyword in keyword_list:
                                keyword_lower = keyword.lower()
                                
                                if (keyword_lower.startswith('"') and keyword_lower.endswith('"')) or \
                                   (keyword_lower.startswith("'") and keyword_lower.endswith("'")):
                                    # 完整短语匹配
                                    phrase = keyword_lower[1:-1]
                                    keyword_conditions.append("LOWER(content) LIKE ?")
                                    query_params.append(f"%{phrase}%")
                                    print(f"🎯 严格匹配短语: '{phrase}'")
                                else:
                                    # 单词边界匹配
                                    keyword_conditions.append("""
                                        (LOWER(content) LIKE ? OR 
                                         LOWER(content) LIKE ? OR 
                                         LOWER(content) LIKE ? OR 
                                         LOWER(content) LIKE ?)
                                    """.strip())
                                    query_params.extend([
                                        f"{keyword_lower} %",
                                        f"% {keyword_lower}",
                                        f"% {keyword_lower} %",
                                        keyword_lower
                                    ])
                                    print(f"🎯 严格匹配单词: '{keyword_lower}' (单词边界)")
                            
                            where_clauses.append(f"({' AND '.join(keyword_conditions)})")
                        else:
                            # 宽松匹配
                            keyword_conditions = []
                            for keyword in keyword_list:
                                keyword_conditions.append("LOWER(content) LIKE ?")
                                query_params.append(f"%{keyword.lower()}%")
                            where_clauses.append(f"({' OR '.join(keyword_conditions)})")
                    
                    # 否定关键词
                    negative_keyword_list = [kw.strip().lower() for kw in negative_keywords.split(",") if kw.strip()]
                    if negative_keyword_list:
                        for neg_keyword in negative_keyword_list:
                            where_clauses.append("LOWER(content) NOT LIKE ?")
                            query_params.append(f"%{neg_keyword}%")
                    
                    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
                    
                    # 执行查询
                    query = f"SELECT txt_path, content, image_path FROM file_cache WHERE {where_clause}"
                    self.cursor.execute(query, query_params)
                    results = self.cursor.fetchall()
                    
                    return results
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    print(f"⚠️ 数据库被锁定，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 2.0)  # 更温和的指数退避
                    continue
                else:
                    print(f"❌ 数据库查询失败: {str(e)}")
                    raise e
            except Exception as e:
                print(f"❌ 查询过程中发生错误: {str(e)}")
                raise e
        
        return []

    def get_search_fingerprint(self, folder_path, keywords, negative_keywords, strict_match, search_entire_database):
        """生成搜索条件的指纹，用于判断搜索条件是否变化"""
        return f"{folder_path}|{keywords}|{negative_keywords}|{strict_match}|{search_entire_database}"
    
    def should_use_cache(self):
        """判断是否应该使用搜索缓存 - 只有在数据库稳定时才使用缓存"""
        # 如果正在构建数据库，不使用缓存
        if self.is_building:
            return False
        
        # 如果缓存结果为空且之前没有搜索过，可能是数据库还在构建时缓存的空结果
        if (self.last_search_results is not None and 
            len(self.last_search_results) == 0 and 
            self.last_search_conditions is not None):
            # 检查数据库中是否现在有数据了
            try:
                with self.db_lock:
                    self.cursor.execute("SELECT COUNT(*) FROM file_cache LIMIT 1")
                    count = self.cursor.fetchone()[0]
                    if count > 0:
                        # 数据库现在有数据了，清空之前的空缓存
                        print("🔄 检测到数据库现在有数据，清空之前的空缓存")
                        self.last_search_conditions = None
                        self.last_search_results = None
                        return False
            except Exception as e:
                print(f"⚠️ 检查数据库状态失败: {str(e)}")
                return False
        
        return True
    
    def clear_search_cache(self, reason=""):
        """清空搜索缓存"""
        if reason:
            print(f"🔄 清空搜索缓存: {reason}")
        self.last_search_conditions = None
        self.last_search_results = None
        
        # 同时清空顺序读取缓存
        if self.sequential_mode:
            self.reset_sequential_mode("搜索缓存清理")

    def reset_sequential_mode(self, reason=""):
        """重置顺序读取模式"""
        if reason:
            print(f"🔄 重置顺序读取模式: {reason}")
        self.sequential_mode = False
        self.sequential_results = []
        self.sequential_index = 0
        self.sequential_folder = ""
        self.sequential_conditions = None

    def should_reset_sequential_mode(self, folder_path, keywords, negative_keywords, strict_match, search_entire_database):
        """判断是否需要重置顺序读取模式"""
        current_conditions = self.get_search_fingerprint(folder_path, keywords, negative_keywords, strict_match, search_entire_database)
        
        # 如果搜索条件变化，需要重置
        if self.sequential_conditions != current_conditions:
            return True
        
        # 如果文件夹变化，需要重置
        if not search_entire_database and self.sequential_folder != folder_path:
            return True
        
        # 如果从整个数据库搜索变为文件夹搜索，需要重置
        if self.sequential_mode and search_entire_database != (self.sequential_folder == ""):
            return True
        
        return False

    def get_sequential_result(self):
        """获取下一个顺序结果"""
        if not self.sequential_results or self.sequential_index >= len(self.sequential_results):
            return None
        
        result = self.sequential_results[self.sequential_index]
        self.sequential_index += 1
        
        # 如果到达列表末尾，循环回到开始
        if self.sequential_index >= len(self.sequential_results):
            self.sequential_index = 0
            print(f"🔄 顺序读取已到达末尾，重新开始循环 (共 {len(self.sequential_results)} 个结果)")
        
        return result

    def ultra_fast_select(self, results, history_set, max_attempts=100):
        """超高速选择 - 优化随机选择算法"""
        if not results:
            return None
        
        if not history_set:
            # 没有历史记录限制，直接随机选择
            import random
            return random.choice(results)
        
        # 有历史记录限制，使用更高效的选择算法
        total_results = len(results)
        
        # 如果历史记录很少，使用随机尝试
        if len(history_set) < total_results * 0.8:
            for _ in range(max_attempts):
                import random
                selected = random.choice(results)
                if selected[0] not in history_set:
                    return selected
        
        # 如果历史记录较多，直接过滤
        print("⚡ 使用预过滤模式选择结果")
        available = [result for result in results if result[0] not in history_set]
        if available:
            import random
            return random.choice(available)
        
        return None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keywords": ("STRING", {"default": ""}),
                "negative_keywords": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "history_depth": ("INT", {"default": 3, "min": 1, "max": 100}),
                "image_extensions": ("STRING", {"default": "png,jpg,jpeg,webp"}),
            },
            "optional": {
                "folder_path": ("STRING", {"default": ""}),
                "strict_match": ("BOOLEAN", {"default": False}),
                "force_cache_update": ("BOOLEAN", {"default": False}),
                "search_entire_database": ("BOOLEAN", {"default": False}),
                "sequential_read": ("BOOLEAN", {"default": False}),
                "generate_missing_txt": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    FUNCTION = "process"
    CATEGORY = "hhy"

    def process(self, keywords, negative_keywords, seed, history_depth, image_extensions, 
                folder_path="", strict_match=False, force_cache_update=False, search_entire_database=False, sequential_read=False, generate_missing_txt=False):
        
        # 严格匹配使用说明
        if strict_match and keywords:
            print("🎯 严格匹配模式已启用！")
            print("💡 使用提示:")
            print("   - 普通关键词: 使用单词边界匹配 (例如: white,panty)")
            print("   - 完整短语: 用引号包围 (例如: \"white panty\",\"black dress\")")
            print("   - 多个条件: 用逗号分隔，所有条件都必须满足 (AND关系)")
            print("   - 示例: 'white,panty' 要求同时包含这两个完整单词")
            print("   - 示例: '\"white panty\"' 要求包含完整短语 'white panty'")
        
        # 顺序读取模式说明
        if sequential_read:
            print("📖 顺序读取模式已启用！")
            print("💡 顺序读取特性:")
            print("   - 不受种子和历史记录影响")
            print("   - 按固定顺序依次输出结果")
            print("   - 到达末尾后自动循环")
            print("   - 搜索条件变化时自动重置")
        
        # 检查folder_path是否为空
        if not folder_path or not folder_path.strip():
            print("📌 未指定文件夹路径，将从整个数据库中搜索数据")
            print("💡 提示：如果需要更新特定文件夹的数据，请指定folder_path参数")
            
            # 线程安全地检查数据库是否为空
            with self.db_lock:
                self.cursor.execute("SELECT COUNT(*) FROM file_cache")
                total_count = self.cursor.fetchone()[0]
            
            if total_count == 0:
                error_msg = ("❌ 数据库为空！请先指定 folder_path 参数来扫描文件夹并构建数据库缓存。\n"
                           "示例：设置 folder_path 为包含文本文件的文件夹路径")
                print(error_msg)
                return (error_msg, self.blank_image)
            
            print(f"📊 数据库中共有 {total_count} 个文件缓存可供搜索")
            
            # 停止任何正在进行的后台构建任务
            if self.is_building:
                print("⏹️ 停止当前正在进行的数据库构建任务...")
                self.stop_background_building()
            
            # 重置last_folder，确保下次输入文件夹时会重新构建
            self.last_folder = ""
            
            # 强制设置为搜索整个数据库
            search_entire_database = True
            folder_path = ""  # 确保folder_path为空字符串
        else:
            folder_path = os.path.normpath(folder_path)
        
        if not search_entire_database and not os.path.exists(folder_path):
            error_msg = f"❌ 路径不存在: {folder_path}"
            print(error_msg)
            return (error_msg, self.blank_image)
        
        # 顺序读取模式下不设置种子
        if not sequential_read:
            torch.manual_seed(seed)
        
        # 检查是否更换了文件夹，如果是则清空搜索缓存
        if not search_entire_database and folder_path != self.last_folder:
            print(f"📁 检测到文件夹变更: {self.last_folder} -> {folder_path}")
            self.clear_search_cache("文件夹变更")
            self.history.clear()  # 清空历史记录，因为文件夹变了
        
        if not search_entire_database:
            with self.db_lock:
                self.cursor.execute("SELECT COUNT(*) FROM file_cache WHERE folder_path = ?", (folder_path,))
                existing_count = self.cursor.fetchone()[0]
                self.cursor.execute("SELECT COUNT(*) FROM file_cache")
                total_count = self.cursor.fetchone()[0]
            
            print(f"📊 数据库状态: 该文件夹已缓存 {existing_count} 个文件，数据库总计 {total_count} 个文件")
            
            if existing_count == 0 and total_count > 0:
                with self.db_lock:
                    self.cursor.execute("SELECT DISTINCT folder_path FROM file_cache LIMIT 5")
                    sample_folders = [row[0] for row in self.cursor.fetchall()]
                print(f"🔧 数据库中的文件夹示例: {sample_folders}")
                print(f"🔧 当前查询文件夹: {folder_path}")
            
            # 如果该文件夹在数据库中没有记录，清空缓存
            if existing_count == 0:
                self.clear_search_cache("文件夹无缓存记录")
                if total_count == 0:
                    error_msg = f"❌ 数据库为空！请等待后台构建完成或检查文件夹路径: {folder_path}"
                else:
                    error_msg = f"❌ 文件夹 {folder_path} 在数据库中没有缓存记录！\n请检查路径是否正确，或等待后台构建完成。"
                print(error_msg)
                # 不直接返回错误，让后台构建有机会运行
        
        # 只有在指定了folder_path且不是搜索整个数据库的情况下才进行后台构建
        if not search_entire_database and folder_path:
            if folder_path != self.last_folder or force_cache_update:
                # 开始新的构建时清空缓存
                if folder_path != self.last_folder:
                    self.clear_search_cache("开始新文件夹构建")
                elif force_cache_update:
                    self.clear_search_cache("强制更新缓存")
                self.start_background_building(folder_path, image_extensions, generate_missing_txt)
                self.last_folder = folder_path
            
            if self.is_building:
                print("🔄 后台正在构建数据库，从现有数据库中搜索...")
            else:
                print("✅ 数据库已是最新状态")

        # 超高速模式：直接从搜索结果中选择，避免预过滤
        if search_entire_database:
            print(f"🌐 从整个数据库中搜索...")
        else:
            print(f"📁 从文件夹 {folder_path} 的现有缓存中搜索...")
        
        keyword_list = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]
        if keyword_list:
            if strict_match:
                print(f"🎯 使用严格匹配模式搜索关键词: {', '.join(keyword_list)}")
            else:
                print(f"🔍 使用模糊匹配模式搜索关键词: {', '.join(keyword_list)}")
        
        # 获取搜索条件指纹和历史记录
        current_conditions = self.get_search_fingerprint(folder_path, keywords, negative_keywords, strict_match, search_entire_database)
        
        # 顺序读取模式处理
        if sequential_read:
            # 检查是否需要重置顺序读取模式
            if self.should_reset_sequential_mode(folder_path, keywords, negative_keywords, strict_match, search_entire_database):
                self.reset_sequential_mode("搜索条件或文件夹变化")
                print("🔄 顺序读取模式已重置")
            
            # 如果顺序读取模式未激活，需要初始化
            if not self.sequential_mode:
                print("📖 初始化顺序读取模式...")
                results = self.fast_query(folder_path, keywords, negative_keywords, strict_match, search_entire_database, [])
                
                if not results:
                    if not search_entire_database and self.is_building:
                        no_result_msg = "💡 后台正在构建数据库，稍后可能有更多结果。请稍等片刻后重试。"
                        print(no_result_msg)
                        return (no_result_msg, self.blank_image)
                    else:
                        no_result_msg = "❌ 没有找到符合条件的结果。请检查关键词或文件夹路径。"
                        print(no_result_msg)
                        return (no_result_msg, self.blank_image)
                
                # 初始化顺序读取
                self.sequential_mode = True
                self.sequential_results = results
                self.sequential_index = 0
                self.sequential_folder = folder_path if not search_entire_database else ""
                self.sequential_conditions = current_conditions
                
                print(f"📖 顺序读取模式已初始化，共 {len(results)} 个结果")
            
            # 获取下一个顺序结果
            selected = self.get_sequential_result()
            if not selected:
                no_result_msg = "❌ 顺序读取模式异常，没有可用结果"
                print(no_result_msg)
                return (no_result_msg, self.blank_image)
            
            print(f"📖 顺序读取模式: 第 {self.sequential_index}/{len(self.sequential_results)} 个结果")
            
        else:
            # 随机模式处理（原有逻辑）
            recent_history_set = set(self.history[-history_depth:])
            
            # 检查是否可以使用缓存结果 - 使用新的缓存判断逻辑
            if (self.should_use_cache() and 
                self.last_search_conditions == current_conditions and 
                self.last_search_results is not None):
                print("⚡ 搜索条件未变化且数据库稳定，使用缓存结果直接选择")
                
                # 检查是否只有一张图片符合条件
                if len(self.last_search_results) == 1:
                    print("📌 只有一张图片符合条件，跳过history_depth限制")
                    selected = self.last_search_results[0]
                else:
                    selected = self.ultra_fast_select(self.last_search_results, recent_history_set)
            else:
                # 搜索条件变化或不能使用缓存，重新查询
                if self.is_building:
                    print("🔍 数据库正在构建中，重新查询现有数据...")
                else:
                    print("🔍 搜索条件已变化或缓存无效，重新查询数据库...")
                
                results = self.fast_query(folder_path, keywords, negative_keywords, strict_match, search_entire_database, [])
                
                # 只有在数据库不在构建时才缓存结果
                if not self.is_building:
                    self.last_search_conditions = current_conditions
                    self.last_search_results = results
                    print(f"💾 已缓存搜索结果 ({len(results)} 个)")
                else:
                    print(f"⚠️ 数据库构建中，不缓存搜索结果 ({len(results)} 个)")
                
                print(f"🎯 数据库查询到 {len(results)} 个结果")
                
                # 检查是否只有一张图片符合条件
                if len(results) == 1:
                    print("📌 只有一张图片符合条件，跳过history_depth限制")
                    selected = results[0]
                else:
                    selected = self.ultra_fast_select(results, recent_history_set)
            
            # 如果没有选中结果，清空历史记录重试
            if not selected and self.history:
                print("🔄 没有可用结果，清空历史记录重试...")
                self.history.clear()
                recent_history_set = set()
                
                # 再次检查是否只有一张图片符合条件
                if self.last_search_results and len(self.last_search_results) == 1:
                    print("📌 重试时发现只有一张图片符合条件，跳过history_depth限制")
                    selected = self.last_search_results[0]
                else:
                    selected = self.ultra_fast_select(self.last_search_results, recent_history_set)
            
            # 处理最终结果
            if not selected:
                if not search_entire_database and self.is_building:
                    no_result_msg = "💡 后台正在构建数据库，稍后可能有更多结果。请稍等片刻后重试。"
                    print(no_result_msg)
                    return (no_result_msg, self.blank_image)
                else:
                    no_result_msg = "❌ 没有找到符合条件的结果。请检查关键词或文件夹路径。"
                    print(no_result_msg)
                    return (no_result_msg, self.blank_image)
            
            # 随机模式下更新历史记录
            txt_path = selected[0]
            
            # 如果只有一张图片符合条件，不更新历史记录，确保下次仍然能选择到它
            if self.last_search_results and len(self.last_search_results) == 1:
                print("📌 只有一张图片，不更新历史记录以确保可重复选择")
            else:
                self.history.append(txt_path)
                if len(self.history) > history_depth * 2:
                    self.history = self.history[-history_depth:]
        
        # 处理选中的结果
        txt_path, content, image_path = selected
        
        if not sequential_read:
            if search_entire_database:
                print(f"📄 选中文件: {txt_path}")
        
        image_tensor = self.blank_image
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(img_array)[None,]
            except Exception as e:
                print(f"⚠️ 图片加载错误 [{image_path}]: {str(e)}，使用空白图片")
                # 图片加载失败时保持使用空白图片
        else:
            # 这种情况理论上不应该发生，因为数据库中的记录保证了图片存在
            print(f"⚠️ 图片文件不存在 [{image_path}]，使用空白图片")
        
        return (content, image_tensor)
    
    def __del__(self):
        self.stop_background_building()
        # 清理顺序读取模式
        if hasattr(self, 'sequential_mode') and self.sequential_mode:
            self.reset_sequential_mode("对象销毁")
        if self.conn:
            with self.db_lock:
                self.conn.close()

class ImageBatchReader:
    def __init__(self):
        self.blank_image = self.create_blank_image()
        
        # 顺序读取相关属性
        self.sequential_mode = False
        self.sequential_results = []
        self.sequential_index = 0
        self.sequential_folder = ""
        self.last_folder = ""
        self.last_extensions = ""
        
        print("🖼️ 图片批量读取器已初始化")
    
    def create_blank_image(self):
        """创建空白图片"""
        blank = Image.new("RGB", (512, 512), (0, 0, 0))
        blank_array = np.array(blank).astype(np.float32) / 255.0
        return torch.from_numpy(blank_array)[None,]
    
    def load_image_safe(self, image_path):
        """安全加载图片"""
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_array)[None,]
        except Exception as e:
            print(f"⚠️ 图片加载错误 [{image_path}]: {str(e)}，使用空白图片")
            return self.blank_image
    
    def reset_sequential_mode(self, reason=""):
        """重置顺序读取模式"""
        if reason:
            print(f"🔄 重置顺序读取模式: {reason}")
        self.sequential_mode = False
        self.sequential_results = []
        self.sequential_index = 0
        self.sequential_folder = ""
    
    def should_reset_sequential_mode(self, folder_path, extensions):
        """判断是否需要重置顺序读取模式"""
        # 如果文件夹变化，需要重置
        if self.sequential_folder != folder_path:
            return True
        
        # 如果扩展名变化，需要重置
        if self.last_extensions != extensions:
            return True
        
        return False
    
    def get_image_files(self, folder_path, extensions):
        """获取文件夹中所有支持的图片文件"""
        if not os.path.exists(folder_path):
            return []
        
        image_files = []
        ext_list = [ext.strip().lower() for ext in extensions.split(',')]
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(f'.{ext}') for ext in ext_list):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)  # 排序确保顺序一致
    
    def get_sequential_batch(self, batch_size):
        """获取下一批顺序结果"""
        if not self.sequential_results:
            return []
        
        batch = []
        for _ in range(batch_size):
            if self.sequential_index >= len(self.sequential_results):
                # 到达末尾，循环回到开始
                self.sequential_index = 0
                print(f"🔄 顺序读取已到达末尾，重新开始循环 (共 {len(self.sequential_results)} 个图片)")
            
            if self.sequential_results:  # 确保列表不为空
                batch.append(self.sequential_results[self.sequential_index])
                self.sequential_index += 1
        
        return batch
    
    def get_random_batch(self, image_files, batch_size, seed):
        """获取随机批次"""
        if not image_files:
            return []
        
        # 设置随机种子
        random.seed(seed)
        
        # 如果批次大小大于或等于总文件数，返回打乱后的所有文件
        if batch_size >= len(image_files):
            shuffled_files = image_files.copy()
            random.shuffle(shuffled_files)
            return shuffled_files
        
        return random.sample(image_files, batch_size)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 50}),
                "image_extensions": ("STRING", {"default": "png,jpg,jpeg,webp,bmp,gif,tiff"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "sequential_read": ("BOOLEAN", {"default": False}),
                "shuffle_on_reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "process"
    CATEGORY = "hhy"

    def process(self, folder_path, batch_size, image_extensions, seed, sequential_read=False, shuffle_on_reset=False):
        
        # 输入验证
        if not folder_path or not folder_path.strip():
            error_msg = "❌ 请指定图片文件夹路径"
            print(error_msg)
            return ([self.blank_image],)
        
        folder_path = os.path.normpath(folder_path.strip())
        
        if not os.path.exists(folder_path):
            error_msg = f"❌ 路径不存在: {folder_path}"
            print(error_msg)
            return ([self.blank_image],)
        
        if not os.path.isdir(folder_path):
            error_msg = f"❌ 路径不是文件夹: {folder_path}"
            print(error_msg)
            return ([self.blank_image],)
        
        # 获取所有图片文件
        print(f"📁 扫描文件夹: {folder_path}")
        image_files = self.get_image_files(folder_path, image_extensions)
        
        if not image_files:
            error_msg = f"❌ 文件夹中没有找到支持的图片文件 (支持格式: {image_extensions})"
            print(error_msg)
            return ([self.blank_image],)
        
        print(f"🖼️ 找到 {len(image_files)} 个图片文件")
        
        # 限制批次大小不超过可用图片数量
        original_batch_size = batch_size
        max_available = len(image_files)
        batch_size = min(batch_size, max_available)
        
        if original_batch_size > max_available:
            print(f"⚠️ 批次大小 ({original_batch_size}) 超过可用图片数量 ({max_available})，已调整为 {batch_size}")
        
        # 顺序读取模式处理
        if sequential_read:
            print("📖 顺序读取模式已启用")
            
            # 检查是否需要重置顺序读取模式
            if self.should_reset_sequential_mode(folder_path, image_extensions):
                self.reset_sequential_mode("文件夹或扩展名变化")
            
            # 如果顺序读取模式未激活，需要初始化
            if not self.sequential_mode:
                print("📖 初始化顺序读取模式...")
                
                # 初始化顺序读取
                self.sequential_mode = True
                self.sequential_results = image_files.copy()
                
                # 如果启用了重置时打乱，则打乱文件列表
                if shuffle_on_reset:
                    random.seed(seed)
                    random.shuffle(self.sequential_results)
                    print("🎲 已打乱文件顺序")
                
                self.sequential_index = 0
                self.sequential_folder = folder_path
                self.last_extensions = image_extensions
                
                print(f"📖 顺序读取模式已初始化，共 {len(self.sequential_results)} 个图片")
            
            # 获取下一批顺序结果
            selected_files = self.get_sequential_batch(batch_size)
            current_start = self.sequential_index - len(selected_files)
            if current_start < 0:
                current_start = len(self.sequential_results) + current_start
            
            print(f"📖 顺序读取: 第 {current_start + 1}-{current_start + len(selected_files)} 个图片 (共 {len(self.sequential_results)} 个)")
            
        else:
            # 随机模式处理
            print("🎲 随机读取模式")
            selected_files = self.get_random_batch(image_files, batch_size, seed)
            print(f"🎯 随机选择了 {len(selected_files)} 个图片")
        
        # 加载选中的图片
        image_tensors = []
        for i, image_path in enumerate(selected_files):
            print(f"📄 加载图片 {i+1}/{len(selected_files)}: {os.path.basename(image_path)}")
            image_tensor = self.load_image_safe(image_path)
            image_tensors.append(image_tensor)
        
        print(f"✅ 成功加载 {len(image_tensors)} 个图片")
        
        return (image_tensors,)

NODE_CLASS_MAPPINGS = {
    "TextWithImageReader": TextWithImageReader,
    "ImageBatchReader": ImageBatchReader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextWithImageReader": "Text-Image Pair Reader",
    "ImageBatchReader": "Image Batch Reader"
}