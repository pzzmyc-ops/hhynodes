import os
import random
import torch
from PIL import Image
import numpy as np

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_image_files(folder_path, extensions):
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
        image_files = get_image_files(folder_path, image_extensions)
        
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
    "ImageBatchReader": ImageBatchReader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchReader": "Image Batch Reader"
} 