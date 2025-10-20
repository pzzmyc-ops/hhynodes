"""
Load Video from URL Node
从 URL 下载视频并加载帧，支持帧插值/减少
"""

import os
import urllib.request
import urllib.parse
import numpy as np
import torch
import cv2
import logging
import hashlib

import folder_paths


# 配置日志
logger = logging.getLogger("LoadVideoFromURL")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov', 'avi', 'flv', 'wmv']


def download_video_from_url(url: str, output_dir: str = None) -> str:
    """从 URL 下载视频到临时目录"""
    if output_dir is None:
        output_dir = folder_paths.get_temp_directory()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 从 URL 解析文件名
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # 如果没有文件名或文件名无效，使用默认名称
    if not filename or '.' not in filename:
        filename = "downloaded_video.mp4"
    
    # 确保文件扩展名是视频格式
    file_ext = filename.split('.')[-1].lower()
    if file_ext not in video_extensions:
        filename += ".mp4"
    
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"开始下载视频: {url}")
    logger.info(f"目标路径: {output_path}")
    
    try:
        with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
            file_size = int(response.headers.get('content-length', 0))
            if file_size > 0:
                logger.info(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
            
            # 读取并写入数据
            chunk_size = 8192
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                
                # 显示进度
                if file_size > 0 and downloaded % (chunk_size * 100) == 0:
                    progress = (downloaded / file_size) * 100
                    logger.info(f"下载进度: {progress:.1f}%")
        
        actual_size = os.path.getsize(output_path)
        logger.info(f"下载完成: {output_path} ({actual_size / (1024 * 1024):.2f} MB)")
        return output_path
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def load_all_frames_from_video(video_path):
    """从视频加载所有帧"""
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise ValueError(f"{video_path} 无法使用 cv2 加载")

    # 提取视频元数据
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if width <= 0 or height <= 0:
        ret, frame = video_cap.read()
        if ret:
            height, width = frame.shape[:2]
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            raise ValueError(f"无法读取视频帧: {video_path}")

    logger.info(f"视频信息: {width}x{height}, {fps:.2f} fps, {total_frames} 帧, {duration:.2f}s")

    # 加载所有帧
    frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        
        # 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为 float32 并归一化到 [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frames.append(frame_normalized)

    video_cap.release()
    
    if not frames:
        raise RuntimeError(f"未能从视频加载任何帧: {video_path}")
    
    logger.info(f"完成加载 {len(frames)} 帧")
    
    # 转换为 torch tensor
    images = torch.from_numpy(np.stack(frames))
    
    return images, fps, width, height, total_frames, duration


def interpolate_frames(images, target_frames):
    """
    帧插值 - 增加帧数
    使用线性插值
    """
    input_frames = images.shape[0]
    
    if target_frames <= input_frames:
        return images
    
    output_frames = []
    
    for i in range(target_frames):
        # 计算在原始序列中的位置
        pos = i * (input_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
        
        # 获取要插值的两帧
        frame_idx = int(pos)
        next_frame_idx = min(frame_idx + 1, input_frames - 1)
        
        # 计算插值权重
        weight = pos - frame_idx
        
        if frame_idx == next_frame_idx or weight == 0:
            interpolated_frame = images[frame_idx]
        else:
            frame1 = images[frame_idx]
            frame2 = images[next_frame_idx]
            interpolated_frame = (1 - weight) * frame1 + weight * frame2
        
        output_frames.append(interpolated_frame)
    
    return torch.stack(output_frames)


def reduce_frames(images, target_frames):
    """
    帧减少 - 均匀采样
    """
    input_frames = images.shape[0]
    
    if target_frames >= input_frames:
        return images
    
    indices = []
    for i in range(target_frames):
        idx = int(i * input_frames / target_frames)
        idx = min(idx, input_frames - 1)
        indices.append(idx)
    
    # 去重并保持顺序
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    
    # 如果去重后不够，均匀采样
    if len(unique_indices) < target_frames:
        step = max(1, input_frames // target_frames)
        unique_indices = list(range(0, input_frames, step))[:target_frames]
        
        while len(unique_indices) < target_frames:
            unique_indices.append(input_frames - 1)
    
    unique_indices = unique_indices[:target_frames]
    selected_frames = [images[idx] for idx in unique_indices]
    
    return torch.stack(selected_frames)


def process_frames(images, force_output_frames):
    """
    处理帧数 - 插值或减少
    """
    input_frame_count = images.shape[0]
    
    if input_frame_count == 0:
        raise ValueError("没有输入图像")
    
    if force_output_frames == input_frame_count:
        processed_images = images
        process_mode = "unchanged"
    elif force_output_frames > input_frame_count:
        processed_images = interpolate_frames(images, force_output_frames)
        process_mode = "interpolated"
    else:
        processed_images = reduce_frames(images, force_output_frames)
        process_mode = "reduced"
    
    logger.info(f"帧处理: {input_frame_count} -> {force_output_frames} ({process_mode})")
    
    return processed_images, process_mode


def load_video_from_url(url, force_rate, force_output_frames):
    """从 URL 加载视频并处理帧"""
    
    # 下载视频
    video_path = download_video_from_url(url)
    
    try:
        # 加载所有原始帧
        images, original_fps, width, height, total_frames, duration = load_all_frames_from_video(video_path)
        
        # 确定输出帧数
        if force_output_frames > 0:
            # 使用指定的帧数（插值或减少）
            processed_images, process_mode = process_frames(images, force_output_frames)
        else:
            # 使用原始帧数（不处理）
            processed_images = images
            process_mode = "original"
        
        # 确定输出帧率
        if force_rate > 0:
            output_fps = force_rate
        else:
            output_fps = original_fps
        
        # 创建处理信息
        if force_rate == 0 and force_output_frames == 0:
            # 两者都为 0，只是拆帧
            process_info = f"模式: 视频拆帧（无处理）\n" \
                          f"帧数: {total_frames} 帧\n" \
                          f"帧率: {original_fps:.2f} fps\n" \
                          f"分辨率: {width}x{height}\n" \
                          f"时长: {duration:.2f}s"
        else:
            process_info = f"原始: {total_frames} 帧 @ {original_fps:.2f} fps\n" \
                          f"输出: {len(processed_images)} 帧 @ {output_fps:.2f} fps\n" \
                          f"处理模式: {process_mode}\n" \
                          f"分辨率: {width}x{height}"
        
        logger.info(f"\n--- 视频处理完成 ---")
        logger.info(process_info)
        logger.info(f"-------------------\n")
        
        return (processed_images, len(processed_images), output_fps, process_info)
        
    except Exception as e:
        logger.error(f"加载视频失败: {e}")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"已清理临时文件: {video_path}")
            except:
                pass


class LoadVideoFromURL:
    """
    从 URL 加载视频节点，支持帧插值/减少
    
    参数说明：
    - force_rate = 0: 使用视频原始帧率
    - force_output_frames = 0: 使用视频原始帧数（不做插值/减少）
    - 两者都为 0: 只做视频拆帧，保持原始参数
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://example.com/video.mp4"
                }),
                "force_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "输出帧率，0 = 使用视频原始帧率"
                }),
                "force_output_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "输出帧数，0 = 使用视频原始帧数（不插值/减少）"
                }),
            },
        }

    CATEGORY = "hhy/video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "fps", "process_info")
    FUNCTION = "load_video"

    def load_video(self, url, force_rate, force_output_frames):
        """加载视频的主方法"""
        
        if not url or not isinstance(url, str):
            raise ValueError("URL 不能为空")
        
        # 验证 URL 格式
        if not url.startswith(('http://', 'https://', 'ftp://')):
            raise ValueError("URL 必须以 http://, https:// 或 ftp:// 开头")
        
        logger.info(f"开始从 URL 加载视频: {url}")
        
        return load_video_from_url(
            url=url,
            force_rate=force_rate,
            force_output_frames=force_output_frames
        )

    @classmethod
    def IS_CHANGED(cls, url, force_rate, force_output_frames, **kwargs):
        # URL 或参数改变时重新加载
        return f"{url}_{force_rate}_{force_output_frames}"


# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoadVideoFromURL": LoadVideoFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFromURL": "Load Video from URL 🌐",
}

