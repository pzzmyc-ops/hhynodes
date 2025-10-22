"""
Images to Video Node
简化版图片到视频转换节点，支持H264编码和MP4格式输出
"""

import os
import subprocess
import numpy as np
import torch
import cv2
import logging
from datetime import datetime
from PIL import Image
import folder_paths

# 配置日志
logger = logging.getLogger("ImagesToVideo")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def tensor_to_bytes(image):
    """将tensor转换为字节数组"""
    if isinstance(image, torch.Tensor):
        # 确保tensor在CPU上并且是float类型
        image = image.cpu().float()
        # 转换为numpy数组并确保值在0-255范围内
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image.numpy(), 0, 255).astype(np.uint8)
    else:
        image = np.array(image)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 如果是3D数组，取第一个元素
    if len(image.shape) == 4:
        image = image[0]
    
    return image

def get_ffmpeg_path():
    """获取ffmpeg可执行文件路径"""
    import shutil
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except:
        # 尝试系统路径
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        # 尝试当前目录
        if os.path.isfile("ffmpeg"):
            return os.path.abspath("ffmpeg")
        if os.path.isfile("ffmpeg.exe"):
            return os.path.abspath("ffmpeg.exe")
        return None

class ImagesToVideoNode:
    def __init__(self):
        self.ffmpeg_path = get_ffmpeg_path()
        if self.ffmpeg_path is None:
            logger.warning("ffmpeg not found. Video output will not be available.")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "images_to_video"
    CATEGORY = "hhy/video"
    OUTPUT_NODE = True

    def images_to_video(self, images, frame_rate, audio=None):
        """
        将图片序列转换为视频
        
        Args:
            images: 图片tensor列表
            frame_rate: 帧率
            audio: 可选的音频文件路径
            
        Returns:
            video_path: 生成的视频文件路径
        """
        if self.ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to use video output.")
        
        if not images or len(images) == 0:
            raise ValueError("No images provided")
        
        # 生成基于当前时间的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"video_{timestamp}"
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, _, _, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        # 确保输出目录存在
        os.makedirs(full_output_folder, exist_ok=True)
        
        # 生成输出文件路径
        output_file = f"{filename}.mp4"
        file_path = os.path.join(full_output_folder, output_file)
        
        try:
            # 获取第一张图片的尺寸
            first_image = tensor_to_bytes(images[0])
            height, width = first_image.shape[:2]
            
            logger.info(f"Converting {len(images)} images to video: {file_path}")
            logger.info(f"Resolution: {width}x{height}, Frame rate: {frame_rate}")
            
            # 构建ffmpeg命令
            cmd = [
                self.ffmpeg_path,
                "-y",  # 覆盖输出文件
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(frame_rate),
                "-i", "-",  # 从stdin读取
                "-c:v", "libx264",  # H264编码
                "-preset", "medium",  # 编码预设
                "-crf", "23",  # 质量参数
                "-pix_fmt", "yuv420p",  # 像素格式
            ]
            
            # 如果有音频，添加音频处理
            if audio is not None:
                cmd.extend([
                    "-i", audio,
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-shortest"  # 以最短的流为准
                ])
            
            cmd.append(file_path)
            
            # 启动ffmpeg进程
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 逐帧写入图片数据
            for i, image_tensor in enumerate(images):
                try:
                    image_bytes = tensor_to_bytes(image_tensor)
                    # 确保图片尺寸一致
                    if image_bytes.shape[:2] != (height, width):
                        image_bytes = cv2.resize(image_bytes, (width, height))
                    
                    # 写入帧数据
                    process.stdin.write(image_bytes.tobytes())
                    process.stdin.flush()
                    
                    if i % 10 == 0:  # 每10帧打印一次进度
                        logger.info(f"Processed {i+1}/{len(images)} frames")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    continue
            
            # 关闭stdin
            process.stdin.close()
            
            # 等待进程完成
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"ffmpeg failed with return code {process.returncode}: {error_msg}")
            
            logger.info(f"Video successfully created: {file_path}")
            
            # 检查文件是否真的创建了
            if not os.path.exists(file_path):
                raise RuntimeError("Video file was not created")
            
            return (file_path,)
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise RuntimeError(f"Failed to create video: {e}")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImagesToVideoNode": ImagesToVideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagesToVideoNode": "Images to Video (H264/MP4)"
}
