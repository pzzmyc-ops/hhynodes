"""
Load Audio from URL Node
从 URL 下载音频文件并加载
"""

import os
import urllib.request
import urllib.parse
import torch
import logging
import subprocess
import re

import folder_paths


# 配置日志
logger = logging.getLogger("LoadAudioFromURL")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# 支持的音频格式
audio_extensions = ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'opus', 'wma', 'aiff', 'ape']

# 音频编码参数
ENCODE_ARGS = ("utf-8", 'backslashreplace')


# 检测ffmpeg路径
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

ffmpeg_path = get_ffmpeg_path()


def download_audio_from_url(url: str, output_dir: str = None) -> str:
    """从 URL 下载音频到临时目录"""
    if output_dir is None:
        output_dir = folder_paths.get_temp_directory()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 从 URL 解析文件名
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # 如果没有文件名或文件名无效，使用默认名称
    if not filename or '.' not in filename:
        filename = "downloaded_audio.mp3"
    
    # 确保文件扩展名是音频格式
    file_ext = filename.split('.')[-1].lower()
    if file_ext not in audio_extensions:
        filename += ".mp3"
    
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"开始下载音频: {url}")
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


def load_audio_from_file(audio_path):
    """从音频文件中加载音频数据"""
    if ffmpeg_path is None:
        raise RuntimeError("未找到ffmpeg，无法加载音频。请安装ffmpeg或imageio-ffmpeg")
    
    args = [ffmpeg_path, "-i", audio_path]
    
    try:
        # 提取音频为32位浮点格式
        res = subprocess.run(args + ["-f", "f32le", "-"],
                            capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        
        # 从stderr中解析音频信息
        stderr_text = res.stderr.decode(*ENCODE_ARGS)
        
        # 解析采样率和声道信息
        match = re.search(r', (\d+) Hz, (\w+)', stderr_text)
        
        if match:
            sample_rate = int(match.group(1))
            # 处理声道数
            channel_type = match.group(2)
            if channel_type == "mono":
                channels = 1
            elif channel_type == "stereo":
                channels = 2
            else:
                # 尝试解析更多声道类型
                if "5.1" in channel_type:
                    channels = 6
                elif "7.1" in channel_type:
                    channels = 8
                else:
                    logger.warning(f"未知声道类型: {channel_type}，尝试推断...")
                    # 尝试从数据推断
                    channels = 2
        else:
            # 使用默认值
            logger.warning("无法从ffmpeg输出解析音频信息，使用默认值")
            sample_rate = 44100
            channels = 2
        
        # 重塑音频数据
        if len(audio) == 0:
            raise RuntimeError("音频文件为空或无法读取")
        
        # 确保数据长度能被声道数整除
        remainder = len(audio) % channels
        if remainder != 0:
            audio = audio[:-remainder]
        
        if len(audio) == 0:
            raise RuntimeError("音频数据处理后为空")
        
        # 重塑为 (batch, channels, samples)
        audio = audio.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
        
        duration_seconds = audio.shape[2] / sample_rate
        
        logger.info(f"音频信息: {sample_rate} Hz, {channels} 声道, {duration_seconds:.2f}s")
        
        return {
            'waveform': audio,
            'sample_rate': sample_rate
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode(*ENCODE_ARGS) if e.stderr else str(e)
        logger.error(f"音频加载失败: {error_msg}")
        raise RuntimeError(f"音频加载失败: {error_msg}")
    except Exception as e:
        logger.error(f"音频加载出错: {e}")
        raise


def load_audio_from_url(url):
    """从 URL 加载音频"""
    
    # 下载音频
    audio_path = download_audio_from_url(url)
    
    try:
        # 加载音频
        audio_data = load_audio_from_file(audio_path)
        
        if audio_data is None:
            raise RuntimeError("音频加载失败")
        
        # 创建处理信息
        audio_info = f"音频信息:\n" \
                    f"采样率: {audio_data['sample_rate']} Hz\n" \
                    f"声道数: {audio_data['waveform'].shape[1]}\n" \
                    f"时长: {audio_data['waveform'].shape[2] / audio_data['sample_rate']:.2f}s"
        
        logger.info(f"\n--- 音频加载完成 ---")
        logger.info(audio_info)
        logger.info(f"-------------------\n")
        
        return (audio_data, audio_info)
        
    except Exception as e:
        logger.error(f"加载音频失败: {e}")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"已清理临时文件: {audio_path}")
            except:
                pass


class LoadAudioFromURL:
    """
    从 URL 加载音频节点
    
    参数说明：
    - url: 音频文件的URL地址
    
    输出：
    - audio: 音频数据
    - info: 音频信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://example.com/audio.mp3"
                }),
            },
        }

    CATEGORY = "hhy/oss"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "load_audio"

    def load_audio(self, url):
        """加载音频的主方法"""
        
        if not url or not isinstance(url, str):
            raise ValueError("URL 不能为空")
        
        # 验证 URL 格式
        if not url.startswith(('http://', 'https://', 'ftp://')):
            raise ValueError("URL 必须以 http://, https:// 或 ftp:// 开头")
        
        logger.info(f"开始从 URL 加载音频: {url}")
        
        return load_audio_from_url(url=url)

    @classmethod
    def IS_CHANGED(cls, url, **kwargs):
        # URL 改变时重新加载
        return url


# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoadAudioFromURL": LoadAudioFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioFromURL": "Load Audio from URL 🔊",
}

