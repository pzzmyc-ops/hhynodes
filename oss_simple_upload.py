import os
import json
import base64
import time
import hmac
import hashlib
import tempfile
import numpy as np
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from urllib.parse import urlparse, quote, quote_plus
from typing import List, Optional, Union, Dict, Any, Set
from fractions import Fraction
import mimetypes
import secrets
import atexit

try:
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_v1_5
    from Crypto import Random
except ImportError:
    from Cryptodome.PublicKey import RSA
    from Cryptodome.Cipher import PKCS1_v1_5
    from Cryptodome import Random

import alibabacloud_oss_v2 as oss
from tqdm import tqdm

# =============================================================================
# 🛡️ 安全配置 - 严格限制上传目录和文件类型
# =============================================================================

# 仅允许的上传目录（绝对路径）
ALLOWED_UPLOAD_DIRS: List[str] = [
    "/workspace/ComfyUI/temp",
    "/data/ComfyUI/personal/output",
    "/tmp"
]

# 允许的文件类型白名单
ALLOWED_EXTENSIONS: Set[str] = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff',  # 图片
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v',            # 视频  
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',            # 音频
    '.txt', '.json', '.csv', '.md',                             # 文本
    '.pdf', '.doc', '.docx', '.zip',                            # 文档
    '.npz', '.npy'                                              # 安全的numpy格式（替代pickle）
}

# 危险文件类型黑名单 - 绝对禁止（包括pickle！）
DANGEROUS_EXTENSIONS: Set[str] = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.msi', '.app',     # 可执行文件
    '.sh', '.bash', '.zsh', '.ps1', '.vbs',                     # 脚本
    '.py', '.rb', '.pl', '.php', '.jsp', '.js', '.ts',          # 代码文件
    '.pkl', '.pickle',                                          # ⚠️ 危险序列化文件
    '.dll', '.so', '.dylib',                                    # 动态链接库
    '.jar', '.war', '.class'                                    # Java文件
}

# 文件大小限制（字节）
MAX_FILE_SIZES = {
    'image': 50 * 1024 * 1024,   # 50MB
    'video': 500 * 1024 * 1024,  # 500MB  
    'audio': 100 * 1024 * 1024,  # 100MB
    'text': 10 * 1024 * 1024,    # 10MB
    'document': 100 * 1024 * 1024, # 100MB
    'default': 50 * 1024 * 1024   # 50MB
}

# =============================================================================
# 🛡️ 安全验证函数
# =============================================================================

class SecurityError(Exception):
    """安全相关错误"""
    pass

class SecureTempFileManager:
    """安全的临时文件管理器"""
    def __init__(self):
        self.temp_files: List[str] = []
        atexit.register(self.cleanup_all)
    
    def create_temp_file(self, prefix: str, suffix: str) -> str:
        """创建安全的临时文件"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=f"secure_{prefix}_",
                suffix=suffix,
                dir=tempfile.gettempdir()
            )
            os.close(temp_fd)  # 立即关闭文件描述符
            
            # 设置安全权限（仅当前用户可读写）
            try:
                os.chmod(temp_path, 0o600)
            except:
                pass  # Windows可能不支持chmod
            
            self.temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            print(f"🚨 创建临时文件失败: {e}")
            raise SecurityError(f"无法创建安全临时文件: {e}")
    
    def cleanup_file(self, file_path: str):
        """安全删除临时文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if file_path in self.temp_files:
                self.temp_files.remove(file_path)
        except Exception as e:
            print(f"🚨 清理临时文件失败: {e}")
    
    def cleanup_all(self):
        """清理所有临时文件"""
        for file_path in self.temp_files.copy():
            self.cleanup_file(file_path)

# 全局临时文件管理器
temp_manager = SecureTempFileManager()

def validate_file_path(file_path: str) -> bool:
    """验证文件路径是否在允许的目录内"""
    try:
        abs_path = os.path.abspath(os.path.normpath(file_path))
        for allowed_dir in ALLOWED_UPLOAD_DIRS:
            allowed_abs = os.path.abspath(os.path.normpath(allowed_dir))
            if abs_path.startswith(allowed_abs + os.sep) or abs_path == allowed_abs:
                return True
        print(f"🚨 安全警告: 路径被拒绝 - {abs_path}")
        return False
    except Exception as e:
        print(f"🚨 路径验证错误: {e}")
        return False

def validate_file_type(file_path: str) -> bool:
    """验证文件类型是否安全"""
    try:
        _, ext = os.path.splitext(file_path.lower())
        if ext in DANGEROUS_EXTENSIONS:
            print(f"🚨 安全警告: 危险文件类型被拒绝 - {ext}")
            return False
        if ext not in ALLOWED_EXTENSIONS:
            print(f"🚨 安全警告: 不支持的文件类型 - {ext}")
            return False
        return True
    except Exception as e:
        print(f"🚨 文件类型验证错误: {e}")
        return False

def validate_file_size(file_path: str) -> bool:
    """验证文件大小是否在限制内"""
    try:
        if not os.path.exists(file_path):
            return False
        file_size = os.path.getsize(file_path)
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}:
            max_size = MAX_FILE_SIZES['image']
        elif ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}:
            max_size = MAX_FILE_SIZES['video']
        elif ext in {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}:
            max_size = MAX_FILE_SIZES['audio']
        elif ext in {'.txt', '.json', '.csv', '.md'}:
            max_size = MAX_FILE_SIZES['text']
        elif ext in {'.pdf', '.doc', '.docx', '.zip'}:
            max_size = MAX_FILE_SIZES['document']
        else:
            max_size = MAX_FILE_SIZES['default']
        
        if file_size > max_size:
            print(f"🚨 安全警告: 文件过大 - {file_size} bytes (限制: {max_size} bytes)")
            return False
        return True
    except Exception as e:
        print(f"🚨 文件大小验证错误: {e}")
        return False

def perform_security_validation(file_path: str) -> None:
    """执行完整的安全验证"""
    print(f"🔒 开始安全验证: {file_path}")
    
    if not os.path.exists(file_path):
        raise SecurityError(f"文件不存在: {file_path}")
    
    if not validate_file_path(file_path):
        raise SecurityError(f"文件路径不在允许范围内: {file_path}")
    
    if not validate_file_type(file_path):
        raise SecurityError(f"不支持的文件类型: {file_path}")
    
    if not validate_file_size(file_path):
        raise SecurityError(f"文件大小超出限制: {file_path}")
    
    print(f"✅ 安全验证通过: {file_path}")

# 尝试导入相关模块
try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, image support disabled")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("soundfile not available, audio support disabled")

try:
    from comfy.comfy_types import IO
    from comfy.comfy_types.node_typing import IO as NodeIO
    from comfy_api.latest import Input, InputImpl
    COMFY_TYPES_AVAILABLE = True
except ImportError:
    try:
        from comfy.comfy_types.node_typing import IO as NodeIO
        COMFY_TYPES_AVAILABLE = True
    except ImportError:
        COMFY_TYPES_AVAILABLE = False
        print("ComfyUI types not available, using fallback")

# 数据类型定义
class Audio:
    def __init__(self, sample_rate: int, waveform: torch.Tensor):
        self.sample_rate = sample_rate
        self.waveform = waveform

# 辅助函数
def save_image_to_temp(img_tensor: torch.Tensor, filename_prefix: str = "image", format: str = "png") -> str:
    """🛡️ 安全地保存图片张量到临时文件"""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image support")
    
    if len(img_tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor (H, W, C), got {img_tensor.shape}")
    
    # 转换张量格式
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    if img_tensor.shape[0] not in (3, 4):
        raise ValueError(f"Image must have 3 or 4 channels, got {img_tensor.shape[0]}")
    
    img_tensor = img_tensor.clamp(0, 1)
    img = F.to_pil_image(img_tensor)
    
    # 🛡️ 使用安全的临时文件管理器
    temp_file = temp_manager.create_temp_file(filename_prefix, f".{format}")
    img.save(temp_file)
    
    return temp_file

def save_audio_to_temp(audio_data: Dict[str, Any], filename_prefix: str = "audio", format: str = "wav") -> str:
    """🛡️ 安全地保存音频数据到临时文件"""
    if not SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required for audio support")
    
    sample_rate = audio_data.get("sample_rate")
    waveform = audio_data.get("waveform")
    
    if not sample_rate or waveform is None:
        raise ValueError("Audio data must contain 'sample_rate' and 'waveform'")
    
    # 处理waveform格式
    if len(waveform.shape) == 3:
        if waveform.shape[0] > 1:
            raise ValueError("Audio batch size must be 1")
        waveform = waveform[0]
    elif len(waveform.shape) == 2:
        pass
    elif len(waveform.shape) == 1:
        waveform = torch.unsqueeze(waveform, 0)
    else:
        raise ValueError(f"Invalid waveform shape: {waveform.shape}")
    
    # 🛡️ 使用安全的临时文件管理器
    temp_file = temp_manager.create_temp_file(filename_prefix, f".{format}")
    
    # 保存音频
    subtype = "FLOAT" if format.lower() == "wav" else None
    sf.write(temp_file, waveform.T.numpy(), sample_rate, subtype=subtype)
    
    return temp_file

def save_video_to_temp(video_input, filename_prefix: str = "video", format: str = "mp4") -> str:
    """🛡️ 安全地保存视频到临时文件"""
    if not COMFY_TYPES_AVAILABLE:
        raise ImportError("ComfyUI types are required for video support")
    
    # 🛡️ 使用安全的临时文件管理器
    temp_file = temp_manager.create_temp_file(filename_prefix, f".{format}")
    
    # 使用视频对象的save方法
    video_input.save_to(temp_file, format=format)
    
    return temp_file

def save_text_to_temp(text: str, filename_prefix: str = "text", format: str = "txt") -> str:
    """🛡️ 安全地保存文本到临时文件"""
    # 🛡️ 使用安全的临时文件管理器
    temp_file = temp_manager.create_temp_file(filename_prefix, f".{format}")
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return temp_file

hhy_key = """-----BEGIN PRIVATE KEY-----
MIIEugIBADANBgkqhkiG9w0BAQEFAASCBKQwggSgAgEAAoIBAQDJAdHcxa0Lu1OY
CS18Mb0UHD6OA4UM+0e0bcYHbxHhDqZllI2y98R53ImN9FiYxGzCF/oh+KF2LhJ+
FevsaRkH5BCBt18KHzSYSg342vRz6xnfwvQwexVRDEpF03kpdD6cmMv7dr4HpYjW
moxeWrF3ctuaeGMCCQ5gZcMZCmrCGfLPoVuyiyl/qe2B+ZsN9o0TZ03BrRgspyYA
GS/rB68trPCTiz2y25jhYfNgnPJzlube2pTGF8kixkwvTFAwkOYwfq+cnJuAl/mN
i1CTrwaDhnJgofzo8g4rpAGxgMGCZjHoAMNB5eyug/REDGBhBL6nfGZugT0ahIgc
uaWTpHMBAgMBAAECggEAB25TGqYe7aqqJmor8Z4B3UQGSsAd74b8XZLrlc+lDcC5
tXKIisXKwF3PIsWmTxb/ZY1G/rGjL/QX9Ev1i+jOf2Z+tvnvSDezPUBLHBClpgGq
dCWQGxj4xrVVa1eMIJSC4k2KiHhJQg18RaFC6FPSYN3o2REDaYJJ0xXeAJ9sRxLH
W8ljrrgFiFk4NeqPXvVoB+h+H/L4LFag93AjzOORcvQzSE09hGIMqBbwNvSRAslO
s9gyE+TLB2cZXCstrBR4MmCvDh6hE9By9IHTfoDw105SjSaNuMa4ZusOfkuW7B/r
ad0Y64FN/VnDssIzIdXwlr4FELwW50ORs0yePPefMQKBgQDi4xCyyoVGFZWG19gl
q/Cs743MtS3qmt+Y4DLNEzmAO5V2J6ZRhS6coagDsj70KXCCE5/+lhJrjfe8iu8m
Jq2u3qz7Ksy3REtb35GlRWgD0hA4rslLJB0lhUWE33TLJ+37Osj6BTBYcQfy4dLC
kmqJibjlOSeB6Ww4CM+VUi0I8QKBgQDizKKtsHgitOUeFHv3s8NZWQPdVnoUgGYv
Kfdb/eCm21vIo0Wn39/zNOs95vxU9iXXFapfVuKk0r6XbsoAAQKaASpKPd9t89oM
rQcRfhwTP+CaUAMJ8ugUs/o8kSlLK8blG8C1K5g9DxdzBxbAt9TFrsYUrEOB+lhQ
nriQN4yLEQKBgGdBs53K8XB97jkaDnLGl5f8xen+ItF8fnpSvov6TdcARvso/FZp
aFc8cvyLqH7yRRPN3qi8n9F3IOIb0M7qF21YRh1g0x4s5KcBToWK2tWySlOhqFac
Lu+egY8BK2Qx3erSTBkNN31oo5d0ErkebYH+vbkEk+hZ1TiDOgXZCknhAoGAKQW2
jxASSsTJhG1UFvOu6+RL7KcNodOvp+xBT6RWFBgtO9c8bCb0TPtPaXz0OzHimkrS
7De8+u8bhiyF4QZNwClhytfyJ+Mpl41cb++NiHXPXFoIkq4bCFOdeYMQIwaiDSK9
8ocWHEU0ipvHo8gcdj0smuSluUbc3og2/e7uPuECfwv51oZlwSV/Lw9RlCGYd6aI
7s7JQqKgO5Uuzkj1vqmq38tj8MY9ABsTrpS5vbNYNN2u6q6/vWWJpV7gd/DBvQKS
OKOj9wa2GCJkZBTx/g4pIqRLAX6rDGAH+m1IH2T48NszoUhR3wYnT31/6CcUkTtj
wHn8/YsYZz89sqcqQlw=
-----END PRIVATE KEY-----"""


def __init__(self, encrypted_config: str):
    self.encrypted_config = encrypted_config
    self.client = None
    self.config = None
    self._init_client()

def _decrypt_config(self, encrypted_config_b64: str) -> dict:
    try:
        key = RSA.import_key(hhy_key)
    except Exception as exc:
        raise ValueError(f"导入密钥失败: {exc}")

    cipher = PKCS1_v1_5.new(key)
    encrypted_bytes = base64.b64decode(encrypted_config_b64)
    sentinel = Random.new().read(16)
    decrypted_bytes = cipher.decrypt(encrypted_bytes, sentinel)
    if decrypted_bytes == sentinel:
        raise ValueError("损坏的oss字符串数据")
    try:
        return json.loads(decrypted_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"解密后的JSON解析失败: {exc}")

def _init_client(self):
    try:
        cfg_dict = _decrypt_config(self, self.encrypted_config)
        cfg_upper = {str(k).upper(): v for k, v in cfg_dict.items()}
        
        self.region = cfg_upper.get("REGION")
        self.bucket_name = cfg_upper.get("BUCKET")
        self.endpoint = cfg_upper.get("ENDPOINT")
        access_key_id = cfg_upper.get("ACCESS_KEY_ID")
        access_key_secret = cfg_upper.get("ACCESS_KEY_SECRET")
        security_token = cfg_upper.get("SECURITY_TOKEN")
        
        if not all([self.region, self.bucket_name, access_key_id, access_key_secret]):
            raise ValueError("oss字符串缺少必要字段")
        
        if self.region and self.region.startswith("oss-"):
            self.region = self.region[4:]
        
        if security_token:
            credentials_provider = oss.credentials.StaticCredentialsProvider(
                access_key_id, access_key_secret, security_token
            )
        else:
            credentials_provider = oss.credentials.StaticCredentialsProvider(
                access_key_id, access_key_secret
            )
        
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = self.region
        try:
            if self.endpoint:
                cfg.endpoint = self.endpoint
        except Exception:
            pass

        self.client = oss.Client(cfg)
        
    except Exception as e:
        print(f"OSS客户端初始化失败: {e}")
        raise

def upload_file_with_progress(self, local_file_path, object_name=None):
    # 🛡️ 执行安全验证
    try:
        perform_security_validation(local_file_path)
    except SecurityError as e:
        raise SecurityError(f"🚨 上传被拒绝: {e}")
    
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"本地文件不存在: {local_file_path}")
    
    if object_name is None:
        cfg_dict = _decrypt_config(self, self.encrypted_config)
        cfg_upper = {str(k).upper(): v for k, v in cfg_dict.items()}
        key_from_config = cfg_upper.get("KEY")
        
        folder = "hhy"
        file_name = os.path.basename(local_file_path)
        
        if key_from_config:
            key_has_ext = os.path.splitext(key_from_config)[1] != ""
            if key_has_ext:
                object_name = key_from_config
            else:
                object_name = f"{folder.strip('/')}/{file_name}"
        else:
            object_name = f"{folder.strip('/')}/{file_name}"
    
    file_size = os.path.getsize(local_file_path)

    print(f"开始上传文件: {local_file_path}")
    print(f"目标对象: {object_name}")
    print(f"文件大小: {_format_file_size(self, file_size)}")
    print("-" * 50)

    PART_SIZE_BYTES = 5 * 1024 * 1024
    MAX_PART_RETRIES = 3
    RETRY_SLEEP_SECONDS = 2

    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc='上传进度', ncols=80)

    init_res = self.client.initiate_multipart_upload(
        oss.InitiateMultipartUploadRequest(bucket=self.bucket_name, key=object_name)
    )
    upload_id = init_res.upload_id

    part_number = 1
    upload_parts: List[oss.UploadPart] = []

    try:
        with open(local_file_path, 'rb') as f:
            for start in range(0, file_size, PART_SIZE_BYTES):
                size = min(PART_SIZE_BYTES, file_size - start)
                reader = oss.io_utils.SectionReader(oss.io_utils.ReadAtReader(f), start, size)

                part_last_written = 0

                def progress_fn(*args):
                    nonlocal part_last_written
                    if len(args) >= 3:
                        n = int(args[0])
                        if n > 0:
                            progress_bar.update(n)
                            part_last_written += n
                    elif len(args) == 2:
                        consumed = int(args[0])
                        delta = max(0, consumed - part_last_written)
                        if delta:
                            progress_bar.update(delta)
                            part_last_written = consumed

                attempts = 0
                while True:
                    try:
                        req = oss.UploadPartRequest(
                            bucket=self.bucket_name,
                            key=object_name,
                            upload_id=upload_id,
                            part_number=part_number,
                            body=reader,
                            progress_fn=progress_fn,
                        )
                        up_res = self.client.upload_part(req)
                        upload_parts.append(oss.UploadPart(part_number=part_number, etag=up_res.etag))
                        break
                    except Exception as e:
                        attempts += 1
                        if attempts >= MAX_PART_RETRIES:
                            raise
                        time.sleep(RETRY_SLEEP_SECONDS)

                part_number += 1

        # 2) 完成分片
        parts_sorted = sorted(upload_parts, key=lambda p: p.part_number)
        comp_req = oss.CompleteMultipartUploadRequest(
            bucket=self.bucket_name,
            key=object_name,
            upload_id=upload_id,
            complete_multipart_upload=oss.CompleteMultipartUpload(parts=parts_sorted),
        )
        result = self.client.complete_multipart_upload(comp_req)

        if progress_bar.n < file_size:
            progress_bar.update(file_size - progress_bar.n)
        progress_bar.close()

        print("\n" + "=" * 50)
        print("上传完成！(分片上传)")
        print(f"状态码: {result.status_code}")
        print(f"请求ID: {result.request_id}")
        print(f"ETag: {result.etag}")
        print(f"CRC64校验码: {result.hash_crc64}")

        presigned_url = _generate_presigned_url(self, object_name)

        return {
            'status_code': result.status_code,
            'request_id': result.request_id,
            'content_md5': getattr(result, 'content_md5', ''),
            'etag': result.etag,
            'hash_crc64': result.hash_crc64,
            'version_id': getattr(result, 'version_id', None),
            'object_name': object_name,
            'presigned_url': presigned_url
        }

    except Exception as e:
        progress_bar.close()
        print(f"\n上传失败: {e}")
        raise

def _generate_presigned_url(self, object_key, expires=3600):
    try:
        cfg_dict = _decrypt_config(self, self.encrypted_config)
        cfg_upper = {str(k).upper(): v for k, v in cfg_dict.items()}
        
        access_key_id = cfg_upper.get("ACCESS_KEY_ID")
        access_key_secret = cfg_upper.get("ACCESS_KEY_SECRET")
        security_token = cfg_upper.get("SECURITY_TOKEN")
        
        quoted_key = quote(object_key.lstrip('/'))
        host = f"oss-{self.region}.aliyuncs.com"
        expires_ts = int(time.time()) + int(expires)
        string_to_sign = f"GET\n\n\n{expires_ts}\n/{self.bucket_name}/{object_key}"
        signature = base64.b64encode(
            hmac.new(access_key_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1).digest()
        ).decode('utf-8')
        query = f"OSSAccessKeyId={quote_plus(access_key_id)}&Expires={expires_ts}&Signature={quote_plus(signature)}"
        if security_token:
            query += f"&security-token={quote_plus(security_token)}"
        
        presigned_url = f"https://{self.bucket_name}.{host}/{quoted_key}?{query}"
        return presigned_url
        
    except Exception as e:
        print(f"生成临时URL失败: {e}")
        return None

def _format_file_size(self, size_bytes):
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


 


class ResourceToFilePaths:
    """资源转文件路径节点，将各种资源（图片list、音频list等）转换为文件路径列表"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resources": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入：支持图片list、音频list、视频list等"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "resource",
                    "tooltip": "文件名前缀（可选）"
                }),
                "output_format": (["auto", "png", "jpg", "wav", "mp4", "txt", "json"], {
                    "default": "auto",
                    "tooltip": "输出文件格式，auto为自动检测"
                }),
                "max_resources": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "最大处理资源数量"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("file_paths", "process_log", "file_count", "file_types")
    FUNCTION = "convert_to_paths"
    CATEGORY = "hhy/oss"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False, False)
    
    def _process_single_resource(self, resource, filename_prefix, index, output_format="auto"):
        """处理单个资源，返回(文件路径, 数据类型, 临时文件路径)"""
        temp_file = None
        data_type = "unknown"
        format_ext = output_format if output_format != "auto" else None
        
        # 为每个资源生成唯一的文件名
        unique_prefix = f"{filename_prefix}_{index}_{int(time.time())}"
        
        # 1. 检查是否是文件路径字符串
        if isinstance(resource, str):
            if os.path.exists(resource):
                return resource, "existing_file", None
            else:
                # 当作文本内容处理
                ext = format_ext or "txt"
                temp_file = save_text_to_temp(resource, unique_prefix, ext)
                return temp_file, "text", temp_file
        
        # 2. 检查是否是torch.Tensor（图片）
        elif isinstance(resource, torch.Tensor):
            if len(resource.shape) >= 3:
                try:
                    # 如果是批量，取第一张
                    if len(resource.shape) == 4:
                        resource = resource[0]
                    ext = format_ext or "png"
                    temp_file = save_image_to_temp(resource, unique_prefix, ext)
                    return temp_file, "image", temp_file
                except Exception:
                    # 🚨 安全策略: 禁用pickle序列化，改用安全的numpy格式
                    temp_file = temp_manager.create_temp_file(unique_prefix, ".npz")
                    try:
                        # 使用numpy的安全压缩格式代替pickle
                        np.savez_compressed(temp_file, data=resource.cpu().numpy())
                        return temp_file, "tensor", temp_file
                    except Exception as e:
                        temp_manager.cleanup_file(temp_file)
                        raise SecurityError(f"无法安全保存tensor数据: {e}")
        
        # 3. 检查是否是音频字典
        elif isinstance(resource, dict) and "sample_rate" in resource and "waveform" in resource:
            ext = format_ext or "wav"
            temp_file = save_audio_to_temp(resource, unique_prefix, ext)
            return temp_file, "audio", temp_file
        
        # 4. 检查是否是视频对象
        elif hasattr(resource, 'save_to'):
            try:
                ext = format_ext or "mp4"
                temp_file = save_video_to_temp(resource, unique_prefix, ext)
                return temp_file, "video", temp_file
            except Exception:
                pass
        
        # 5. 检查基本数据类型
        elif isinstance(resource, (int, float, bool)):
            content = str(resource)
            ext = format_ext or "txt"
            temp_file = save_text_to_temp(content, unique_prefix, ext)
            return temp_file, "numeric", temp_file
        
        # 6. 检查是否是列表或字典
        elif isinstance(resource, (list, dict, tuple)):
            try:
                content = json.dumps(resource, indent=2, ensure_ascii=False)
                ext = format_ext or "json"
                temp_file = save_text_to_temp(content, unique_prefix, ext)
                return temp_file, "json", temp_file
            except Exception:
                content = str(resource)
                ext = format_ext or "txt"
                temp_file = save_text_to_temp(content, unique_prefix, ext)
                return temp_file, "data", temp_file
        
        # 7. 其他类型
        else:
            try:
                content = json.dumps(resource, indent=2, ensure_ascii=False)
                ext = format_ext or "json"
                temp_file = save_text_to_temp(content, unique_prefix, ext)
                return temp_file, "serialized", temp_file
            except Exception:
                try:
                    content = str(resource)
                    ext = format_ext or "txt"
                    temp_file = save_text_to_temp(content, unique_prefix, ext)
                    return temp_file, "string", temp_file
                except Exception:
                    # 🚨 安全策略: 禁用pickle序列化，拒绝处理未知类型
                    raise SecurityError(
                        f"🚨 不支持的数据类型: {type(resource).__name__}\n"
                        f"为了安全，已禁用pickle序列化功能。\n"
                        f"支持的类型: 字符串、图片、视频、音频、基本数据类型、JSON兼容的对象"
                    )

    def convert_to_paths(self, resources, filename_prefix="resource", output_format="auto", max_resources=20):
        """将资源列表转换为文件路径列表"""
        try:
            # 处理输入参数（可能是列表格式）
            if isinstance(filename_prefix, list):
                filename_prefix = filename_prefix[0] if filename_prefix else "resource"
            if isinstance(output_format, list):
                output_format = output_format[0] if output_format else "auto"
            if isinstance(max_resources, list):
                max_resources = max_resources[0] if max_resources else 20
            
            print(f"[ResourceToFilePaths] 处理 {len(resources)} 个资源，前缀: {filename_prefix}")
            
            file_paths = []
            process_logs = []
            file_types = []
            
            # 限制处理数量
            resources_to_process = resources[:max_resources]
            
            for i, resource in enumerate(resources_to_process):
                try:
                    file_path, data_type, temp_path = self._process_single_resource(
                        resource, filename_prefix, i+1, output_format
                    )
                    
                    if file_path:
                        file_paths.append(file_path)
                        file_types.append(data_type)
                        
                        log_msg = f"资源{i+1}: {data_type} -> {os.path.basename(file_path)}"
                        process_logs.append(log_msg)
                        print(f"[ResourceToFilePaths] {log_msg}")
                    else:
                        error_msg = f"资源{i+1}: 处理失败"
                        process_logs.append(error_msg)
                        print(f"[ResourceToFilePaths] {error_msg}")
                        
                except Exception as e:
                    error_msg = f"资源{i+1}: 处理异常 - {str(e)}"
                    process_logs.append(error_msg)
                    print(f"[ResourceToFilePaths] {error_msg}")
            
            # 组合结果
            paths_str = "\n".join(file_paths) if file_paths else ""
            logs_str = "\n".join(process_logs)
            types_str = "\n".join(file_types) if file_types else ""
            file_count = len(file_paths)
            
            summary = f"\n=== 处理完成 ===\n成功: {file_count}/{len(resources_to_process)} 个资源"
            logs_str += summary
            print(f"[ResourceToFilePaths] {summary}")
            
            return (paths_str, logs_str, file_count, types_str)
            
        except Exception as exc:
            error_msg = f"资源处理失败: {str(exc)}"
            print(f"[ResourceToFilePaths] {error_msg}")
            return ("", error_msg, 0, "")


class OSSUploadFromPaths:
    """OSS文件路径上传节点，专门处理文件路径输入，支持多个路径批量上传"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "oss_encrypted_config": ("STRING", {
                    "multiline": False,
                    "tooltip": "加密的OSS配置字符串"
                }),
                "file_paths": ("STRING", {
                    "multiline": True,
                    "tooltip": "文件路径输入，每行一个路径或用分号分隔（可连接ResourceToFilePaths节点）"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "",
                    "tooltip": "文件名前缀（可选，为空则使用原文件名）"
                }),
                "max_files": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "最大上传文件数量"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("urls", "upload_logs", "valid_paths", "uploaded_count")
    FUNCTION = "upload_from_paths"
    CATEGORY = "hhy/oss"
    
    def _parse_file_paths(self, file_paths_input):
        """解析文件路径输入，支持换行符和分号分隔"""
        if not file_paths_input:
            return []
        
        # 先按换行符分割，再按分号分割
        paths = []
        lines = file_paths_input.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # 按分号分割
                sub_paths = [p.strip() for p in line.split(';') if p.strip()]
                paths.extend(sub_paths)
        
        return paths
    
    def upload_from_paths(self, oss_encrypted_config, file_paths, filename_prefix="", max_files=10):
        """从文件路径批量上传到OSS"""
        try:
            # 解析文件路径
            paths = self._parse_file_paths(file_paths)
            
            if not paths:
                return ("", "没有提供有效的文件路径", "", 0)
            
            # 限制文件数量
            paths = paths[:max_files]
            
            # 初始化OSS客户端
            self.encrypted_config = oss_encrypted_config
            _init_client(self)
            
            results = []
            valid_paths = []
            upload_logs = []
            
            for i, file_path in enumerate(paths):
                try:
                    # 检查文件是否存在
                    if not os.path.exists(file_path):
                        error_log = f"文件{i+1}: 路径不存在 - {file_path}"
                        upload_logs.append(error_log)
                        print(f"[OSS上传] {error_log}")
                        continue
                    
                    # 🛡️ 执行安全验证
                    try:
                        perform_security_validation(file_path)
                    except SecurityError as e:
                        error_log = f"文件{i+1}: 安全验证失败 - {os.path.basename(file_path)} | 错误: {str(e)}"
                        upload_logs.append(error_log)
                        print(f"[OSS上传] {error_log}")
                        continue
                    
                    # 上传文件（使用原文件名，不添加前缀）
                    result = upload_file_with_progress(self, file_path)
                    
                    # 生成URL
                    url = result.get('presigned_url')
                    if not url:
                        object_name = result.get('object_name')
                        if getattr(self, 'endpoint', None) and self.endpoint.startswith('http'):
                            url = f"{self.endpoint.rstrip('/')}/{object_name}"
                        else:
                            url = f"https://{self.bucket_name}.oss-{self.region}.aliyuncs.com/{object_name}"
                    
                    results.append(url)
                    valid_paths.append(file_path)
                    
                    # 生成日志
                    log_msg = f"文件{i+1}: 上传成功 - {os.path.basename(file_path)} | {result.get('etag', '')[:8]}..."
                    upload_logs.append(log_msg)
                    print(f"[OSS上传] {log_msg}")
                    
                except Exception as e:
                    error_log = f"文件{i+1}: 上传失败 - {os.path.basename(file_path)} | 错误: {str(e)}"
                    upload_logs.append(error_log)
                    print(f"[OSS上传] {error_log}")
            
            # 组合结果
            urls_str = "\n".join(results) if results else ""
            logs_str = "\n".join(upload_logs)
            paths_str = "\n".join(valid_paths)
            uploaded_count = len(results)
            
            summary_log = f"\n=== 上传完成 ===\n成功: {uploaded_count}/{len(paths)} 个文件"
            logs_str += summary_log
            print(f"[OSS上传] {summary_log}")
            
            return (urls_str, logs_str, paths_str, uploaded_count)
            
        except Exception as exc:
            error_msg = f"批量上传失败: {str(exc)}"
            return ("", error_msg, "", 0)


class VideoCombineToPath:
    """视频合成节点，将图片batch和音频合成为视频并输出文件路径"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "图片batch输入"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "视频帧率(FPS)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "video",
                    "tooltip": "输出文件名前缀"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "音频输入（可选）"
                }),
                "video_format": (["mp4", "avi", "mov", "webm"], {
                    "default": "mp4",
                    "tooltip": "视频格式"
                }),
                "quality": (["high", "medium", "low"], {
                    "default": "medium",
                    "tooltip": "视频质量"
                }),
                "loop_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "循环次数，0为不循环"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "process_log", "frame_count", "duration")
    FUNCTION = "combine_video"
    CATEGORY = "hhy/oss"

    def _tensor_to_pil(self, tensor):
        """将tensor转换为PIL Image"""
        # 确保tensor在CPU上且为numpy格式
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            tensor = tensor.numpy()
        
        # 转换数据类型和范围
        if tensor.dtype != np.uint8:
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).astype(np.uint8)
            else:
                tensor = tensor.astype(np.uint8)
        
        return Image.fromarray(tensor)

    def _save_frames_as_temp_video(self, images, frame_rate, filename_prefix, video_format, quality, audio=None):
        """🛡️ 安全地保存帧序列为临时视频文件"""
        # 🛡️ 使用安全的临时文件管理器
        video_file = temp_manager.create_temp_file(filename_prefix, f".{video_format}")
        
        try:
            # 检查是否有可用的视频编码库
            try:
                import cv2
                return self._save_with_opencv(images, frame_rate, video_file, audio)
            except ImportError:
                pass
            
            try:
                import imageio
                return self._save_with_imageio(images, frame_rate, video_file, quality, audio)
            except ImportError:
                pass
            
            # 如果都没有，使用PIL保存为GIF
            return self._save_as_gif(images, frame_rate, filename_prefix)
            
        except Exception as e:
            print(f"视频保存失败: {e}")
            raise

    def _save_with_opencv(self, images, frame_rate, video_file, audio=None):
        """使用OpenCV保存视频"""
        import cv2
        
        # 获取第一帧的尺寸
        first_frame = self._tensor_to_pil(images[0])
        width, height = first_frame.size
        
        # 创建视频编写器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
        
        for image_tensor in images:
            pil_image = self._tensor_to_pil(image_tensor)
            # PIL转OpenCV格式 (RGB -> BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            out.write(opencv_image)
        
        out.release()
        
        # 如果有音频，需要用ffmpeg合并
        if audio is not None:
            return self._add_audio_with_ffmpeg(video_file, audio, frame_rate)
        
        return video_file

    def _save_with_imageio(self, images, frame_rate, video_file, quality, audio=None):
        """使用imageio保存视频"""
        import imageio
        
        # 设置质量参数
        quality_map = {"high": 9, "medium": 5, "low": 2}
        crf = quality_map.get(quality, 5)
        
        # 转换图片为numpy数组列表
        frames = []
        for image_tensor in images:
            pil_image = self._tensor_to_pil(image_tensor)
            frames.append(np.array(pil_image))
        
        # 保存视频
        imageio.mimsave(video_file, frames, fps=frame_rate, 
                       macro_block_size=None, codec='libx264', 
                       quality=crf, pixelformat='yuv420p')
        
        # 如果有音频，需要用ffmpeg合并
        if audio is not None:
            return self._add_audio_with_ffmpeg(video_file, audio, frame_rate)
        
        return video_file

    def _save_as_gif(self, images, frame_rate, filename_prefix):
        """🛡️ 安全地保存为GIF格式（后备方案）"""
        # 🛡️ 使用安全的临时文件管理器
        gif_file = temp_manager.create_temp_file(filename_prefix, ".gif")
        
        pil_images = [self._tensor_to_pil(img) for img in images]
        duration = int(1000 / frame_rate)  # 毫秒
        
        pil_images[0].save(
            gif_file,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0
        )
        
        return gif_file

    def _add_audio_with_ffmpeg(self, video_file, audio, frame_rate):
        """使用ffmpeg添加音频到视频"""
        try:
            import subprocess
            import shutil
            
            # 检查ffmpeg是否可用
            if not shutil.which('ffmpeg'):
                print("ffmpeg不可用，跳过音频合成")
                return video_file
            
            # 🛡️ 创建安全的临时音频文件
            audio_file = temp_manager.create_temp_file("temp_audio", ".wav")
            
            # 保存音频
            sample_rate = audio.get("sample_rate", 44100)
            waveform = audio.get("waveform")
            
            if waveform is not None:
                # 转换waveform为numpy
                if hasattr(waveform, 'cpu'):
                    waveform = waveform.cpu()
                if hasattr(waveform, 'numpy'):
                    waveform = waveform.numpy()
                
                # 保存为wav文件
                import soundfile as sf
                if len(waveform.shape) == 3 and waveform.shape[0] == 1:
                    waveform = waveform[0]  # 移除batch维度
                
                sf.write(audio_file, waveform.T, sample_rate)
                
                # 使用ffmpeg合并视频和音频
                output_file = video_file.replace('.mp4', '_with_audio.mp4')
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_file,
                    '-i', audio_file,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    output_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # 删除临时文件
                    try:
                        os.remove(audio_file)
                        os.remove(video_file)
                    except:
                        pass
                    return output_file
                else:
                    print(f"ffmpeg错误: {result.stderr}")
                    return video_file
            
        except Exception as e:
            print(f"音频合成失败: {e}")
        
        return video_file

    def combine_video(self, images, frame_rate, filename_prefix, audio=None, video_format="mp4", quality="medium", loop_count=0):
        """合成视频主函数"""
        try:
            # 验证输入
            if images is None or len(images) == 0:
                return ("", "错误: 没有输入图片", 0, 0.0)
            
            frame_count = len(images)
            duration = frame_count / frame_rate
            
            print(f"[VideoCombine] 开始合成视频: {frame_count}帧, {frame_rate}fps, 时长{duration:.2f}秒")
            
            # 合成视频
            video_path = self._save_frames_as_temp_video(
                images, frame_rate, filename_prefix, video_format, quality, audio
            )
            
            # 生成日志
            log_items = [
                f"frames={frame_count}",
                f"fps={frame_rate}",
                f"duration={duration:.2f}s",
                f"format={video_format}",
                f"quality={quality}",
                f"has_audio={audio is not None}",
                f"output={os.path.basename(video_path)}"
            ]
            process_log = " | ".join(log_items)
            
            print(f"[VideoCombine] 合成完成: {os.path.basename(video_path)}")
            
            return (video_path, process_log, frame_count, duration)
            
        except Exception as exc:
            error_msg = f"视频合成失败: {str(exc)}"
            print(f"[VideoCombine] {error_msg}")
            return ("", error_msg, 0, 0.0)


class OSSUploadAny:
    """万能OSS上传节点，支持任意类型输入"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "oss_encrypted_config": ("STRING", {
                    "multiline": False,
                    "tooltip": "加密的OSS配置字符串"
                }),
                "source": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "任意类型的数据输入"
                }),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "comfyui",
                    "tooltip": "文件名前缀（可选）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("url", "upload_log", "local_file_path")
    FUNCTION = "upload"
    CATEGORY = "hhy/oss"
    
    def _detect_source_type_and_save(self, source, filename_prefix="comfyui"):
        """🛡️ 安全地检测输入源类型并保存为文件"""
        temp_file = None
        data_type = "unknown"
        
        # 1. 检查是否是文件路径字符串
        if isinstance(source, str):
            if os.path.exists(source):
                # 🛡️ 验证文件路径安全性
                try:
                    perform_security_validation(source)
                    return source, "file", None
                except SecurityError as e:
                    raise SecurityError(f"🚨 文件路径验证失败: {e}")
            else:
                # 当作文本内容处理
                temp_file = save_text_to_temp(source, filename_prefix, "txt")
                return temp_file, "text", temp_file
        
        # 2. 检查是否是torch.Tensor（图片）
        elif isinstance(source, torch.Tensor):
            if len(source.shape) >= 3:
                # 可能是图片张量
                try:
                    # 如果是批量，取第一张
                    if len(source.shape) == 4:
                        source = source[0]
                    temp_file = save_image_to_temp(source, filename_prefix, "png")
                    return temp_file, "image", temp_file
                except Exception:
                    # 🚨 安全策略: 禁用pickle序列化，改用安全的numpy格式
                    temp_file = temp_manager.create_temp_file(filename_prefix, ".npz")
                    try:
                        # 使用numpy的安全压缩格式代替pickle
                        np.savez_compressed(temp_file, data=source.cpu().numpy())
                        return temp_file, "tensor", temp_file
                    except Exception as e:
                        temp_manager.cleanup_file(temp_file)
                        raise SecurityError(f"无法安全保存tensor数据: {e}")
        
        # 3. 检查是否是音频字典
        elif isinstance(source, dict) and "sample_rate" in source and "waveform" in source:
            temp_file = save_audio_to_temp(source, filename_prefix, "wav")
            return temp_file, "audio", temp_file
        
        # 4. 检查是否是视频对象
        elif hasattr(source, 'save_to'):
            try:
                temp_file = save_video_to_temp(source, filename_prefix, "mp4")
                return temp_file, "video", temp_file
            except Exception:
                pass
        
        # 5. 检查基本数据类型
        elif isinstance(source, (int, float, bool)):
            content = str(source)
            temp_file = save_text_to_temp(content, filename_prefix, "txt")
            return temp_file, "numeric", temp_file
        
        # 6. 检查是否是列表或字典
        elif isinstance(source, (list, dict, tuple)):
            try:
                content = json.dumps(source, indent=2, ensure_ascii=False)
                temp_file = save_text_to_temp(content, filename_prefix, "json")
                return temp_file, "json", temp_file
            except Exception:
                content = str(source)
                temp_file = save_text_to_temp(content, filename_prefix, "txt")
                return temp_file, "data", temp_file
        
        # 7. 尝试序列化其他类型
        else:
            try:
                # 尝试JSON序列化
                content = json.dumps(source, indent=2, ensure_ascii=False)
                temp_file = save_text_to_temp(content, filename_prefix, "json")
                return temp_file, "serialized", temp_file
            except Exception:
                try:
                    # 尝试字符串转换
                    content = str(source)
                    temp_file = save_text_to_temp(content, filename_prefix, "txt")
                    return temp_file, "string", temp_file
                except Exception:
                    # 🚨 安全策略: 禁用pickle序列化，拒绝处理未知类型
                    raise SecurityError(
                        f"🚨 不支持的数据类型: {type(source).__name__}\n"
                        f"为了安全，已禁用pickle序列化功能。\n"
                        f"支持的类型: 字符串、图片、视频、音频、基本数据类型、JSON兼容的对象"
                    )

    def upload(self, oss_encrypted_config, source, filename_prefix="comfyui"):
        temp_file = None
        try:
            # 检测源类型并保存
            local_file, data_type, temp_file = self._detect_source_type_and_save(source, filename_prefix)

            # 初始化OSS客户端
            self.encrypted_config = oss_encrypted_config
            _init_client(self)

            # 上传文件
            result = upload_file_with_progress(self, local_file)

            # 生成URL
            url = result.get('presigned_url')
            if not url:
                object_name = result.get('object_name')
                if getattr(self, 'endpoint', None) and self.endpoint.startswith('http'):
                    url = f"{self.endpoint.rstrip('/')}/{object_name}"
                else:
                    url = f"https://{self.bucket_name}.oss-{self.region}.aliyuncs.com/{object_name}"

            # 生成日志
            log_items = [
                f"type={data_type}",
                f"source_type={type(source).__name__}",
                f"status_code={result.get('status_code')}",
                f"request_id={result.get('request_id')}",
                f"etag={result.get('etag')}",
                f"crc64={result.get('hash_crc64')}",
                f"object={result.get('object_name')}"
            ]
            upload_log = " | ".join([str(x) for x in log_items if x is not None])

            return (str(url), str(upload_log), str(local_file))
            
        except Exception as exc:
            error_msg = f"上传失败: {exc}"
            return ("", error_msg, temp_file or "")
        finally:
            # 保留临时文件供用户查看
            pass


NODE_CLASS_MAPPINGS = {
    "ResourceToFilePaths": ResourceToFilePaths,
    "OSSUploadFromPaths": OSSUploadFromPaths,
    "VideoCombineToPath": VideoCombineToPath,
    "OSSUploadAny": OSSUploadAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResourceToFilePaths": "资源转文件路径",
    "OSSUploadFromPaths": "OSS 从路径上传",
    "VideoCombineToPath": "视频合并到路径",
    "OSSUploadAny": "OSS 任意上传(即将弃用)"
}