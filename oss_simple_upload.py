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
from typing import List, Optional, Union, Dict, Any
from fractions import Fraction

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
    """保存图片张量到临时文件"""
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
    
    # 创建临时文件
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    img.save(temp_file)
    
    return temp_file

def save_audio_to_temp(audio_data: Dict[str, Any], filename_prefix: str = "audio", format: str = "wav") -> str:
    """保存音频数据到临时文件"""
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
    
    # 创建临时文件
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    
    # 保存音频
    subtype = "FLOAT" if format.lower() == "wav" else None
    sf.write(temp_file, waveform.T.numpy(), sample_rate, subtype=subtype)
    
    return temp_file

def save_video_to_temp(video_input, filename_prefix: str = "video", format: str = "mp4") -> str:
    """保存视频到临时文件"""
    if not COMFY_TYPES_AVAILABLE:
        raise ImportError("ComfyUI types are required for video support")
    
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    
    # 使用视频对象的save方法
    video_input.save_to(temp_file, format=format)
    
    return temp_file

def save_text_to_temp(text: str, filename_prefix: str = "text", format: str = "txt") -> str:
    """保存文本到临时文件"""
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    
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
                "resources_1": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入1：支持图片batch/list、音频list、视频list、文本list等，批量图片会自动拆分为单独文件"
                }),
            },
            "optional": {
                "resources_2": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入2：支持图片batch/list、音频list、视频list、文本list等"
                }),
                "resources_3": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入3：支持图片batch/list、音频list、视频list、文本list等"
                }),
                "resources_4": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入4：支持图片batch/list、音频list、视频list、文本list等"
                }),
                "resources_5": (NodeIO.ANY if COMFY_TYPES_AVAILABLE else "ANY", {
                    "tooltip": "资源输入5：支持图片batch/list、音频list、视频list、文本list等"
                }),
                "filename_prefix": ("STRING", {
                    "default": "resource",
                    "tooltip": "文件名前缀（可选）"
                }),
                "output_format": (["auto", "png", "jpg", "wav", "mp4", "txt", "json"], {
                    "default": "auto",
                    "tooltip": "输出文件格式，auto为自动检测"
                }),
                "max_resources": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "tooltip": "最大处理资源数量"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("file_paths", "process_log", "file_count", "file_types")
    FUNCTION = "convert_to_paths"
    CATEGORY = "hhy"
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
                    # 如果不是图片，尝试保存为numpy数组
                    import pickle
                    temp_file = os.path.join(tempfile.gettempdir(), f"{unique_prefix}_tensor.pkl")
                    with open(temp_file, 'wb') as f:
                        pickle.dump(resource.cpu().numpy(), f)
                    return temp_file, "tensor", temp_file
        
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
                    import pickle
                    temp_file = os.path.join(tempfile.gettempdir(), f"{unique_prefix}_object.pkl")
                    with open(temp_file, 'wb') as f:
                        pickle.dump(resource, f)
                    return temp_file, "pickle", temp_file

    def _expand_resource(self, resource):
        """展开单个资源，如果是批量tensor则拆分为多个"""
        # 处理图片batch (4D tensor: [batch, height, width, channels])
        if isinstance(resource, torch.Tensor) and len(resource.shape) == 4:
            # 拆分batch中的每一张图片
            return [resource[i] for i in range(resource.shape[0])]
        
        # 处理音频列表 (可能是list of dict)
        elif isinstance(resource, list):
            # 检查是否是音频列表
            if all(isinstance(item, dict) and "sample_rate" in item and "waveform" in item for item in resource):
                return resource  # 已经是列表了
            # 检查是否是视频列表
            elif all(hasattr(item, 'save_to') for item in resource):
                return resource  # 已经是列表了
            # 其他列表类型，展开
            else:
                return resource
        
        # 其他单个资源，包装成列表
        else:
            return [resource]
    
    def convert_to_paths(self, resources_1, resources_2=None, resources_3=None, resources_4=None, resources_5=None, filename_prefix="resource", output_format="auto", max_resources=50):
        """将资源列表转换为文件路径列表"""
        try:
            # 处理输入参数（可能是列表格式）
            if isinstance(filename_prefix, list):
                filename_prefix = filename_prefix[0] if filename_prefix else "resource"
            if isinstance(output_format, list):
                output_format = output_format[0] if output_format else "auto"
            if isinstance(max_resources, list):
                max_resources = max_resources[0] if max_resources else 50
            
            # 合并所有输入的资源
            all_resources = []
            
            # 添加必需的资源输入
            if resources_1 is not None:
                if isinstance(resources_1, list):
                    # 对列表中的每个资源进行展开
                    for res in resources_1:
                        expanded = self._expand_resource(res)
                        all_resources.extend(expanded)
                else:
                    # 单个资源也进行展开
                    expanded = self._expand_resource(resources_1)
                    all_resources.extend(expanded)
            
            # 添加可选的资源输入
            for resource_input in [resources_2, resources_3, resources_4, resources_5]:
                if resource_input is not None:
                    if isinstance(resource_input, list):
                        # 对列表中的每个资源进行展开
                        for res in resource_input:
                            expanded = self._expand_resource(res)
                            all_resources.extend(expanded)
                    else:
                        # 单个资源也进行展开
                        expanded = self._expand_resource(resource_input)
                        all_resources.extend(expanded)
            
            print(f"[ResourceToFilePaths] 处理 {len(all_resources)} 个资源，前缀: {filename_prefix}")
            
            file_paths = []
            process_logs = []
            file_types = []
            
            # 限制处理数量
            resources_to_process = all_resources[:max_resources]
            
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
    CATEGORY = "hhy"
    
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
                "images_1": ("IMAGE", {
                    "tooltip": "图片batch输入1"
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
                "audio_1": ("AUDIO", {
                    "tooltip": "音频输入1（对应images_1，可选）"
                }),
                "images_2": ("IMAGE", {
                    "tooltip": "图片batch输入2（可选）"
                }),
                "audio_2": ("AUDIO", {
                    "tooltip": "音频输入2（对应images_2，可选）"
                }),
                "images_3": ("IMAGE", {
                    "tooltip": "图片batch输入3（可选）"
                }),
                "audio_3": ("AUDIO", {
                    "tooltip": "音频输入3（对应images_3，可选）"
                }),
                "quality": (["high", "medium", "low"], {
                    "default": "medium",
                    "tooltip": "视频质量 (H264编码)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("video_paths", "process_log", "total_videos", "total_duration")
    FUNCTION = "combine_video"
    CATEGORY = "hhy"

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

    def _tensor_to_bytes(self, tensor):
        """将tensor转换为字节数组"""
        # 转换为numpy数组
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        elif not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        
        # 确保是3维数组 [H, W, C]
        if len(tensor.shape) == 4:
            # 如果是4维，取第一个
            tensor = tensor[0]
        
        # 转换数据类型和范围
        if tensor.dtype != np.uint8:
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).astype(np.uint8)
            else:
                tensor = tensor.astype(np.uint8)
        
        return tensor.tobytes()

    def _save_frames_as_temp_video(self, images, frame_rate, filename_prefix, video_format, quality, audio=None):
        """保存帧序列为临时视频文件，使用ffmpeg和H264编码"""
        import subprocess
        import shutil
        
        # 获取ffmpeg路径
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except:
                pass
        
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to use video output.")
        
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        video_file = os.path.join(temp_dir, f"{filename_prefix}_{timestamp}.{video_format}")
        
        # 检查第一张图片的尺寸和通道数
        first_tensor = images[0]
        if isinstance(first_tensor, torch.Tensor):
            if len(first_tensor.shape) == 4:
                first_tensor = first_tensor[0]
            has_alpha = len(first_tensor.shape) == 3 and first_tensor.shape[2] == 4
        else:
            has_alpha = len(first_tensor.shape) == 3 and first_tensor.shape[2] == 4
        
        # 获取图片尺寸
        first_pil = self._tensor_to_pil(images[0])
        width, height = first_pil.size
        
        # 设置质量参数
        quality_map = {"high": "18", "medium": "23", "low": "28"}
        crf = quality_map.get(quality, "23")
        
        # 像素格式
        if has_alpha:
            pix_fmt = "rgba"
        else:
            pix_fmt = "rgb24"
        
        # 构建ffmpeg命令
        cmd = [
            ffmpeg_path,
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}",
            "-r", str(frame_rate),
            "-i", "-",
        ]
        
        # H264编码参数
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", crf,
            "-pix_fmt", "yuv420p",
        ])
        
        # 如果有音频，后续单独处理
        has_audio_input = audio is not None and audio.get("waveform") is not None
        
        # 添加输出文件
        cmd.append(video_file)
        
        # 收集所有帧的数据到内存
        print(f"[VideoCombine] 正在处理 {len(images)} 帧数据...")
        frame_data = b''
        for i, image_tensor in enumerate(images):
            image_bytes = self._tensor_to_bytes(image_tensor)
            frame_data += image_bytes
        
        print(f"[VideoCombine] 数据准备完成，总大小: {len(frame_data) / 1024 / 1024:.2f} MB")
        
        # 一次性写入所有数据到ffmpeg
        result = subprocess.run(
            cmd,
            input=frame_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg failed: {error_msg}")
        
        if result.stderr:
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            if stderr_output.strip():
                print(f"[VideoCombine] ffmpeg警告: {stderr_output}")
        
        # 如果有音频，使用二次处理添加音频
        if has_audio_input:
            return self._add_audio_to_video(video_file, audio, frame_rate)
        
        return video_file

    def _add_audio_to_video(self, video_file, audio, frame_rate):
        """为视频添加音频"""
        import subprocess
        import shutil
        
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except:
                pass
        
        if not ffmpeg_path:
            return video_file
        
        # 创建临时音频文件
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        audio_file = os.path.join(temp_dir, f"temp_audio_{timestamp}.wav")
        
        try:
            import soundfile as sf
            
            sample_rate = audio.get("sample_rate", 44100)
            waveform = audio.get("waveform")
            
            if waveform is not None:
                if hasattr(waveform, 'cpu'):
                    waveform = waveform.cpu()
                if hasattr(waveform, 'numpy'):
                    waveform = waveform.numpy()
                
                if len(waveform.shape) == 3 and waveform.shape[0] == 1:
                    waveform = waveform[0]
                
                sf.write(audio_file, waveform.T, sample_rate)
                
                # 创建带音频的输出文件
                output_file = video_file.replace('.mp4', '_audio.mp4')
                channels = waveform.shape[0]
                
                cmd = [
                    ffmpeg_path, "-v", "error", "-n",
                    "-i", video_file,
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-f", "f32le", "-i", "-",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    output_file
                ]
                
                audio_data = waveform.squeeze(0).transpose(0, 1).numpy().tobytes()
                
                result = subprocess.run(cmd, input=audio_data, capture_output=True)
                
                if result.returncode == 0:
                    # 删除原文件和临时音频文件
                    try:
                        os.remove(video_file)
                        os.remove(audio_file)
                    except:
                        pass
                    return output_file
                else:
                    print(f"Audio add failed: {result.stderr.decode()}")
        except Exception as e:
            print(f"Audio processing failed: {e}")
        
        return video_file


    def combine_video(self, images_1, frame_rate, filename_prefix, audio_1=None, images_2=None, audio_2=None, images_3=None, audio_3=None, quality="medium"):
        """合成视频主函数 - 支持多个图片-音频对，使用H264编码"""
        try:
            # 收集所有的图片-音频对
            video_pairs = []
            
            # 第一个是必需的
            if images_1 is None or len(images_1) == 0:
                return ("", "错误: 没有输入图片", 0, 0.0)
            video_pairs.append((images_1, audio_1, 1))
            
            # 添加可选的图片-音频对
            if images_2 is not None and len(images_2) > 0:
                video_pairs.append((images_2, audio_2, 2))
            if images_3 is not None and len(images_3) > 0:
                video_pairs.append((images_3, audio_3, 3))
            
            video_paths = []
            log_items = []
            total_duration = 0.0
            
            print(f"[VideoCombine] 开始合成 {len(video_pairs)} 个视频 (H264/MP4)")
            
            # 为每个图片-音频对生成视频
            for images, audio, index in video_pairs:
                frame_count = len(images)
                duration = frame_count / frame_rate
                total_duration += duration
                
                # 生成带序号的文件名
                indexed_prefix = f"{filename_prefix}_{index}" if len(video_pairs) > 1 else filename_prefix
                
                print(f"[VideoCombine] 视频{index}: {frame_count}帧, {frame_rate}fps, 时长{duration:.2f}秒")
                
                # 合成视频 (固定使用mp4格式，H264编码)
                video_path = self._save_frames_as_temp_video(
                    images, frame_rate, indexed_prefix, "mp4", quality, audio
                )
                
                video_paths.append(video_path)
                
                # 生成单个视频的日志
                video_log = f"Video{index}: {frame_count}frames, {duration:.2f}s, audio={audio is not None}, H264/MP4, output={os.path.basename(video_path)}"
                log_items.append(video_log)
                
                print(f"[VideoCombine] 视频{index}合成完成: {os.path.basename(video_path)}")
            
            # 合并所有视频路径为分号分隔的字符串
            combined_paths = ";".join(video_paths)
            
            # 生成总体日志
            process_log = " || ".join(log_items)
            summary_log = f"Total: {len(video_pairs)} videos (H264/MP4), {total_duration:.2f}s, quality={quality}"
            final_log = f"{summary_log} || {process_log}"
            
            print(f"[VideoCombine] 全部合成完成: {len(video_pairs)}个视频, 总时长{total_duration:.2f}秒")
            
            return (combined_paths, final_log, len(video_pairs), total_duration)
            
        except Exception as exc:
            error_msg = f"视频合成失败: {str(exc)}"
            print(f"[VideoCombine] {error_msg}")
            return ("", error_msg, 0, 0.0)


NODE_CLASS_MAPPINGS = {
    "ResourceToFilePaths": ResourceToFilePaths,
    "OSSUploadFromPaths": OSSUploadFromPaths,
    "VideoCombineToPath": VideoCombineToPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResourceToFilePaths": "Resources to File Paths",
    "OSSUploadFromPaths": "OSS Upload from Paths",
    "VideoCombineToPath": "Video Combine to Path"
}