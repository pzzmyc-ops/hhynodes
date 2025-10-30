from PIL import Image
import numpy as np
import torch
import requests
import os
import base64
import datetime
import hashlib
import hmac
import json
import time
import av
from fractions import Fraction
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile
import folder_paths
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
import urllib3

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_closest_aspect_ratio(width, height):
    """获取最接近的标准视频比例 - 仅返回API支持的比例"""
    # API支持的标准比例
    standard_ratios = {
        '16:9': 16/9,
        '4:3': 4/3,
        '1:1': 1,
        '3:4': 3/4,
        '9:16': 9/16,
        '21:9': 21/9
    }
    
    # 计算输入图片的比例
    image_ratio = width / height
    
    # 找到最接近的标准比例并返回（总是返回API支持的值）
    closest_ratio = min(standard_ratios.items(), key=lambda x: abs(x[1] - image_ratio))
    return closest_ratio[0]

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, dateStamp, regionName, serviceName):
    kDate = sign(key.encode('utf-8'), dateStamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'request')
    return kSigning

def extract_frames_from_video(video_path):
    """从视频中提取所有帧"""
    frames = []
    
    try:
        container = av.open(video_path)
        for frame in container.decode(video=0):
            # 将PyAV帧转换为PIL图像
            img = frame.to_image()
            # 将PIL图像转换为tensor
            tensor = pil2tensor(img)
            frames.append(tensor)
        
        container.close()
        
        # 如果没有帧，返回空列表
        if not frames:
            return None
            
        # 堆叠所有帧，创建批量tensor [N, C, H, W]
        return torch.cat(frames, dim=0)
        
    except Exception as e:
        print(f"提取视频帧时出错: {str(e)}")
        return None

class JimengImageGenerate(ComfyNodeABC):
    """
    即梦生图API节点
    
    支持的模型:
    - doubao-seedream-3.0-t2i: 文本生图，支持seed和guidance_scale
    - doubao-seededit-3.0-i2i: 图生图，支持seed和guidance_scale  
    - doubao-seedream-4-0-250828: 最新模型，支持4K输出、组图、流式输出、多图输入
    
    尺寸支持:
    方式1 - 分辨率模式: 1K, 2K, 4K (仅doubao-seedream-4-0-250828支持4K)
    方式2 - 像素模式: 具体像素值，如2048x2048
    
    像素模式限制:
    - 总像素范围: [1024x1024, 4096x4096]
    - 宽高比范围: [1/16, 16]
    
    高级功能:
    - 组图生成: 设置sequential_image_generation为"auto"，可生成多张相关图片
    - 流式输出: 启用stream可实时接收生成的图片，改善等待体验
    - 多图输入: doubao-seedream-4-0-250828支持最多10张参考图融合生成
    
    多图输入使用方法:
    - 使用batch image节点将多张图片合并为批量输入
    - doubao-seedream-4-0-250828: 支持1-10张参考图
    - 其他模型: 自动使用第一张图片
    """
    
    def __init__(self):
        self.temp_dir = os.path.join(folder_paths.get_temp_directory(), "jimeng")
        os.makedirs(self.temp_dir, exist_ok=True)
        # 创建自定义的http连接管理器
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.http = PoolManager(
            retries=retry_strategy,
            maxsize=1  # 限制连接池大小
        )
        # 定义API相关配置 - 使用ARK API
        self.endpoint = 'https://ark.cn-beijing.volces.com/api/v3/images/generations'
    

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "model": (["doubao-seedream-3.0-t2i", "doubao-seededit-3.0-i2i", "doubao-seedream-4-0-250828"], {"default": "doubao-seedream-4-0-250828"}),
            "ark_api_key": ("STRING", {"default": ""})
        },
        "optional": {
            "reference_image": ("IMAGE",),
            "size": (["1K", "2K", "4K", "1024x1024", "1024x768", "768x1024", "1152x896", "896x1152", "2048x2048", "3072x3072", "4096x4096", "2048x1024", "1024x2048", "3072x1536", "1536x3072", "4096x2048", "2048x4096"], {"default": "2K"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            "sequential_image_generation": (["auto", "disabled"], {"default": "disabled"}),
            "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "stream": ("BOOLEAN", {"default": False}),
            "response_format": (["url", "b64_json"], {"default": "url"}),
            "watermark": ("BOOLEAN", {"default": True})
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "hhy/api"

    def make_request(self, method, url, headers=None, data=None, stream=False):
        """统一的请求处理函数"""
        try:
            if headers is None:
                headers = {}
            
            response = self.http.request(
                method,
                url,
                headers=headers,
                body=data,
                preload_content=not stream
            )
            
            if not stream:
                return response.data, response.status
            return response
            
        except Exception as e:
            print(f"请求出错: {str(e)}")
            raise e


    def encode_image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def parse_sse_response(self, response):
        """解析Server-Sent Events响应"""
        images = []
        usage_info = None
        
        try:
            for line in response.data.decode('utf-8').split('\n'):
                line = line.strip()
                if line.startswith('data: ') and not line.endswith('[DONE]'):
                    try:
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        data = json.loads(data_str)
                        
                        if data.get('type') == 'image_generation.partial_succeeded':
                            image_info = {
                                'url': data.get('url'),
                                'size': data.get('size'),
                                'index': data.get('image_index', 0),
                                'created': data.get('created')
                            }
                            images.append(image_info)
                            print(f"收到第 {data.get('image_index', 0) + 1} 张图片: {data.get('size')}")
                        
                        elif data.get('type') == 'image_generation.completed':
                            usage_info = data.get('usage', {})
                            print(f"生成完成，共 {usage_info.get('generated_images', 0)} 张图片")
                            
                    except json.JSONDecodeError as e:
                        print(f"解析SSE数据失败: {e}")
                        continue
                        
        except Exception as e:
            print(f"处理流式响应时出错: {e}")
            
        return images, usage_info

    def generate_image(self, prompt, model, ark_api_key, reference_image=None, size="1024x1024", seed=-1, guidance_scale=2.5, sequential_image_generation="disabled", max_images=1, stream=False, response_format="url", watermark=True):
        # 创建临时文件路径
        temp_image = os.path.join(self.temp_dir, f"temp_jimeng_image_{int(time.time())}.jpg")
        temp_ref_image = []
        
        try:
            # 根据模型确定配置
            model_config = {
                "doubao-seedream-3.0-t2i": {
                    "supports_i2i": False,
                    "supports_seed": True,
                    "supports_guidance": True,
                    "supports_sequential": False,
                    "supports_stream": False,
                    "default_guidance": 2.5
                },
                "doubao-seededit-3.0-i2i": {
                    "supports_i2i": True,
                    "supports_seed": True,
                    "supports_guidance": True,
                    "supports_sequential": False,
                    "supports_stream": False,
                    "default_guidance": 5.5
                },
                "doubao-seedream-4-0-250828": {
                    "supports_i2i": True,  # 支持多图输入
                    "supports_seed": False,
                    "supports_guidance": False,
                    "supports_sequential": True,
                    "supports_stream": True,
                    "supports_4k": True,  # 支持4K输出
                    "supports_resolution_modes": True,  # 支持两种尺寸模式
                    "default_guidance": None
                }
            }
            
            config = model_config.get(model, model_config["doubao-seedream-4-0-250828"])
            
            # 处理尺寸参数
            # 检查是否使用分辨率模式（1K, 2K, 4K）还是像素模式（WxH）
            resolution_modes = ["1K", "2K", "4K"]
            is_resolution_mode = size in resolution_modes
            
            # 验证4K支持
            if size == "4K" and not config.get("supports_4k", False):
                print(f"警告: 模型 {model} 不支持4K输出，将使用2K")
                size = "2K"
            
            # 准备请求Body
            body_params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "response_format": response_format,
                "watermark": watermark
            }
            
            # 如果使用分辨率模式，添加提示说明
            if is_resolution_mode:
                print(f"使用分辨率模式: {size}")
                print("提示: 在分辨率模式下，建议在prompt中用自然语言描述图片宽高比、图片形状或图片用途")
            else:
                print(f"使用像素模式: {size}")
                # 验证像素值范围
                if 'x' in size:
                    try:
                        width, height = map(int, size.split('x'))
                        total_pixels = width * height
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        # 检查总像素范围 [1024x1024, 4096x4096]
                        min_pixels = 1024 * 1024
                        max_pixels = 4096 * 4096
                        
                        if total_pixels < min_pixels or total_pixels > max_pixels:
                            print(f"警告: 像素值 {size} 超出范围 [1024x1024, 4096x4096]")
                        
                        # 检查宽高比范围 [1/16, 16]
                        if aspect_ratio > 16:
                            print(f"警告: 宽高比 {aspect_ratio:.2f} 超出范围 [1/16, 16]")
                            
                    except ValueError:
                        print(f"警告: 无效的像素格式 {size}")
            
            
            # 添加种子参数（仅支持的模型）
            if config["supports_seed"] and seed != -1:
                body_params["seed"] = seed
            
            # 添加引导比例（仅支持的模型）
            if config["supports_guidance"]:
                if guidance_scale != config["default_guidance"]:
                    body_params["guidance_scale"] = guidance_scale
            
            # 添加组图参数（仅doubao-seedream-4-0-250828支持）
            if config["supports_sequential"]:
                body_params["sequential_image_generation"] = sequential_image_generation
                
                # 如果启用组图功能，添加组图选项
                if sequential_image_generation == "auto":
                    body_params["sequential_image_generation_options"] = {
                        "max_images": max_images
                    }
                    print(f"启用组图功能，最大图片数量: {max_images}")
            
            # 添加流式输出参数（仅doubao-seedream-4-0-250828支持）
            if config["supports_stream"]:
                body_params["stream"] = stream
                if stream:
                    print("启用流式输出模式")
            
            # 处理参考图像（支持批量图像）
            if reference_image is not None:
                if config["supports_i2i"]:
                    # 检查是否为批量图像
                    if len(reference_image.shape) == 4:  # [N, C, H, W] 批量图像
                        batch_size = reference_image.shape[0]
                        print(f"检测到批量参考图像，数量: {batch_size}")
                        
                        # doubao-seedream-4-0-250828最多支持10张参考图
                        max_ref_images = 10 if model == "doubao-seedream-4-0-250828" else 1
                        if batch_size > max_ref_images:
                            print(f"警告: 参考图像数量 {batch_size} 超过模型限制 {max_ref_images}，将只使用前 {max_ref_images} 张")
                            reference_image = reference_image[:max_ref_images]
                            batch_size = max_ref_images
                        
                        # 处理多张图像
                        image_data_list = []
                        temp_ref_images = []
                        
                        for i in range(batch_size):
                            temp_ref_image = os.path.join(self.temp_dir, f"temp_ref_image_{int(time.time())}_{i}.jpg")
                            temp_ref_images.append(temp_ref_image)
                            
                            pil_ref_image = tensor2pil(reference_image[i])
                            pil_ref_image.save(temp_ref_image)
                            
                            # 转换为base64
                            with open(temp_ref_image, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode('utf-8')
                                image_data_list.append(f"data:image/jpeg;base64,{image_data}")
                        
                        # 根据模型设置图像参数
                        if model == "doubao-seedream-4-0-250828":
                            # doubao-seedream-4-0支持多图数组
                            body_params["image"] = image_data_list
                            print(f"使用 {len(image_data_list)} 张参考图像进行多图融合生成")
                        else:
                            # 其他模型只支持单图，使用第一张
                            body_params["image"] = image_data_list[0]
                            print(f"模型 {model} 只支持单图输入，使用第一张参考图像")
                        
                        # 保存临时文件列表以便后续清理
                        temp_ref_image = temp_ref_images
                        
                    else:  # [C, H, W] 单张图像
                        temp_ref_image = os.path.join(self.temp_dir, f"temp_ref_image_{int(time.time())}.jpg")
                        pil_ref_image = tensor2pil(reference_image)
                        pil_ref_image.save(temp_ref_image)
                        
                        # 使用base64格式传递图像
                        with open(temp_ref_image, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            body_params["image"] = f"data:image/jpeg;base64,{image_data}"
                        
                        print(f"使用单张参考图像进行图生图，模型: {model}")
                        temp_ref_image = [temp_ref_image]  # 统一为列表格式
                else:
                    print(f"警告: 模型 {model} 不支持图生图功能，将忽略参考图像")
                    temp_ref_image = []
            
            formatted_body = json.dumps(body_params)
            print(f"使用模型: {model}")
            print(f"请求参数: {body_params}")
            
            # 准备请求头 - 使用ARK API Key认证
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {ark_api_key}'
            }
            
            # 根据是否启用流式输出选择不同的处理方式
            if stream:
                # 流式输出模式
                response = self.make_request('POST', self.endpoint, headers=headers, data=formatted_body, stream=True)
                
                if response.status != 200:
                    error_msg = f"流式请求失败，状态码: {response.status}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # 解析流式响应
                images_info, usage_info = self.parse_sse_response(response)
                response.release_conn()
                
                if not images_info:
                    raise Exception("流式响应中未获取到任何图片")
                
                # 下载所有图片并转换为tensor
                image_tensors = []
                for i, img_info in enumerate(images_info):
                    temp_img_path = os.path.join(self.temp_dir, f"temp_jimeng_stream_{int(time.time())}_{i}.jpg")
                    
                    print(f"下载第 {i+1} 张图片: {img_info['url']}")
                    img_response = self.make_request('GET', img_info['url'], stream=True)
                    
                    with open(temp_img_path, 'wb') as f:
                        while True:
                            chunk = img_response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                    
                    img_response.release_conn()
                    
                    # 转换为tensor
                    pil_image = Image.open(temp_img_path)
                    image_tensor = pil2tensor(pil_image)
                    image_tensors.append(image_tensor)
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                
                # 合并所有图片tensor
                if len(image_tensors) > 1:
                    combined_tensor = torch.cat(image_tensors, dim=0)
                else:
                    combined_tensor = image_tensors[0]
                
            else:
                # 非流式输出模式
                response_data, status = self.make_request('POST', self.endpoint, headers=headers, data=formatted_body)
                response_data = json.loads(response_data)
                
                if status != 200:
                    error_msg = f"请求失败，状态码: {status}\n错误信息: {response_data}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                print(f"API响应: {response_data}")
                
                # 检查是否有错误
                if 'error' in response_data:
                    error_info = response_data['error']
                    error_msg = f"API返回错误: {error_info.get('message', '未知错误')}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # 处理响应数据
                if 'data' not in response_data:
                    error_msg = f"响应数据格式异常: {response_data}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                data = response_data['data']
                
                # 处理多张图片（组图模式）
                image_tensors = []
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        temp_img_path = os.path.join(self.temp_dir, f"temp_jimeng_batch_{int(time.time())}_{i}.jpg")
                        
                        if response_format == "b64_json":
                            # Base64格式
                            image_data = item.get('b64_json')
                            if image_data:
                                image_bytes = base64.b64decode(image_data)
                                with open(temp_img_path, 'wb') as f:
                                    f.write(image_bytes)
                                print(f"从Base64数据保存第 {i+1} 张图片到: {temp_img_path}")
                        else:
                            # URL格式
                            image_url = item.get('url')
                            if image_url:
                                print(f"下载第 {i+1} 张图片: {image_url}")
                                img_response = self.make_request('GET', image_url, stream=True)
                                
                                with open(temp_img_path, 'wb') as f:
                                    while True:
                                        chunk = img_response.read(8192)
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                
                                img_response.release_conn()
                        
                        # 转换为tensor
                        pil_image = Image.open(temp_img_path)
                        image_tensor = pil2tensor(pil_image)
                        image_tensors.append(image_tensor)
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_img_path)
                        except:
                            pass
                
                # 合并所有图片tensor
                if len(image_tensors) > 1:
                    combined_tensor = torch.cat(image_tensors, dim=0)
                elif len(image_tensors) == 1:
                    combined_tensor = image_tensors[0]
                else:
                    raise Exception("未获取到任何图片")
            
            # 清理临时文件
            try:
                if temp_ref_image:
                    if isinstance(temp_ref_image, list):
                        for temp_file in temp_ref_image:
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                    else:
                        os.remove(temp_ref_image)
            except:
                pass
            
            return (combined_tensor,)
            
        except Exception as e:
            print(f"错误: {str(e)}")
            raise e


class JimengVideoGenerate(ComfyNodeABC):
    """
    即梦视频生成API节点 - jimeng_ti2v_v30_pro
    
    功能:
    - 图生视频：支持单张图片生成视频
    - 支持自定义种子值（-1表示随机，有效范围: 0 到 2147483647）
    - 支持选择视频时长（5秒或10秒）
    - 可手动指定或自动选择视频长宽比
    - 异步任务处理，自动轮询直到视频生成完成
    - 返回视频文件、提取的视频帧和详细日志信息
    
    视频分辨率:
    - 21:9 → 2176x928
    - 16:9 → 1920x1088
    - 4:3 → 1664x1248
    - 1:1 → 1440x1440
    - 3:4 → 1248x1664
    - 9:16 → 1088x1920
    """
    MAX_WAIT_TIME = 15 * 60  # 15分钟
    QUERY_INTERVAL = 5  # 5秒查询一次
    
    # 配置参数
    _cfg_a = "AKLTYTNhY2MzMjk5Zjk0NDY2NDhjMTA1YThjNjk2MGEyYzI"
    _cfg_b = "TTJFeFpqQXlaVE5tWkRNM05ESm1NMkpqTldSaE9XSTFORGMwWldaaFkyUQ=="
    
    def __init__(self):
        self.temp_dir = os.path.join(folder_paths.get_temp_directory(), "jimeng")
        os.makedirs(self.temp_dir, exist_ok=True)
        # 创建自定义的http连接管理器
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.http = PoolManager(
            retries=retry_strategy,
            maxsize=1  # 限制连接池大小
        )
        # 定义API相关配置
        self.method = 'POST'
        self.host = 'visual.volcengineapi.com'
        self.region = 'cn-north-1'
        self.endpoint = 'https://visual.volcengineapi.com'
        self.service = 'cv'
    
    def formatQuery(self, parameters):
        request_parameters_init = ''
        for key in sorted(parameters):
            request_parameters_init += key + '=' + parameters[key] + '&'
        return request_parameters_init[:-1]
    
    def encode_image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def signV4Request(self, access_key, secret_key, req_query, req_body):
        t = datetime.datetime.utcnow()
        current_date = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d')
        canonical_uri = '/'
        canonical_querystring = req_query
        signed_headers = 'content-type;host;x-content-sha256;x-date'
        payload_hash = hashlib.sha256(req_body.encode('utf-8')).hexdigest()
        content_type = 'application/json'
        canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + self.host + \
            '\n' + 'x-content-sha256:' + payload_hash + \
            '\n' + 'x-date:' + current_date + '\n'
        canonical_request = self.method + '\n' + canonical_uri + '\n' + canonical_querystring + \
            '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash

        algorithm = 'HMAC-SHA256'
        credential_scope = datestamp + '/' + self.region + '/' + self.service + '/' + 'request'
        string_to_sign = algorithm + '\n' + current_date + '\n' + credential_scope + '\n' + hashlib.sha256(
            canonical_request.encode('utf-8')).hexdigest()

        signing_key = getSignatureKey(secret_key, datestamp, self.region, self.service)
        signature = hmac.new(signing_key, (string_to_sign).encode(
            'utf-8'), hashlib.sha256).hexdigest()

        authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + \
            credential_scope + ', ' + 'SignedHeaders=' + \
            signed_headers + ', ' + 'Signature=' + signature

        headers = {
            'X-Date': current_date,
            'Authorization': authorization_header,
            'X-Content-Sha256': payload_hash,
            'Content-Type': content_type,
            'Host': self.host
        }
        return headers

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True})
        },
        "optional": {
            "aspect_ratio": (["auto", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"], {"default": "auto"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),  # API要求: [0, 2^31)
            "frames": (["121", "241"], {"default": "121"}),  # 121=5秒, 241=10秒
        }}
    
    RETURN_TYPES = (IO.VIDEO, "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "log")
    FUNCTION = "generate_video"
    CATEGORY = "hhy"

    def make_request(self, method, url, headers=None, data=None, stream=False):
        """统一的请求处理函数"""
        try:
            if headers is None:
                headers = {}
            
            response = self.http.request(
                method,
                url,
                headers=headers,
                body=data,
                preload_content=not stream
            )
            
            if not stream:
                return response.data, response.status
            return response
            
        except Exception as e:
            print(f"请求出错: {str(e)}")
            raise e

    def query_task_status(self, task_id):
        query_params = {
            'Action': 'CVSync2AsyncGetResult',
            'Version': '2022-08-31',
        }
        formatted_query = self.formatQuery(query_params)

        body_params = {
            "req_key": "jimeng_ti2v_v30_pro",
            "task_id": task_id
        }
        formatted_body = json.dumps(body_params)
        
        headers = self.signV4Request(self._cfg_a, self._cfg_b, formatted_query, formatted_body)
        request_url = self.endpoint + '?' + formatted_query
        
        try:
            response_data, status = self.make_request('POST', request_url, headers=headers, data=formatted_body)
            response_data = json.loads(response_data)
            
            if status != 200:
                error_msg = f"请求失败，状态码: {status}\n错误信息: {response_data}"
                print(error_msg)
                raise Exception(error_msg)
                
            if 'data' in response_data:
                data = response_data['data']
                status = data.get('status', 'unknown')
                print(f"当前状态: {status}")
                
                if status == 'done' and 'video_url' in data:
                    return data['video_url']
            else:
                error_msg = f"响应数据异常: {response_data}"
                print(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"查询出错: {str(e)}")
            raise e
        
        return None

    def generate_video(self, image, prompt, aspect_ratio="auto", seed=-1, frames="121"):
        # 记录开始时间
        start_time = time.time()
        
        # 创建临时文件路径
        temp_image = os.path.join(self.temp_dir, "temp_jimeng_input.jpg")
        temp_video = os.path.join(self.temp_dir, f"temp_jimeng_output_{int(time.time())}.mp4")
        
        # 视频比例对应的分辨率
        aspect_ratio_resolutions = {
            "21:9": "2176x928",
            "16:9": "1920x1088",
            "4:3": "1664x1248",
            "1:1": "1440x1440",
            "3:4": "1248x1664",
            "9:16": "1088x1920"
        }
        
        try:
            # 保存tensor图像为临时文件
            pil_image = tensor2pil(image)
            pil_image.save(temp_image)

            # 处理长宽比
            width, height = pil_image.size
            if aspect_ratio == "auto":
                # 自动根据图片尺寸选择最接近的视频比例
                aspect_ratio = get_closest_aspect_ratio(width, height)
                print(f"输入图片尺寸: {width}x{height}, 自动选择视频比例: {aspect_ratio}")
            else:
                # 使用用户指定的长宽比
                print(f"输入图片尺寸: {width}x{height}, 手动指定视频比例: {aspect_ratio}")
            
            frames_int = int(frames)
            duration = 5 if frames_int == 121 else 10
            
            # 获取视频分辨率
            video_resolution = aspect_ratio_resolutions.get(aspect_ratio, "未知")
            
            # 计算计费（一秒一元）
            cost = duration * 1.0
            
            # 构建日志信息（JSON格式）
            log_data = {
                "time": {},
                "info": {
                    "model": "jimeng_ti2v_v30_pro",
                    "prompt": prompt,
                    "video_aspect_ratio": aspect_ratio,
                    "video_resolution": video_resolution,
                    "duration_seconds": duration,
                    "total_frames": frames_int,
                    "seed": seed if seed != -1 else "random"
                },
                "billing": {
                    "cost_yuan": cost
                }
            }

            # 准备请求参数
            query_params = {
                'Action': 'CVSync2AsyncSubmitTask',
                'Version': '2022-08-31',
            }
            formatted_query = self.formatQuery(query_params)

            # 准备请求Body
            body_params = {
                "req_key": "jimeng_ti2v_v30_pro",
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "frames": frames_int,
                "binary_data_base64": [self.encode_image_to_base64(temp_image)]
            }
            
            # 添加seed参数（-1表示随机，不传该参数）
            if seed != -1:
                # API要求 seed 必须在 [0, 2^31) 范围内
                if seed < 0 or seed >= 2147483648:
                    raise ValueError(f"seed 值必须在 [0, 2147483647] 范围内，当前值: {seed}")
                body_params["seed"] = seed  # 使用int类型，不转字符串
            
            # 打印发送请求信息（用于控制台查看）
            print(f"\n发送API请求: model={log_data['info']['model']}, prompt={prompt[:50]}...\n")
            
            formatted_body = json.dumps(body_params)
            
            # 发送请求
            headers = self.signV4Request(self._cfg_a, self._cfg_b, formatted_query, formatted_body)
            request_url = self.endpoint + '?' + formatted_query
            
            response_data, status = self.make_request('POST', request_url, headers=headers, data=formatted_body)
            response_data = json.loads(response_data)
            
            if status != 200:
                error_msg = f"请求失败，状态码: {status}\n错误信息: {response_data}"
                print(error_msg)
                raise Exception(error_msg)
            
            if 'data' not in response_data or 'task_id' not in response_data['data']:
                error_msg = f"未能获取到任务ID\n响应内容: {response_data}"
                print(error_msg)
                raise Exception(error_msg)
                
            task_id = response_data['data']['task_id']
            print(f"获取到任务ID: {task_id}")
            
            # 循环查询任务状态
            video_url = None
            max_attempts = self.MAX_WAIT_TIME // self.QUERY_INTERVAL
            print(f"\n开始查询任务状态，最长等待时间: {self.MAX_WAIT_TIME}秒 ({self.MAX_WAIT_TIME//60}分钟)")
            
            for i in range(max_attempts):
                print(f"\n第 {i+1}/{max_attempts} 次查询状态...")
                try:
                    video_url = self.query_task_status(task_id)
                    if video_url:
                        break
                    time.sleep(self.QUERY_INTERVAL)
                except Exception as e:
                    print(f"查询失败，停止执行: {str(e)}")
                    raise e
            
            if not video_url:
                error_msg = f"任务超时，等待时间超过 {self.MAX_WAIT_TIME//60} 分钟"
                print(error_msg)
                raise Exception(error_msg)

            print(f"\n获取到视频URL: {video_url}")
            # 下载视频到临时文件
            print(f"开始保存视频到: {temp_video}")
            response = self.make_request('GET', video_url, stream=True)
            
            with open(temp_video, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            print(f"视频已保存到: {temp_video}")
            response.release_conn()
            
            # 创建VideoInput对象
            video = VideoFromFile(temp_video)
            
            # 提取视频帧
            print("开始提取视频帧...")
            frames = extract_frames_from_video(temp_video)
            if frames is None:
                print("警告: 无法提取视频帧")
                # 如果无法提取帧，返回一个空的图像批次
                frames = torch.zeros((1, 3, height, width))
            else:
                print(f"成功提取 {frames.shape[0]} 帧")
            
            # 计算总耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 添加完成信息到日志
            log_data["time"]["time_spend"] = round(elapsed_time, 2)
            log_data["status"] = "success"
            log_data["result"] = {
                "extracted_frames": frames.shape[0] if frames is not None else 0,
                "message": "视频生成成功"
            }
            
            # 转换为JSON字符串（使用log代码块格式）
            log_output = "```log\n" + json.dumps(log_data, ensure_ascii=False, indent=2) + "\n```"
            
            return (video, frames, log_output)
            
        except Exception as e:
            # 计算错误发生时的耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 添加错误信息到日志
            if 'log_data' in locals():
                log_data["time"]["time_spend"] = round(elapsed_time, 2)
                log_data["status"] = "error"
                log_data["error"] = {
                    "message": str(e),
                    "type": type(e).__name__
                }
                error_log = "```log\n" + json.dumps(log_data, ensure_ascii=False, indent=2) + "\n```"
            else:
                error_log = "```log\n" + json.dumps({
                    "time": {
                        "time_spend": round(elapsed_time, 2)
                    },
                    "status": "error",
                    "error": {
                        "message": str(e),
                        "type": type(e).__name__
                    }
                }, ensure_ascii=False, indent=2) + "\n```"
            print(f"错误: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "JimengImageGenerate": JimengImageGenerate,
    "JimengVideoGenerate": JimengVideoGenerate
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengImageGenerate": "Jimeng Image Generate",
    "JimengVideoGenerate": "Jimeng Video Generate"
}
