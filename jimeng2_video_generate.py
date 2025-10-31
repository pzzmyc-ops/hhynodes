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

# 注意: keys_config 模块由 __init__.py 在运行时注入，不需要显式导入
# 如果看到 "keys_config is not defined" 的 linter 警告，可以忽略

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

class Jimeng2VideoGenerate(ComfyNodeABC):
    """
    即梦2.0视频生成API节点
    
    功能:
    - 文生视频：不输入图片，仅用提示词生成视频 (jimeng_t2v_v30)
    - 首帧生成：输入1张图片，使用首帧生成视频 (jimeng_i2v_first_v30)
    - 首尾帧生成：输入2张图片，使用首尾帧生成视频 (jimeng_i2v_first_tail_v30)
    
    参数:
    - prompt: 用于生成视频的提示词，建议在400字以内，不超过800字
    - seed: 随机种子，-1表示随机，有效范围: [0, 2147483647]
    - frames: 生成的总帧数，121=5秒，241=10秒
    - aspect_ratio: 视频长宽比（仅文生视频支持）
    
    注意:
    - 图片格式：JPEG、PNG
    - 图片大小：最大 4.7MB
    - 图片分辨率：最大 4096 * 4096，最短边不低于320
    - 图片长边与短边比例在3以内
    - 尾帧图片需与首帧图片比例相同
    """
    MAX_WAIT_TIME = 15 * 60  # 15分钟
    QUERY_INTERVAL = 5  # 5秒查询一次
    
    def __init__(self):
        self.temp_dir = os.path.join(folder_paths.get_temp_directory(), "jimeng2")
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
            "prompt": ("STRING", {"default": "", "multiline": True})
        },
        "optional": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            "frames": (["121", "241"], {"default": "121"}),  # 121=5秒, 241=10秒
            "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"], {"default": "16:9"}),  # 仅文生视频使用
        }}
    
    RETURN_TYPES = (IO.VIDEO, "IMAGE", "STRING")
    RETURN_NAMES = ("video", "frames", "log")
    FUNCTION = "generate_video"
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

    def query_task_status(self, task_id, req_key):
        query_params = {
            'Action': 'CVSync2AsyncGetResult',
            'Version': '2022-08-31',
        }
        formatted_query = self.formatQuery(query_params)

        body_params = {
            "req_key": req_key,
            "task_id": task_id
        }
        formatted_body = json.dumps(body_params)
        
        # 从keys_config获取密钥（由__init__.py注入）
        access_key = keys_config.JIMENG_VIDEO_ACCESS_KEY if 'keys_config' in globals() else ""
        secret_key = keys_config.JIMENG_VIDEO_SECRET_KEY if 'keys_config' in globals() else ""
        
        if not access_key or not secret_key:
            raise Exception("密钥配置未加载，请确认密钥文件是否存在")
        
        headers = self.signV4Request(access_key, secret_key, formatted_query, formatted_body)
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

    def generate_video(self, prompt, image1=None, image2=None, seed=-1, frames="121", aspect_ratio="16:9"):
        # 记录开始时间
        start_time = time.time()
        
        # 确定使用的模型和图片数量
        has_image1 = image1 is not None
        has_image2 = image2 is not None
        
        if has_image1 and has_image2:
            # 两张图片：使用首尾帧生成
            req_key = "jimeng_i2v_first_tail_v30"
            model_name = "即梦2.0 - 首尾帧生成"
            num_images = 2
        elif has_image1:
            # 一张图片：使用首帧生成
            req_key = "jimeng_i2v_first_v30"
            model_name = "即梦2.0 - 首帧生成"
            num_images = 1
        else:
            # 没有图片：使用文生视频
            req_key = "jimeng_t2v_v30"
            model_name = "即梦2.0 - 文生视频"
            num_images = 0
        
        
        # 创建临时文件路径
        temp_images = []
        temp_video = os.path.join(self.temp_dir, f"temp_jimeng2_output_{int(time.time())}.mp4")
        
        try:
            # 准备请求参数
            query_params = {
                'Action': 'CVSync2AsyncSubmitTask',
                'Version': '2022-08-31',
            }
            formatted_query = self.formatQuery(query_params)

            # 准备请求Body
            body_params = {
                "req_key": req_key,
                "prompt": prompt,
                "frames": int(frames),
            }
            
            # 添加seed参数（-1表示随机，不传该参数）
            if seed != -1:
                # API要求 seed 必须在 [0, 2^31) 范围内
                if seed < 0 or seed >= 2147483648:
                    raise ValueError(f"seed 值必须在 [0, 2147483647] 范围内，当前值: {seed}")
                body_params["seed"] = seed
            
            # 根据模型类型添加不同的参数
            if num_images == 0:
                # 文生视频：添加长宽比参数
                body_params["aspect_ratio"] = aspect_ratio
                
            elif num_images == 1:
                # 首帧生成：添加1张图片
                temp_image1 = os.path.join(self.temp_dir, f"temp_image1_{int(time.time())}.jpg")
                temp_images.append(temp_image1)
                
                pil_image1 = tensor2pil(image1)
                pil_image1.save(temp_image1)
                
                # 检查图片约束
                width, height = pil_image1.size
                file_size = os.path.getsize(temp_image1) / (1024 * 1024)  # MB
                aspect_ratio_val = max(width, height) / min(width, height)
                
                if file_size > 4.7:
                    raise ValueError(f"图片文件大小 {file_size:.2f}MB 超过4.7MB限制")
                if max(width, height) > 4096:
                    raise ValueError(f"图片最大边 {max(width, height)} 超过4096限制")
                if min(width, height) < 320:
                    raise ValueError(f"图片最小边 {min(width, height)} 低于320限制")
                if aspect_ratio_val > 3:
                    raise ValueError(f"图片长宽比 {aspect_ratio_val:.2f} 超过3:1限制")
                
                body_params["binary_data_base64"] = [self.encode_image_to_base64(temp_image1)]
                
            elif num_images == 2:
                # 首尾帧生成：添加2张图片
                temp_image1 = os.path.join(self.temp_dir, f"temp_image1_{int(time.time())}.jpg")
                temp_image2 = os.path.join(self.temp_dir, f"temp_image2_{int(time.time())}.jpg")
                temp_images.extend([temp_image1, temp_image2])
                
                pil_image1 = tensor2pil(image1)
                pil_image2 = tensor2pil(image2)
                pil_image1.save(temp_image1)
                pil_image2.save(temp_image2)
                
                # 检查图片约束
                width1, height1 = pil_image1.size
                width2, height2 = pil_image2.size
                file_size1 = os.path.getsize(temp_image1) / (1024 * 1024)  # MB
                file_size2 = os.path.getsize(temp_image2) / (1024 * 1024)  # MB
                aspect_ratio_val1 = max(width1, height1) / min(width1, height1)
                aspect_ratio_val2 = max(width2, height2) / min(width2, height2)
                
                # 验证图片约束
                for i, (w, h, size, ratio) in enumerate([(width1, height1, file_size1, aspect_ratio_val1), 
                                                           (width2, height2, file_size2, aspect_ratio_val2)], 1):
                    if size > 4.7:
                        raise ValueError(f"图片{i}文件大小 {size:.2f}MB 超过4.7MB限制")
                    if max(w, h) > 4096:
                        raise ValueError(f"图片{i}最大边 {max(w, h)} 超过4096限制")
                    if min(w, h) < 320:
                        raise ValueError(f"图片{i}最小边 {min(w, h)} 低于320限制")
                    if ratio > 3:
                        raise ValueError(f"图片{i}长宽比 {ratio:.2f} 超过3:1限制")
                
                # 检查两张图片比例是否相同
                ratio1 = width1 / height1
                ratio2 = width2 / height2
                
                body_params["binary_data_base64"] = [
                    self.encode_image_to_base64(temp_image1),
                    self.encode_image_to_base64(temp_image2)
                ]
            
            frames_int = int(frames)
            duration = 5 if frames_int == 121 else 10
            
            # 计算计费（即梦AI-视频生成3.0 720P: 0.28元/秒）
            cost_per_second = 0.28
            cost = duration * cost_per_second
            
            # 构建日志信息（JSON格式）
            log_data = {
                "time": {},
                "info": {
                    "model": req_key,
                    "prompt": prompt,
                    "duration_seconds": duration,
                    "total_frames": frames_int,
                    "seed": seed if seed != -1 else "random"
                },
                "billing": {
                    "cost_yuan": round(cost, 2)
                }
            }
            
            # 文生视频时添加长宽比信息
            if num_images == 0:
                log_data["info"]["video_aspect_ratio"] = aspect_ratio
            
            # 打印发送请求信息（用于控制台查看）
            print(f"\n发送API请求: model={model_name}, prompt={prompt[:50]}...\n")
            
            formatted_body = json.dumps(body_params)
            
            # 从keys_config获取密钥（由__init__.py注入）
            access_key = keys_config.JIMENG_VIDEO_ACCESS_KEY if 'keys_config' in globals() else ""
            secret_key = keys_config.JIMENG_VIDEO_SECRET_KEY if 'keys_config' in globals() else ""
            
            if not access_key or not secret_key:
                raise Exception("密钥配置未加载，请确认密钥文件是否存在")
            
            # 发送请求
            headers = self.signV4Request(access_key, secret_key, formatted_query, formatted_body)
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
            
            for i in range(max_attempts):
                print(f"\n第 {i+1}/{max_attempts} 次查询状态...")
                try:
                    video_url = self.query_task_status(task_id, req_key)
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

            # 下载视频到临时文件
            response = self.make_request('GET', video_url, stream=True)
            
            with open(temp_video, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            response.release_conn()
            
            # 创建VideoInput对象
            video = VideoFromFile(temp_video)
            
            # 提取视频帧
            extracted_frames = extract_frames_from_video(temp_video)
            if extracted_frames is None:
                # 如果无法提取帧，返回一个空的图像批次
                extracted_frames = torch.zeros((1, 3, 512, 512))
            
            # 清理临时图片文件
            for temp_img in temp_images:
                try:
                    os.remove(temp_img)
                except:
                    pass
            
            # 计算总耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 添加完成信息到日志
            log_data["time"]["time_spend"] = round(elapsed_time, 2)
            log_data["status"] = "success"
            log_data["result"] = {
                "extracted_frames": extracted_frames.shape[0] if extracted_frames is not None else 0,
                "message": "视频生成成功"
            }
            
            # 转换为JSON字符串（使用log代码块格式）
            log_output = "```log\n" + json.dumps(log_data, ensure_ascii=False, indent=2) + "\n```"
            
            return (video, extracted_frames, log_output)
            
        except Exception as e:
            # 清理临时文件
            for temp_img in temp_images:
                try:
                    os.remove(temp_img)
                except:
                    pass
            
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
    "Jimeng2VideoGenerate": Jimeng2VideoGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng2VideoGenerate": "Jimeng first last frame Video Generate  "
}

