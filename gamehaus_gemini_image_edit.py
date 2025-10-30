import os
import json
import base64
import io
import requests
import torch
from PIL import Image
import numpy as np

# 注意: keys_config 模块由 __init__.py 在运行时注入，不需要显式导入
# 如果看到 "keys_config is not defined" 的 linter 警告，可以忽略

def tensor2pil(image):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_data = buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')


class GameHausGeminiImageEditNode:
    """ComfyUI node for GameHaus Gemini image editing API"""
    
    def __init__(self):
        pass
    
    def _create_debug_payload(self, payload):
        """创建调试用的payload，截断base64数据"""
        debug_payload = json.loads(json.dumps(payload))  # 深拷贝
        
        if 'contents' in debug_payload:
            for content in debug_payload['contents']:
                if 'parts' in content:
                    for part in content['parts']:
                        if 'inline_data' in part and 'data' in part['inline_data']:
                            original_data = part['inline_data']['data']
                            if len(original_data) > 100:
                                part['inline_data']['data'] = f"{original_data[:50]}...[TRUNCATED {len(original_data)} chars]...{original_data[-50:]}"
        
        return debug_payload
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_prompt": ("STRING", {
                    "default": "Generate or edit images", 
                    "multiline": True
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "留空则使用配置文件中的API Key"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "text_response")
    FUNCTION = "edit_image_gamehaus"
    CATEGORY = "hhy/api"
    INPUT_IS_LIST = True

    def edit_image_gamehaus(self, 
                           edit_prompt,
                           api_key=None,
                           image1=None,
                           image2=None,
                           image3=None,
                           seed=0):
        """
        Edit image using GameHaus Gemini API
        Note: seed parameter is used to trigger ComfyUI regeneration, not sent to API
        Note: INPUT_IS_LIST = True, so all parameters come as lists
        """
        # Process list inputs - extract first value for non-image parameters
        if isinstance(edit_prompt, list):
            edit_prompt = edit_prompt[0] if edit_prompt else ""
        if isinstance(api_key, list):
            api_key = api_key[0] if api_key else ""
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
            
        # 尝试从keys_config获取API key（如果用户未提供）
        if not api_key or api_key.strip() == "":
            if 'keys_config' in globals() and hasattr(keys_config, 'GAMEHAUS_GEMINI_CONFIG'):
                gemini_config = keys_config.GAMEHAUS_GEMINI_CONFIG
                api_key = gemini_config.get('api_key', '')
                if api_key:
                    print("[GameHaus Gemini] 使用keys_config中的Gemini API密钥")
                else:
                    print("[GameHaus Gemini] ❌ keys_config中未找到Gemini API密钥")
                    empty_image = torch.zeros((1, 3, 512, 512))
                    return (empty_image, "Error: API key not configured in keys_config")
            else:
                print("[GameHaus Gemini] ❌ 未找到keys_config配置且未提供API key")
                empty_image = torch.zeros((1, 3, 512, 512))
                return (empty_image, "Error: No API key provided and keys_config not found")
        else:
            print("[GameHaus Gemini] 使用用户提供的API key")
        
        # 先判断是否为纯文本生成，以备错误处理使用
        is_text_only = (image1 is None or (isinstance(image1, list) and len(image1) == 0)) and \
                       (image2 is None or (isinstance(image2, list) and len(image2) == 0)) and \
                       (image3 is None or (isinstance(image3, list) and len(image3) == 0))
        
        try:
            # 收集所有提供的图像
            images = []
            
            # Process image1 as list - can contain multiple images
            if image1 is not None and isinstance(image1, list) and len(image1) > 0:
                print(f"[GameHaus Gemini] image1 contains {len(image1)} images in list")
                for img_item in image1:
                    if img_item is not None:
                        # Each item can be batch [B, H, W, C] or single [H, W, C]
                        if len(img_item.shape) == 4:
                            # Batch - add all images
                            for i in range(img_item.shape[0]):
                                images.append(img_item[i])
                        elif len(img_item.shape) == 3:
                            # Single image
                            images.append(img_item)
            
            # Process image2 and image3 - extract from list and use first image
            if image2 is not None and isinstance(image2, list) and len(image2) > 0:
                img2_item = image2[0]
                if img2_item is not None:
                    img2 = img2_item[0] if len(img2_item.shape) == 4 else img2_item
                    images.append(img2)
            
            if image3 is not None and isinstance(image3, list) and len(image3) > 0:
                img3_item = image3[0]
                if img3_item is not None:
                    img3 = img3_item[0] if len(img3_item.shape) == 4 else img3_item
                    images.append(img3)
            
            total_images = len(images)
            print(f"[GameHaus Gemini] Processing {total_images} input images in total")
            
            # 判断是否为纯文本生成
            is_text_only = total_images == 0
            if is_text_only:
                print("[GameHaus Gemini] Text-only generation mode (no input images)")
            else:
                print("[GameHaus Gemini] Image editing mode")
            
            # Prepare parts array
            parts = [
                {
                    "text": edit_prompt
                }
            ]
            
            # Add all input images
            for i, image in enumerate(images):
                # Process each image (should be 3D tensor [H, W, C] at this point)
                if len(image.shape) == 3:
                    input_tensor = torch.unsqueeze(image, 0)  # Add batch dim [1, H, W, C]
                else:
                    input_tensor = image
                
                input_pil = tensor2pil(input_tensor)
                input_base64 = pil_to_base64(input_pil, "PNG")
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": input_base64
                    }
                })
                print(f"[GameHaus Gemini] Added image {i+1}/{total_images}")
            
            # Prepare GameHaus API payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }
            
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": api_key
            }
            
            # GameHaus API URL
            api_url = "https://api-robot-v1.gamehaus.com/v02/vertex/models/gemini-2.5-flash-image-preview:predictLongRunning"
            
            print(f"[GameHaus Gemini] Calling GameHaus API...")
            print(f"[GameHaus Gemini] API URL: {api_url}")
            print(f"[GameHaus Gemini] Edit prompt: {edit_prompt}")
            safe_headers = dict(headers)
            if 'Authorization' in safe_headers:
                safe_headers['Authorization'] = '[REDACTED]'
            print(f"[GameHaus Gemini] Request headers: {safe_headers}")
            print(f"[GameHaus Gemini] Payload structure: contents with {len(parts)} parts")
            
            # 打印完整请求（截断base64部分）
            debug_payload = self._create_debug_payload(payload)
            print(f"[GameHaus Gemini] Complete request payload:")
            print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
            
            # Make API request
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120  # 2分钟超时，因为可能需要较长时间生成
            )
            
            print(f"[GameHaus Gemini] Response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                print(f"[GameHaus Gemini] ❌ API request failed")
                print(f"[GameHaus Gemini] Status code: {response.status_code}")
                
                # 尝试解析错误响应
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        error_msg = error_json['error']
                        print(f"[GameHaus Gemini] Error: {error_msg}")
                        # 根据是否为纯文本生成返回合适的fallback
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            # Return first image from collected images as fallback
                            first_img = images[0]
                            fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                        return (fallback_tensor, f"API Error: {error_msg}")
                except:
                    pass
                
                # 限制错误文本长度
                if len(error_text) > 200:
                    error_display = error_text[:200] + "..."
                else:
                    error_display = error_text
                    
                print(f"[GameHaus Gemini] Error response: {error_display}")
                # 根据是否为纯文本生成返回合适的fallback
                if is_text_only:
                    fallback_tensor = torch.zeros((1, 3, 512, 512))
                else:
                    # Return first image from collected images as fallback
                    first_img = images[0]
                    fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                return (fallback_tensor, f"Error: API request failed with status {response.status_code}")
            
            # Process successful response
            print(f"[GameHaus Gemini] Response content length: {len(response.content)}")
            try:
                hdrs = dict(response.headers)
                if 'Authorization' in hdrs:
                    hdrs['Authorization'] = '[REDACTED]'
                print(f"[GameHaus Gemini] Response headers: {hdrs}")
            except Exception:
                print("[GameHaus Gemini] Response headers: <unavailable>")
            
            # 检查响应是否为空
            if not response.content or len(response.content) == 0:
                print("[GameHaus Gemini] ❌ Empty response received")
                if is_text_only:
                    fallback_tensor = torch.zeros((1, 3, 512, 512))
                else:
                    first_img = images[0]
                    fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                return (fallback_tensor, "Error: Empty response from API")
            
            # 尝试解析JSON
            try:
                result = response.json()
                print(f"[GameHaus Gemini] Response received and parsed successfully")
            except json.JSONDecodeError as e:
                print(f"[GameHaus Gemini] ❌ JSON decode error: {e}")
                print(f"[GameHaus Gemini] Raw response: {response.text[:500]}...")
                if is_text_only:
                    fallback_tensor = torch.zeros((1, 3, 512, 512))
                else:
                    first_img = images[0]
                    fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                return (fallback_tensor, f"Error: Invalid JSON response - {str(e)}")
            
            if result.get('success') == True and 's3_url' in result:
                s3_url = result['s3_url']
                print(f"[GameHaus Gemini] S3 URL: {s3_url}")
                
                # Download image from S3
                try:
                    print("[GameHaus Gemini] Downloading image from S3...")
                    img_response = requests.get(s3_url, timeout=60)
                    
                    if img_response.status_code == 200:
                        # Convert downloaded image to PIL
                        image_data = img_response.content
                        edited_pil = Image.open(io.BytesIO(image_data))
                        result_tensor = pil2tensor(edited_pil)
                        
                        print(f"[GameHaus Gemini] Successfully downloaded and processed image: {edited_pil.size}")
                        return (result_tensor, f"Image generated successfully from GameHaus API")
                    else:
                        print(f"[GameHaus Gemini] ❌ Failed to download image from S3: {img_response.status_code}")
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            first_img = images[0]
                            fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                        return (fallback_tensor, f"Error: Failed to download generated image")
                        
                except Exception as e:
                    print(f"[GameHaus Gemini] ❌ Error downloading image: {e}")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    return (fallback_tensor, f"Error: Failed to download image - {str(e)}")
            else:
                # Handle API error response
                if 'error' in result:
                    error_msg = result['error']
                    print(f"[GameHaus Gemini] API Error: {error_msg}")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    return (fallback_tensor, f"API Error: {error_msg}")
                else:
                    print("[GameHaus Gemini] Unexpected response format")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    return (fallback_tensor, "Error: Unexpected API response format")
            
        except requests.exceptions.Timeout:
            print("[GameHaus Gemini] ❌ Request timeout")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, "Error: Request timeout")
        except requests.exceptions.RequestException as e:
            print(f"[GameHaus Gemini] ❌ Request exception: {e}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, f"Error: Request failed - {str(e)}")
        except Exception as e:
            print(f"[GameHaus Gemini] ❌ Error in image editing: {str(e)}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, f"Error: {str(e)}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GameHausGeminiImageEdit": GameHausGeminiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GameHausGeminiImageEdit": "GameHaus Gemini Image Edit",
}

