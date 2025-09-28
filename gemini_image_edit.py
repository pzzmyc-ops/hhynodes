import os
import json
import base64
import mimetypes
import io
import requests
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
import numpy as np

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Gemini Image Edit] Warning: google-genai not installed. Please install with: pip install google-genai")

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

def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def get_mime_type(format: str) -> str:
    """Get MIME type from image format"""
    format_map = {
        "PNG": "image/png",
        "JPEG": "image/jpeg", 
        "JPG": "image/jpeg",
        "WEBP": "image/webp"
    }
    return format_map.get(format.upper(), "image/png")


# Removed GeminiImageEditor class - functionality merged into node


class GeminiImageEditNode:
    """ComfyUI node for Gemini image editing with advanced features"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "edit_prompt": ("STRING", {
                    "default": "Edit this image", 
                    "multiline": True
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "model": (["gemini-2.5-flash-image-preview"], {
                    "default": "gemini-2.5-flash-image-preview"
                }),
                "include_text_response": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "text_response")
    FUNCTION = "edit_image"
    CATEGORY = "hhy/api"

    def edit_image(self, 
                   input_image,
                   edit_prompt,
                   api_key,
                   reference_image=None,
                   model="gemini-2.5-flash-image-preview",
                   include_text_response=False):
        """
        Edit image using Gemini - matches the original source code structure
        """
        if not GEMINI_AVAILABLE:
            print("[Gemini Image Edit] Gemini client not available")
            empty_image = torch.zeros((1, 3, 512, 512))
            return empty_image, "Error: Gemini not available"
        
        # Validate API key
        if not api_key or api_key.strip() == "":
            print("[Gemini Image Edit] ❌ API key is empty")
            empty_image = torch.zeros((1, 3, 512, 512))
            return empty_image, "Error: Empty API key"
        
        try:
            # Initialize client
            client = genai.Client(api_key=api_key)
            
            # Process input image
            input_tensor = input_image[0] if len(input_image.shape) == 4 else input_image
            input_tensor = torch.unsqueeze(input_tensor, 0)
            input_pil = tensor2pil(input_tensor)
            
            # Prepare content parts (matching source code structure)
            content_parts = [
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(pil_to_base64(input_pil, "PNG"))
                )
            ]
            
            # Add reference image if provided
            if reference_image is not None:
                reference_tensor = reference_image[0] if len(reference_image.shape) == 4 else reference_image
                reference_tensor = torch.unsqueeze(reference_tensor, 0)
                reference_pil = tensor2pil(reference_tensor)
                content_parts.append(types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(pil_to_base64(reference_pil, "PNG"))
                ))
            
            # Add text prompt
            content_parts.append(types.Part.from_text(text=edit_prompt))
            
            # Prepare contents (simplified structure like source code)
            contents = [
                types.Content(
                    role="user",
                    parts=content_parts
                )
            ]
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"] if include_text_response else ["IMAGE"]
            )
            
            print(f"[Gemini Image Edit] Calling Gemini API (model: {model})...")
            print(f"[Gemini Image Edit] Edit prompt: {edit_prompt}")
            if reference_image is not None:
                print("[Gemini Image Edit] Using reference image")
            
            # Generate content
            edited_images = []
            text_responses = []
            
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                # Handle image data
                if (chunk.candidates[0].content.parts[0].inline_data and 
                    chunk.candidates[0].content.parts[0].inline_data.data):
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    
                    # Convert binary data to PIL Image
                    edited_pil = Image.open(io.BytesIO(data_buffer))
                    edited_images.append(edited_pil)
                    print(f"[Gemini Image Edit] Received edited image: {edited_pil.size}")
                
                # Handle text response
                elif chunk.text:
                    text_responses.append(chunk.text)
                    print(f"[Gemini Image Edit] Text response: {chunk.text}")
            
            # Process results
            if edited_images:
                result_image = edited_images[0]
                result_tensor = pil2tensor(result_image)
                text_response = " ".join(text_responses) if text_responses else ""
                
                print(f"[Gemini Image Edit] Successfully edited image")
                return result_tensor, text_response
            else:
                print("[Gemini Image Edit] No edited image received, returning original")
                text_response = " ".join(text_responses) if text_responses else "No image generated"
                return input_tensor, text_response
            
        except Exception as e:
            print(f"[Gemini Image Edit] Error in image editing: {str(e)}")
            input_tensor = input_image[0] if len(input_image.shape) == 4 else input_image
            input_tensor = torch.unsqueeze(input_tensor, 0)
            return input_tensor, f"Error: {str(e)}"



 


class GameHausGeminiImageEditNode:
    """ComfyUI node for GameHaus Gemini image editing API"""
    
    # 写死的API key - 需要找负责人要
    FIXED_API_KEY = "627300aa-3742-29e0-69d4-bb2dd9081965"  # TODO: 需要替换为实际的API key
    
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
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
                "include_text_response": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "text_response")
    FUNCTION = "edit_image_gamehaus"
    CATEGORY = "hhy/api"

    def edit_image_gamehaus(self, 
                           edit_prompt,
                           image1=None,
                           image2=None,
                           image3=None,
                           temperature=1.0,
                           top_p=1.0,
                           max_tokens=2048,
                           seed=0,
                           include_text_response=True):
        """
        Edit image using GameHaus Gemini API
        """
        # 使用固定的API key
        api_key = self.FIXED_API_KEY
        print("[GameHaus Gemini] 使用内置API key")
        
        if api_key == "需要找负责人要":
            print("[GameHaus Gemini] ❌ API key未设置，请联系负责人获取")
            empty_image = torch.zeros((1, 3, 512, 512))
            return empty_image, "Error: API key not configured"
        
        # 先判断是否为纯文本生成，以备错误处理使用
        is_text_only = image1 is None and image2 is None and image3 is None
        
        try:
            # 收集所有提供的图像
            images = []
            if image1 is not None:
                images.append(image1)
            if image2 is not None:
                images.append(image2)
            if image3 is not None:
                images.append(image3)
            
            print(f"[GameHaus Gemini] Processing {len(images)} input images")
            
            # 判断是否为纯文本生成
            is_text_only = len(images) == 0
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
                # Process each image
                input_tensor = image[0] if len(image.shape) == 4 else image
                input_tensor = torch.unsqueeze(input_tensor, 0)
                input_pil = tensor2pil(input_tensor)
                input_base64 = pil_to_base64(input_pil, "PNG")
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": input_base64
                    }
                })
                print(f"[GameHaus Gemini] Added image{i+1}")
            
            # Prepare GameHaus API payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxOutputTokens": max_tokens,
                    "seed": seed,
                    "responseModalities": ["TEXT", "IMAGE"] if include_text_response else ["IMAGE"]
                },
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
            print(f"[GameHaus Gemini] Temperature: {temperature}, Top-P: {top_p}")
            print(f"[GameHaus Gemini] Max tokens: {max_tokens}, Seed: {seed}")
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
                            fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                        return fallback_tensor, f"API Error: {error_msg}"
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
                    fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                return fallback_tensor, f"Error: API request failed with status {response.status_code}"
            
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
                    fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                return fallback_tensor, "Error: Empty response from API"
            
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
                    fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                return fallback_tensor, f"Error: Invalid JSON response - {str(e)}"
            
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
                        return result_tensor, f"Image generated successfully from GameHaus API"
                    else:
                        print(f"[GameHaus Gemini] ❌ Failed to download image from S3: {img_response.status_code}")
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                        return fallback_tensor, f"Error: Failed to download generated image"
                        
                except Exception as e:
                    print(f"[GameHaus Gemini] ❌ Error downloading image: {e}")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                    return fallback_tensor, f"Error: Failed to download image - {str(e)}"
            else:
                # Handle API error response
                if 'error' in result:
                    error_msg = result['error']
                    print(f"[GameHaus Gemini] API Error: {error_msg}")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                    return fallback_tensor, f"API Error: {error_msg}"
                else:
                    print("[GameHaus Gemini] Unexpected response format")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
                    return fallback_tensor, "Error: Unexpected API response format"
            
        except requests.exceptions.Timeout:
            print("[GameHaus Gemini] ❌ Request timeout")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
            return fallback_tensor, "Error: Request timeout"
        except requests.exceptions.RequestException as e:
            print(f"[GameHaus Gemini] ❌ Request exception: {e}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
            return fallback_tensor, f"Error: Request failed - {str(e)}"
        except Exception as e:
            print(f"[GameHaus Gemini] ❌ Error in image editing: {str(e)}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                fallback_tensor = torch.unsqueeze(image1[0], 0) if len(image1.shape) == 4 else torch.unsqueeze(image1, 0)
            return fallback_tensor, f"Error: {str(e)}"


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit": GeminiImageEditNode,
    "GameHausGeminiImageEdit": GameHausGeminiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEdit": "Gemini Image Edit",
    "GameHausGeminiImageEdit": "GameHaus Gemini Image Edit",
}
