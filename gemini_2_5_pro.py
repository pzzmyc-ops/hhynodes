import os
import json
import base64
import io
import requests
import torch
import tempfile
import time
from PIL import Image
import numpy as np

# 注意: keys_config 模块由 __init__.py 在运行时注入，不需要显式导入
# 如果看到 "keys_config is not defined" 的 linter 警告，可以忽略

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("[Gemini 2.5 Pro] soundfile not available, audio support disabled")

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

def file_to_base64(file_path: str) -> str:
    """Convert file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_audio_to_temp(audio_data: dict, filename_prefix: str = "audio", format: str = "wav") -> str:
    """Save audio data to temporary file"""
    if not SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required for audio support")
    
    sample_rate = audio_data.get("sample_rate")
    waveform = audio_data.get("waveform")
    
    if not sample_rate or waveform is None:
        raise ValueError("Audio data must contain 'sample_rate' and 'waveform'")
    
    # Handle different waveform shapes
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
    
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    
    subtype = "FLOAT" if format.lower() == "wav" else None
    sf.write(temp_file, waveform.T, sample_rate, subtype=subtype)
    
    return temp_file

def save_video_to_temp(video_input, filename_prefix: str = "video", format: str = "mp4") -> str:
    """Save video data to temporary file"""
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{filename_prefix}_{int(time.time())}.{format}")
    
    # Use ComfyUI's video save method if available
    if hasattr(video_input, 'save_to'):
        video_input.save_to(temp_file, format=format)
    else:
        raise ValueError("Video input must have 'save_to' method")
    
    return temp_file


class Gemini25ProNode:
    """ComfyUI node for Gemini 2.5 Pro via GameHaus API"""
    
    def __init__(self):
        pass
    
    def _create_debug_payload(self, payload):
        """创建调试用的payload，截断base64数据"""
        debug_payload = json.loads(json.dumps(payload))  # 深拷贝
        
        if 'messages' in debug_payload:
            for message in debug_payload['messages']:
                if 'content' in message and isinstance(message['content'], list):
                    for item in message['content']:
                        if isinstance(item, dict) and 'image_url' in item:
                            if 'url' in item['image_url']:
                                original_url = item['image_url']['url']
                                if len(original_url) > 100:
                                    item['image_url']['url'] = f"{original_url[:50]}...[TRUNCATED {len(original_url)} chars]...{original_url[-50:]}"
        
        return debug_payload
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Describe this image", 
                    "multiline": True
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking mode for deeper reasoning"
                }),
                "thinking_budget_tokens": ("INT", {
                    "default": 2000,
                    "min": 100,
                    "max": 10000,
                    "tooltip": "Maximum tokens for thinking process"
                }),
                "audio_timestamp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable timestamp understanding for audio inputs"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("output_image", "text_response", "thinking_content")
    FUNCTION = "generate"
    CATEGORY = "hhy/api"
    INPUT_IS_LIST = True

    def generate(self, 
                 prompt,
                 image1=None,
                 image2=None,
                 image3=None,
                 video=None,
                 audio=None,
                 enable_thinking=False,
                 thinking_budget_tokens=2000,
                 audio_timestamp=True,
                 seed=0):
        """
        Generate with Gemini 2.5 Pro via GameHaus API
        Note: seed parameter is used to trigger ComfyUI regeneration, not sent to API
        Note: INPUT_IS_LIST = True, so all parameters come as lists
        """
        # Process list inputs - extract first value for non-image parameters
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if isinstance(enable_thinking, list):
            enable_thinking = enable_thinking[0] if enable_thinking else False
        if isinstance(thinking_budget_tokens, list):
            thinking_budget_tokens = thinking_budget_tokens[0] if thinking_budget_tokens else 2000
        if isinstance(audio_timestamp, list):
            audio_timestamp = audio_timestamp[0] if audio_timestamp else True
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
            
        # 从keys_config获取API key
        api_key = ""
        if 'keys_config' in globals() and hasattr(keys_config, 'GAMEHAUS_GEMINI_CONFIG'):
            gemini_config = keys_config.GAMEHAUS_GEMINI_CONFIG
            api_key = gemini_config.get('api_key', '')
            if api_key:
                print("[Gemini 2.5 Pro] 使用keys_config中的Gemini API密钥")
            else:
                print("[Gemini 2.5 Pro] ❌ keys_config中未找到Gemini API密钥")
                empty_image = torch.zeros((1, 3, 512, 512))
                return (empty_image, "Error: API key not configured in keys_config", "")
        else:
            print("[Gemini 2.5 Pro] ❌ 未找到keys_config配置")
            empty_image = torch.zeros((1, 3, 512, 512))
            return (empty_image, "Error: keys_config not found", "")
        
        # 先判断是否为纯文本生成，以备错误处理使用
        is_text_only = (image1 is None or (isinstance(image1, list) and len(image1) == 0)) and \
                       (image2 is None or (isinstance(image2, list) and len(image2) == 0)) and \
                       (image3 is None or (isinstance(image3, list) and len(image3) == 0)) and \
                       (video is None or (isinstance(video, list) and len(video) == 0)) and \
                       (audio is None or (isinstance(audio, list) and len(audio) == 0))
        
        try:
            # 收集所有提供的图像
            images = []
            
            # Process image1 as list - can contain multiple images
            if image1 is not None and isinstance(image1, list) and len(image1) > 0:
                print(f"[Gemini 2.5 Pro] image1 contains {len(image1)} images in list")
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
            print(f"[Gemini 2.5 Pro] Processing {total_images} input images")
            
            # 判断是否为纯文本生成
            is_text_only = total_images == 0
            if is_text_only:
                print("[Gemini 2.5 Pro] Text-only generation mode")
            else:
                print("[Gemini 2.5 Pro] Multi-modal mode")
            
            # Prepare content array (OpenAI-style format)
            content_array = []
            
            # Add text content
            content_array.append({
                "type": "text",
                "text": prompt
            })
            
            # Add all input images
            for i, image in enumerate(images):
                # Process each image (should be 3D tensor [H, W, C] at this point)
                if len(image.shape) == 3:
                    input_tensor = torch.unsqueeze(image, 0)  # Add batch dim [1, H, W, C]
                else:
                    input_tensor = image
                
                input_pil = tensor2pil(input_tensor)
                input_base64 = pil_to_base64(input_pil, "PNG")
                
                content_array.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{input_base64}"
                    }
                })
                print(f"[Gemini 2.5 Pro] Added image {i+1}/{total_images}")
            
            # Process video inputs
            temp_files = []  # Track temp files for cleanup
            
            if video is not None and isinstance(video, list) and len(video) > 0:
                print(f"[Gemini 2.5 Pro] Processing {len(video)} video inputs")
                for i, video_item in enumerate(video):
                    if video_item is not None:
                        try:
                            # Save video to temp file
                            temp_video = save_video_to_temp(video_item, f"gemini_video_{i}")
                            temp_files.append(temp_video)
                            
                            # Convert to base64
                            video_base64 = file_to_base64(temp_video)
                            
                            # Get file extension to determine MIME type
                            ext = os.path.splitext(temp_video)[1].lower()
                            mime_type = "video/mp4" if ext == ".mp4" else f"video/{ext[1:]}"
                            
                            content_array.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{video_base64}"
                                }
                            })
                            print(f"[Gemini 2.5 Pro] Added video {i+1}/{len(video)} ({mime_type})")
                        except Exception as e:
                            print(f"[Gemini 2.5 Pro] ⚠ Failed to process video {i+1}: {e}")
            
            # Process audio inputs
            if audio is not None and isinstance(audio, list) and len(audio) > 0:
                if not SOUNDFILE_AVAILABLE:
                    print("[Gemini 2.5 Pro] ⚠ Audio input detected but soundfile not available")
                else:
                    print(f"[Gemini 2.5 Pro] Processing {len(audio)} audio inputs")
                    for i, audio_item in enumerate(audio):
                        if audio_item is not None:
                            try:
                                # Convert audio dict to expected format if needed
                                if isinstance(audio_item, dict):
                                    audio_data = audio_item
                                else:
                                    print(f"[Gemini 2.5 Pro] ⚠ Unexpected audio format for item {i+1}")
                                    continue
                                
                                # Save audio to temp file
                                temp_audio = save_audio_to_temp(audio_data, f"gemini_audio_{i}")
                                temp_files.append(temp_audio)
                                
                                # Convert to base64
                                audio_base64 = file_to_base64(temp_audio)
                                
                                # Get file extension to determine MIME type
                                ext = os.path.splitext(temp_audio)[1].lower()
                                mime_type = "audio/wav" if ext == ".wav" else f"audio/{ext[1:]}"
                                
                                content_array.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{audio_base64}"
                                    }
                                })
                                print(f"[Gemini 2.5 Pro] Added audio {i+1}/{len(audio)} ({mime_type})")
                            except Exception as e:
                                print(f"[Gemini 2.5 Pro] ⚠ Failed to process audio {i+1}: {e}")
            
            # Prepare API payload (OpenAI-compatible format)
            try:
                # Determine if we should use content_array or plain text
                # Use content_array if we have multiple items (text + media)
                has_media = len(content_array) > 1
                
                payload = {
                    "model": "MaaS_2.5_pro_20250617",
                    "messages": [
                        {
                            "role": "user",
                            "content": content_array if has_media else prompt
                        }
                    ]
                }
                
                # Add thinking configuration if enabled
                if enable_thinking:
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": int(thinking_budget_tokens)
                    }
                    print(f"[Gemini 2.5 Pro] Thinking mode enabled with budget: {thinking_budget_tokens} tokens")
                
                # Add audio timestamp configuration if enabled
                if audio_timestamp:
                    payload["config"] = {
                        "audio_timestamp": True
                    }
                    print(f"[Gemini 2.5 Pro] Audio timestamp understanding enabled")
                
                # Set up headers
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": api_key
                }
                
                # GameHaus API URL with MaaS 2.5 Pro
                api_url = "https://api-robot-v1.gamehaus.com/v02/v1/maas_2.5_pro_20250617"
                
                print(f"[Gemini 2.5 Pro] Calling API...")
                print(f"[Gemini 2.5 Pro] API URL: {api_url}")
                print(f"[Gemini 2.5 Pro] Prompt: {prompt}")
                safe_headers = dict(headers)
                if 'Authorization' in safe_headers:
                    safe_headers['Authorization'] = '[REDACTED]'
                print(f"[Gemini 2.5 Pro] Request headers: {safe_headers}")
                print(f"[Gemini 2.5 Pro] Payload structure: messages with {len(content_array) if has_media else 1} content items")
                print(f"[Gemini 2.5 Pro] Has media: {has_media}, Content array length: {len(content_array)}")
                if enable_thinking:
                    print(f"[Gemini 2.5 Pro] Thinking: enabled, budget={thinking_budget_tokens}")
                
                # 打印完整请求（截断base64部分）
                debug_payload = self._create_debug_payload(payload)
                print(f"[Gemini 2.5 Pro] Complete request payload:")
                print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
                
                # Make API request
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=120  # 2分钟超时
                )
                
                print(f"[Gemini 2.5 Pro] Response status: {response.status_code}")
            
                if response.status_code != 200:
                    error_text = response.text
                    print(f"[Gemini 2.5 Pro] ❌ API request failed")
                    print(f"[Gemini 2.5 Pro] Status code: {response.status_code}")
                    
                    # 尝试解析错误响应
                    try:
                        error_json = json.loads(error_text)
                        if 'error' in error_json:
                            error_msg = error_json['error']
                            print(f"[Gemini 2.5 Pro] Error: {error_msg}")
                            # 根据是否为纯文本生成返回合适的fallback
                            if is_text_only:
                                fallback_tensor = torch.zeros((1, 3, 512, 512))
                            else:
                                # Return first image from collected images as fallback
                                first_img = images[0]
                                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                            return (fallback_tensor, f"API Error: {error_msg}", "")
                    except:
                        pass
                    
                    # 限制错误文本长度
                    if len(error_text) > 200:
                        error_display = error_text[:200] + "..."
                    else:
                        error_display = error_text
                        
                    print(f"[Gemini 2.5 Pro] Error response: {error_display}")
                    # 根据是否为纯文本生成返回合适的fallback
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        # Return first image from collected images as fallback
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    return (fallback_tensor, f"Error: API request failed with status {response.status_code}", "")
                
                # Process successful response
                print(f"[Gemini 2.5 Pro] Response content length: {len(response.content)}")
                try:
                    hdrs = dict(response.headers)
                    if 'Authorization' in hdrs:
                        hdrs['Authorization'] = '[REDACTED]'
                    print(f"[Gemini 2.5 Pro] Response headers: {hdrs}")
                except Exception:
                    print("[Gemini 2.5 Pro] Response headers: <unavailable>")
                
                # 检查响应是否为空
                if not response.content or len(response.content) == 0:
                    print("[Gemini 2.5 Pro] ❌ Empty response received")
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    return (fallback_tensor, "Error: Empty response from API", "")
                
                # 尝试解析JSON
                result = response.json()
                print(f"[Gemini 2.5 Pro] Response received and parsed successfully")
                print(f"[Gemini 2.5 Pro] Response data: {json.dumps(result, indent=2, ensure_ascii=False)[:1000]}")
                
                # Parse OpenAI-compatible response format
                # Expected format: {"choices": [{"message": {"content": "...", "thinking": "..."}}]}
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    message = choice.get('message', {})
                    
                    # Print full message object for debugging
                    print(f"[Gemini 2.5 Pro] ===== Full Message Object =====")
                    print(json.dumps(message, indent=2, ensure_ascii=False))
                    print(f"[Gemini 2.5 Pro] ===== Message Keys: {list(message.keys())} =====")
                    
                    message_content = message.get('content', '')
                    
                    # Extract reasoning content
                    thinking_content = message.get('reasoning_content', '')
                    
                    if thinking_content:
                        print(f"[Gemini 2.5 Pro] Reasoning length: {len(thinking_content)} chars")
                    
                    print(f"[Gemini 2.5 Pro] Response length: {len(message_content)} chars")
                    
                    # Return the text response with a placeholder image
                    if is_text_only:
                        fallback_tensor = torch.zeros((1, 3, 512, 512))
                    else:
                        first_img = images[0]
                        fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                    
                    return (fallback_tensor, message_content, thinking_content)
                
                # Fallback: check for old S3 format (in case API still returns this)
                elif result.get('success') == True and 's3_url' in result:
                    s3_url = result['s3_url']
                    print(f"[Gemini 2.5 Pro] S3 URL: {s3_url}")
                    
                    # Download image from S3
                    try:
                        print("[Gemini 2.5 Pro] Downloading image from S3...")
                        img_response = requests.get(s3_url, timeout=60)
                        
                        if img_response.status_code == 200:
                            # Convert downloaded image to PIL
                            image_data = img_response.content
                            output_pil = Image.open(io.BytesIO(image_data))
                            result_tensor = pil2tensor(output_pil)
                            
                            print(f"[Gemini 2.5 Pro] Successfully downloaded and processed image: {output_pil.size}")
                            return (result_tensor, f"Image generated successfully", "")
                        else:
                            print(f"[Gemini 2.5 Pro] ❌ Failed to download image from S3: {img_response.status_code}")
                            if is_text_only:
                                fallback_tensor = torch.zeros((1, 3, 512, 512))
                            else:
                                first_img = images[0]
                                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                            return (fallback_tensor, f"Error: Failed to download generated image", "")
                            
                    except Exception as e:
                        print(f"[Gemini 2.5 Pro] ❌ Error downloading image: {e}")
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            first_img = images[0]
                            fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                        return (fallback_tensor, f"Error: Failed to download image - {str(e)}", "")
                else:
                    # Handle API error response
                    if 'error' in result:
                        error_msg = result['error']
                        print(f"[Gemini 2.5 Pro] API Error: {error_msg}")
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            first_img = images[0]
                            fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                        return (fallback_tensor, f"API Error: {error_msg}", "")
                    else:
                        print("[Gemini 2.5 Pro] Unexpected response format")
                        if is_text_only:
                            fallback_tensor = torch.zeros((1, 3, 512, 512))
                        else:
                            first_img = images[0]
                            fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
                        return (fallback_tensor, "Error: Unexpected API response format", "")
                
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print(f"[Gemini 2.5 Pro] Cleaned up temp file: {os.path.basename(temp_file)}")
                    except Exception as e:
                        print(f"[Gemini 2.5 Pro] Failed to clean up {temp_file}: {e}")
            
        except requests.exceptions.Timeout:
            print("[Gemini 2.5 Pro] ❌ Request timeout")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, "Error: Request timeout", "")
        except requests.exceptions.RequestException as e:
            print(f"[Gemini 2.5 Pro] ❌ Request exception: {e}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, f"Error: Request failed - {str(e)}", "")
        except Exception as e:
            print(f"[Gemini 2.5 Pro] ❌ Error: {str(e)}")
            if is_text_only:
                fallback_tensor = torch.zeros((1, 3, 512, 512))
            else:
                first_img = images[0]
                fallback_tensor = torch.unsqueeze(first_img, 0) if len(first_img.shape) == 3 else first_img
            return (fallback_tensor, f"Error: {str(e)}", "")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Gemini25Pro": Gemini25ProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini25Pro": "Gemini 2.5 Pro by hhy",
}

