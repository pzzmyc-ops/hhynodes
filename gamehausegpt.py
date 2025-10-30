#!/usr/bin/env python3
"""
Gamehaus API Image Batch Processor - ComfyUI Node

This ComfyUI node processes all images in a folder using the Gamehaus API,
following the same pattern as the OpenAI images/edits API.
"""

import os
import sys
import json
import time
import logging
import requests
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import tempfile
from PIL import Image
import numpy as np
import torch

# 注意: keys_config 模块由 __init__.py 在运行时注入，不需要显式导入
# 如果看到 "keys_config is not defined" 的 linter 警告，可以忽略

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tensor2pil(image):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GamehausAPICaller:
    def __init__(self, config_path: str = None):
        """Initialize the Gamehaus API caller with configuration."""
        # Set default config path relative to this file's directory
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        self.config = self.load_config(config_path)
        
        # 尝试从keys_config获取敏感信息（如果config.yaml未提供）
        default_keys = {}
        if 'keys_config' in globals() and hasattr(keys_config, 'GAMEHAUS_CONFIG'):
            default_keys = keys_config.GAMEHAUS_CONFIG
            logger.info("使用keys_config中的Gamehaus密钥")
        
        # Gamehaus API settings（敏感信息优先使用config.yaml，其次keys_config）
        gamehaus_config = self.config.get('gamehaus', {})
        self.api_key = gamehaus_config.get('api_key') or default_keys.get('api_key')
        self.base_url = gamehaus_config.get('base_url') or default_keys.get('base_url')
        self.account = gamehaus_config.get('account') or default_keys.get('account')
        
        # Image generation settings（非敏感配置，直接使用config.yaml或代码默认值）
        self.model = gamehaus_config.get('model', 'gpt-image-1')
        self.size = gamehaus_config.get('size', '1024x1024')
        self.quality = gamehaus_config.get('quality', 'medium')
        
        # Processing settings
        self.retry_attempts = self.config.get('processing', {}).get('retry_attempts', 3)
        self.retry_delay = self.config.get('processing', {}).get('retry_delay', 1)
        
        # Validation
        if not self.api_key:
            logger.warning("No API key found in config, will try without authentication")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Configuration file '{config_path}' not found, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file '{config_path}': {e}")
            return {}
    
    def edit_image(self, 
                   image_path: str, 
                   prompt: str, 
                   size: str = None, 
                   quality: str = None,
                   model: str = None) -> Optional[Tuple[str, bytes]]:
        """
        Edit an image using the Gamehaus API.
        
        Args:
            image_path: Path to the input image file
            prompt: Text prompt describing the desired changes
            size: Image size (e.g., '1024x1024' or 'auto')
            quality: Image quality ('low', 'medium', 'high')
            model: Model to use for generation
            
        Returns:
            Tuple of (format, image_data) or None if failed
        """
        # Use provided parameters or defaults
        # Support 'auto' mode for size by passing through 'auto', or omitting size entirely when not provided
        size_input = size if (size is not None and str(size).strip() != "") else self.size
        quality = quality or self.quality
        model = model or self.model
        
        use_auto_size = isinstance(size_input, str) and size_input.strip().lower() == 'auto'
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        if not prompt:
            logger.error("No prompt provided for image editing")
            return None
        
        # Build the API endpoint
        endpoint = f"{self.base_url}/v01/v1/images/edits"
        params = {'account': self.account}
        
        # Prepare headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Read the input image
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read image file: {e}")
            return None
        
        # Prepare multipart form data (similar to OpenAI API)
        files = [
            ('image[]', (Path(image_path).name, image_data, 'image/png'))
        ]
        
        # Prepare form data
        data = {
            'model': model,
            'prompt': prompt,
            'quality': quality,
            'n': 1
        }
        
        # Handle size logic: include 'auto' explicitly, otherwise include concrete sizes; allow omitting
        if use_auto_size:
            data['size'] = 'auto'
        elif size_input:
            data['size'] = size_input
        
        # Add any additional parameters that might be needed
        if self.config.get('gamehaus', {}).get('additional_params'):
            data.update(self.config['gamehaus']['additional_params'])
        
        logger.info(f"Calling Gamehaus API: {endpoint}")
        logger.info(f"Parameters: {data}")
        logger.info(f"Image: {image_path}")
        
        # Make API call with retry logic
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.retry_attempts}")
                
                response = requests.post(
                    endpoint,
                    params=params,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=300  # 5 minutes timeout
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Try to parse response
                    try:
                        response_data = response.json()
                        logger.info("Received JSON response")
                        
                        # Handle different response formats
                        if 'data' in response_data and len(response_data['data']) > 0:
                            result = response_data['data'][0]
                            
                            # Check for base64 image data
                            if 'b64_json' in result:
                                logger.info("Received base64 image data")
                                img_data = base64.b64decode(result['b64_json'])
                                return ('base64', img_data)
                            
                            # Check for URL
                            elif 'url' in result:
                                image_url = result['url']
                                logger.info(f"Downloading image from URL: {image_url}")
                                
                                img_response = requests.get(image_url, timeout=60)
                                if img_response.status_code == 200:
                                    return ('url', img_response.content)
                                else:
                                    logger.error(f"Failed to download image: {img_response.status_code}")
                            
                            # Check for direct image data
                            elif 'image' in result:
                                logger.info("Received direct image data")
                                return ('direct', result['image'])
                            
                            else:
                                logger.error("No valid image data in response")
                                logger.debug(f"Response structure: {response_data}")
                        
                        else:
                            logger.error("No data field in API response")
                            logger.debug(f"Response: {response_data}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.debug(f"Raw response: {response.text}")
                        
                        # Try to handle non-JSON responses (e.g., direct image data)
                        if response.headers.get('content-type', '').startswith('image/'):
                            logger.info("Received direct image response")
                            return ('direct', response.content)
                
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Rate limit hit - waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    logger.error("Authentication failed - check your API key")
                    return None
                
                elif response.status_code == 400:
                    logger.error(f"Bad request: {response.text}")
                    return None
                
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return None
            
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out on attempt {attempt + 1}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return None
            
            except Exception as e:
                logger.error(f"Exception during API call: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return None
        
        return None
    
    def generate_image(self, 
                      prompt: str, 
                      size: str = None, 
                      quality: str = None,
                      model: str = None) -> Optional[Tuple[str, bytes]]:
        """
        Generate an image using the Gamehaus API.
        
        Args:
            prompt: Text prompt describing the desired image
            size: Image size (e.g., '1024x1024' or 'auto')
            quality: Image quality ('low', 'medium', 'high')
            model: Model to use for generation
            
        Returns:
            Tuple of (format, image_data) or None if failed
        """
        # Use provided parameters or defaults
        size_input = size if (size is not None and str(size).strip() != "") else self.size
        quality = quality or self.quality
        model = model or self.model
        
        use_auto_size = isinstance(size_input, str) and size_input.strip().lower() == 'auto'
        
        if not prompt:
            logger.error("No prompt provided for image generation")
            return None
        
        # Build the API endpoint for image generation
        endpoint = f"{self.base_url}/v01/v1/images/generations"
        params = {'account': self.account}
        
        # Prepare headers for JSON request
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare JSON data for generation
        data = {
            'model': model,
            'prompt': prompt,
            'quality': quality,
            'n': 1
        }
        
        # Handle size logic: include 'auto' explicitly, otherwise include concrete sizes; allow omitting
        if use_auto_size:
            data['size'] = 'auto'
        elif size_input:
            data['size'] = size_input
        
        # Add any additional parameters that might be needed
        if self.config.get('gamehaus', {}).get('additional_params'):
            data.update(self.config['gamehaus']['additional_params'])
        
        logger.info(f"Calling Gamehaus API for image generation: {endpoint}")
        logger.info(f"Parameters: {data}")
        
        # Make API call with retry logic
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.retry_attempts}")
                
                response = requests.post(
                    endpoint,
                    params=params,
                    headers=headers,
                    json=data,  # Use json parameter instead of data
                    timeout=300  # 5 minutes timeout
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Try to parse response
                    try:
                        response_data = response.json()
                        logger.info("Received JSON response")
                        
                        # Handle different response formats
                        if 'data' in response_data and len(response_data['data']) > 0:
                            result = response_data['data'][0]
                            
                            # Check for base64 image data
                            if 'b64_json' in result:
                                logger.info("Received base64 image data")
                                img_data = base64.b64decode(result['b64_json'])
                                return ('base64', img_data)
                            
                            # Check for URL
                            elif 'url' in result:
                                image_url = result['url']
                                logger.info(f"Downloading image from URL: {image_url}")
                                
                                img_response = requests.get(image_url, timeout=60)
                                if img_response.status_code == 200:
                                    return ('url', img_response.content)
                                else:
                                    logger.error(f"Failed to download image: {img_response.status_code}")
                            
                            # Check for direct image data
                            elif 'image' in result:
                                logger.info("Received direct image data")
                                return ('direct', result['image'])
                            
                            else:
                                logger.error("No valid image data in response")
                                logger.debug(f"Response structure: {response_data}")
                        
                        else:
                            logger.error("No data field in API response")
                            logger.debug(f"Response: {response_data}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.debug(f"Raw response: {response.text}")
                        
                        # Try to handle non-JSON responses (e.g., direct image data)
                        if response.headers.get('content-type', '').startswith('image/'):
                            logger.info("Received direct image response")
                            return ('direct', response.content)
                
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Rate limit hit - waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    logger.error("Authentication failed - check your API key")
                    return None
                
                elif response.status_code == 400:
                    logger.error(f"Bad request: {response.text}")
                    return None
                
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return None
            
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out on attempt {attempt + 1}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return None
            
            except Exception as e:
                logger.error(f"Exception during API call: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return None
        
        return None
    
    def save_result(self, result: Tuple[str, bytes], output_path: str) -> bool:
        """Save the API result to a file."""
        try:
            if result[0] == 'base64' or result[0] == 'url' or result[0] == 'direct':
                with open(output_path, 'wb') as f:
                    f.write(result[1])
                logger.info(f"Result saved to: {output_path}")
                return True
            else:
                logger.error(f"Unknown result format: {result[0]}")
                return False
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return False

    def process_folder(self, 
                      folder_path: str, 
                      prompt: str, 
                      size: str = None, 
                      quality: str = None,
                      model: str = None,
                      output_folder: str = None) -> Dict[str, bool]:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to the input folder
            prompt: Text prompt describing the desired changes
            size: Image size (e.g., '1024x1024')
            quality: Image quality ('low', 'medium', 'high')
            model: Model to use for generation
            output_folder: Output folder path (optional)
            
        Returns:
            Dictionary mapping input files to success status
        """
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Folder not found or not a directory: {folder_path}")
            return {}
        
        # Set output folder
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_folder = folder_path / "edited"
            output_folder.mkdir(exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            # Use case-insensitive pattern matching to avoid duplicates
            image_files.extend(folder_path.glob(f"*{ext}"))
            # Also check for uppercase extensions
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_image_files = []
        for img_file in image_files:
            if str(img_file) not in seen:
                seen.add(str(img_file))
                unique_image_files.append(img_file)
        image_files = unique_image_files
        
        if not image_files:
            logger.warning(f"No image files found in folder: {folder_path}")
            return {}
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Check for existing files in output folder and filter out already processed files
        files_to_process = []
        skipped_files = []
        
        for image_file in image_files:
            output_path = output_folder / f"{image_file.name}"
            if output_path.exists():
                skipped_files.append(image_file)
                logger.info(f"Skipping already processed file: {image_file.name}")
            else:
                files_to_process.append(image_file)
        
        # Log statistics
        total_files = len(image_files)
        files_to_process_count = len(files_to_process)
        skipped_count = len(skipped_files)
        
        logger.info(f"Processing statistics:")
        logger.info(f"  Total files found: {total_files}")
        logger.info(f"  Files to process: {files_to_process_count}")
        logger.info(f"  Files skipped (already exist): {skipped_count}")
        
        if files_to_process_count == 0:
            logger.info("All files have already been processed. Nothing to do.")
            return {}
        
        # Process each image
        results = {}
        for i, image_file in enumerate(files_to_process, 1):
            logger.info(f"Processing image {i}/{files_to_process_count}: {image_file.name}")
            
            # Generate output path
            output_path = output_folder / f"{image_file.name}"
            
            # Process the image
            result = self.edit_image(
                image_path=str(image_file),
                prompt=prompt,
                size=size,
                quality=quality,
                model=model
            )
            
            if result:
                # Save the result
                if self.save_result(result, str(output_path)):
                    results[str(image_file)] = True
                    logger.info(f"Successfully processed: {image_file.name}")
                else:
                    results[str(image_file)] = False
                    logger.error(f"Failed to save result for: {image_file.name}")
            else:
                results[str(image_file)] = False
                logger.error(f"Failed to process: {image_file.name}")
            
            # Add delay between API calls to avoid rate limiting
            if i < files_to_process_count:
                logger.info("Waiting 2 seconds before next image...")
                time.sleep(2)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        processed = len(results)
        logger.info(f"Processing complete: {successful}/{processed} images processed successfully")
        logger.info(f"Total summary: {skipped_count} files skipped, {processed} files processed, {successful} files successful")
        
        return results


class GamehausImageBatchProcessor:
    """ComfyUI Node for processing images with Gamehaus API - supports both single image and batch processing"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "remove image's text and ui and make it more clear, do not change the image's art style", 
                    "multiline": True,
                    "placeholder": "输入处理提示词"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "input_folder": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "输入图片文件夹路径（批量处理）"
                }),
                "size": ("STRING", {
                    "default": "auto", 
                    "multiline": False,
                    "placeholder": "图片尺寸 (如: 1024x1024 或 auto)"
                }),
                "quality": (["medium", "low", "high"], {"default": "medium"}),
                "model": ("STRING", {
                    "default": "gpt-image-1", 
                    "multiline": False,
                    "placeholder": "模型名称"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "result_summary")
    OUTPUT_NODE = True
    FUNCTION = "process_images"
    CATEGORY = "hhy/api"

    def process_images(self, prompt, seed, image=None, input_folder="", size="auto", quality="medium", model="gpt-image-1"):
        """Process single image or batch process images in folder using Gamehaus API"""
        
        if not prompt or not prompt.strip():
            error_msg = "❌ 提示词不能为空"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        has_image_input = image is not None
        has_folder = input_folder and input_folder.strip()
        
        
        if has_image_input and has_folder:
            error_msg = "❌ 请只选择一种输入模式：图片输入、文件夹批量处理 或 纯文本生成"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        try:
            # Initialize the API caller
            caller = GamehausAPICaller()
            
            if has_image_input:
                # Single image editing mode
                return self._process_single_image_tensor(caller, image, prompt, size, quality, model, seed)
            elif has_folder:
                # Folder batch processing mode
                return self._process_folder(caller, input_folder.strip(), prompt, size, quality, model, seed)
            else:
                # Text-to-image generation mode (no image input)
                return self._generate_image_from_text(caller, prompt, size, quality, model, seed)
                
        except Exception as e:
            error_msg = f"❌ 处理过程中发生错误: {str(e)}"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
    
    def _process_single_image_tensor(self, caller, image_tensor, prompt, size, quality, model, seed):
        """Process a single image from tensor input"""
        try:
            # Convert tensor to PIL image
            if len(image_tensor.shape) == 4:
                # Batch of images, take the first one
                pil_image = tensor2pil(image_tensor[0])
            else:
                pil_image = tensor2pil(image_tensor)
            
            # Save PIL image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                pil_image.save(temp_path, 'PNG')
            
            print(f"🔄 开始处理单个图片...")
            print(f"💬 提示词: {prompt}")
            print(f"🎲 种子: {seed}")
            print(f"📏 尺寸: {size}")
            print(f"🎨 质量: {quality}")
            print(f"🤖 模型: {model}")
            
            # Process the image
            result = caller.edit_image(
                image_path=temp_path,
                prompt=prompt,
                size=size if size and size.strip() else None,
                quality=quality,
                model=model
            )
            
            if result:
                # Save result to temporary file and convert back to tensor
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_temp:
                    output_path = output_temp.name
                
                if caller.save_result(result, output_path):
                    # Load processed image and convert to tensor
                    processed_pil = Image.open(output_path)
                    processed_tensor = pil2tensor(processed_pil)
                    
                    # Clean up temporary files
                    try:
                        os.unlink(temp_path)
                        os.unlink(output_path)
                    except:
                        pass
                    
                    success_msg = f"✅ 单个图片处理完成！"
                    print(success_msg)
                    return (processed_tensor, success_msg)
                else:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    error_msg = f"❌ 保存处理结果失败"
                    print(error_msg)
                    return (pil2tensor(pil_image), error_msg)
            else:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                error_msg = f"❌ 图片处理失败"
                print(error_msg)
                return (pil2tensor(pil_image), error_msg)
                
        except Exception as e:
            error_msg = f"❌ 处理图片时发生错误: {str(e)}"
            print(error_msg)
            # Return original image on error
            if len(image_tensor.shape) == 4:
                return (image_tensor[0:1], error_msg)
            else:
                return (image_tensor.unsqueeze(0), error_msg)
    
    def _generate_image_from_text(self, caller, prompt, size, quality, model, seed):
        """Generate an image from text prompt using Gamehaus API"""
        try:
            print(f"🔄 开始从文本生成图片...")
            print(f"💬 提示词: {prompt}")
            print(f"🎲 种子: {seed}")
            print(f"📏 尺寸: {size}")
            print(f"🎨 质量: {quality}")
            print(f"🤖 模型: {model}")
            
            # Generate the image
            result = caller.generate_image(
                prompt=prompt,
                size=size if size and size.strip() else None,
                quality=quality,
                model=model
            )
            
            if result:
                # Save result to temporary file and convert to tensor
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_temp:
                    output_path = output_temp.name
                
                if caller.save_result(result, output_path):
                    # Load generated image and convert to tensor
                    generated_pil = Image.open(output_path)
                    generated_tensor = pil2tensor(generated_pil)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(output_path)
                    except:
                        pass
                    
                    success_msg = f"✅ 图片生成完成！"
                    print(success_msg)
                    return (generated_tensor, success_msg)
                else:
                    error_msg = f"❌ 保存生成结果失败"
                    print(error_msg)
                    empty_image = torch.zeros((1, 512, 512, 3))
                    return (empty_image, error_msg)
            else:
                error_msg = f"❌ 图片生成失败"
                print(error_msg)
                empty_image = torch.zeros((1, 512, 512, 3))
                return (empty_image, error_msg)
                
        except Exception as e:
            error_msg = f"❌ 生成图片时发生错误: {str(e)}"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
    
    def _process_folder(self, caller, input_folder, prompt, size, quality, model, seed):
        """Process all images in a folder (original batch processing logic)"""
        # Normalize the input folder path
        input_folder = os.path.normpath(input_folder)
        
        if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
            error_msg = f"❌ 输入文件夹不存在或不是有效目录: {input_folder}"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        # Set output folder to be at the same level as input folder
        input_path = Path(input_folder)
        output_folder = input_path.parent / f"{input_path.name}_edited"
        
        print(f"🔄 开始批量处理图片...")
        print(f"📁 输入文件夹: {input_folder}")
        print(f"📁 输出文件夹: {output_folder}")
        print(f"💬 提示词: {prompt}")
        print(f"🎲 种子: {seed}")
        print(f"📏 尺寸: {size}")
        print(f"🎨 质量: {quality}")
        print(f"🤖 模型: {model}")
        
        # Process the folder
        results = caller.process_folder(
            folder_path=input_folder,
            prompt=prompt,
            size=size if size and size.strip() else None,
            quality=quality,
            model=model,
            output_folder=str(output_folder)
        )
        
        # For folder processing, return a placeholder image and summary
        empty_image = torch.zeros((1, 512, 512, 3))
        
        if results:
            successful = sum(1 for success in results.values() if success)
            processed = len(results)
            
            # Count existing files in output folder for skip statistics
            input_path = Path(input_folder)
            output_folder_path = input_path.parent / f"{input_path.name}_edited"
            
            # Count total input files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            total_input_files = 0
            for ext in image_extensions:
                total_input_files += len(list(input_path.glob(f"*{ext}")))
                total_input_files += len(list(input_path.glob(f"*{ext.upper()}")))
            
            # Remove duplicates count (rough estimate)
            skipped = max(0, total_input_files - processed)
            
            if successful == processed:
                success_msg = f"✅ 批量处理完成！成功处理 {successful}/{processed} 张图片"
                if skipped > 0:
                    success_msg += f"，跳过 {skipped} 张已存在的图片"
                success_msg += f"\n输出目录: {output_folder_path}"
                print(success_msg)
                return (empty_image, success_msg)
            else:
                partial_msg = f"⚠️ 部分处理完成：成功处理 {successful}/{processed} 张图片"
                if skipped > 0:
                    partial_msg += f"，跳过 {skipped} 张已存在的图片"
                partial_msg += f"\n输出目录: {output_folder_path}"
                print(partial_msg)
                return (empty_image, partial_msg)
        else:
            # Check if all files were skipped
            input_path = Path(input_folder)
            output_folder_path = input_path.parent / f"{input_path.name}_edited"
            
            if output_folder_path.exists():
                # Count files in both directories to see if all were skipped
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                input_count = 0
                for ext in image_extensions:
                    input_count += len(list(input_path.glob(f"*{ext}")))
                    input_count += len(list(input_path.glob(f"*{ext.upper()}")))
                
                output_count = 0
                for ext in image_extensions:
                    output_count += len(list(output_folder_path.glob(f"*{ext}")))
                    output_count += len(list(output_folder_path.glob(f"*{ext.upper()}")))
                
                if output_count > 0:
                    skip_msg = f"ℹ️ 所有文件都已处理过，跳过了 {input_count} 张图片\n输出目录: {output_folder_path}"
                    print(skip_msg)
                    return (empty_image, skip_msg)
            
            error_msg = "❌ 没有找到可处理的图片文件"
            print(error_msg)
            return (empty_image, error_msg)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "GamehausImageBatchProcessor": GamehausImageBatchProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GamehausImageBatchProcessor": "Gamehaus gptimage api"
}
