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
        
        # Gamehaus API settings
        self.api_key = self.config.get('gamehaus', {}).get('api_key', 'dee57439-c3b7-4bdc-963d-cf007223d73d')
        self.base_url = self.config.get('gamehaus', {}).get('base_url', 'https://api-robot-v1.gamehaus.com')
        self.account = self.config.get('gamehaus', {}).get('account', '08')
        
        # Image generation settings
        self.model = self.config.get('gamehaus', {}).get('model', 'gpt-image-1')
        self.size = self.config.get('gamehaus', {}).get('size', '1024x1024')
        self.quality = self.config.get('gamehaus', {}).get('quality', 'medium')
        
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
        
        # Process each image
        results = {}
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {image_file.name}")
            
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
            if i < len(image_files):
                logger.info("Waiting 2 seconds before next image...")
                time.sleep(2)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Processing complete: {successful}/{total} images processed successfully")
        
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
    CATEGORY = "hhy/image_processing"

    def process_images(self, prompt, image=None, input_folder="", size="auto", quality="medium", model="gpt-image-1"):
        """Process single image or batch process images in folder using Gamehaus API"""
        
        if not prompt or not prompt.strip():
            error_msg = "❌ 提示词不能为空"
            print(error_msg)
            # Return empty image tensor and error message
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        # Check which input mode to use
        has_image_input = image is not None
        has_folder = input_folder and input_folder.strip()
        
        if not has_image_input and not has_folder:
            error_msg = "❌ 请提供图片输入或文件夹路径"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        if has_image_input and has_folder:
            error_msg = "❌ 请只选择一种输入模式：图片输入 或 文件夹批量处理"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
        
        try:
            # Initialize the API caller
            caller = GamehausAPICaller()
            
            if has_image_input:
                # Single image processing mode
                return self._process_single_image_tensor(caller, image, prompt, size, quality, model)
            else:
                # Folder batch processing mode
                return self._process_folder(caller, input_folder.strip(), prompt, size, quality, model)
                
        except Exception as e:
            error_msg = f"❌ 处理过程中发生错误: {str(e)}"
            print(error_msg)
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, error_msg)
    
    def _process_single_image_tensor(self, caller, image_tensor, prompt, size, quality, model):
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
    
    def _process_folder(self, caller, input_folder, prompt, size, quality, model):
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
            total = len(results)
            
            if successful == total:
                success_msg = f"✅ 批量处理完成！成功处理 {successful}/{total} 张图片\n输出目录: {output_folder}"
                print(success_msg)
                return (empty_image, success_msg)
            else:
                partial_msg = f"⚠️ 部分处理完成：成功处理 {successful}/{total} 张图片\n输出目录: {output_folder}"
                print(partial_msg)
                return (empty_image, partial_msg)
        else:
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


# Keep the original main function for standalone usage
def main():
    """Run the Gamehaus API caller with folder or single file processing."""
    # Defaults
    config_path = "config.yaml"
    
    print("Gamehaus API 图片处理工具")
    print("=" * 40)
    print("1. 处理单个图片文件")
    print("2. 处理文件夹中的所有图片")
    print("=" * 40)
    
    choice = input("请选择处理模式 (1 或 2): ").strip()
    
    if choice == "1":
        # Single file processing
        image_path = input("请输入图片路径 (例如: input.png 或 /path/to/image.jpg): ").strip()
        if not image_path:
            logger.error("图片路径不能为空")
            return 1
        
        # Remove quotes if user accidentally included them
        image_path = image_path.strip('"\'')
        
        prompt = input("请输入处理提示词 (默认: change this image's style to realistic photograph): ").strip()
        if not prompt:
            prompt = "change this image's style to realistic photograph"
        
        size = input("请输入图片尺寸 (默认: 1024x1024，可输入 auto 自动选择/保持原尺寸): ").strip()
        if not size:
            size = None
        
        quality = input("请输入图片质量 (low/medium/high, 默认: medium): ").strip()
        if not quality:
            quality = None
        
        model = input("请输入模型名称 (默认: gpt-image-1): ").strip()
        if not model:
            model = None

        # Initialize API caller
        caller = GamehausAPICaller(config_path=config_path)

        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.name}"

        logger.info(f"Input image: {image_path}")
        logger.info(f"Output image: {output_path}")
        logger.info(f"Prompt: {prompt}")

        if not input_path.exists():
            logger.error(f"Input image not found: {input_path.resolve()}")
            return 1

        # Call the API
        logger.info("Calling Gamehaus API...")
        result = caller.edit_image(
            image_path=image_path,
            prompt=prompt,
            size=size,
            quality=quality,
            model=model
        )

        if result:
            logger.info("API call successful!")

            # Save the result
            if caller.save_result(result, output_path):
                logger.info("Image editing completed successfully!")
                return 0
            else:
                logger.error("Failed to save the result")
                return 1
        else:
            logger.error("API call failed")
            return 1
    
    elif choice == "2":
        # Folder processing
        folder_path = input("请输入图片文件夹路径: ").strip()
        if not folder_path:
            logger.error("文件夹路径不能为空")
            return 1
        
        # Remove quotes if user accidentally included them
        folder_path = folder_path.strip('"\'')
        
        prompt = input("请输入处理提示词 (默认: change this image's style to realistic photograph): ").strip()
        if not prompt:
            prompt = "remove image's text and ui and make it more clear,do not change the image's art style"
        
        size = input("请输入图片尺寸 (默认: 1024x1024，可输入 auto 自动选择/保持原尺寸): ").strip()
        if not size:
            size = None
        
        quality = input("请输入图片质量 (low/medium/high, 默认: medium): ").strip()
        if not quality:
            quality = None
        
        model = input("请输入模型名称 (默认: gpt-image-1): ").strip()
        if not model:
            model = None
        
        output_folder = input("请输入输出文件夹路径 (默认: 在输入文件夹内创建 'edited' 子文件夹): ").strip()
        if not output_folder:
            output_folder = None
        
        # Initialize API caller
        caller = GamehausAPICaller(config_path=config_path)
        
        logger.info(f"Processing folder: {folder_path}")
        logger.info(f"Prompt: {prompt}")
        if output_folder:
            logger.info(f"Output folder: {output_folder}")
        
        # Process the folder
        results = caller.process_folder(
            folder_path=folder_path,
            prompt=prompt,
            size=size,
            quality=quality,
            model=model,
            output_folder=output_folder
        )
        
        if results:
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"Folder processing complete: {successful}/{total} images processed successfully")
            return 0 if successful == total else 1
        else:
            logger.error("No images were processed")
            return 1
    
    else:
        logger.error("无效的选择，请输入 1 或 2")
        return 1


if __name__ == "__main__":
    sys.exit(main())