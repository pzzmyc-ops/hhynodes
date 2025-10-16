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
# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit": GeminiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEdit": "Gemini Image Edit",
}
