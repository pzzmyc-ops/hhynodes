import os
import json
from typing import List, Dict, Any
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Gemini Detection] Warning: google-genai not installed. Please install with: pip install google-genai")

def tensor2pil(image):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image):
    """Convert PIL Image to mask tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def add_mask(mask1, mask2):
    """Add two masks together"""
    return torch.clamp(mask1 + mask2, 0.0, 1.0)

def draw_bbox_on_image(image, bboxes, labels=None, colors=None):
    """Draw bounding boxes on image"""
    if not isinstance(image, Image.Image):
        image = tensor2pil(image)
    
    # Create a copy to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Default colors
    if colors is None:
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            if font:
                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width, text_height = len(label) * 10, 15
            
            # Draw background for text
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], fill=color)
            
            # Draw text
            if font:
                draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
            else:
                draw.text((x1 + 5, y1 - text_height - 2), label, fill="white")
    
    return draw_image

def create_mask_from_bboxes(image_size, bboxes, merge_masks=False):
    """Create masks from bounding boxes"""
    width, height = image_size
    individual_masks = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Create a mask for this bounding box
        mask = Image.new('L', (width, height), "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x1, y1, x2, y2], fill="white")
        individual_masks.append(image2mask(mask))
    
    if merge_masks and individual_masks:
        # Merge all masks
        final_mask = individual_masks[0]
        for i in range(1, len(individual_masks)):
            final_mask = add_mask(final_mask, individual_masks[i])
        return [final_mask]
    else:
        return individual_masks

def parse_gemini_response(response_text: str, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """Parse Gemini response and convert normalized coordinates to absolute coordinates"""
    try:
        response_data = json.loads(response_text)
        print(f"[Gemini Detection] 响应数据类型: {type(response_data)}")
        
        # 处理不同的响应格式
        bounding_boxes = []
        
        if isinstance(response_data, dict):
            # 如果是单个对象，转换为列表
            if "box_2d" in response_data:
                bounding_boxes = [response_data]
            else:
                print("[Gemini Detection] 响应中没有找到box_2d字段")
                return []
        elif isinstance(response_data, list):
            # 如果已经是列表，直接使用
            bounding_boxes = response_data
        else:
            print(f"[Gemini Detection] 未知的响应格式: {type(response_data)}")
            return []
        
        items = []
        print(f"[Gemini Detection] 图像尺寸: {img_width} x {img_height}")
        
        for bounding_box in bounding_boxes:
            if isinstance(bounding_box, dict):
                # Gemini 2.0+ returns [ymin, xmin, ymax, xmax] normalized to [0, 1000]
                box_2d = bounding_box.get("box_2d", [])
                label = bounding_box.get("label", "detected_object")
                
                if len(box_2d) >= 4:
                    # Convert normalized coordinates (0-1000) to absolute coordinates
                    # Gemini format: [ymin, xmin, ymax, xmax]
                    abs_y1 = int(box_2d[0] / 1000 * img_height)  # ymin
                    abs_x1 = int(box_2d[1] / 1000 * img_width)   # xmin
                    abs_y2 = int(box_2d[2] / 1000 * img_height)  # ymax
                    abs_x2 = int(box_2d[3] / 1000 * img_width)   # xmax
                    
                    # Ensure coordinates are in correct order
                    if abs_x1 > abs_x2:
                        abs_x1, abs_x2 = abs_x2, abs_x1
                    if abs_y1 > abs_y2:
                        abs_y1, abs_y2 = abs_y2, abs_y1
                    
                    # Return in [x1, y1, x2, y2] format for consistency
                    items.append({"bbox": [abs_x1, abs_y1, abs_x2, abs_y2], "label": label})
                    print(f"[Gemini Detection] 转换坐标: {label}")
                    print(f"  原始 (归一化): {box_2d}")
                    print(f"  转换后 (绝对): [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
                else:
                    print(f"[Gemini Detection] box_2d格式不正确: {box_2d}")
            else:
                print(f"[Gemini Detection] 检测结果不是字典格式: {bounding_box}")
        
        return items
    except Exception as e:
        print(f"[Gemini Detection] Error parsing response: {e}")
        return []


class GeminiDetector:
    def __init__(self):
        self.client = None
        if not GEMINI_AVAILABLE:
            print("[Gemini Detection] Gemini not available - missing google-genai package")

    def detect_objects(self, image, api_key, target="all prominent items", 
                      bbox_selection="all", merge_boxes=False):
        if not GEMINI_AVAILABLE:
            print("[Gemini Detection] Gemini client not available - missing google-genai package")
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            return empty_image, empty_mask
        
        # 验证API密钥
        if not api_key or api_key.strip() == "":
            print("[Gemini Detection] ❌ API密钥为空，请输入有效的Gemini API密钥")
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            return empty_image, empty_mask
        
        # 初始化客户端（每次调用时使用提供的API密钥）
        try:
            client = genai.Client(api_key=api_key)
            print("[Gemini Detection] Gemini client initialized successfully")
        except Exception as e:
            print(f"[Gemini Detection] ❌ Failed to initialize Gemini client: {e}")
            print("[Gemini Detection] 请检查API密钥是否有效")
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            return empty_image, empty_mask
        
        try:
            detection_images = []
            output_masks = []
            
            # Process each image in the batch
            for img_tensor in image:
                img_tensor_single = torch.unsqueeze(img_tensor, 0)
                pil_image = tensor2pil(img_tensor_single)
                
                # Prepare prompt
                prompt = f"Detect the {target} in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
                
                # Configure response format
                config = types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
                
                # Generate content with Gemini (always use gemini-2.5-flash)
                print(f"[Gemini Detection] 调用Gemini API (模型: gemini-2.5-flash)...")
                print(f"[Gemini Detection] 检测目标: {target}")
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[pil_image, prompt],
                    config=config
                )
                
                print(f"[Gemini Detection] API调用成功!")
                print(f"[Gemini Detection] 原始响应: {response.text}")
                
                # Parse response
                items = parse_gemini_response(response.text, pil_image.width, pil_image.height)
                
                # Apply bbox selection
                selection = bbox_selection.strip().lower()
                boxes = items
                if selection != "all" and selection:
                    idxs = []
                    for part in selection.replace(",", " ").split():
                        try:
                            idxs.append(int(part))
                        except Exception:
                            continue
                    boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

                # Apply merge boxes if requested
                if merge_boxes and boxes:
                    x1 = min(b["bbox"][0] for b in boxes)
                    y1 = min(b["bbox"][1] for b in boxes)
                    x2 = max(b["bbox"][2] for b in boxes)
                    y2 = max(b["bbox"][3] for b in boxes)
                    label = boxes[0].get("label", target)
                    boxes = [{"bbox": [x1, y1, x2, y2], "label": label}]

                # Create detection image with bboxes drawn
                if boxes:
                    bboxes = [b["bbox"] for b in boxes]
                    labels = [b.get('label', 'object') for b in boxes]
                    detection_image = draw_bbox_on_image(pil_image, bboxes, labels)
                    detection_images.append(pil2tensor(detection_image))
                    
                    # Create masks from bboxes
                    individual_masks = create_mask_from_bboxes(pil_image.size, bboxes, merge_boxes)
                    output_masks.extend(individual_masks)
                    
                    print(f"[Gemini Detection] Detected {len(boxes)} objects: {[b.get('label', 'object') for b in boxes]}")
                else:
                    # No detection, return original image and empty mask
                    detection_images.append(pil2tensor(pil_image))
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    output_masks.append(empty_mask)
                    print("[Gemini Detection] No objects detected")
            
            # Convert lists to tensors
            detection_images_tensor = torch.cat(detection_images, dim=0) if detection_images else torch.zeros((1, 3, 512, 512))
            output_masks_tensor = torch.cat(output_masks, dim=0) if output_masks else torch.zeros((1, 512, 512))
            
            return detection_images_tensor, output_masks_tensor
            
        except Exception as e:
            print(f"[Gemini Detection] Error in object detection: {str(e)}")
            # Return empty outputs on error
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            return empty_image, empty_mask


detector = GeminiDetector()


class GeminiDetectionNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "target": ("STRING", {"default": "all prominent items"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("detection_image", "mask")
    FUNCTION = "detect_objects"
    CATEGORY = "hhy/api"

    def detect_objects(self, image, api_key, target="all prominent items", 
                      bbox_selection="all", merge_boxes=False):
        detection_image, mask = detector.detect_objects(
            image, api_key, target, bbox_selection, merge_boxes
        )
        return (detection_image, mask)


NODE_CLASS_MAPPINGS = {
    "GeminiDetection": GeminiDetectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiDetection": "Gemini Object Detection",
} 