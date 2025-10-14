import os
import ast
import json
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    Qwen3VLMoeForConditionalGeneration,
)
import gc
import tqdm


def tensor2pil(image: torch.Tensor) -> Image.Image:
    try:
        numpy_image = image.cpu().numpy()
        if len(numpy_image.shape) == 4:
            numpy_image = numpy_image[0]
        elif len(numpy_image.shape) == 3:
            if numpy_image.shape[0] == 3 and numpy_image.shape[0] < numpy_image.shape[1]:
                numpy_image = np.transpose(numpy_image, (1, 2, 0))
        elif len(numpy_image.shape) == 2:
            pass
        elif len(numpy_image.shape) == 1:
            return Image.new('RGB', (64, 64), color='black')
        else:
            numpy_image = numpy_image.squeeze()
            if len(numpy_image.shape) not in [2, 3]:
                return Image.new('RGB', (64, 64), color='black')
        if numpy_image.max() <= 1.0:
            numpy_image = numpy_image * 255.0
        numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
        if len(numpy_image.shape) == 2:
            return Image.fromarray(numpy_image, mode='L')
        elif len(numpy_image.shape) == 3:
            if numpy_image.shape[2] == 3:
                return Image.fromarray(numpy_image, mode='RGB')
            elif numpy_image.shape[2] == 4:
                return Image.fromarray(numpy_image, mode='RGBA')
            elif numpy_image.shape[2] == 1:
                return Image.fromarray(numpy_image.squeeze(2), mode='L')
            else:
                return Image.new('RGB', (64, 64), color='black')
        else:
            return Image.new('RGB', (64, 64), color='black')
    except Exception:
        return Image.new('RGB', (64, 64), color='black')


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image2mask(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def add_mask(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return torch.clamp(mask1 + mask2, 0.0, 1.0)


def draw_bbox_on_image(image, bboxes, labels=None, colors=None):
    if not isinstance(image, Image.Image):
        image = tensor2pil(image)
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    if colors is None:
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if labels and i < len(labels):
            label = labels[i]
            if font:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width, text_height = len(label) * 10, 15
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], fill=color)
            if font:
                draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
            else:
                draw.text((x1 + 5, y1 - text_height - 2), label, fill="white")
    return draw_image


def create_mask_from_bboxes(image_size, bboxes, merge_masks=False):
    width, height = image_size
    individual_masks = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask = Image.new('L', (width, height), "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x1, y1, x2, y2], fill="white")
        individual_masks.append(image2mask(mask))
    if merge_masks and individual_masks:
        final_mask = individual_masks[0]
        for i in range(1, len(individual_masks)):
            final_mask = add_mask(final_mask, individual_masks[i])
        return [final_mask]
    else:
        return individual_masks


def parse_json(json_output: str) -> str:
    if "```json" in json_output:
        json_output = json_output.split("```json", 1)[1]
        json_output = json_output.split("```", 1)[0]
    try:
        parsed = json.loads(json_output)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


def rel_to_px(coord, W, H):
    """将相对坐标转换为像素坐标
    
    Args:
        coord: 输入的相对坐标 (x, y)，基于1000x1000坐标系
        W: 目标图像的宽度（像素）
        H: 目标图像的高度（像素）
    
    Returns:
        tuple: 转换后的像素坐标 (px_x, px_y)
    """
    x, y = coord
    return int(round(x/1000 * (W-1))), int(round(y/1000 * (H-1)))


def parse_boxes_qwen3(text: str, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            return []
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, (str, list, dict)):
            try:
                data = ast.literal_eval(inner) if isinstance(inner, str) else inner
            except Exception:
                data = []
        else:
            data = []
    items = []
    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            # 使用rel_to_px函数进行坐标转换
            # 假设Qwen3VL输出的坐标是基于1000x1000坐标系
            px_x1, px_y1 = rel_to_px((x1, y1), img_width, img_height)
            px_x2, px_y2 = rel_to_px((x2, y2), img_width, img_height)
            
            # 确保坐标顺序正确
            if px_x1 > px_x2:
                px_x1, px_x2 = px_x2, px_x1
            if px_y1 > px_y2:
                px_y1, px_y2 = px_y2, px_y1
            
            # 边界检查
            px_x1 = max(0, min(img_width - 1, px_x1))
            px_x2 = max(0, min(img_width - 1, px_x2))
            px_y1 = max(0, min(img_height - 1, px_y1))
            px_y2 = max(0, min(img_height - 1, px_y2))
            
            items.append({"bbox": [px_x1, px_y1, px_x2, px_y2], "label": label})
    return items


class Qwen3VLDetector:
    def __init__(self):
        self.device = None
        self.model_loaded = False
        self.processor = None
        self.model = None
        self.model_path = None
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.device_map = {"": 0}
        else:
            self.device = "cpu"
            self.device_map = "cpu"

    def load_model(self, model_path, attention="flash_attention_2"):
        if not model_path:
            model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        if not self.model_loaded or self.model_path != model_path:
            if self.model_loaded:
                self.unload_model()
            self.model_path = model_path
            
            attn_impl = attention
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).eval()
            
            self.model_loaded = True

    def unload_model(self):
        if self.model_loaded:
            if self.device == "cuda:0":
                self.model = self.model.to("cpu")
            del self.model
            torch.cuda.empty_cache()
            self.model_loaded = False
            self.model_path = None

    def _build_messages(self, prompt_text: str, pil_image: Image.Image = None):
        if pil_image is not None:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

    def generate_text(self, prompt_text, image=None, model_path="", max_new_tokens=128,
                      attention="flash_attention_2", unload_model=False):
        self.load_model(model_path, attention)
        if image is not None:
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                pil_image = tensor2pil(image.unsqueeze(0))
            else:
                pil_image = image
        else:
            pil_image = None
        messages = self._build_messages(prompt_text, pil_image)
        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        if unload_model:
            self.unload_model()
        return output_text

    def detect_objects(self, image, model_path, prompt_text="object", bbox_selection="all",
                       merge_boxes=False, attention="flash_attention_2", unload_model=False):
        self.load_model(model_path, attention)
        detection_images = []
        output_masks = []
        all_bboxes = []
        for img_tensor in image:
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            if any(keyword in prompt_text.lower() for keyword in ["locate", "bbox", "detection", "find", "detect"]):
                prompt = prompt_text
            else:
                prompt = f"Locate the {prompt_text} and output bbox in JSON"
            messages = self._build_messages(prompt, pil_image)
            with torch.no_grad():
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                    output_ids = self.model.generate(**inputs, max_new_tokens=1024)
                    gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs["input_ids"], output_ids)]
                    output_text = self.processor.batch_decode(
                        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )[0]
                    items = parse_boxes_qwen3(output_text, pil_image.width, pil_image.height)
                except Exception:
                    items = []
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
                if merge_boxes and boxes:
                    x1 = min(b["bbox"][0] for b in boxes)
                    y1 = min(b["bbox"][1] for b in boxes)
                    x2 = max(b["bbox"][2] for b in boxes)
                    y2 = max(b["bbox"][3] for b in boxes)
                    label = boxes[0].get("label", prompt_text)
                    boxes = [{"bbox": [x1, y1, x2, y2], "label": label}]
                if boxes:
                    bboxes = [b["bbox"] for b in boxes]
                    labels = [b.get('label', prompt_text) for b in boxes]
                    all_bboxes.append([bboxes])
                    detection_image = draw_bbox_on_image(pil_image, bboxes, labels)
                    detection_tensor = pil2tensor(detection_image)
                    detection_images.append(detection_tensor)
                    individual_masks = create_mask_from_bboxes(pil_image.size, bboxes, merge_boxes)
                    output_masks.extend(individual_masks)
                else:
                    detection_images.append(pil2tensor(pil_image))
                    output_masks.append(torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32))
                    all_bboxes.append([[]])
        if unload_model:
            self.unload_model()
        detection_images_tensor = torch.cat(detection_images, dim=0) if detection_images else torch.zeros((1, 3, 512, 512), dtype=torch.float32)
        if output_masks:
            # Normalize mask shapes
            target_shape = output_masks[0].shape
            normalized_masks = []
            for mask in output_masks:
                if mask.shape != target_shape:
                    mask_resized = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(target_shape[1], target_shape[2]),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                    normalized_masks.append(mask_resized.unsqueeze(0))
                else:
                    normalized_masks.append(mask)
            output_masks = normalized_masks
            output_masks_tensor = torch.cat(output_masks, dim=0)
        else:
            output_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        return detection_images_tensor, output_masks_tensor, all_bboxes


detector = Qwen3VLDetector()


class Qwen3VLTextGenerationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "object_detection",
                    "image_description",
                    "text_only",
                ], {"default": "image_description"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "object"}),
                "model_path": ("STRING", {"default": "Qwen/Qwen3-VL-30B-A3B-Instruct"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 4096, "step": 1}),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], {"default": "flash_attention_2"}),
                "unload_model": ("BOOLEAN", {"default": False}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "MASK", "BBOXES")
    RETURN_NAMES = ("generated_text", "detection_image", "mask", "bboxes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, False)
    FUNCTION = "process"
    CATEGORY = "hhy/Vision"

    def process(self, mode, prompt_text, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", max_new_tokens=128,
               attention="flash_attention_2", unload_model=False,
               bbox_selection="all", merge_boxes=False, image=None):
        if isinstance(mode, list):
            mode = mode[0] if mode else "image_description"
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0] if prompt_text else "object"
        if isinstance(model_path, list):
            model_path = model_path[0] if model_path else "Qwen/Qwen3-VL-30B-A3B-Instruct"
        if isinstance(max_new_tokens, list):
            max_new_tokens = max_new_tokens[0] if max_new_tokens else 128
        if isinstance(attention, list):
            attention = attention[0] if attention else "flash_attention_2"
        if isinstance(unload_model, list):
            unload_model = unload_model[0] if unload_model else False
        if isinstance(bbox_selection, list):
            bbox_selection = bbox_selection[0] if bbox_selection else "all"
        if isinstance(merge_boxes, list):
            merge_boxes = merge_boxes[0] if merge_boxes else False
        if image is not None and isinstance(image, list):
            combined_images = []
            for img_batch in image:
                if img_batch is not None:
                    if len(img_batch.shape) == 3:
                        img_batch = img_batch.unsqueeze(0)
                    combined_images.append(img_batch)
            if combined_images:
                image = torch.cat(combined_images, dim=0)
            else:
                image = None
        generated_text = ""
        detection_images = []
        masks = []
        bboxes = [[[]]]
        if mode == "object_detection":
            if image is None:
                generated_text = "Error: Image is required for object detection mode."
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, [empty_image], [empty_mask], bboxes)
            detection_prompt = prompt_text if prompt_text.strip() and prompt_text != "Describe this image." else "object"
            detection_image_tensor, mask_tensor, bboxes = detector.detect_objects(
                image, model_path, detection_prompt, bbox_selection, merge_boxes,
                attention, unload_model
            )
            generated_text = f"Object detection completed for: {detection_prompt}"
            # 将tensor分解为列表
            for i in range(detection_image_tensor.shape[0]):
                detection_images.append(detection_image_tensor[i:i+1])
            for i in range(mask_tensor.shape[0]):
                masks.append(mask_tensor[i:i+1])
        elif mode == "image_description":
            if image is None:
                generated_text = "Error: Image is required for image description mode."
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, [empty_image], [empty_mask], bboxes)
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    img_for_desc = image[0]
                else:
                    img_for_desc = image
                pil_image = tensor2pil(img_for_desc.unsqueeze(0))
            else:
                pil_image = image
            if not prompt_text or prompt_text.strip() == "":
                prompt_text = "Describe this image."
            generated_text = detector.generate_text(
                prompt_text, pil_image, model_path, max_new_tokens,
                attention, unload_model
            )
            # 将图像分解为列表
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    for i in range(image.shape[0]):
                        detection_images.append(image[i:i+1])
                        masks.append(torch.zeros((1, image.shape[2], image.shape[3])))
                else:
                    detection_images.append(image.unsqueeze(0))
                    masks.append(torch.zeros((1, image.shape[1], image.shape[2])))
            else:
                detection_images.append(torch.zeros((1, 3, 512, 512)))
                masks.append(torch.zeros((1, 512, 512)))
        elif mode == "text_only":
            if not prompt_text or prompt_text.strip() == "":
                generated_text = "Error: Prompt text is required for text-only mode."
            else:
                generated_text = detector.generate_text(
                    prompt_text, None, model_path, max_new_tokens,
                    attention, unload_model
                )
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_images.append(empty_image)
            masks.append(empty_mask)
        else:
            generated_text = f"Error: Unknown mode '{mode}'."
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_images.append(empty_image)
            masks.append(empty_mask)
        return (generated_text, detection_images, masks, bboxes)


class Qwen3BboxProcessorNode:
    """专门处理Qwen3输出的bbox JSON的节点"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_json": ("STRING", {"multiline": True, "default": ""}),
                "merge_masks": ("BOOLEAN", {"default": False}),
                "bbox_color": ("STRING", {"default": "red"}),
                "line_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_with_bbox", "mask")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "process_bbox"
    CATEGORY = "Qwen3VL"

    def process_bbox(self, image, bbox_json, merge_masks=False, bbox_color="red", line_width=3):
        """处理bbox JSON并生成带bbox的图片和mask"""
        
        # 处理list输入参数
        if isinstance(bbox_json, list):
            bbox_json = bbox_json[0] if bbox_json else ""
        if isinstance(merge_masks, list):
            merge_masks = merge_masks[0] if merge_masks else False
        if isinstance(bbox_color, list):
            bbox_color = bbox_color[0] if bbox_color else "red"
        if isinstance(line_width, list):
            line_width = line_width[0] if line_width else 3
            
        # 处理image list输入
        if isinstance(image, list):
            combined_images = []
            for img_batch in image:
                if img_batch is not None:
                    if len(img_batch.shape) == 3:
                        img_batch = img_batch.unsqueeze(0)
                    combined_images.append(img_batch)
            if combined_images:
                image = torch.cat(combined_images, dim=0)
            else:
                return ([], [])
        
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        if not bbox_json.strip():
            # 如果没有bbox数据，返回原图和空mask列表
            result_images = []
            result_masks = []
            for i in range(image.shape[0]):
                result_images.append(image[i:i+1])
                result_masks.append(torch.zeros((1, image.shape[2], image.shape[3]), dtype=torch.float32))
            return (result_images, result_masks)
        
        # 处理每张图片
        processed_images = []
        output_masks = []
        
        for i, img_tensor in enumerate(image):
            # 转换为PIL图像
            pil_image = tensor2pil(img_tensor.unsqueeze(0))
            img_width, img_height = pil_image.size
            
            # 解析bbox
            bboxes = parse_boxes_qwen3(bbox_json, img_width, img_height)
            
            if bboxes:
                # 提取bbox坐标和标签
                bbox_coords = [bbox["bbox"] for bbox in bboxes]
                bbox_labels = []
                
                # 处理标签
                for j, bbox in enumerate(bboxes):
                    if bbox.get("label"):
                        bbox_labels.append(bbox["label"])
                    else:
                        bbox_labels.append(f"Object {j+1}")
                
                # 生成mask
                individual_masks = create_mask_from_bboxes(pil_image.size, bbox_coords, merge_masks)
                # 直接扩展mask列表，不合并不同尺寸的mask
                output_masks.extend(individual_masks)
                
                # 绘制bbox
                colors = [bbox_color] * len(bbox_coords)
                
                # 重新绘制以设置线条宽度
                draw_image = pil_image.copy()
                draw = ImageDraw.Draw(draw_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except Exception:
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                
                for j, (bbox, label) in enumerate(zip(bbox_coords, bbox_labels)):
                    x1, y1, x2, y2 = bbox
                    color = colors[j % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                    
                    if font:
                        bbox_text = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        text_width, text_height = len(label) * 10, 15
                    
                    draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], fill=color)
                    if font:
                        draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
                    else:
                        draw.text((x1 + 5, y1 - text_height - 2), label, fill="white")
                
                image_with_bbox = draw_image
                processed_images.append(pil2tensor(image_with_bbox))
            else:
                # 没有检测到bbox，返回原图
                processed_images.append(img_tensor.unsqueeze(0))
                output_masks.append(torch.zeros((1, img_height, img_width), dtype=torch.float32))
        
        # 返回列表而不是合并的tensor
        return (processed_images, output_masks)


NODE_CLASS_MAPPINGS = {
    "Qwen3VLTextGeneration": Qwen3VLTextGenerationNode,
    "Qwen3BboxProcessor": Qwen3BboxProcessorNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLTextGeneration": "Qwen3-VL Unified (Detection/Description/Chat)",
    "Qwen3BboxProcessor": "Qwen3-VL Bbox Processor",
}
