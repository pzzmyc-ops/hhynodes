import os
import ast
import json
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration,
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
        # 检查是否有 "content" 键（包装格式）
        inner = data.get("content")
        if isinstance(inner, (str, list, dict)):
            try:
                data = ast.literal_eval(inner) if isinstance(inner, str) else inner
            except Exception:
                data = []
        # 检查是否直接包含 bbox_2d 或 bbox（单个对象格式）
        elif "bbox_2d" in data or "bbox" in data:
            data = [data]
        else:
            data = []
    
    # 确保 data 是列表
    if not isinstance(data, list):
        data = []
    
    items = []
    for item in data:
        # 确保 item 是字典
        if not isinstance(item, dict):
            continue
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
            )

            # Use appropriate Qwen3-VL model class based on model path
            model_path_lower = model_path.lower()
            
            # Check if it's a MoE model (30B uses MoE architecture)
            if "30b" in model_path_lower or "moe" in model_path_lower or "a3b" in model_path_lower:
                # MoE models (e.g., Qwen3-VL-30B-A3B-Instruct)
                self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation=attn_impl,
                ).eval()
            elif "Qwen3-VL" in model_path or "qwen3-vl" in model_path_lower:
                # Standard Qwen3-VL models (e.g., Qwen3-VL-8B-Instruct)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation=attn_impl,
                ).eval()
            else:
                # Legacy support for other models
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype="auto",
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
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "MASK", "BBOXES", "STRING", "STRING")
    RETURN_NAMES = ("generated_text", "detection_image", "mask", "bboxes", "merged_text", "log")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, False, False, False)
    FUNCTION = "process"
    CATEGORY = "hhy/qwen3"

    def process(self, mode, prompt_text, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", max_new_tokens=128,
               attention="flash_attention_2", unload_model=False,
               bbox_selection="all", merge_boxes=False, image=None, image_list=None):
        # 处理标量参数
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
        
        # 决定使用哪个输入：优先使用 image_list（逐张处理），否则使用 image（批量处理）
        use_list_mode = False
        processed_images = []
        
        if image_list is not None and isinstance(image_list, list):
            # 使用 image_list 模式：逐张处理，允许不同尺寸
            use_list_mode = True
            for img_batch in image_list:
                if img_batch is not None:
                    # 将每个batch中的图片拆分为单独的图片
                    if len(img_batch.shape) == 4:
                        for i in range(img_batch.shape[0]):
                            processed_images.append(img_batch[i])
                    elif len(img_batch.shape) == 3:
                        processed_images.append(img_batch)
        elif image is not None and isinstance(image, list):
            # 使用 image 模式：批量处理，需要合并
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
        elif image is not None:
            # image 不是 list，转换为标准格式
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        generated_texts = []
        detection_images = []
        masks = []
        bboxes = [[[]]]
        log_messages = []
        
        # 记录输入信息
        log_messages.append(f"Mode: {mode}")
        log_messages.append(f"Prompt: {prompt_text}")
        log_messages.append(f"Model: {model_path}")
        log_messages.append(f"Use list mode: {use_list_mode}")
        if use_list_mode:
            log_messages.append(f"Images in list: {len(processed_images)}")
        elif image is not None:
            log_messages.append(f"Images in batch: {image.shape[0] if len(image.shape) == 4 else 1}")
        
        if mode == "object_detection":
            if use_list_mode:
                # image_list 模式：逐张处理
                if not processed_images:
                    generated_texts.append("Error: Image is required for object detection mode.")
                    empty_image = torch.zeros((1, 3, 512, 512))
                    empty_mask = torch.zeros((1, 512, 512))
                    log_messages.append("ERROR: No images provided")
                    merged_text = "\n\n---\n\n".join(generated_texts) if generated_texts else ""
                    return (generated_texts, [empty_image], [empty_mask], bboxes, merged_text, "\n".join(log_messages))
                
                # 加载模型
                detector.load_model(model_path, attention)
                
                detection_prompt = prompt_text if prompt_text.strip() and prompt_text != "Describe this image." else "object"
                all_bboxes_list = []
                
                # 逐张图片处理
                for img_tensor in processed_images:
                    img_tensor_single = img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor
                    pil_image = tensor2pil(img_tensor_single)
                    
                    # 使用与 detect_objects 相同的逻辑处理单张图片
                    if any(keyword in detection_prompt.lower() for keyword in ["locate", "bbox", "detection", "find", "detect"]):
                        prompt = detection_prompt
                    else:
                        prompt = f"Locate the {detection_prompt} and output bbox in JSON"
                    
                    messages = detector._build_messages(prompt, pil_image)
                    
                    with torch.no_grad():
                        try:
                            inputs = detector.processor.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors="pt",
                            )
                            inputs = {k: (v.to(detector.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                            output_ids = detector.model.generate(**inputs, max_new_tokens=1024)
                            gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs["input_ids"], output_ids)]
                            output_text = detector.processor.batch_decode(
                                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )[0]
                            items = parse_boxes_qwen3(output_text, pil_image.width, pil_image.height)
                        except Exception:
                            items = []
                    
                    # 处理bbox选择
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
                        label = boxes[0].get("label", detection_prompt)
                        boxes = [{"bbox": [x1, y1, x2, y2], "label": label}]
                    
                    if boxes:
                        bboxes_coords = [b["bbox"] for b in boxes]
                        labels = [b.get('label', detection_prompt) for b in boxes]
                        all_bboxes_list.append([bboxes_coords])
                        
                        detection_image = draw_bbox_on_image(pil_image, bboxes_coords, labels)
                        detection_tensor = pil2tensor(detection_image)
                        detection_images.append(detection_tensor)
                        
                        individual_masks = create_mask_from_bboxes(pil_image.size, bboxes_coords, merge_boxes)
                        masks.extend(individual_masks)
                    else:
                        detection_images.append(pil2tensor(pil_image))
                        masks.append(torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32))
                        all_bboxes_list.append([[]])
                
                if unload_model:
                    detector.unload_model()
                
                bboxes = all_bboxes_list
                # 为每张图片生成文本
                for i in range(len(processed_images)):
                    generated_texts.append(f"Object detection completed for image {i+1}: {detection_prompt}")
                
                log_messages.append(f"Detection completed for {len(processed_images)} images")
                log_messages.append(f"Total masks generated: {len(masks)}")
            else:
                # image 模式：批量处理
                if image is None:
                    generated_texts.append("Error: Image is required for object detection mode.")
                    empty_image = torch.zeros((1, 3, 512, 512))
                    empty_mask = torch.zeros((1, 512, 512))
                    log_messages.append("ERROR: No images provided (batch mode)")
                    merged_text = "\n\n---\n\n".join(generated_texts) if generated_texts else ""
                    return (generated_texts, [empty_image], [empty_mask], bboxes, merged_text, "\n".join(log_messages))
                detection_prompt = prompt_text if prompt_text.strip() and prompt_text != "Describe this image." else "object"
                detection_image_tensor, mask_tensor, bboxes = detector.detect_objects(
                    image, model_path, detection_prompt, bbox_selection, merge_boxes,
                    attention, unload_model
                )
                # 将tensor分解为列表
                for i in range(detection_image_tensor.shape[0]):
                    detection_images.append(detection_image_tensor[i:i+1])
                    generated_texts.append(f"Object detection completed for image {i+1}: {detection_prompt}")
                for i in range(mask_tensor.shape[0]):
                    masks.append(mask_tensor[i:i+1])
                
                log_messages.append(f"Detection completed for {detection_image_tensor.shape[0]} images (batch mode)")
                log_messages.append(f"Total masks generated: {mask_tensor.shape[0]}")
        elif mode == "image_description":
            if use_list_mode:
                # image_list 模式：为每张图片生成描述
                if not processed_images:
                    generated_texts.append("Error: Image is required for image description mode.")
                    empty_image = torch.zeros((1, 3, 512, 512))
                    empty_mask = torch.zeros((1, 512, 512))
                    log_messages.append("ERROR: No images provided")
                    merged_text = "\n\n---\n\n".join(generated_texts) if generated_texts else ""
                    return (generated_texts, [empty_image], [empty_mask], bboxes, merged_text, "\n".join(log_messages))
                
                if not prompt_text or prompt_text.strip() == "":
                    prompt_text = "Describe this image."
                
                print(f"=== Image Description List Mode ===")
                print(f"Total images to process: {len(processed_images)}")
                
                # 为每张图片生成描述
                for idx, img_tensor in enumerate(processed_images):
                    print(f"Processing image {idx+1}/{len(processed_images)}")
                    pil_image = tensor2pil(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                    
                    text = detector.generate_text(
                        prompt_text, pil_image, model_path, max_new_tokens,
                        attention, False  # 不在循环中卸载模型
                    )
                    generated_texts.append(text)
                    print(f"Generated text for image {idx+1}: {text[:100]}...")
                    
                    # 添加图片和mask
                    if len(img_tensor.shape) == 3:
                        detection_images.append(img_tensor.unsqueeze(0))
                        masks.append(torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2])))
                    else:
                        detection_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 2 else img_tensor)
                        masks.append(torch.zeros((1, img_tensor.shape[1] if len(img_tensor.shape) >= 2 else 512, 
                                                 img_tensor.shape[2] if len(img_tensor.shape) >= 3 else 512)))
                
                # 循环结束后卸载模型
                if unload_model:
                    detector.unload_model()
                
                print(f"Total texts generated: {len(generated_texts)}")
                print(f"Total images: {len(detection_images)}")
                print(f"Total masks: {len(masks)}")
                print("=" * 40)
                
                log_messages.append(f"Description completed for {len(processed_images)} images")
                log_messages.append(f"Texts generated: {len(generated_texts)}")
            else:
                # image 模式：只处理第一张图片
                if image is None:
                    generated_texts.append("Error: Image is required for image description mode.")
                    empty_image = torch.zeros((1, 3, 512, 512))
                    empty_mask = torch.zeros((1, 512, 512))
                    log_messages.append("ERROR: No images provided (batch mode)")
                    merged_text = "\n\n---\n\n".join(generated_texts) if generated_texts else ""
                    return (generated_texts, [empty_image], [empty_mask], bboxes, merged_text, "\n".join(log_messages))
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
                text = detector.generate_text(
                    prompt_text, pil_image, model_path, max_new_tokens,
                    attention, unload_model
                )
                # 将图像分解为列表，所有图片共享同一个描述
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:
                        for i in range(image.shape[0]):
                            detection_images.append(image[i:i+1])
                            generated_texts.append(text if i == 0 else f"[Same as image 1] {text}")
                            masks.append(torch.zeros((1, image.shape[2], image.shape[3])))
                    else:
                        detection_images.append(image.unsqueeze(0))
                        generated_texts.append(text)
                        masks.append(torch.zeros((1, image.shape[1], image.shape[2])))
                else:
                    detection_images.append(torch.zeros((1, 3, 512, 512)))
                    generated_texts.append(text)
                    masks.append(torch.zeros((1, 512, 512)))
                
                log_messages.append(f"Description completed for {len(detection_images)} images (batch mode)")
        elif mode == "text_only":
            if not prompt_text or prompt_text.strip() == "":
                generated_texts.append("Error: Prompt text is required for text-only mode.")
            else:
                text = detector.generate_text(
                    prompt_text, None, model_path, max_new_tokens,
                    attention, unload_model
                )
                generated_texts.append(text)
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_images.append(empty_image)
            masks.append(empty_mask)
        else:
            generated_texts.append(f"Error: Unknown mode '{mode}'.")
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_images.append(empty_image)
            masks.append(empty_mask)
        
        # 生成最终 log 和合并文本
        log_messages.append(f"\nFinal output:")
        log_messages.append(f"- Texts: {len(generated_texts)}")
        log_messages.append(f"- Images: {len(detection_images)}")
        log_messages.append(f"- Masks: {len(masks)}")
        final_log = "\n".join(log_messages)
        
        # 合并所有文本，用分隔符连接
        merged_text = "\n\n---\n\n".join(generated_texts) if generated_texts else ""
        
        return (generated_texts, detection_images, masks, bboxes, merged_text, final_log)


class Qwen3BboxProcessorNode:
    """专门处理Qwen3输出的bbox JSON的节点"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_json": ("STRING", {"multiline": True, "default": ""}),
                "merge_masks": ("BOOLEAN", {"default": False}),
                "bbox_color": ("STRING", {"default": "red"}),
                "line_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image_with_bbox", "mask", "log")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "process_bbox"
    CATEGORY = "hhy/qwen3"

    def process_bbox(self, bbox_json, merge_masks=False, bbox_color="red", line_width=3, image=None, image_list=None):
        """处理bbox JSON并生成带bbox的图片和mask"""
        
        # 处理 bbox_json - 可能是列表（每张图片不同的json）或单个字符串（所有图片相同）
        bbox_json_list = []
        if isinstance(bbox_json, list):
            bbox_json_list = bbox_json if bbox_json else [""]
        else:
            bbox_json_list = [bbox_json]
        
        # 处理其他标量参数
        if isinstance(merge_masks, list):
            merge_masks = merge_masks[0] if merge_masks else False
        if isinstance(bbox_color, list):
            bbox_color = bbox_color[0] if bbox_color else "red"
        if isinstance(line_width, list):
            line_width = line_width[0] if line_width else 3
        
        # 决定使用哪个输入：优先使用 image_list（逐张处理），否则使用 image（批量处理）
        use_list_mode = False
        processed_images = []
        log_messages = []
        boundary_issues = []  # 记录边界问题
        
        log_messages.append(f"Bbox JSON count: {len(bbox_json_list)}")
        log_messages.append(f"Merge masks: {merge_masks}")
        log_messages.append(f"Bbox color: {bbox_color}")
        log_messages.append(f"Line width: {line_width}")
        
        if image_list is not None and isinstance(image_list, list):
            # 使用 image_list 模式：逐张处理，允许不同尺寸
            use_list_mode = True
            for img_batch in image_list:
                if img_batch is not None:
                    # 将每个batch中的图片拆分为单独的图片
                    if len(img_batch.shape) == 4:
                        for i in range(img_batch.shape[0]):
                            processed_images.append(img_batch[i])
                    elif len(img_batch.shape) == 3:
                        processed_images.append(img_batch)
        elif image is not None and isinstance(image, list):
            # 使用 image 模式：批量处理，需要合并
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
        elif image is not None:
            # image 不是 list，转换为标准格式
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        
        # 检查是否所有 bbox_json 都为空
        all_empty = all(not json_str.strip() for json_str in bbox_json_list)
        
        if all_empty:
            # 如果没有bbox数据，返回原图和空mask列表
            result_images = []
            result_masks = []
            if use_list_mode:
                # image_list 模式
                for img_tensor in processed_images:
                    if len(img_tensor.shape) == 3:
                        result_images.append(img_tensor.unsqueeze(0))
                        result_masks.append(torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32))
                    else:
                        result_images.append(img_tensor.unsqueeze(0))
                        result_masks.append(torch.zeros((1, img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32))
            else:
                # image 模式
                if image is not None:
                    for i in range(image.shape[0]):
                        result_images.append(image[i:i+1])
                        result_masks.append(torch.zeros((1, image.shape[2], image.shape[3]), dtype=torch.float32))
            
            log_messages.append("All bbox_json are empty, returning original images")
            log_messages.append(f"Returned {len(result_images)} images with empty masks")
            return (result_images, result_masks, "\n".join(log_messages))
        
        # 处理每张图片
        result_images = []
        output_masks = []
        
        images_to_process = processed_images if use_list_mode else [image[i] for i in range(image.shape[0])]
        
        print(f"=== Qwen3BboxProcessor Debug ===")
        print(f"Total images to process: {len(images_to_process)}")
        print(f"Total bbox_json provided: {len(bbox_json_list)}")
        
        for idx, img_tensor in enumerate(images_to_process):
            # 为每张图片选择对应的 bbox_json
            # 如果 bbox_json_list 只有一个，则所有图片使用相同的
            # 如果有多个，则使用对应索引的 bbox_json
            current_bbox_json = bbox_json_list[idx] if idx < len(bbox_json_list) else bbox_json_list[-1]
            
            print(f"Processing image {idx+1}/{len(images_to_process)}, using bbox_json index: {min(idx, len(bbox_json_list)-1)}")
            
            # 转换为PIL图像
            if len(img_tensor.shape) == 3:
                pil_image = tensor2pil(img_tensor.unsqueeze(0))
            else:
                pil_image = tensor2pil(img_tensor)
            img_width, img_height = pil_image.size
            
            # 先解析JSON以进行边界检测
            try:
                bbox_data_raw = json.loads(parse_json(current_bbox_json))
            except Exception:
                try:
                    bbox_data_raw = ast.literal_eval(parse_json(current_bbox_json))
                except Exception:
                    bbox_data_raw = []
            
            # 对原始坐标进行边界检测
            # 规范化 bbox_data_raw 为列表格式
            if isinstance(bbox_data_raw, dict):
                # 如果是单个字典（如 {"bbox_2d": [...], "dialogue": [...]}），将其包装为列表
                bbox_data_raw = [bbox_data_raw]
            
            if bbox_data_raw:
                log_messages.append(f"\nImage {idx+1} ({img_width}x{img_height}):")
                for i, item in enumerate(bbox_data_raw):
                    # 确保 item 是字典
                    if not isinstance(item, dict):
                        continue
                    box = item.get("bbox_2d") or item.get("bbox") or item
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        px1 = rel_to_px((x1, y1), img_width, img_height)
                        px2 = rel_to_px((x2, y2), img_width, img_height)
                        
                        # 记录坐标转换
                        log_messages.append(f"  Bbox {i+1}: [{x1:4d},{y1:4d},{x2:4d},{y2:4d}] -> [{px1[0]:4d},{px1[1]:4d},{px2[0]:4d},{px2[1]:4d}]")
                        
                        # 边界检测
                        issues = []
                        if x1 < 0 or x1 > 1000 or x2 < 0 or x2 > 1000:
                            issues.append("X坐标超出[0,1000]")
                        if y1 < 0 or y1 > 1000 or y2 < 0 or y2 > 1000:
                            issues.append("Y坐标超出[0,1000]")
                        if px1[0] < 0 or px1[0] >= img_width or px2[0] < 0 or px2[0] >= img_width:
                            issues.append(f"X像素超出[0,{img_width-1}]")
                        if px1[1] < 0 or px1[1] >= img_height or px2[1] < 0 or px2[1] >= img_height:
                            issues.append(f"Y像素超出[0,{img_height-1}]")
                        if px1[0] > px2[0] or px1[1] > px2[1]:
                            issues.append("坐标顺序错误")
                        
                        if issues:
                            log_messages.append(f"    ✗ 问题: {', '.join(issues)}")
                            boundary_issues.append({
                                "image": idx + 1,
                                "bbox": i + 1,
                                "issues": issues
                            })

            bboxes = parse_boxes_qwen3(current_bbox_json, img_width, img_height)
            
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
                result_images.append(pil2tensor(image_with_bbox))
            else:
                # 没有检测到bbox，返回原图
                if len(img_tensor.shape) == 3:
                    result_images.append(img_tensor.unsqueeze(0))
                else:
                    result_images.append(img_tensor.unsqueeze(0))
                output_masks.append(torch.zeros((1, img_height, img_width), dtype=torch.float32))
        
        # 返回列表而不是合并的tensor
        print(f"Total images processed: {len(result_images)}")
        print(f"Total masks generated: {len(output_masks)}")
        print("=" * 40)
        
        log_messages.append(f"\nUse list mode: {use_list_mode}")
        log_messages.append(f"\nProcessing completed:")
        log_messages.append(f"- Images processed: {len(result_images)}")
        log_messages.append(f"- Masks generated: {len(output_masks)}")
        
        # 边界检测总结
        if boundary_issues:
            log_messages.append(f"\n⚠ Boundary Issues Detected:")
            log_messages.append(f"- Total issues: {len(boundary_issues)}")
            affected_images = len(set(item['image'] for item in boundary_issues))
            log_messages.append(f"- Affected images: {affected_images}")
            for issue_item in boundary_issues:
                log_messages.append(f"  Image {issue_item['image']} Bbox {issue_item['bbox']}: {', '.join(issue_item['issues'])}")
        else:
            log_messages.append(f"\n✓ All bbox coordinates are valid")
        
        final_log = "\n".join(log_messages)
        
        return (result_images, output_masks, final_log)


class Qwen3VLImageFilterNode:
    """Qwen3-VL图片过滤节点，基于description模式输出yes/no来过滤图片"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": "Does this image contain what I'm looking for? Answer yes or no."}),
                "model_path": ("STRING", {"default": "Qwen/Qwen3-VL-30B-A3B-Instruct"}),
                "max_new_tokens": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], {"default": "flash_attention_2"}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("filtered_images", "filter_results", "log")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "filter_images"
    CATEGORY = "hhy/qwen3"

    def filter_images(self, prompt_text, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", 
                     max_new_tokens=10, attention="flash_attention_2", 
                     unload_model=False, image_list=None):
        """过滤图片：收集yes/no结果，保留结果为no的图片"""
        
        # 处理标量参数
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0] if prompt_text else "Does this image contain what I'm looking for? Answer yes or no."
        if isinstance(model_path, list):
            model_path = model_path[0] if model_path else "Qwen/Qwen3-VL-30B-A3B-Instruct"
        if isinstance(max_new_tokens, list):
            max_new_tokens = max_new_tokens[0] if max_new_tokens else 10
        if isinstance(attention, list):
            attention = attention[0] if attention else "flash_attention_2"
        if isinstance(unload_model, list):
            unload_model = unload_model[0] if unload_model else False
        
        # 处理图片列表
        processed_images = []
        if image_list is not None and isinstance(image_list, list):
            for img_batch in image_list:
                if img_batch is not None:
                    # 将每个batch中的图片拆分为单独的图片
                    if len(img_batch.shape) == 4:
                        for i in range(img_batch.shape[0]):
                            processed_images.append(img_batch[i])
                    elif len(img_batch.shape) == 3:
                        processed_images.append(img_batch)
        
        log_messages = []
        log_messages.append(f"Filter prompt: {prompt_text}")
        log_messages.append(f"Model: {model_path}")
        log_messages.append(f"Total images to filter: {len(processed_images)}")
        
        if not processed_images:
            log_messages.append("ERROR: No images provided")
            return ([], [], "\n".join(log_messages))
        
        # 使用共享的detector实例，确保模型已加载
        detector.load_model(model_path, attention)
        
        filtered_images = []
        filter_results = []
        
        print(f"=== Qwen3VL Image Filter ===")
        print(f"Total images to process: {len(processed_images)}")
        
        # 逐张图片处理
        all_results = []  # 存储所有结果
        for idx, img_tensor in enumerate(processed_images):
            print(f"Processing image {idx+1}/{len(processed_images)}")
            
            # 转换为PIL图像
            if len(img_tensor.shape) == 3:
                pil_image = tensor2pil(img_tensor.unsqueeze(0))
            else:
                pil_image = tensor2pil(img_tensor)
            
            # 使用与原版一致的推理逻辑
            text = detector.generate_text(
                prompt_text, pil_image, model_path, max_new_tokens,
                attention, False  # 不在循环中卸载模型
            )
            
            # 解析结果，查找yes/no
            result_text = text.strip().lower()
            is_yes = "yes" in result_text and "no" not in result_text
            is_no = "no" in result_text and "yes" not in result_text
            
            # 记录结果
            if is_yes:
                all_results.append("yes")
            elif is_no:
                all_results.append("no")
            else:
                all_results.append("unclear")
            
            print(f"Image {idx+1} result: {text.strip()}")
        
        # 检查是否所有结果都是yes或都是no
        unique_results = set(all_results)
        if len(unique_results) == 1 and unique_results.pop() in ["yes", "no"]:
            # 如果所有结果都是yes或都是no，返回所有图片
            log_messages.append(f"所有图片结果都是{all_results[0].upper()}，返回所有原始图片")
            for idx, img_tensor in enumerate(processed_images):
                filtered_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                filter_results.append(f"Image {idx+1}: {all_results[idx].upper()} - ALL RETURNED")
                log_messages.append(f"Image {idx+1}: {all_results[idx].upper()} - ALL RETURNED")
        else:
            # 正常过滤逻辑
            for idx, img_tensor in enumerate(processed_images):
                result = all_results[idx]
                if result == "no":
                    filtered_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                    filter_results.append(f"Image {idx+1}: NO - KEPT")
                    log_messages.append(f"Image {idx+1}: NO - KEPT")
                elif result == "yes":
                    filter_results.append(f"Image {idx+1}: YES - FILTERED OUT")
                    log_messages.append(f"Image {idx+1}: YES - FILTERED OUT")
                else:
                    # 如果无法确定，默认保留（保守策略）
                    filtered_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                    filter_results.append(f"Image {idx+1}: UNCLEAR - KEPT (default)")
                    log_messages.append(f"Image {idx+1}: UNCLEAR - KEPT (default)")
        
        # 循环结束后卸载模型（如果设置了卸载选项）
        if unload_model:
            detector.unload_model()
        
        print(f"Original images: {len(processed_images)}")
        print(f"Filtered images: {len(filtered_images)}")
        print(f"Filtered out: {len(processed_images) - len(filtered_images)}")
        print("=" * 40)
        
        log_messages.append(f"\nFilter Summary:")
        log_messages.append(f"- Original images: {len(processed_images)}")
        log_messages.append(f"- Filtered images: {len(filtered_images)}")
        log_messages.append(f"- Filtered out: {len(processed_images) - len(filtered_images)}")
        
        final_log = "\n".join(log_messages)
        merged_results = "\n".join(filter_results)
        
        return (filtered_images, [merged_results], final_log)


class Qwen3VLJsonProcessorNode:
    """Qwen3-VL JSON处理节点，处理图片过滤后的对话重新分配"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_json": ("STRING", {"multiline": True, "default": ""}),
                "merge_strategy": ([
                    "nearest_bbox",
                    "first_remaining",
                    "last_remaining",
                ], {"default": "nearest_bbox"}),
            },
            "optional": {
                "source_image": ("IMAGE",),  # 第一组：原图（1张）
                "pre_filter_images": ("IMAGE",),  # 第二组：筛选前的图片
                "post_filter_images": ("IMAGE",),  # 第三组：筛选后的图片
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_json", "log")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "process_json"
    CATEGORY = "hhy/qwen3"

    def process_json(self, original_json, merge_strategy="nearest_bbox", source_image=None, pre_filter_images=None, post_filter_images=None):
        """处理JSON数据，重新分配被过滤图片的对话到剩余bbox中"""
        
        # 处理标量参数
        if isinstance(original_json, list):
            original_json = original_json[0] if original_json else ""
        if isinstance(merge_strategy, list):
            merge_strategy = merge_strategy[0] if merge_strategy else "nearest_bbox"
        
        log_messages = []
        
        if not original_json.strip():
            log_messages.append("错误：未提供JSON数据")
            return ("", "\n".join(log_messages))
        
        try:
            # 清理JSON输入
            cleaned_json = parse_json(original_json)
            json_data = json.loads(cleaned_json)
            if not isinstance(json_data, list):
                log_messages.append("错误：JSON数据不是列表格式")
                return ("", "\n".join(log_messages))
            
            # 获取原图
            source_img = None
            if source_image and isinstance(source_image, list) and len(source_image) > 0:
                source_img = source_image[0]
                if len(source_img.shape) == 4:
                    source_img = source_img[0]
            
            # 获取筛选前的图片列表
            pre_filter_list = []
            if pre_filter_images and isinstance(pre_filter_images, list):
                for img_batch in pre_filter_images:
                    if img_batch is not None:
                        if len(img_batch.shape) == 4:
                            for i in range(img_batch.shape[0]):
                                pre_filter_list.append(img_batch[i])
                        elif len(img_batch.shape) == 3:
                            pre_filter_list.append(img_batch)
            
            # 获取筛选后的图片列表
            post_filter_list = []
            if post_filter_images and isinstance(post_filter_images, list):
                for img_batch in post_filter_images:
                    if img_batch is not None:
                        if len(img_batch.shape) == 4:
                            for i in range(img_batch.shape[0]):
                                post_filter_list.append(img_batch[i])
                        elif len(img_batch.shape) == 3:
                            post_filter_list.append(img_batch)
            
            # 输出图片数量信息
            log_messages.append(f"筛选前图片数量：{len(pre_filter_list)}")
            log_messages.append(f"筛选后图片数量：{len(post_filter_list)}")
            
            # 检查是否有图片被过滤
            if len(post_filter_list) >= len(json_data):
                log_messages.append("没有图片被过滤，返回原始JSON")
                return (original_json, "\n".join(log_messages))
            
            filtered_out_count = len(json_data) - len(post_filter_list)
            log_messages.append(f"被过滤的图片数量：{filtered_out_count}")
            
            # 步骤6: 通过图片大小匹配筛选前后的图片
            log_messages.append("\n通过图片大小匹配筛选前后的图片：")
            
            # 记录筛选前图片的大小信息（这些对应JSON中的7个项目）
            pre_filter_sizes = []
            for i, img in enumerate(pre_filter_list):
                if len(img.shape) >= 2:
                    height, width = img.shape[0], img.shape[1]
                    pre_filter_sizes.append((height, width, i))
                    log_messages.append(f"筛选前图片{i+1}：大小{width}x{height}")
            
            # 记录筛选后图片的大小信息（这些是从筛选前图片中选出的5张）
            post_filter_sizes = []
            for i, img in enumerate(post_filter_list):
                if len(img.shape) >= 2:
                    height, width = img.shape[0], img.shape[1]
                    post_filter_sizes.append((height, width, i))
                    log_messages.append(f"筛选后图片{i+1}：大小{width}x{height}")
            
            # 通过大小匹配找到筛选后图片对应的筛选前图片索引
            matched_indices = []
            for post_size in post_filter_sizes:
                post_h, post_w, post_idx = post_size
                best_match_idx = -1
                min_size_diff = float('inf')
                
                for pre_size in pre_filter_sizes:
                    pre_h, pre_w, pre_idx = pre_size
                    # 计算大小差异（使用面积差异）
                    size_diff = abs(post_h * post_w - pre_h * pre_w)
                    if size_diff < min_size_diff:
                        min_size_diff = size_diff
                        best_match_idx = pre_idx
                
                matched_indices.append(best_match_idx)
                log_messages.append(f"筛选后图片{post_idx+1}匹配到筛选前图片{best_match_idx+1}（大小差异：{min_size_diff}）")
            
            # 输出筛选后图片在原图中的坐标
            log_messages.append("\n筛选后图片在原图中的坐标：")
            for i, pre_filter_idx in enumerate(matched_indices):
                if pre_filter_idx < len(json_data):
                    bbox = json_data[pre_filter_idx].get('bbox_2d', [])
                    if len(bbox) >= 4:
                        log_messages.append(f"图片{i+1}：坐标[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
            # 步骤7: 根据匹配结果重新构建JSON
            log_messages.append("\n根据匹配结果重新构建JSON：")
            
            # 创建新的JSON数据 - 只包含筛选后图片的bbox
            processed_data = []
            removed_dialogues = []
            
            # 根据匹配的索引构建新的JSON（matched_indices是筛选后图片对应的筛选前图片索引）
            for i, pre_filter_idx in enumerate(matched_indices):
                if pre_filter_idx < len(json_data):
                    item = json_data[pre_filter_idx]
                    new_item = {
                        "bbox_2d": item.get('bbox_2d', []),
                        "dialogue": item.get('dialogue', []).copy()  # 保留原始对话
                    }
                    processed_data.append(new_item)
                    bbox = item.get('bbox_2d', [])
                    log_messages.append(f"保留项目{pre_filter_idx+1}：坐标[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
            # 提取被过滤图片的对话
            kept_indices_set = set(matched_indices)
            for i, item in enumerate(json_data):
                if i not in kept_indices_set:
                    # 被过滤的图片，提取其对话
                    if isinstance(item, dict) and "dialogue" in item:
                        dialogues = item["dialogue"]
                        if isinstance(dialogues, list):
                            removed_dialogues.extend(dialogues)
                            bbox = item.get('bbox_2d', [])
                            log_messages.append(f"移除项目{i+1}：坐标[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
            # 重新分配被移除的对话
            log_messages.append(f"\n对话重新分配：")
            
            if removed_dialogues and processed_data:
                if merge_strategy == "nearest_bbox":
                    # 根据坐标距离分配对话到最近的bbox
                    # 为每个被移除的对话找到最近的保留项目
                    for dialogue in removed_dialogues:
                        # 获取被移除对话的原始bbox（这里简化处理，假设按顺序）
                        removed_index = len(processed_data) + removed_dialogues.index(dialogue)
                        if removed_index < len(json_data):
                            removed_bbox = json_data[removed_index].get('bbox_2d', [])
                            if len(removed_bbox) >= 4:
                                removed_center_x = (removed_bbox[0] + removed_bbox[2]) / 2
                                removed_center_y = (removed_bbox[1] + removed_bbox[3]) / 2
                                
                                # 计算到每个保留项目的距离
                                min_distance = float('inf')
                                nearest_index = 0
                                
                                for i, item in enumerate(processed_data):
                                    bbox = item.get('bbox_2d', [])
                                    if len(bbox) >= 4:
                                        center_x = (bbox[0] + bbox[2]) / 2
                                        center_y = (bbox[1] + bbox[3]) / 2
                                        distance = ((removed_center_x - center_x) ** 2 + (removed_center_y - center_y) ** 2) ** 0.5
                                        
                                        if distance < min_distance:
                                            min_distance = distance
                                            nearest_index = i
                                
                                # 将对话添加到最近的保留项目
                                processed_data[nearest_index]["dialogue"].append(dialogue)
                                dialogue_text = dialogue.get('dialogue', '')[:20] + "..." if len(dialogue.get('dialogue', '')) > 20 else dialogue.get('dialogue', '')
                                log_messages.append(f"对话「{dialogue_text}」分配到图片{nearest_index+1}（距离：{min_distance:.1f}）")
                    
                    # 记录最终结果
                    log_messages.append(f"\n最终分配结果：")
                    for i, item in enumerate(processed_data):
                        log_messages.append(f"图片{i+1}：共{len(item['dialogue'])}个对话")
                
                elif merge_strategy == "first_remaining":
                    # 将所有对话添加到第一个剩余item
                    if processed_data and isinstance(processed_data[0], dict) and "dialogue" in processed_data[0]:
                        processed_data[0]["dialogue"].extend(removed_dialogues)
                        log_messages.append(f"所有对话分配到图片1")
                
                elif merge_strategy == "last_remaining":
                    # 将所有对话添加到最后一个剩余item
                    if processed_data and isinstance(processed_data[-1], dict) and "dialogue" in processed_data[-1]:
                        processed_data[-1]["dialogue"].extend(removed_dialogues)
                        log_messages.append(f"所有对话分配到图片{len(processed_data)}")
            
            # 生成处理后的JSON
            processed_json = json.dumps(processed_data, ensure_ascii=False, indent=2)
            
            final_log = "\n".join(log_messages)
            return (processed_json, final_log)
            
        except json.JSONDecodeError as e:
            log_messages.append(f"错误：JSON格式无效 - {str(e)}")
            return ("", "\n".join(log_messages))
        except Exception as e:
            log_messages.append(f"错误：处理失败 - {str(e)}")
            return ("", "\n".join(log_messages))


NODE_CLASS_MAPPINGS = {
    "Qwen3VLTextGeneration": Qwen3VLTextGenerationNode,
    "Qwen3BboxProcessor": Qwen3BboxProcessorNode,
    "Qwen3VLImageFilter": Qwen3VLImageFilterNode,
    "Qwen3VLJsonProcessor": Qwen3VLJsonProcessorNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLTextGeneration": "Qwen3-VL Unified (Detection/Description/Chat)",
    "Qwen3BboxProcessor": "Qwen3-VL Bbox Processor",
    "Qwen3VLImageFilter": "Qwen3-VL Image Filter",
    "Qwen3VLJsonProcessor": "Qwen3-VL JSON Processor",
}
