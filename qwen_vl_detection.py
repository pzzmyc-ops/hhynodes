import os
import ast
import json
from typing import List, Dict, Any
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
import numpy as np

# Import for vision processing
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[QwenVL] Warning: qwen_vl_utils not found. Text-only mode will use basic processing.")
    def process_vision_info(messages):
        """Fallback function if qwen_vl_utils is not available"""
        image_inputs = []
        video_inputs = []
        for message in messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "image":
                        if isinstance(content.get("image"), str):
                            # URL or path - for now just pass None
                            image_inputs.append(None)
                        else:
                            # PIL Image
                            image_inputs.append(content.get("image"))
        return image_inputs, video_inputs

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

def parse_json(json_output: str) -> str:
    """Extract the JSON payload from a model response string."""
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


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    
    items = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 * y_scale)
        abs_x1 = int(x1 * x_scale)
        abs_y2 = int(y2 * y_scale)
        abs_x2 = int(x2 * x_scale)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        items.append({"bbox": [abs_x1, abs_y1, abs_x2, abs_y2], "label": label})
    
    return items


class QwenVLDetector:
    def __init__(self):
        self.device = None
        self.model_loaded = False
        self.processor = None
        self.model = None
        self.model_path = None
        
        # Set HF_HOME to current directory and use mirror
        self.base_path = Path(__file__).parent
        self.hf_home = str(self.base_path / "hf_cache")
        os.environ["HF_HOME"] = self.hf_home
        os.environ["TRANSFORMERS_CACHE"] = self.hf_home
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.device_map = {"": 0}
        else:
            self.device = "cpu"
            self.device_map = "cpu"
            
        print(f"[QwenVL] Initialized with:")
        print(f"[QwenVL] - HF Home: {self.hf_home}")
        print(f"[QwenVL] - Device: {self.device}")

    def load_model(self, model_path, precision="BF16", attention="flash_attention_2"):
        # Use default model if not specified
        if not model_path or model_path == "你的Qwen2.5-VL模型路径":
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        if not self.model_loaded or self.model_path != model_path:
            if self.model_loaded:
                self.unload_model()
                
            print(f"[QwenVL] Loading model from {model_path} on {self.device}...")
            self.model_path = model_path
            
            try:
                # Check if it's a local path
                if os.path.exists(model_path):
                    model_dir = model_path
                    print(f"[QwenVL] Using local model path: {model_dir}")
                else:
                    # Download from HuggingFace
                    model_dir = os.path.join(self.hf_home, "models", model_path.replace("/", "_"))
                    print(f"[QwenVL] Downloading model to: {model_dir}")
                    snapshot_download(
                        repo_id=model_path,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )

                precision = precision.upper()
                dtype_map = {
                    "BF16": torch.bfloat16,
                    "FP16": torch.float16,
                    "FP32": torch.float32,
                }
                torch_dtype = dtype_map.get(precision, torch.bfloat16)
                quant_config = None
                if precision == "INT4":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                elif precision == "INT8":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)

                attn_impl = attention
                if precision == "FP32" and attn_impl == "flash_attention_2":
                    # FlashAttention doesn't support fp32. Fall back to SDPA.
                    attn_impl = "sdpa"

                self.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    cache_dir=self.hf_home,
                    trust_remote_code=True
                )
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=self.device_map,
                    attn_implementation=attn_impl,
                    cache_dir=self.hf_home,
                    trust_remote_code=True
                ).eval()
                
                self.model_loaded = True
                print(f"[QwenVL] Model loaded successfully on {self.device}")
            except Exception as e:
                print(f"[QwenVL] Failed to load model: {e}")
                raise e

    def unload_model(self):
        if self.model_loaded:
            print(f"[QwenVL] Unloading model from {self.device}...")
            if self.device == "cuda:0":
                self.model = self.model.to("cpu")
            del self.model
            torch.cuda.empty_cache()
            self.model_loaded = False
            self.model_path = None

    def generate_text(self, prompt_text, image=None, model_path="", max_new_tokens=128, 
                     precision="BF16", attention="flash_attention_2", unload_model=False):
        """Generate text with optional image input"""
        try:
            self.load_model(model_path, precision, attention)
            
            # Prepare messages based on whether image is provided
            if image is not None:
                # Image + Text mode
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:  # Batch of images, take first one
                        image = image[0]
                    pil_image = tensor2pil(image.unsqueeze(0))
                else:
                    pil_image = image
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
            else:
                # Text-only mode
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
            
            with torch.no_grad():
                # Preparation for inference
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)

                # Inference: Generation of the output
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                print(f"[QwenVL] Generated text: {output_text}")
                
                if unload_model:
                    self.unload_model()
                
                return output_text
                
        except Exception as e:
            print(f"[QwenVL] Error in text generation: {str(e)}")
            return ""

    def detect_objects(self, image, model_path, prompt_text="object", bbox_selection="all", 
                      merge_boxes=False, precision="BF16", 
                      attention="flash_attention_2", unload_model=False):
        try:
            self.load_model(model_path, precision, attention)
            
            detection_images = []
            output_masks = []
            
            # Process each image in the batch
            for img_tensor in image:
                img_tensor_single = torch.unsqueeze(img_tensor, 0)
                pil_image = tensor2pil(img_tensor_single)
                
                # 如果prompt_text包含"Locate"或"bbox"关键词，直接使用，否则构建检测prompt
                if any(keyword in prompt_text.lower() for keyword in ["locate", "bbox", "detection", "find", "detect"]):
                    prompt = prompt_text
                else:
                    prompt = f"Locate the {prompt_text} and output bbox in JSON"

                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": pil_image}]},
                ]
                
                with torch.no_grad():
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(self.device)
                    output_ids = self.model.generate(**inputs, max_new_tokens=1024)
                    gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
                    output_text = self.processor.batch_decode(
                        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )[0]
                    
                    input_h = inputs['image_grid_thw'][0][1] * 14
                    input_w = inputs['image_grid_thw'][0][2] * 14
                    items = parse_boxes(
                        output_text,
                        pil_image.width,
                        pil_image.height,
                        input_w,
                        input_h,
                    )

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

                    # Create detection image with bboxes drawn
                    if boxes:
                        bboxes = [b["bbox"] for b in boxes]
                        labels = [b.get('label', prompt_text) for b in boxes]
                        detection_image = draw_bbox_on_image(pil_image, bboxes, labels)
                        detection_images.append(pil2tensor(detection_image))
                        
                        # Create masks from bboxes
                        individual_masks = create_mask_from_bboxes(pil_image.size, bboxes, merge_boxes)
                        output_masks.extend(individual_masks)
                        
                        print(f"[QwenVL] Detected {len(boxes)} objects: {[b.get('label', prompt_text) for b in boxes]}")
                    else:
                        # No detection, return original image and empty mask
                        detection_images.append(pil2tensor(pil_image))
                        empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                        output_masks.append(empty_mask)
                        print("[QwenVL] No objects detected")
                
                if unload_model:
                    self.unload_model()
                
                # Convert lists to tensors
                detection_images_tensor = torch.cat(detection_images, dim=0) if detection_images else torch.zeros((1, 3, 512, 512))
                output_masks_tensor = torch.cat(output_masks, dim=0) if output_masks else torch.zeros((1, 512, 512))
                
                return detection_images_tensor, output_masks_tensor
                
        except Exception as e:
            print(f"[QwenVL] Error in object detection: {str(e)}")
            # Return empty outputs on error
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            return empty_image, empty_mask


detector = QwenVLDetector()






class QwenVLGenderFilterBatchNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "gender_filter": (["woman", "man"], {"default": "woman"}),
                "model_path": ("STRING", {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "precision": ([
                    "INT4",
                    "INT8", 
                    "BF16",
                    "FP16",
                    "FP32",
                ], {"default": "BF16"}),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], {"default": "flash_attention_2"}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": "The person in image is woman or man? Output only woman or man"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("filtered_images", "detection_results", "selected_gender")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False)
    FUNCTION = "filter_batch_by_gender"
    CATEGORY = "hhy/Vision"

    def filter_batch_by_gender(self, images, gender_filter, model_path="Qwen/Qwen2.5-VL-7B-Instruct", 
                              precision="BF16", attention="flash_attention_2", unload_model=False, 
                              custom_prompt="The person in image is woman or man? Output only woman or man"):
        
        # 当使用 INPUT_IS_LIST = True 时，所有参数都会被包装成列表
        if isinstance(gender_filter, list):
            gender_filter = gender_filter[0] if gender_filter else "woman"
        if isinstance(model_path, list):
            model_path = model_path[0] if model_path else "Qwen/Qwen2.5-VL-7B-Instruct"
        if isinstance(precision, list):
            precision = precision[0] if precision else "BF16"
        if isinstance(attention, list):
            attention = attention[0] if attention else "flash_attention_2"
        if isinstance(unload_model, list):
            unload_model = unload_model[0] if unload_model else False
        if isinstance(custom_prompt, list):
            custom_prompt = custom_prompt[0] if custom_prompt else "The person in image is woman or man? Output only woman or man"
        
        filtered_images = []  # 存储所有匹配的图片和其尺寸信息 [(img_tensor, area, img_info), ...]
        detection_results = []
        total_processed = 0
        
        print(f"[GenderFilterBatch] ===== BATCH PROCESSING =====")
        print(f"[GenderFilterBatch] Input type: {type(images)}")
        print(f"[GenderFilterBatch] Number of input batches: {len(images)}")
        print(f"[GenderFilterBatch] Filter: {gender_filter}")
        print(f"[GenderFilterBatch] Model: {model_path}")
        
        # 处理每个输入批次
        for batch_idx, img_batch in enumerate(images):
            if img_batch is None:
                print(f"[GenderFilterBatch] Batch {batch_idx+1}: None (skipped)")
                continue
                
            print(f"[GenderFilterBatch] Processing batch {batch_idx+1}: shape {img_batch.shape}")
            
            # 确保是4D张量 [batch, H, W, C]
            if len(img_batch.shape) == 3:  # [H, W, C]
                img_batch = img_batch.unsqueeze(0)  # [1, H, W, C]
            elif len(img_batch.shape) != 4:
                print(f"[GenderFilterBatch] Batch {batch_idx+1}: Unexpected shape {img_batch.shape} (skipped)")
                continue
            
            batch_size = img_batch.shape[0]
            
            # 处理批次中的每张图片
            for i in range(batch_size):
                img_tensor = img_batch[i]  # [H, W, C]
                total_processed += 1
                
                print(f"[GenderFilterBatch] Processing image {total_processed} (batch {batch_idx+1}, image {i+1}): shape {img_tensor.shape}")
                
                # 转换为PIL图像
                img_tensor_single = torch.unsqueeze(img_tensor, 0)  # [1, H, W, C]
                pil_image = tensor2pil(img_tensor_single)
                
                # 使用自定义prompt或默认prompt进行性别检测
                prompt = custom_prompt if custom_prompt.strip() else "The person in image is woman or man? Output only woman or man"
                
                # 调用模型进行性别检测
                result = detector.generate_text(
                    prompt, pil_image, model_path, 32, 
                    precision, attention, False  # 不在每次调用后卸载模型
                )
                
                # 清理和分析结果
                result_lower = result.lower().strip()
                detected_gender = None
                
                # 查找关键词
                if "woman" in result_lower or "female" in result_lower or "girl" in result_lower or "lady" in result_lower:
                    detected_gender = "woman"
                elif "man" in result_lower or "male" in result_lower or "boy" in result_lower or "gentleman" in result_lower:
                    detected_gender = "man"
                
                # 判断是否匹配筛选条件
                is_match = detected_gender == gender_filter
                match_status = "✓ MATCH" if is_match else "✗ NO MATCH"
                
                detection_results.append(f"Image {total_processed}: '{result.strip()}' -> {detected_gender} -> {match_status}")
                print(f"[GenderFilterBatch] Image {total_processed}: Raw='{result.strip()}' -> Detected={detected_gender} -> Filter={gender_filter} -> {match_status}")
                
                # 只收集匹配的图片信息（图片、面积、描述）
                if is_match:
                    height, width = img_tensor.shape[0], img_tensor.shape[1]
                    area = height * width
                    img_info = f"Image {total_processed} ({height}x{width}, area={area})"
                    filtered_images.append((img_tensor, area, img_info))
                    print(f"[GenderFilterBatch] ✓ Added image {total_processed} to filtered results: {height}x{width} (area={area})")
        
        # 在所有图片处理完成后卸载模型（如果需要）
        if unload_model:
            detector.unload_model()
        
        # 处理结果 - 选择最大面积的图片
        count = len(filtered_images)
        print(f"[GenderFilterBatch] ===== RESULTS =====")
        print(f"[GenderFilterBatch] Total processed: {total_processed}")
        print(f"[GenderFilterBatch] Matches found: {count}")
        print(f"[GenderFilterBatch] Filter: {gender_filter}")
        
        if filtered_images:
            # 找到面积最大的图片
            largest_img_data = max(filtered_images, key=lambda x: x[1])  # x[1] 是面积
            largest_img_tensor = largest_img_data[0]  # 图片tensor
            largest_img_info = largest_img_data[2]    # 图片信息
            
            # 转换为批次格式 [1, H, W, C] （只有一张图片）
            filtered_images_tensor = largest_img_tensor.unsqueeze(0)
            
            print(f"[GenderFilterBatch] Selected largest image: {largest_img_info}")
            print(f"[GenderFilterBatch] Output tensor shape: {filtered_images_tensor.shape}")
            
            # 更新检测结果，标明选择了哪张图片
            detection_results.append(f"SELECTED: {largest_img_info} (largest among {count} matches)")
        else:
            # 如果没有匹配的图片，返回一个小的占位符
            filtered_images_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            print(f"[GenderFilterBatch] No matches found, returning small placeholder: {filtered_images_tensor.shape}")
        
        detection_results_text = "\n".join(detection_results)
        print(f"[GenderFilterBatch] ==================")
        
        return (filtered_images_tensor, detection_results_text, gender_filter)


class QwenVLTextGenerationNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "object_detection",   # 对象检测模式
                    "image_description",  # 图片打标/描述模式
                    "text_only",         # 纯文本对话模式
                ], {"default": "image_description"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "object"}),
                "model_path": ("STRING", {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 4096, "step": 1}),
                "precision": ([
                    "INT4",
                    "INT8", 
                    "BF16",
                    "FP16",
                    "FP32",
                ], {"default": "BF16"}),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], {"default": "flash_attention_2"}),
                "unload_model": ("BOOLEAN", {"default": False}),
                # 对象检测相关参数
                "bbox_selection": ("STRING", {"default": "all"}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("generated_text", "detection_image", "mask")
    FUNCTION = "process"
    CATEGORY = "hhy/Vision"

    def process(self, mode, prompt_text, model_path="Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens=128, 
               precision="BF16", attention="flash_attention_2", unload_model=False, 
               bbox_selection="all", merge_boxes=False, image=None):
        
        # 初始化返回值
        generated_text = ""
        detection_image = None
        mask = None
        
        if mode == "object_detection":
            # 对象检测模式
            if image is None:
                generated_text = "Error: Image is required for object detection mode."
                # 返回空的图像和mask
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, empty_image, empty_mask)
            
            # 如果prompt_text为空或默认值，使用通用的检测prompt
            detection_prompt = prompt_text if prompt_text.strip() and prompt_text != "Describe this image." else "object"
            
            detection_image, mask = detector.detect_objects(
                image, model_path, detection_prompt, bbox_selection, merge_boxes, 
                precision, attention, unload_model
            )
            # 保持原有的检测结果文本输出，由detect_objects方法内部的print提供反馈
            generated_text = f"Object detection completed for: {detection_prompt}"
            
        elif mode == "image_description":
            # 图片描述模式
            if image is None:
                generated_text = "Error: Image is required for image description mode."
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, empty_image, empty_mask)
            
            # Convert image tensor to PIL
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # Batch of images, take first one
                    img_for_desc = image[0]
                else:
                    img_for_desc = image
                pil_image = tensor2pil(img_for_desc.unsqueeze(0))
            else:
                pil_image = image
            
            # 如果没有提供prompt，使用默认的图片描述prompt
            if not prompt_text or prompt_text.strip() == "":
                prompt_text = "Describe this image."
            
            generated_text = detector.generate_text(
                prompt_text, pil_image, model_path, max_new_tokens, 
                precision, attention, unload_model
            )
            
            # 对于图片描述模式，返回原始图片
            detection_image = image
            # 创建空mask
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    batch_size, _, height, width = image.shape
                    mask = torch.zeros((batch_size, height, width))
                else:
                    _, height, width = image.shape
                    mask = torch.zeros((1, height, width))
            else:
                mask = torch.zeros((1, 512, 512))
                
        elif mode == "text_only":
            # 纯文本模式
            if not prompt_text or prompt_text.strip() == "":
                generated_text = "Error: Prompt text is required for text-only mode."
            else:
                generated_text = detector.generate_text(
                    prompt_text, None, model_path, max_new_tokens, 
                    precision, attention, unload_model
                )
            
            # 对于纯文本模式，返回空的图像和mask
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_image = empty_image
            mask = empty_mask
            
        else:
            generated_text = f"Error: Unknown mode '{mode}'."
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            detection_image = empty_image
            mask = empty_mask
        
        return (generated_text, detection_image, mask)


NODE_CLASS_MAPPINGS = {
    "QwenVLTextGeneration": QwenVLTextGenerationNode,
    "QwenVLGenderFilterBatch": QwenVLGenderFilterBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLTextGeneration": "Qwen2.5-VL Unified (Detection/Description/Chat)",
    "QwenVLGenderFilterBatch": "Qwen2.5-VL Gender Filter",
} 