import os
import ast
import json
from typing import List, Dict, Any
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
import numpy as np

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[QwenVL] Warning: qwen_vl_utils not found. Text-only mode will use basic processing.")
    def process_vision_info(messages):
        image_inputs = []
        video_inputs = []
        for message in messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "image":
                        if isinstance(content.get("image"), str):
                            image_inputs.append(None)
                        else:
                            image_inputs.append(content.get("image"))
        return image_inputs, video_inputs

def tensor2pil(image):
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
            print(f"[QwenVL] Warning: Invalid tensor shape {numpy_image.shape} for image conversion")
            return Image.new('RGB', (64, 64), color='black')
        else:
            print(f"[QwenVL] Warning: Unexpected tensor shape {numpy_image.shape} for image conversion")
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
                print(f"[QwenVL] Warning: Unsupported channel count {numpy_image.shape[2]}")
                return Image.new('RGB', (64, 64), color='black')
        else:
            print(f"[QwenVL] Warning: Cannot convert shape {numpy_image.shape} to PIL Image")
            return Image.new('RGB', (64, 64), color='black')
    except Exception as e:
        print(f"[QwenVL] Error in tensor2pil conversion: {e}")
        print(f"[QwenVL] Input tensor shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        return Image.new('RGB', (64, 64), color='black')

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def add_mask(mask1, mask2):
    return torch.clamp(mask1 + mask2, 0.0, 1.0)

def draw_bbox_on_image(image, bboxes, labels=None, colors=None):
    if not isinstance(image, Image.Image):
        image = tensor2pil(image)
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
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

def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
) -> List[Dict[str, Any]]:
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            print(f"[QwenVL] Non-JSON output: {text[:200]}...")
            return []
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
        # 完全离线模式，不使用任何在线功能
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.device_map = {"": 0}
        else:
            self.device = "cpu"
            self.device_map = "cpu"
        print(f"[QwenVL] Initialized with:")
        print(f"[QwenVL] - Mode: Offline only")
        print(f"[QwenVL] - Device: {self.device}")

    def load_model(self, model_path, precision="BF16", attention="flash_attention_2"):
        # 固定使用本地模型路径
        if not model_path or model_path in ["你的Qwen2.5-VL模型路径", "Qwen/Qwen2.5-VL-7B-Instruct"]:
            model_path = "models/LLM/Qwen2.5-VL-7B-Instruct"
        
        if not self.model_loaded or self.model_path != model_path:
            if self.model_loaded:
                self.unload_model()
            print(f"[QwenVL] Loading model from {model_path} on {self.device}...")
            self.model_path = model_path
            try:
                # 只支持本地模型路径
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found at: {model_path}. Please ensure the model is downloaded to this local path.")
                
                model_dir = model_path
                print(f"[QwenVL] Using local model path: {model_dir}")
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
                    attn_impl = "sdpa"
                self.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=self.device_map,
                    attn_implementation=attn_impl,
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
        try:
            self.load_model(model_path, precision, attention)
            if image is not None:
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:
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
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
            with torch.no_grad():
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # 修复：根据是否有图像来正确处理输入
                if image is not None:
                    inputs = self.processor(
                        text=[text],
                        images=[pil_image] if image is not None else None,
                        return_tensors="pt",
                        padding=True
                    )
                else:
                    # 纯文本模式
                    inputs = self.processor(
                        text=[text],
                        images=None,
                        videos=None,
                        return_tensors="pt",
                        padding=True
                    )
                inputs = inputs.to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                else:
                    generated_ids_trimmed = generated_ids
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
            all_bboxes = []
            for img_tensor in image:
                img_tensor_single = torch.unsqueeze(img_tensor, 0)
                pil_image = tensor2pil(img_tensor_single)
                if any(keyword in prompt_text.lower() for keyword in ["locate", "bbox", "detection", "find", "detect"]):
                    prompt = prompt_text
                else:
                    prompt = f"Locate the {prompt_text} and output bbox in JSON"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": pil_image}]},
                ]
                with torch.no_grad():
                    try:
                        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(self.device)
                        output_ids = self.model.generate(**inputs, max_new_tokens=1024)
                        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
                        output_text = self.processor.batch_decode(
                            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )[0]
                        print(f"[QwenVL] Model output: {output_text[:200]}...")
                        input_h = inputs['image_grid_thw'][0][1] * 14
                        input_w = inputs['image_grid_thw'][0][2] * 14
                        items = parse_boxes(
                            output_text,
                            pil_image.width,
                            pil_image.height,
                            input_w,
                            input_h,
                        )
                    except Exception as e:
                        print(f"[QwenVL] Error during model inference: {e}")
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
                        try:
                            bboxes = [b["bbox"] for b in boxes]
                            labels = [b.get('label', prompt_text) for b in boxes]
                            all_bboxes.append([bboxes])
                            detection_image = draw_bbox_on_image(pil_image, bboxes, labels)
                            detection_tensor = pil2tensor(detection_image)
                            if len(detection_tensor.shape) == 4 and detection_tensor.shape[1] > 0 and detection_tensor.shape[2] > 0:
                                detection_images.append(detection_tensor)
                            else:
                                print(f"[QwenVL] Warning: Invalid detection tensor shape {detection_tensor.shape}, using original image")
                                detection_images.append(pil2tensor(pil_image))
                            individual_masks = create_mask_from_bboxes(pil_image.size, bboxes, merge_boxes)
                            valid_masks = []
                            for mask in individual_masks:
                                if isinstance(mask, torch.Tensor) and len(mask.shape) == 3 and mask.shape[1] > 0 and mask.shape[2] > 0:
                                    valid_masks.append(mask)
                                else:
                                    print(f"[QwenVL] Warning: Invalid mask shape {mask.shape if hasattr(mask, 'shape') else 'unknown'}")
                                    valid_masks.append(torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32))
                            output_masks.extend(valid_masks)
                            print(f"[QwenVL] Detected {len(boxes)} objects: {[b.get('label', prompt_text) for b in boxes]}")
                        except Exception as e:
                            print(f"[QwenVL] Error processing detection results: {e}")
                            detection_images.append(pil2tensor(pil_image))
                            empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                            output_masks.append(empty_mask)
                            all_bboxes.append([[]])
                    else:
                        detection_images.append(pil2tensor(pil_image))
                        empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                        output_masks.append(empty_mask)
                        all_bboxes.append([[]])
                        print("[QwenVL] No objects detected")
            if unload_model:
                self.unload_model()
            if detection_images:
                try:
                    shapes = [img.shape for img in detection_images]
                    if len(set(shapes)) > 1:
                        print(f"[QwenVL] Warning: Inconsistent tensor shapes: {shapes}")
                        target_shape = shapes[0]
                        normalized_images = []
                        for img in detection_images:
                            if img.shape != target_shape:
                                pil_img = tensor2pil(img)
                                pil_img = pil_img.resize((target_shape[3], target_shape[2]))
                                img = pil2tensor(pil_img)
                            normalized_images.append(img)
                        detection_images = normalized_images
                    detection_images_tensor = torch.cat(detection_images, dim=0)
                    if len(detection_images_tensor.shape) != 4:
                        print(f"[QwenVL] Warning: Invalid detection tensor shape {detection_images_tensor.shape}, creating fallback")
                        detection_images_tensor = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
                except Exception as e:
                    print(f"[QwenVL] Error concatenating detection images: {e}")
                    detection_images_tensor = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
            else:
                detection_images_tensor = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
            if output_masks:
                try:
                    mask_shapes = [mask.shape for mask in output_masks]
                    if len(set(mask_shapes)) > 1:
                        print(f"[QwenVL] Warning: Inconsistent mask shapes: {mask_shapes}")
                        target_shape = mask_shapes[0]
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
                    if len(output_masks_tensor.shape) != 3:
                        print(f"[QwenVL] Warning: Invalid mask tensor shape {output_masks_tensor.shape}, creating fallback")
                        output_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
                except Exception as e:
                    print(f"[QwenVL] Error concatenating masks: {e}")
                    output_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
            else:
                output_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
            return detection_images_tensor, output_masks_tensor, all_bboxes
        except Exception as e:
            print(f"[QwenVL] Error in object detection: {str(e)}")
            empty_image = torch.zeros((1, 3, 512, 512))
            empty_mask = torch.zeros((1, 512, 512))
            empty_bboxes = [[[]]]
            return empty_image, empty_mask, empty_bboxes

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
                "model_path": ("STRING", {"default": "models/LLM/Qwen2.5-VL-7B-Instruct"}),
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

    def filter_batch_by_gender(self, images, gender_filter, model_path="models/LLM/Qwen2.5-VL-7B-Instruct", 
                              precision="BF16", attention="flash_attention_2", unload_model=False, 
                              custom_prompt="The person in image is woman or man? Output only woman or man"):
        if isinstance(gender_filter, list):
            gender_filter = gender_filter[0] if gender_filter else "woman"
        if isinstance(model_path, list):
            model_path = model_path[0] if model_path else "models/LLM/Qwen2.5-VL-7B-Instruct"
        if isinstance(precision, list):
            precision = precision[0] if precision else "BF16"
        if isinstance(attention, list):
            attention = attention[0] if attention else "flash_attention_2"
        if isinstance(unload_model, list):
            unload_model = unload_model[0] if unload_model else False
        if isinstance(custom_prompt, list):
            custom_prompt = custom_prompt[0] if custom_prompt else "The person in image is woman or man? Output only woman or man"
        filtered_images = []
        detection_results = []
        total_processed = 0
        print(f"[GenderFilterBatch] ===== BATCH PROCESSING =====")
        print(f"[GenderFilterBatch] Input type: {type(images)}")
        print(f"[GenderFilterBatch] Number of input batches: {len(images)}")
        print(f"[GenderFilterBatch] Filter: {gender_filter}")
        print(f"[GenderFilterBatch] Model: {model_path}")
        for batch_idx, img_batch in enumerate(images):
            if img_batch is None:
                print(f"[GenderFilterBatch] Batch {batch_idx+1}: None (skipped)")
                continue
            print(f"[GenderFilterBatch] Processing batch {batch_idx+1}: shape {img_batch.shape}")
            if len(img_batch.shape) == 3:
                img_batch = img_batch.unsqueeze(0)
            elif len(img_batch.shape) != 4:
                print(f"[GenderFilterBatch] Batch {batch_idx+1}: Unexpected shape {img_batch.shape} (skipped)")
                continue
            batch_size = img_batch.shape[0]
            for i in range(batch_size):
                img_tensor = img_batch[i]
                total_processed += 1
                print(f"[GenderFilterBatch] Processing image {total_processed} (batch {batch_idx+1}, image {i+1}): shape {img_tensor.shape}")
                img_tensor_single = torch.unsqueeze(img_tensor, 0)
                pil_image = tensor2pil(img_tensor_single)
                prompt = custom_prompt if custom_prompt.strip() else "The person in image is woman or man? Output only woman or man"
                result = detector.generate_text(
                    prompt, pil_image, model_path, 32, 
                    precision, attention, False
                )
                result_lower = result.lower().strip()
                detected_gender = None
                if "woman" in result_lower or "female" in result_lower or "girl" in result_lower or "lady" in result_lower:
                    detected_gender = "woman"
                elif "man" in result_lower or "male" in result_lower or "boy" in result_lower or "gentleman" in result_lower:
                    detected_gender = "man"
                is_match = detected_gender == gender_filter
                match_status = "✓ MATCH" if is_match else "✗ NO MATCH"
                detection_results.append(f"Image {total_processed}: '{result.strip()}' -> {detected_gender} -> {match_status}")
                print(f"[GenderFilterBatch] Image {total_processed}: Raw='{result.strip()}' -> Detected={detected_gender} -> Filter={gender_filter} -> {match_status}")
                if is_match:
                    height, width = img_tensor.shape[0], img_tensor.shape[1]
                    area = height * width
                    img_info = f"Image {total_processed} ({height}x{width}, area={area})"
                    filtered_images.append((img_tensor, area, img_info))
                    print(f"[GenderFilterBatch] ✓ Added image {total_processed} to filtered results: {height}x{width} (area={area})")
        if unload_model:
            detector.unload_model()
        count = len(filtered_images)
        print(f"[GenderFilterBatch] ===== RESULTS =====")
        print(f"[GenderFilterBatch] Total processed: {total_processed}")
        print(f"[GenderFilterBatch] Matches found: {count}")
        print(f"[GenderFilterBatch] Filter: {gender_filter}")
        if filtered_images:
            largest_img_data = max(filtered_images, key=lambda x: x[1])
            largest_img_tensor = largest_img_data[0]
            largest_img_info = largest_img_data[2]
            filtered_images_tensor = largest_img_tensor.unsqueeze(0)
            print(f"[GenderFilterBatch] Selected largest image: {largest_img_info}")
            print(f"[GenderFilterBatch] Output tensor shape: {filtered_images_tensor.shape}")
            detection_results.append(f"SELECTED: {largest_img_info} (largest among {count} matches)")
        else:
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
                    "object_detection",
                    "image_description",
                    "text_only",
                ], {"default": "image_description"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "object"}),
                "model_path": ("STRING", {"default": "models/LLM/Qwen2.5-VL-7B-Instruct"}),
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
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "process"
    CATEGORY = "hhy"

    def process(self, mode, prompt_text, model_path="models/LLM/Qwen2.5-VL-7B-Instruct", max_new_tokens=128, 
               precision="BF16", attention="flash_attention_2", unload_model=False, 
               bbox_selection="all", merge_boxes=False, image=None):
        if isinstance(mode, list):
            mode = mode[0] if mode else "image_description"
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0] if prompt_text else "object"
        if isinstance(model_path, list):
            model_path = model_path[0] if model_path else "models/LLM/Qwen2.5-VL-7B-Instruct"
        if isinstance(max_new_tokens, list):
            max_new_tokens = max_new_tokens[0] if max_new_tokens else 128
        if isinstance(precision, list):
            precision = precision[0] if precision else "BF16"
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
        detection_image = None
        mask = None
        bboxes = [[[]]]
        if mode == "object_detection":
            if image is None:
                generated_text = "Error: Image is required for object detection mode."
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, empty_image, empty_mask, bboxes)
            detection_prompt = prompt_text if prompt_text.strip() and prompt_text != "Describe this image." else "object"
            detection_image, mask, bboxes = detector.detect_objects(
                image, model_path, detection_prompt, bbox_selection, merge_boxes, 
                precision, attention, unload_model
            )
            generated_text = f"Object detection completed for: {detection_prompt}"
        elif mode == "image_description":
            if image is None:
                generated_text = "Error: Image is required for image description mode."
                empty_image = torch.zeros((1, 3, 512, 512))
                empty_mask = torch.zeros((1, 512, 512))
                return (generated_text, empty_image, empty_mask, bboxes)
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
                precision, attention, unload_model
            )
            detection_image = image
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
            if not prompt_text or prompt_text.strip() == "":
                generated_text = "Error: Prompt text is required for text-only mode."
            else:
                generated_text = detector.generate_text(
                    prompt_text, None, model_path, max_new_tokens, 
                    precision, attention, unload_model
                )
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
        return (generated_text, detection_image, mask, bboxes)

NODE_CLASS_MAPPINGS = {
    "QwenVLTextGeneration": QwenVLTextGenerationNode,
    "QwenVLGenderFilterBatch": QwenVLGenderFilterBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLTextGeneration": "Qwen2.5-VL 统一 (检测/描述/聊天)",
    "QwenVLGenderFilterBatch": "Qwen2.5-VL 性别过滤器",
} 