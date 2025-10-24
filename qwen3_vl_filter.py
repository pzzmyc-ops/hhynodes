import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
import gc


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


class Qwen3VLFilter:
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

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
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


filter_detector = Qwen3VLFilter()


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
                "max_new_tokens": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
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
        
        # 加载模型
        filter_detector.load_model(model_path, attention)
        
        filtered_images = []
        filter_results = []
        
        print(f"=== Qwen3VL Image Filter ===")
        print(f"Total images to process: {len(processed_images)}")
        
        # 逐张图片处理
        for idx, img_tensor in enumerate(processed_images):
            print(f"Processing image {idx+1}/{len(processed_images)}")
            
            # 转换为PIL图像
            if len(img_tensor.shape) == 3:
                pil_image = tensor2pil(img_tensor.unsqueeze(0))
            else:
                pil_image = tensor2pil(img_tensor)
            
            # 使用与原版一致的推理逻辑
            text = filter_detector.generate_text(
                prompt_text, pil_image, model_path, max_new_tokens,
                attention, False  # 不在循环中卸载模型
            )
            
            # 解析结果，查找yes/no
            result_text = text.strip().lower()
            is_yes = "yes" in result_text and "no" not in result_text
            is_no = "no" in result_text and "yes" not in result_text
            
            # 如果明确是no，则保留图片
            if is_no:
                filtered_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                filter_results.append(f"Image {idx+1}: NO - KEPT")
                log_messages.append(f"Image {idx+1}: NO - KEPT")
            elif is_yes:
                filter_results.append(f"Image {idx+1}: YES - FILTERED OUT")
                log_messages.append(f"Image {idx+1}: YES - FILTERED OUT")
            else:
                # 如果无法确定，默认保留（保守策略）
                filtered_images.append(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                filter_results.append(f"Image {idx+1}: UNCLEAR - KEPT (default)")
                log_messages.append(f"Image {idx+1}: UNCLEAR - KEPT (default)")
            
            print(f"Image {idx+1} result: {text.strip()}")
        
        # 循环结束后卸载模型
        if unload_model:
            filter_detector.unload_model()
        
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


NODE_CLASS_MAPPINGS = {
    "Qwen3VLImageFilter": Qwen3VLImageFilterNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLImageFilter": "Qwen3-VL Image Filter",
}
