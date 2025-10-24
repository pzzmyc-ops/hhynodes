import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# 导入Qwen3VL检测器的核心功能
from .qwen3_vl_detection import Qwen3VLDetector, tensor2pil, pil2tensor

# 使用全局检测器实例
detector = Qwen3VLDetector()


class Qwen3VLImageFilter:
    """Qwen3-VL图片过滤器节点
    使用description模式判断图片是否符合条件，过滤掉结果为"yes"的图片
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE", {
                    "tooltip": "要过滤的图片列表"
                }),
                "filter_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Does this image contain inappropriate content?",
                    "tooltip": "过滤提示词，模型需要回答yes或no"
                }),
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                    "tooltip": "Qwen3-VL模型路径"
                }),
                "max_new_tokens": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "最大生成token数，过滤任务通常只需要很少的token"
                }),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], {
                    "default": "flash_attention_2",
                    "tooltip": "注意力机制"
                }),
                "unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "处理完成后是否卸载模型"
                }),
            },
            "optional": {
                "encrypted_config": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "认证配置"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("filtered_images", "filter_log", "detailed_results", "total_images", "filtered_count")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "filter_images"
    CATEGORY = "hhy/qwen3"

    def filter_images(self, image_list, filter_prompt, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", 
                     max_new_tokens=10, attention="flash_attention_2", unload_model=True, 
                     encrypted_config=None):
        """
        过滤图片：使用Qwen3-VL判断图片是否符合条件
        """
        try:
            # 处理输入参数
            if isinstance(filter_prompt, list):
                filter_prompt = filter_prompt[0] if filter_prompt else "Does this image contain inappropriate content?"
            if isinstance(model_path, list):
                model_path = model_path[0] if model_path else "Qwen/Qwen3-VL-30B-A3B-Instruct"
            if isinstance(max_new_tokens, list):
                max_new_tokens = max_new_tokens[0] if max_new_tokens else 10
            if isinstance(attention, list):
                attention = attention[0] if attention else "flash_attention_2"
            if isinstance(unload_model, list):
                unload_model = unload_model[0] if unload_model else True

            # 处理图片列表
            processed_images = []
            for img_batch in image_list:
                if img_batch is not None:
                    if len(img_batch.shape) == 4:
                        for i in range(img_batch.shape[0]):
                            processed_images.append(img_batch[i])
                    elif len(img_batch.shape) == 3:
                        processed_images.append(img_batch)

            if not processed_images:
                return ([], "错误: 没有提供图片", "没有图片需要处理", 0, 0)

            # 构建过滤提示词，确保模型回答yes或no
            if not any(word in filter_prompt.lower() for word in ["yes", "no", "answer"]):
                enhanced_prompt = f"{filter_prompt} Please answer with 'yes' or 'no' only."
            else:
                enhanced_prompt = filter_prompt

            # 加载模型
            detector.load_model(model_path, attention)

            # 记录日志
            log_messages = []
            detailed_results = []
            filtered_images = []
            total_images = len(processed_images)
            filtered_count = 0

            log_messages.append(f"=== Qwen3-VL Image Filter ===")
            log_messages.append(f"Filter prompt: {enhanced_prompt}")
            log_messages.append(f"Model: {model_path}")
            log_messages.append(f"Total images to process: {total_images}")
            log_messages.append(f"Max tokens: {max_new_tokens}")

            # 逐张处理图片
            for idx, img_tensor in enumerate(processed_images):
                try:
                    # 转换为PIL图像
                    pil_image = tensor2pil(img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor)
                    
                    # 使用Qwen3-VL进行判断
                    result_text = detector.generate_text(
                        enhanced_prompt, 
                        pil_image, 
                        model_path, 
                        max_new_tokens,
                        attention, 
                        False  # 不在循环中卸载模型
                    )
                    
                    # 清理结果文本
                    result_text = result_text.strip().lower()
                    
                    # 判断结果
                    is_filtered = False
                    if "yes" in result_text:
                        is_filtered = True
                        filtered_count += 1
                        action = "FILTERED OUT"
                    elif "no" in result_text:
                        is_filtered = False
                        action = "KEPT"
                        filtered_images.append(img_tensor)
                    else:
                        # 如果回答不明确，默认保留
                        is_filtered = False
                        action = "KEPT (unclear response)"
                        filtered_images.append(img_tensor)
                    
                    # 记录详细结果
                    result_info = f"Image {idx+1}: {action} - Response: '{result_text}'"
                    detailed_results.append(result_info)
                    log_messages.append(result_info)
                    
                    print(f"[Filter] Image {idx+1}/{total_images}: {action} - '{result_text}'")
                    
                except Exception as e:
                    error_msg = f"Image {idx+1}: Error processing - {str(e)}"
                    detailed_results.append(error_msg)
                    log_messages.append(error_msg)
                    print(f"[Filter] {error_msg}")
                    
                    # 出错时默认保留图片
                    filtered_images.append(img_tensor)

            # 卸载模型
            if unload_model:
                detector.unload_model()

            # 生成最终日志
            log_messages.append(f"\n=== Filter Results ===")
            log_messages.append(f"Total images: {total_images}")
            log_messages.append(f"Filtered out: {filtered_count}")
            log_messages.append(f"Kept: {len(filtered_images)}")
            log_messages.append(f"Filter rate: {filtered_count/total_images*100:.1f}%")

            final_log = "\n".join(log_messages)
            detailed_log = "\n".join(detailed_results)

            print(f"[Filter] Completed: {len(filtered_images)}/{total_images} images kept")
            
            return (filtered_images, final_log, detailed_log, total_images, filtered_count)

        except Exception as e:
            error_msg = f"Filter processing failed: {str(e)}"
            print(f"[Filter] {error_msg}")
            return ([], error_msg, error_msg, 0, 0)

    @classmethod
    def IS_CHANGED(cls, image_list, filter_prompt, model_path, max_new_tokens, attention, unload_model, encrypted_config=None):
        """检测输入是否发生变化"""
        return f"{len(image_list) if image_list else 0}_{filter_prompt}_{model_path}_{max_new_tokens}_{attention}_{unload_model}"


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLImageFilter": Qwen3VLImageFilter,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLImageFilter": "Qwen3-VL Image Filter",
}
