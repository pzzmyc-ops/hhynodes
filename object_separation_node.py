#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class SimpleObjectSeparationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),
                "original_image": ("IMAGE",),
            },
            "optional": {
                "min_area": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("object_1", "object_2")
    OUTPUT_NODE = True
    FUNCTION = "separate_objects"
    CATEGORY = "hhy/image"

    def separate_objects(self, mask_image, original_image, min_area=100, padding=10):
        """
        根据黑白掩码图片从原始图片中分离对象
        
        Args:
            mask_image: 黑白掩码图片张量
            original_image: 原始彩色图片张量
            min_area: 最小面积阈值，小于此面积的对象将被忽略
            padding: 提取对象时的边距
            
        Returns:
            从原始图片中分离后的对象图片列表
        """
        # 转换掩码张量为PIL图像
        mask_pil = tensor2pil(mask_image)
        
        # 转换原始图片张量为PIL图像
        original_pil = tensor2pil(original_image)
        
        # 转换掩码为灰度图像（如果不是的话）
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        
        # 确保原始图片和掩码尺寸一致
        if mask_pil.size != original_pil.size:
            mask_pil = mask_pil.resize(original_pil.size, Image.LANCZOS)
        
        # 转换为numpy数组
        mask_array = np.array(mask_pil)
        original_array = np.array(original_pil)
        
        # 二值化处理掩码，确保只有0和255两个值
        _, binary = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # 查找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        separated_objects = []
        
        # 遍历每个连通组件（跳过背景，标签0）
        for i in range(1, num_labels):
            # 获取当前组件的统计信息
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 过滤小面积对象
            if area < min_area:
                continue
                
            # 获取边界框
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 添加边距
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(mask_array.shape[1], x + w + padding)
            y_end = min(mask_array.shape[0], y + h + padding)
            
            # 创建只包含当前对象的掩码
            object_mask = (labels == i).astype(np.uint8)
            
            # 从原始图片中提取对象区域
            object_region = original_array[y_start:y_end, x_start:x_end].copy()
            mask_region = object_mask[y_start:y_end, x_start:x_end]
            
            # 创建白色背景，只保留掩码区域的原始像素
            if len(object_region.shape) == 3:  # RGB图像
                # 将非掩码区域设为白色
                object_region[mask_region == 0] = [255, 255, 255]
            else:  # 灰度图像
                object_region[mask_region == 0] = 255
            
            # 转换为PIL图像
            object_pil = Image.fromarray(object_region)
            
            # 确保是RGB模式
            if object_pil.mode != 'RGB':
                object_pil = object_pil.convert('RGB')
            
            # 转换回张量格式
            object_tensor = pil2tensor(object_pil)
            separated_objects.append(object_tensor)
        
        # 确保总是返回两个图像
        white_image = Image.new('RGB', original_pil.size, (255, 255, 255))
        white_tensor = pil2tensor(white_image)
        
        # 如果没有找到对象，返回两个白色图像
        if len(separated_objects) == 0:
            return (white_tensor, white_tensor)
        # 如果只找到一个对象，第二个返回白色图像
        elif len(separated_objects) == 1:
            return (separated_objects[0], white_tensor)
        # 如果找到两个或更多对象，返回前两个
        else:
            return (separated_objects[0], separated_objects[1])


NODE_CLASS_MAPPINGS = {
    "SimpleObjectSeparationNode": SimpleObjectSeparationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleObjectSeparationNode": "Simple Object Separation by hhy"
}
