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
    CATEGORY = "hhy/image_processing"

    def separate_objects(self, mask_image, original_image, min_area=100, padding=10):
        mask_pil = tensor2pil(mask_image)
        original_pil = tensor2pil(original_image)
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        if mask_pil.size != original_pil.size:
            mask_pil = mask_pil.resize(original_pil.size, Image.LANCZOS)
        mask_array = np.array(mask_pil)
        original_array = np.array(original_pil)
        _, binary = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        separated_objects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(mask_array.shape[1], x + w + padding)
            y_end = min(mask_array.shape[0], y + h + padding)
            object_mask = (labels == i).astype(np.uint8)
            object_region = original_array[y_start:y_end, x_start:x_end].copy()
            mask_region = object_mask[y_start:y_end, x_start:x_end]
            if len(object_region.shape) == 3:
                object_region[mask_region == 0] = [255, 255, 255]
            else:
                object_region[mask_region == 0] = 255
            object_pil = Image.fromarray(object_region)
            if object_pil.mode != 'RGB':
                object_pil = object_pil.convert('RGB')
            object_tensor = pil2tensor(object_pil)
            separated_objects.append(object_tensor)
        white_image = Image.new('RGB', original_pil.size, (255, 255, 255))
        white_tensor = pil2tensor(white_image)
        if len(separated_objects) == 0:
            return (white_tensor, white_tensor)
        elif len(separated_objects) == 1:
            return (separated_objects[0], white_tensor)
        else:
            return (separated_objects[0], separated_objects[1])


NODE_CLASS_MAPPINGS = {
    "SimpleObjectSeparationNode": SimpleObjectSeparationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleObjectSeparationNode": "Simple Object Separation by hhy"
}
