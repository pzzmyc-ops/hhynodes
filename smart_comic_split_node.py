import numpy as np
from PIL import Image
import torch
import comfy.utils

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def is_uniform_row(row, color_tolerance=5):
    row_std = np.std(row, axis=0)
    return np.mean(row_std) < color_tolerance

def find_uniform_rows(img_array, color_tolerance=5, min_height=3):
    height = img_array.shape[0]
    uniform_regions = []
    
    in_uniform = False
    start_y = 0
    
    for y in range(height):
        row = img_array[y]
        is_uniform = is_uniform_row(row, color_tolerance)
        
        if is_uniform and not in_uniform:
            in_uniform = True
            start_y = y
        elif not is_uniform and in_uniform:
            if y - start_y >= min_height:
                uniform_regions.append((start_y, y))
            in_uniform = False
    
    if in_uniform and height - start_y >= min_height:
        uniform_regions.append((start_y, height))
    
    return uniform_regions

def split_at_uniform_regions(img_array, regions, min_panel_height=50):
    if not regions:
        return [img_array]
    
    height = img_array.shape[0]
    split_points = []
    
    for start_y, end_y in regions:
        mid_y = (start_y + end_y) // 2
        split_points.append(mid_y)
    
    panel_ranges = []
    last_split = 0
    
    for split_y in split_points:
        if split_y - last_split >= min_panel_height:
            panel_ranges.append((last_split, split_y))
            last_split = split_y
    
    if height - last_split >= min_panel_height:
        panel_ranges.append((last_split, height))
    
    panels = []
    for start, end in panel_ranges:
        panel = img_array[start:end]
        panels.append(panel)
    
    return panels

class SmartComicSplit:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "height_ratio_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "color_tolerance": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "min_height": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "min_panel_height": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "split_comic"
    CATEGORY = "hhy/image"
    
    def split_comic(self, images, height_ratio_threshold, color_tolerance, 
                    min_height, min_panel_height):
        
        height_ratio_threshold = height_ratio_threshold[0] if isinstance(height_ratio_threshold, list) else height_ratio_threshold
        color_tolerance = color_tolerance[0] if isinstance(color_tolerance, list) else color_tolerance
        min_height = min_height[0] if isinstance(min_height, list) else min_height
        min_panel_height = min_panel_height[0] if isinstance(min_panel_height, list) else min_panel_height
        
        image_arrays = []
        
        if isinstance(images, list):
            for img_item in images:
                if img_item is None:
                    continue
                
                if len(img_item.shape) == 3:
                    img_item = img_item.unsqueeze(0)
                
                for i in range(img_item.shape[0]):
                    single_img = img_item[i]
                    img_pil = tensor2pil(single_img)
                    img_array = np.array(img_pil)
                    image_arrays.append(img_array)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            for i in range(images.shape[0]):
                single_img = images[i]
                img_pil = tensor2pil(single_img)
                img_array = np.array(img_pil)
                image_arrays.append(img_array)
        
        if not image_arrays:
            return ([],)
        
        all_panels = []
        
        for img_idx, img_array in enumerate(image_arrays, 1):
            height, width = img_array.shape[:2]
            ratio = height / width if width > 0 else 0
            
            if ratio < height_ratio_threshold:
                all_panels.append(pil2tensor(Image.fromarray(img_array)))
                continue
            
            regions = find_uniform_rows(img_array, color_tolerance, min_height)
            panels = split_at_uniform_regions(img_array, regions, min_panel_height)
            
            for panel_array in panels:
                panel_pil = Image.fromarray(panel_array)
                panel_tensor = pil2tensor(panel_pil)
                all_panels.append(panel_tensor)
        
        return (all_panels,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartComicSplit": SmartComicSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartComicSplit": "Smart Comic Split",
}

