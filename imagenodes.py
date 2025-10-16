from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import requests
import json
import io
import re

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageCropByMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "crop_by_mask"
    CATEGORY = "hhy/image"
    OUTPUT_IS_LIST = (True, True)

    def crop_by_mask(self, image, mask):
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # 确定处理策略
        num_images = image.shape[0]
        num_masks = mask.shape[0]
        
        if num_images == num_masks:
            pairs = [(i, i) for i in range(num_images)]
        elif num_images == 1:
            pairs = [(0, i) for i in range(num_masks)]
        elif num_masks == 1:
            pairs = [(i, 0) for i in range(num_images)]
        else:
            max_count = max(num_images, num_masks)
            pairs = [(i % num_images, i % num_masks) for i in range(max_count)]
        
        # 预分配列表，避免频繁扩容
        result_images = []
        result_masks = []
        
        # 批量处理，避免频繁的tensor<->PIL转换
        for img_idx, mask_idx in pairs:
            current_image = image[img_idx]  # [H, W, C]
            current_mask = mask[mask_idx]   # [H, W]
            
            # 直接在tensor上找到mask的边界框，避免转换为PIL
            mask_indices = (current_mask > 0).nonzero(as_tuple=True)
            
            if len(mask_indices[0]) == 0:
                # 如果mask为空，返回原图
                result_images.append(current_image.unsqueeze(0))
                result_masks.append(current_mask.unsqueeze(0))
                continue
            
            # 使用torch操作计算边界框
            y_coords, x_coords = mask_indices
            y_min, y_max = y_coords.min().item(), y_coords.max().item()
            x_min, x_max = x_coords.min().item(), x_coords.max().item()
            
            # 直接在tensor上裁剪，避免PIL转换
            cropped_image = current_image[y_min:y_max+1, x_min:x_max+1, :]
            
            H, W = current_mask.shape
            position_mask = torch.zeros((H, W), dtype=current_mask.dtype, device=current_mask.device)
            position_mask[y_min:y_max+1, x_min:x_max+1] = 1.0
            
            result_images.append(cropped_image.unsqueeze(0))
            result_masks.append(position_mask.unsqueeze(0))

        return (result_images, result_masks)

class ImagePasteByMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "mask": ("MASK",),
                "feather_amount": ("INT", {"default": 10, "min": 0, "max": 100}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_by_mask"
    CATEGORY = "hhy/image"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def paste_by_mask(self, original_image, cropped_image, mask, feather_amount, opacity):
        # 处理可能的列表输入
        if isinstance(feather_amount, list):
            feather_amount = feather_amount[0]
        if isinstance(opacity, list):
            opacity = opacity[0]

        # 简化输入处理，减少内存分配
        def process_input_tensor(input_tensor):
            if isinstance(input_tensor, list):
                # 合并所有tensor到一个批次中
                all_tensors = []
                for tensor in input_tensor:
                    if len(tensor.shape) == 3:
                        tensor = tensor.unsqueeze(0)
                    all_tensors.append(tensor)
                return torch.cat(all_tensors, dim=0) if all_tensors else None
            else:
                if len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                return input_tensor

        original_images = process_input_tensor(original_image)
        cropped_images = process_input_tensor(cropped_image)
        
        # 处理mask输入
        if isinstance(mask, list):
            mask_tensors = []
            for m in mask:
                if len(m.shape) == 2:
                    m = m.unsqueeze(0)
                mask_tensors.append(m)
            masks = torch.cat(mask_tensors, dim=0) if mask_tensors else None
        else:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            masks = mask

        if original_images is None or cropped_images is None or masks is None:
            return ([],)

        # 调试信息 - 减少输出
        print(f"=== ImagePasteByMask Debug Info ===")
        print(f"Batch sizes - Original: {original_images.shape[0]}, Cropped: {cropped_images.shape[0]}, Masks: {masks.shape[0]}")
        print("=" * 40)

        # 确定处理策略
        num_originals = original_images.shape[0]
        num_cropped = cropped_images.shape[0]
        num_masks = masks.shape[0]
        
        # 优先匹配cropped_image和mask的数量关系
        if num_cropped == num_masks:
            if num_originals == 1:
                pairs = [(0, i, i) for i in range(num_cropped)]
            elif num_originals == num_cropped:
                pairs = [(i, i, i) for i in range(num_cropped)]
            else:
                pairs = [(i % num_originals, i, i) for i in range(num_cropped)]
        elif num_cropped == 1:
            if num_originals == num_masks:
                pairs = [(i, 0, i) for i in range(num_masks)]
            elif num_originals == 1:
                pairs = [(0, 0, i) for i in range(num_masks)]
            else:
                pairs = [(i % num_originals, 0, i) for i in range(num_masks)]
        elif num_masks == 1:
            if num_originals == num_cropped:
                pairs = [(i, i, 0) for i in range(num_cropped)]
            elif num_originals == 1:
                pairs = [(0, i, 0) for i in range(num_cropped)]
            else:
                pairs = [(i % num_originals, i, 0) for i in range(num_cropped)]
        else:
            max_count = max(num_originals, num_cropped, num_masks)
            pairs = [(i % num_originals, i % num_cropped, i % num_masks) for i in range(max_count)]

        print(f"Processing {len(pairs)} paste operations")
        
        result_images = []
        
        # 优化的处理循环，减少tensor<->PIL转换
        for orig_idx, crop_idx, mask_idx in pairs:
            # 直接在tensor上操作，避免转换
            base_img_tensor = original_images[orig_idx].clone()  # [H, W, C]
            crop_img_tensor = cropped_images[crop_idx]  # [H, W, C]
            mask_tensor = masks[mask_idx]  # [H, W]
            
            # 找到mask的边界框 - 直接在tensor上操作
            mask_indices = (mask_tensor > 0).nonzero(as_tuple=True)
            
            if len(mask_indices[0]) == 0:
                # 如果mask为空，返回原图
                result_images.append(base_img_tensor.unsqueeze(0))
                continue
            
            y_coords, x_coords = mask_indices
            y_min, y_max = y_coords.min().item(), y_coords.max().item()
            x_min, x_max = x_coords.min().item(), x_coords.max().item()
            
            expected_height = y_max - y_min + 1
            expected_width = x_max - x_min + 1
            
            # 检查尺寸是否匹配，如果需要则调整
            crop_h, crop_w = crop_img_tensor.shape[:2]
            if (crop_h, crop_w) != (expected_height, expected_width):
                # 使用torch的插值进行调整大小
                crop_img_pil = tensor2pil(crop_img_tensor)
                crop_img_pil = crop_img_pil.resize((expected_width, expected_height), Image.LANCZOS)
                crop_img_tensor = pil2tensor(crop_img_pil).squeeze(0)
            
            # 创建羽化mask - 如果需要
            if feather_amount > 0:
                # 只在需要羽化时进行PIL转换
                crop_img_pil = tensor2pil(crop_img_tensor)
                feather_mask = self._create_feather_mask(crop_img_tensor.shape[:2], feather_amount, opacity)
                base_img_pil = tensor2pil(base_img_tensor)
                base_img_pil.paste(crop_img_pil, (x_min, y_min), feather_mask)
                result_images.append(pil2tensor(base_img_pil))
            else:
                # 直接在tensor上粘贴，无羽化
                base_img_tensor[y_min:y_max+1, x_min:x_max+1, :] = crop_img_tensor * opacity + base_img_tensor[y_min:y_max+1, x_min:x_max+1, :] * (1 - opacity)
                result_images.append(base_img_tensor.unsqueeze(0))

        return (result_images,)
    
    def _create_feather_mask(self, shape, feather_amount, opacity):
        """创建羽化遮罩"""
        h, w = shape
        feather_mask = Image.new('L', (w, h), 0)
        feather_mask_array = np.array(feather_mask)
        
        # 计算羽化渐变
        for y in range(h):
            for x in range(w):
                dist_to_edge = min(x, y, w-1-x, h-1-y)
                if dist_to_edge < feather_amount:
                    alpha = (dist_to_edge / feather_amount) * 255
                    feather_mask_array[y, x] = alpha
                else:
                    feather_mask_array[y, x] = 255

        feather_mask = Image.fromarray(feather_mask_array.astype(np.uint8))
        
        # 应用整体不透明度
        if opacity < 1.0:
            fm = np.array(feather_mask).astype(np.float32)
            fm = np.clip(fm * float(opacity), 0, 255).astype(np.uint8)
            feather_mask = Image.fromarray(fm)
        
        return feather_mask

class ImageCropByRatio:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 1536, "min": 1, "max": 8192}),
                "crop_position": (["center", "top", "bottom"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "mask", "resized_original",)
    FUNCTION = "crop_by_ratio"
    CATEGORY = "hhy/image"

    def crop_by_ratio(self, image, width, height, crop_position):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # 使用torch的插值功能替代PIL操作，减少内存使用
        import torch.nn.functional as F
        
        batch_size = image.shape[0]
        result_images = []
        result_masks = []
        resized_originals = []
        
        target_ratio = width / height
        
        for i in range(batch_size):
            current_image = image[i]  # [H, W, C]
            orig_height, orig_width = current_image.shape[:2]
            img_ratio = orig_width / orig_height
            
            # 计算调整后的尺寸
            if target_ratio > img_ratio:
                new_width = width
                new_height = int(orig_height * (width / orig_width))
            else:
                new_height = height
                new_width = int(orig_width * (height / orig_height))
            
            # 使用torch进行插值调整大小，避免PIL转换
            # 将tensor从[H,W,C]转换为[1,C,H,W]用于插值
            img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)
            resized_tensor = F.interpolate(img_for_resize, size=(new_height, new_width), 
                                         mode='bilinear', align_corners=False)
            resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)  # 转回[H,W,C]
            
            resized_originals.append(resized_tensor.unsqueeze(0))
            
            # 计算裁剪尺寸
            if new_width / new_height > target_ratio:
                crop_width = int(new_height * target_ratio)
                crop_height = new_height
            else:
                crop_width = new_width
                crop_height = int(new_width / target_ratio)
            
            # 计算裁剪坐标
            left = (new_width - crop_width) // 2
            
            if crop_position == "center":
                top = (new_height - crop_height) // 2
            elif crop_position == "top":
                top = 0
            elif crop_position == "bottom":
                top = new_height - crop_height
            
            right = left + crop_width
            bottom = top + crop_height
            
            # 直接在tensor上裁剪
            cropped_tensor = resized_tensor[top:bottom, left:right, :]
            
            # 使用torch调整到目标尺寸
            crop_for_resize = cropped_tensor.permute(2, 0, 1).unsqueeze(0)
            final_tensor = F.interpolate(crop_for_resize, size=(height, width), 
                                       mode='bilinear', align_corners=False)
            final_tensor = final_tensor.squeeze(0).permute(1, 2, 0)
            
            # 创建mask - 直接在tensor上操作
            mask_tensor = torch.zeros((new_height, new_width), dtype=torch.float32, device=current_image.device)
            mask_tensor[top:bottom, left:right] = 1.0
            
            result_images.append(final_tensor.unsqueeze(0))
            result_masks.append(mask_tensor.unsqueeze(0))

        if not result_images:
            return (image, torch.zeros_like(image[:, :, :, 0]), image,)
            
        result_image = torch.cat(result_images, dim=0)
        result_mask = torch.cat(result_masks, dim=0)
        resized_original = torch.cat(resized_originals, dim=0)
        return (result_image, result_mask, resized_original,)

class ImageMatchCropToMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT",)
    RETURN_NAMES = ("original_image", "cropped_image", "matched_mask", "mask_index",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "match_crop_to_mask"
    CATEGORY = "hhy/image"

    def match_crop_to_mask(self, original_image, cropped_image, masks):
        # 当使用 INPUT_IS_LIST = True 时，所有参数都会被包装成列表
        # 处理original_image
        if isinstance(original_image, list):
            original_image = original_image[0] if original_image else None
        if original_image is None:
            raise ValueError("original_image is required")
        
        # 处理cropped_image  
        if isinstance(cropped_image, list):
            cropped_image = cropped_image[0] if cropped_image else None
        if cropped_image is None:
            raise ValueError("cropped_image is required")
            
        # 处理masks - 这里可能是多个mask的列表
        if isinstance(masks, list):
            # 如果masks是列表，我们需要将所有mask合并成一个批次
            all_masks = []
            for mask_item in masks:
                if mask_item is not None:
                    if len(mask_item.shape) == 2:  # 单个mask [H, W]
                        mask_item = mask_item.unsqueeze(0)  # [1, H, W]
                    # 如果是批次mask [B, H, W]，拆分成单个mask
                    for i in range(mask_item.shape[0]):
                        all_masks.append(mask_item[i:i+1])  # 保持 [1, H, W] 格式
            
            if not all_masks:
                raise ValueError("No valid masks provided")
            
            # 将所有mask合并为一个批次
            masks = torch.cat(all_masks, dim=0)  # [total_masks, H, W]
        else:
            # 确保masks是批量格式
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(0)

        # 确保original_image和cropped_image是批量格式
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        if len(cropped_image.shape) == 3:
            cropped_image = cropped_image.unsqueeze(0)

        # 只处理第一张原图和第一张裁剪图
        single_original = original_image[0:1]  # 取第一张原图
        single_cropped = cropped_image[0:1]    # 取第一张裁剪图
        
        # 获取裁剪图片的尺寸 - 直接从tensor获取，避免PIL转换
        crop_height, crop_width = single_cropped[0].shape[:2]
        crop_ratio = crop_width / crop_height
        
        print(f"=== ImageMatchCropToMask Debug Info ===")
        print(f"Total masks to check: {len(masks)}")
        print(f"Final shapes - Original: {single_original.shape}, Cropped: {single_cropped.shape}, Masks: {masks.shape}")
        print(f"Cropped image size: {crop_width} x {crop_height}")
        print(f"Cropped image ratio: {crop_ratio:.4f}")
        
        best_match_idx = -1
        best_ratio_diff = float('inf')
        
        # 遍历所有mask，找到比例最匹配的 - 直接在tensor上操作
        for i in range(len(masks)):
            current_mask = masks[i]  # [H, W]
            
            # 直接在tensor上找到mask的边界框
            mask_indices = (current_mask > 0).nonzero(as_tuple=True)
            
            if len(mask_indices[0]) == 0:
                print(f"Mask {i} is empty, skipping...")
                continue
            
            y_coords, x_coords = mask_indices
            y_min, y_max = y_coords.min().item(), y_coords.max().item()
            x_min, x_max = x_coords.min().item(), x_coords.max().item()
            
            # 计算mask区域的尺寸和比例
            mask_width = x_max - x_min + 1
            mask_height = y_max - y_min + 1
            mask_ratio = mask_width / mask_height
            
            # 计算比例差异
            ratio_diff = abs(crop_ratio - mask_ratio)
            
            print(f"Mask {i}: size {mask_width} x {mask_height}, ratio {mask_ratio:.4f}, diff {ratio_diff:.4f}")
            
            # 更新最佳匹配
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_match_idx = i
        
        if best_match_idx == -1:
            print("No valid mask found! Returning first mask as fallback.")
            best_match_idx = 0
        
        print(f"Best match: Mask {best_match_idx} with ratio difference {best_ratio_diff:.4f}")
        print(f"Final output shapes - Original: {single_original.shape}, Cropped: {single_cropped.shape}, Matched mask: {masks[best_match_idx:best_match_idx+1].shape}")
        print("=" * 40)
        
        # 返回单个结果：一张原图、一张裁剪图、一个匹配的mask
        matched_mask = masks[best_match_idx:best_match_idx+1]
        mask_index = torch.tensor([best_match_idx], dtype=torch.int32)
        
        return (single_original, single_cropped, matched_mask, mask_index)

class ImageCropByShape:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background_type": (["transparent", "black"], {"default": "transparent"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_by_shape"
    CATEGORY = "hhy/image"
    OUTPUT_IS_LIST = (True,)

    def crop_by_shape(self, image, mask, background_type):
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        result_images = []
        
        # 确定处理策略
        num_images = len(image)
        num_masks = len(mask)
        
        if num_images == num_masks:
            # 一对一处理
            pairs = [(i, i) for i in range(num_images)]
        elif num_images == 1:
            # 一张图片对应多个mask
            pairs = [(0, i) for i in range(num_masks)]
        elif num_masks == 1:
            # 多张图片对应一个mask
            pairs = [(i, 0) for i in range(num_images)]
        else:
            # 数量不匹配且都大于1，使用循环匹配
            max_count = max(num_images, num_masks)
            pairs = [(i % num_images, i % num_masks) for i in range(max_count)]
        
        # 逐一处理每对图片和mask - 直接在tensor上操作
        for img_idx, mask_idx in pairs:
            current_image = image[img_idx]  # [H, W, C]
            current_mask = mask[mask_idx]   # [H, W]
            
            # 检查mask是否为空
            mask_indices = (current_mask > 0).nonzero(as_tuple=True)
            if len(mask_indices[0]) == 0:
                # 如果mask为空，返回原图
                result_images.append(current_image.unsqueeze(0))
                continue

            # 直接在tensor上操作，避免PIL转换
            H, W, C = current_image.shape
            
            if background_type == "transparent":
                # 创建RGBA图像，添加alpha通道
                if C == 3:  # RGB -> RGBA
                    alpha_channel = torch.zeros((H, W, 1), dtype=current_image.dtype, device=current_image.device)
                    result_img = torch.cat([current_image, alpha_channel], dim=2)  # [H, W, 4]
                else:
                    result_img = current_image.clone()
                
                # 根据mask设置alpha通道
                mask_bool = current_mask > 0
                if C == 3:
                    # 复制原图的RGB通道到有mask的位置
                    result_img[mask_bool, :3] = current_image[mask_bool, :]
                    result_img[mask_bool, 3] = 1.0  # 设置alpha为不透明
                else:
                    result_img[mask_bool] = current_image[mask_bool]
                    
            else:
                # 黑色背景
                result_img = torch.zeros_like(current_image)
                mask_bool = current_mask > 0
                result_img[mask_bool] = current_image[mask_bool]
            
            result_images.append(result_img.unsqueeze(0))

        return (result_images,)

class ImagePasteByShape:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "origin_images": ("IMAGE",),
                "crop_images": ("IMAGE",),
                "masks": ("MASK",),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_by_shape"
    CATEGORY = "hhy/image"
    OUTPUT_IS_LIST = (True,)

    def paste_by_shape(self, origin_images, crop_images, masks, feather_amount):
        # 确保输入是批量格式
        if len(origin_images.shape) == 3:
            origin_images = origin_images.unsqueeze(0)
        if len(crop_images.shape) == 3:
            crop_images = crop_images.unsqueeze(0)
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        result_images = []
        
        # 确定处理策略
        num_origins = len(origin_images)
        num_crops = len(crop_images)
        num_masks = len(masks)
        
        # 优先匹配crop_images和masks的数量关系
        if num_crops == num_masks:
            # 一对一处理
            if num_origins == 1:
                # 一张原图对应多个crop+mask对
                pairs = [(0, i, i) for i in range(num_crops)]
            elif num_origins == num_crops:
                # 一对一对一处理
                pairs = [(i, i, i) for i in range(num_crops)]
            else:
                # 循环匹配原图
                pairs = [(i % num_origins, i, i) for i in range(num_crops)]
        elif num_crops == 1:
            # 一个crop对应多个mask，需要多个原图或重复使用原图
            if num_origins == num_masks:
                pairs = [(i, 0, i) for i in range(num_masks)]
            elif num_origins == 1:
                pairs = [(0, 0, i) for i in range(num_masks)]
            else:
                pairs = [(i % num_origins, 0, i) for i in range(num_masks)]
        elif num_masks == 1:
            # 多个crop对应一个mask，需要多个原图或重复使用原图
            if num_origins == num_crops:
                pairs = [(i, i, 0) for i in range(num_crops)]
            elif num_origins == 1:
                pairs = [(0, i, 0) for i in range(num_crops)]
            else:
                pairs = [(i % num_origins, i, 0) for i in range(num_crops)]
        else:
            # 数量都不匹配，使用最大数量进行循环匹配
            max_count = max(num_origins, num_crops, num_masks)
            pairs = [(i % num_origins, i % num_crops, i % num_masks) for i in range(max_count)]
        
        # 处理每一对 - 直接在tensor上操作
        for orig_idx, crop_idx, mask_idx in pairs:
            base_img_tensor = origin_images[orig_idx].clone()  # [H, W, C]
            crop_img_tensor = crop_images[crop_idx]  # [H, W, C]
            mask_tensor = masks[mask_idx]  # [H, W]
            
            # 检查mask是否为空
            mask_indices = (mask_tensor > 0).nonzero(as_tuple=True)
            if len(mask_indices[0]) == 0:
                # 如果mask为空，返回原图
                result_images.append(base_img_tensor.unsqueeze(0))
                continue

            # 创建mask布尔索引
            mask_bool = mask_tensor > 0
            
            if feather_amount > 0:
                # 只在需要羽化时才使用复杂的处理
                # 简化的羽化效果 - 使用高斯模糊近似
                import torch.nn.functional as F
                
                # 将mask转换为浮点数并添加batch和channel维度用于模糊
                mask_for_blur = mask_tensor.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # 使用平均池化近似羽化效果
                kernel_size = min(feather_amount * 2 + 1, 15)  # 限制kernel大小
                if kernel_size > 1:
                    feathered_mask = F.avg_pool2d(mask_for_blur, kernel_size, stride=1, 
                                                padding=kernel_size//2)
                    feathered_mask = feathered_mask.squeeze(0).squeeze(0)  # [H, W]
                else:
                    feathered_mask = mask_tensor.float()
                
                # 应用羽化的混合
                alpha = feathered_mask.unsqueeze(2)  # [H, W, 1]
                result_tensor = base_img_tensor * (1 - alpha) + crop_img_tensor * alpha
            else:
                # 无羽化，直接复制
                result_tensor = base_img_tensor.clone()
                result_tensor[mask_bool] = crop_img_tensor[mask_bool]
            
            result_images.append(result_tensor.unsqueeze(0))

        return (result_images,)

class PasteImageIntoMaskRect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "origin": ("IMAGE",),
                "anchor": (["top_left", "top_center", "center", "top_right"], {"default": "top_left"}),
                "size_mode": (["unresized", "resize_to_mask"], {"default": "unresized"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "paste_image"
    CATEGORY = "hhy/image"

    def _find_white_rectangle_bbox(self, mask_tensor: torch.Tensor):
        """Find the bounding box of the white region in a mask tensor.
        Returns (x1, y1, x2, y2) inclusive coordinates or None if not found.
        """
        mask_indices = (mask_tensor > 0.5).nonzero(as_tuple=True)
        
        if len(mask_indices[0]) == 0:
            return None
            
        y_coords, x_coords = mask_indices
        y1, y2 = y_coords.min().item(), y_coords.max().item()
        x1, x2 = x_coords.min().item(), x_coords.max().item()
        
        return x1, y1, x2, y2

    def _calculate_paste_position(self, bbox, img_shape, anchor):
        """Calculate the paste position based on anchor point.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box of mask rectangle
            img_shape: (height, width) of image to paste
            anchor: anchor point string
            
        Returns:
            (x, y) position to paste image
        """
        x1, y1, x2, y2 = bbox
        img_height, img_width = img_shape[:2]
        rect_width = x2 - x1 + 1
        rect_height = y2 - y1 + 1
        
        if anchor == "top_left":
            return (x1, y1)
        elif anchor == "top_center":
            return (x1 + (rect_width - img_width) // 2, y1)
        elif anchor == "center":
            return (x1 + (rect_width - img_width) // 2, y1 + (rect_height - img_height) // 2)
        elif anchor == "top_right":
            return (x1 + rect_width - img_width, y1)
        else:
            return (x1, y1)  # default to top_left

    def paste_image(self, image: torch.Tensor, mask: torch.Tensor, origin: torch.Tensor, anchor: str, size_mode: str):
        # 确保所有输入都是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(origin.shape) == 3:
            origin = origin.unsqueeze(0)

        # 确定批次大小
        num_images = image.shape[0]
        num_masks = mask.shape[0] 
        num_origins = origin.shape[0]
        batch_size = max(num_images, num_masks, num_origins)

        result_images = []

        for i in range(batch_size):
            # 选择索引
            img_idx = i if i < num_images else num_images - 1
            msk_idx = i if i < num_masks else num_masks - 1
            org_idx = i if i < num_origins else num_origins - 1

            current_image = image[img_idx]  # [H, W, C]
            current_mask = mask[msk_idx]    # [H, W]
            current_origin = origin[org_idx]  # [H, W, C]

            # 找到白色矩形边界框 - 直接在tensor上操作
            bbox = self._find_white_rectangle_bbox(current_mask)

            # 从原图开始作为基础
            result = current_origin.clone()

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect_width = x2 - x1 + 1
                rect_height = y2 - y1 + 1

                if size_mode == "resize_to_mask":
                    # 使用torch插值调整图像大小以完全适合mask矩形
                    import torch.nn.functional as F
                    
                    # 将图像从[H,W,C]转换为[1,C,H,W]用于插值
                    img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)
                    resized_tensor = F.interpolate(img_for_resize, size=(rect_height, rect_width), 
                                                 mode='bilinear', align_corners=False)
                    resized_image = resized_tensor.squeeze(0).permute(1, 2, 0)  # 转回[H,W,C]
                    
                    # 直接粘贴到bbox位置
                    result[y1:y2+1, x1:x2+1, :] = resized_image
                else:
                    # 未调整大小：根据锚点计算粘贴位置
                    paste_x, paste_y = self._calculate_paste_position(bbox, current_image.shape, anchor)
                    
                    # 计算实际可粘贴的区域，避免超出边界
                    img_h, img_w = current_image.shape[:2]
                    origin_h, origin_w = current_origin.shape[:2]
                    
                    # 计算粘贴区域的边界
                    paste_x_end = min(paste_x + img_w, origin_w)
                    paste_y_end = min(paste_y + img_h, origin_h)
                    paste_x = max(0, paste_x)
                    paste_y = max(0, paste_y)
                    
                    # 计算要粘贴的图像区域
                    crop_w = paste_x_end - paste_x
                    crop_h = paste_y_end - paste_y
                    
                    if crop_w > 0 and crop_h > 0:
                        # 裁剪图像以适合可用空间
                        img_crop = current_image[:crop_h, :crop_w, :]
                        result[paste_y:paste_y_end, paste_x:paste_x_end, :] = img_crop

            result_images.append(result.unsqueeze(0))

        # 连接结果
        final_result = torch.cat(result_images, dim=0)
        return (final_result,)

class ImageConcatMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "left_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.1}),
                "concat_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat_mask"
    CATEGORY = "hhy/image"

    def image_concat_mask(self, image1, left_ratio=0.5, concat_direction="horizontal", image2=None, mask=None):
        import torch.nn.functional as F
        
        # 确保输入是批量格式
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if image2 is not None and len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)
        if mask is not None and len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        processed_images = []
        masks = []
        
        batch_size = image1.shape[0]
        
        for idx in range(batch_size):
            current_image1 = image1[idx]  # [H, W, C]
            H1, W1, C = current_image1.shape
            
            if concat_direction == "horizontal":
                if image2 is not None and idx < image2.shape[0]:
                    current_image2 = image2[idx]  # [H, W, C]
                    H2, W2, _ = current_image2.shape
                    
                    if left_ratio == 0.0:
                        # 直接连接，使用较大的高度
                        final_height = max(H1, H2)
                        
                        # 使用torch插值调整高度
                        img1_resized = F.interpolate(
                            current_image1.permute(2, 0, 1).unsqueeze(0), 
                            size=(final_height, int(W1 * final_height / H1)), 
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        img2_resized = F.interpolate(
                            current_image2.permute(2, 0, 1).unsqueeze(0),
                            size=(final_height, int(W2 * final_height / H2)),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        # 水平连接
                        combined_image = torch.cat([img1_resized, img2_resized], dim=1)
                        target_left_width = img1_resized.shape[1]
                        total_width = combined_image.shape[1]
                    else:
                        # 按比例调整
                        right_ratio = 1.0 - left_ratio
                        
                        # 以image2为基准计算目标尺寸
                        target_right_width = W2
                        target_right_height = H2
                        total_width = int(target_right_width / right_ratio)
                        target_left_width = int(total_width * left_ratio)
                        final_height = target_right_height
                        
                        # 调整image2到目标尺寸
                        img2_resized = F.interpolate(
                            current_image2.permute(2, 0, 1).unsqueeze(0),
                            size=(final_height, target_right_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        # 调整image1保持纵横比
                        width_scale = target_left_width / W1
                        height_scale = final_height / H1
                        scale_factor = min(width_scale, height_scale)
                        
                        new_width = int(W1 * scale_factor)
                        new_height = int(H1 * scale_factor)
                        
                        img1_resized = F.interpolate(
                            current_image1.permute(2, 0, 1).unsqueeze(0),
                            size=(new_height, new_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        # 创建带白色填充的左侧图像
                        img1_final = torch.ones((final_height, target_left_width, C), 
                                              dtype=current_image1.dtype, device=current_image1.device)
                        
                        # 计算居中位置
                        paste_x = (target_left_width - new_width) // 2
                        paste_y = (final_height - new_height) // 2
                        
                        # 粘贴调整后的图像
                        img1_final[paste_y:paste_y+new_height, paste_x:paste_x+new_width, :] = img1_resized
                        
                        # 水平连接
                        combined_image = torch.cat([img1_final, img2_resized], dim=1)
                else:
                    # 没有第二张图像
                    if left_ratio == 0.0:
                        combined_image = current_image1
                        target_left_width = W1
                        total_width = W1
                        final_height = H1
                    else:
                        total_width = int(W1 / left_ratio)
                        target_left_width = W1
                        target_right_width = total_width - target_left_width
                        final_height = H1
                        
                        # 创建白色右侧区域
                        white_right = torch.ones((final_height, target_right_width, C),
                                               dtype=current_image1.dtype, device=current_image1.device)
                        combined_image = torch.cat([current_image1, white_right], dim=1)
                
                # 创建mask (0为左侧图像区域，1为右侧图像区域)
                mask_tensor = torch.zeros((final_height, total_width), dtype=torch.float32, device=current_image1.device)
                if total_width > target_left_width:
                    mask_tensor[:, target_left_width:] = 1.0
                
                # 如果提供了输入mask，从右侧减去它
                if mask is not None and idx < mask.shape[0] and total_width > target_left_width:
                    input_mask_tensor = mask[idx]  # [H, W]
                    # 调整输入mask到右侧区域尺寸
                    target_right_width = total_width - target_left_width
                    resized_input_mask = F.interpolate(
                        input_mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(final_height, target_right_width),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # 从右侧区域减去输入mask
                    mask_tensor[:, target_left_width:] *= (1.0 - resized_input_mask)
                    
            else:  # vertical concatenation
                # 类似的逻辑，但是垂直方向
                if image2 is not None and idx < image2.shape[0]:
                    current_image2 = image2[idx]
                    H2, W2, _ = current_image2.shape
                    
                    if left_ratio == 0.0:
                        # 直接连接，使用较大的宽度
                        final_width = max(W1, W2)
                        
                        img1_resized = F.interpolate(
                            current_image1.permute(2, 0, 1).unsqueeze(0),
                            size=(int(H1 * final_width / W1), final_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        img2_resized = F.interpolate(
                            current_image2.permute(2, 0, 1).unsqueeze(0),
                            size=(int(H2 * final_width / W2), final_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        # 垂直连接
                        combined_image = torch.cat([img1_resized, img2_resized], dim=0)
                        target_top_height = img1_resized.shape[0]
                        total_height = combined_image.shape[0]
                    else:
                        # 按比例调整
                        bottom_ratio = 1.0 - left_ratio
                        
                        target_bottom_width = W2
                        target_bottom_height = H2
                        total_height = int(target_bottom_height / bottom_ratio)
                        target_top_height = int(total_height * left_ratio)
                        final_width = target_bottom_width
                        
                        # 类似水平方向的处理，但是垂直方向
                        img2_resized = F.interpolate(
                            current_image2.permute(2, 0, 1).unsqueeze(0),
                            size=(target_bottom_height, final_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        width_scale = final_width / W1
                        height_scale = target_top_height / H1
                        scale_factor = min(width_scale, height_scale)
                        
                        new_width = int(W1 * scale_factor)
                        new_height = int(H1 * scale_factor)
                        
                        img1_resized = F.interpolate(
                            current_image1.permute(2, 0, 1).unsqueeze(0),
                            size=(new_height, new_width),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                        
                        img1_final = torch.ones((target_top_height, final_width, C),
                                              dtype=current_image1.dtype, device=current_image1.device)
                        
                        paste_x = (final_width - new_width) // 2
                        paste_y = (target_top_height - new_height) // 2
                        
                        img1_final[paste_y:paste_y+new_height, paste_x:paste_x+new_width, :] = img1_resized
                        
                        combined_image = torch.cat([img1_final, img2_resized], dim=0)
                else:
                    # 没有第二张图像的垂直情况
                    if left_ratio == 0.0:
                        combined_image = current_image1
                        target_top_height = H1
                        total_height = H1
                        final_width = W1
                    else:
                        total_height = int(H1 / left_ratio)
                        target_top_height = H1
                        target_bottom_height = total_height - target_top_height
                        final_width = W1
                        
                        white_bottom = torch.ones((target_bottom_height, final_width, C),
                                                dtype=current_image1.dtype, device=current_image1.device)
                        combined_image = torch.cat([current_image1, white_bottom], dim=0)
                
                # 创建mask (0为顶部图像区域，1为底部图像区域)
                mask_tensor = torch.zeros((total_height, final_width), dtype=torch.float32, device=current_image1.device)
                if total_height > target_top_height:
                    mask_tensor[target_top_height:, :] = 1.0
                
                # 如果提供了输入mask，从底部减去它
                if mask is not None and idx < mask.shape[0] and total_height > target_top_height:
                    input_mask_tensor = mask[idx]
                    target_bottom_height = total_height - target_top_height
                    resized_input_mask = F.interpolate(
                        input_mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(target_bottom_height, final_width),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    mask_tensor[target_top_height:, :] *= (1.0 - resized_input_mask)
            
            processed_images.append(combined_image.unsqueeze(0))
            masks.append(mask_tensor.unsqueeze(0))
        
        final_images = torch.cat(processed_images, dim=0)
        final_masks = torch.cat(masks, dim=0)
        
        return (final_images, final_masks)

class ImageResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "target_size": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                "multiple_of": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_resize"
    CATEGORY = "hhy/image"

    def image_resize(self, image, resampling="lanczos", target_size=1024, multiple_of=64, reference_image=None):
        import torch.nn.functional as F
        
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        resized_images = []
        
        # 如果提供了参考图像，使用其尺寸
        if reference_image is not None:
            if len(reference_image.shape) == 3:
                reference_image = reference_image.unsqueeze(0)
                
            ref_H, ref_W = reference_image.shape[1:3]
            
            for i in range(image.shape[0]):
                current_image = image[i]  # [H, W, C]
                
                # 使用torch插值调整大小
                img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                
                # 选择插值模式
                mode = 'bilinear' if resampling in ['bilinear', 'lanczos'] else 'nearest'
                
                resized_tensor = F.interpolate(img_for_resize, size=(ref_H, ref_W), 
                                             mode=mode, align_corners=False if mode != 'nearest' else None)
                resized_image = resized_tensor.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                
                resized_images.append(resized_image.unsqueeze(0))
        else:
            # 原始调整大小逻辑，使用torch操作
            for i in range(image.shape[0]):
                current_image = image[i]  # [H, W, C]
                current_height, current_width = current_image.shape[:2]
                
                # 计算缩放比例
                ratio = min(target_size / current_width, target_size / current_height)
                new_width = round(current_width * ratio)
                new_height = round(current_height * ratio)

                # 调整为multiple_of的倍数
                new_width = new_width - (new_width % multiple_of)
                new_height = new_height - (new_height % multiple_of)

                # 确保最小尺寸
                new_width = max(multiple_of, new_width)
                new_height = max(multiple_of, new_height)

                # 使用torch插值
                img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                
                mode = 'bilinear' if resampling in ['bilinear', 'lanczos'] else 'nearest'
                
                resized_tensor = F.interpolate(img_for_resize, size=(new_height, new_width), 
                                             mode=mode, align_corners=False if mode != 'nearest' else None)
                resized_image = resized_tensor.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                
                resized_images.append(resized_image.unsqueeze(0))
        
        return (torch.cat(resized_images, dim=0),)

class ImageResizeProportional:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "targetsize": ("INT", {
                    "default": 1024,
                    "min": 8,
                    "max": 4096,
                    "display": "number"
                }),
                "step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "display": "number"
                }),
                "resize_mode": (["Lanczos", "Nearest", "Bilinear", "Bicubic"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "hhy/image"

    def resize_image(self, image, targetsize, step, resize_mode):
        import torch.nn.functional as F
        
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        batch_size = image.shape[0]
        result = []
        
        # 选择插值模式
        mode_map = {
            "Lanczos": "bilinear",  # torch没有lanczos，用bilinear近似
            "Nearest": "nearest",
            "Bilinear": "bilinear", 
            "Bicubic": "bicubic"
        }
        mode = mode_map.get(resize_mode, "bilinear")
        
        for i in range(batch_size):
            current_image = image[i]  # [H, W, C]
            original_height, original_width = current_image.shape[:2]
            aspect_ratio = original_width / original_height
            
            # 计算目标像素
            target_pixels = targetsize ** 2
            
            # 计算理想尺寸
            scale = (target_pixels / (original_width * original_height)) ** 0.5
            ideal_width = original_width * scale
            ideal_height = original_height * scale
            
            # 生成候选尺寸
            width_candidates = []
            width_candidates.append(round(ideal_width / step) * step)
            width_candidates.append((int(ideal_width) // step) * step)
            width_candidates.append(((int(ideal_width) + step - 1) // step) * step)
            
            # 评估每个候选尺寸
            best_size = None
            best_diff = float('inf')
            
            for w in set(width_candidates):
                w = max(step, w)
                h = w / aspect_ratio
                h_candidates = [
                    round(h / step) * step,
                    (int(h) // step) * step,
                    ((int(h) + step - 1) // step) * step
                ]
                
                for h_candidate in set(h_candidates):
                    h_candidate = max(step, h_candidate)
                    current_pixels = w * h_candidate
                    diff = abs(current_pixels - target_pixels)
                    
                    if diff < best_diff or (diff == best_diff and current_pixels < best_size[0]*best_size[1]):
                        best_diff = diff
                        best_size = (int(w), int(h_candidate))
            
            # 使用torch调整图像大小
            new_width, new_height = best_size
            img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            
            resized_tensor = F.interpolate(img_for_resize, size=(new_height, new_width), 
                                         mode=mode, align_corners=False if mode != 'nearest' else None)
            resized_image = resized_tensor.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
            result.append(resized_image.unsqueeze(0))
        
        return (torch.cat(result, dim=0),)

class ImageResizeToReferencePixels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "resize_mode": (["Lanczos", "Nearest", "Bilinear", "Bicubic"],),
                "step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "display": "number"
                }),
                "ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_to_reference_pixels"
    CATEGORY = "hhy/image"

    def resize_to_reference_pixels(self, image, reference_image, resize_mode, step, ratio):
        import torch.nn.functional as F
        
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(reference_image.shape) == 3:
            reference_image = reference_image.unsqueeze(0)
            
        # 获取参考图像尺寸并计算目标像素
        ref_height, ref_width = reference_image.shape[1:3]
        target_pixels = int(ref_width * ref_height * ratio)
        
        batch_size = image.shape[0]
        result = []
        
        # 选择插值模式
        mode_map = {
            "Lanczos": "bilinear",
            "Nearest": "nearest", 
            "Bilinear": "bilinear",
            "Bicubic": "bicubic"
        }
        mode = mode_map.get(resize_mode, "bilinear")
        
        for i in range(batch_size):
            current_image = image[i]  # [H, W, C]
            original_height, original_width = current_image.shape[:2]
            aspect_ratio = original_width / original_height
            
            # 计算缩放因子以匹配目标像素
            scale = (target_pixels / (original_width * original_height)) ** 0.5
            ideal_width = original_width * scale
            ideal_height = original_height * scale
            
            # 生成候选尺寸
            width_candidates = []
            width_candidates.append(round(ideal_width / step) * step)
            width_candidates.append((int(ideal_width) // step) * step)
            width_candidates.append(((int(ideal_width) + step - 1) // step) * step)
            
            # 评估每个候选尺寸
            best_size = None
            best_diff = float('inf')
            
            for w in set(width_candidates):
                w = max(step, w)
                h = w / aspect_ratio
                h_candidates = [
                    round(h / step) * step,
                    (int(h) // step) * step,
                    ((int(h) + step - 1) // step) * step
                ]
                
                for h_candidate in set(h_candidates):
                    h_candidate = max(step, h_candidate)
                    current_pixels = w * h_candidate
                    diff = abs(current_pixels - target_pixels)
                    
                    if diff < best_diff or (diff == best_diff and current_pixels < best_size[0]*best_size[1]):
                        best_diff = diff
                        best_size = (int(w), int(h_candidate))
            
            # 使用torch调整图像大小
            new_width, new_height = best_size
            img_for_resize = current_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            
            resized_tensor = F.interpolate(img_for_resize, size=(new_height, new_width), 
                                         mode=mode, align_corners=False if mode != 'nearest' else None)
            resized_image = resized_tensor.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
            result.append(resized_image.unsqueeze(0))
        
        return (torch.cat(result, dim=0),)

class LoadImageFromURL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images_from_url"
    CATEGORY = "hhy/image"
    OUTPUT_IS_LIST = (True,)

    def load_images_from_url(self, urls):
        """
        Load images from URLs.
        Supports:
        1. JSON format: ["url1", "url2"]
        2. Plain text format with one URL per line
        3. Concatenated URLs (split by http:// or https://)
        """
        # Parse the input
        url_list = []
        urls = urls.strip()
        
        if not urls:
            raise ValueError("No URLs provided")
        
        # Try to parse as JSON first
        try:
            parsed = json.loads(urls)
            if isinstance(parsed, list):
                url_list = [str(url).strip() for url in parsed if url]
            elif isinstance(parsed, str):
                url_list = [parsed.strip()]
            else:
                raise ValueError("Invalid JSON format")
        except json.JSONDecodeError:
            # Check if it contains concatenated URLs (multiple http/https without separators)
            # Use regex to find all URLs starting with http:// or https://
            concatenated_urls = re.findall(r'https?://[^\s]+', urls)
            
            if len(concatenated_urls) > 1:
                # Multiple URLs found concatenated together
                url_list = concatenated_urls
            else:
                # If not concatenated, treat as plain text with one URL per line
                url_list = [line.strip() for line in urls.split('\n') if line.strip()]
        
        if not url_list:
            raise ValueError("No valid URLs found in input")
        
        print(f"Loading {len(url_list)} images from URLs...")
        
        # Download and process images
        result_images = []
        for idx, url in enumerate(url_list):
            try:
                print(f"Downloading image {idx + 1}/{len(url_list)}: {url}")
                
                # Download the image
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Open image from bytes
                img = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary (handle RGBA, L, etc.)
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to tensor
                img_tensor = pil2tensor(img)
                result_images.append(img_tensor)
                
                print(f"Successfully loaded image {idx + 1}: {img.size}")
                
            except Exception as e:
                error_msg = f"Failed to load image {idx + 1}/{len(url_list)} from URL: {url}\nError: {str(e)}"
                print(error_msg)
                raise ValueError(error_msg) from e
        
        print(f"Successfully loaded {len(result_images)} images")
        return (result_images,)

NODE_CLASS_MAPPINGS = {
    "Image Crop By Mask": ImageCropByMask,
    "Image Paste By Mask": ImagePasteByMask,
    "Image Crop By Ratio": ImageCropByRatio,
    "Image Match Crop To Mask": ImageMatchCropToMask,
    "Image Crop By Shape": ImageCropByShape,
    "Image Paste By Shape": ImagePasteByShape,
    "PasteImageIntoMaskRect": PasteImageIntoMaskRect,
    "image concat mask": ImageConcatMask,
    "image resize": ImageResize,
    "ImageResizeProportional": ImageResizeProportional,
    "ImageResizeToReferencePixels": ImageResizeToReferencePixels,
    "LoadImageFromURL": LoadImageFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Crop By Mask": "Image Crop By Mask",
    "Image Paste By Mask": "Image Paste By Mask",
    "Image Crop By Ratio": "Image Crop By Ratio",
    "Image Match Crop To Mask": "Image Match Crop To Mask",
    "Image Crop By Shape": "Image Crop By Shape",
    "Image Paste By Shape": "Image Paste By Shape",
    "PasteImageIntoMaskRect": "Paste Image Into Mask Rect",
    "image concat mask": "Image Concat with Mask",
    "image resize": "Image Resize",
    "ImageResizeProportional": "Proportional Image Resizer",
    "ImageResizeToReferencePixels": "Resize to Reference Pixels",
    "LoadImageFromURL": "Load Image from URL"
} 