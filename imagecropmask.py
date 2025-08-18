from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

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
    CATEGORY = "hhy"
    OUTPUT_IS_LIST = (True, True)

    def crop_by_mask(self, image, mask):
        # 确保输入是批量格式
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        result_images = []
        result_masks = []
        
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
        
        # 逐一处理每对图片和mask
        for img_idx, mask_idx in pairs:
            img = tensor2pil(image[img_idx])
            msk = tensor2pil(mask[mask_idx])

            non_zero = np.array(msk) > 0
            if not non_zero.any():
                # 如果mask为空，返回原图
                result_images.append(image[img_idx:img_idx+1])
                result_masks.append(mask[mask_idx:mask_idx+1])
                continue

            rows = np.any(non_zero, axis=1)
            cols = np.any(non_zero, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # 裁剪原图
            cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            
            # 创建对应的mask，保持原图大小，只在裁剪区域标记为白色
            original_size = img.size
            position_mask = Image.new('L', original_size, 0)
            position_mask_array = np.array(position_mask)
            position_mask_array[y_min:y_max+1, x_min:x_max+1] = 255
            position_mask = Image.fromarray(position_mask_array)
            
            result_images.append(pil2tensor(cropped_img))
            result_masks.append(pil2tensor(position_mask))

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_by_mask"
    CATEGORY = "hhy"
    INPUT_IS_LIST = True

    def paste_by_mask(self, original_image, cropped_image, mask, feather_amount):
        # 处理可能的列表输入
        if isinstance(original_image, list):
            original_image = original_image[0]  # 使用第一张原图
        if isinstance(feather_amount, list):
            feather_amount = feather_amount[0]  # 使用第一个羽化值
            
        # 确保输入是批量格式
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)

        # 处理cropped_image输入 - 现在应该是列表，但不要合并
        if isinstance(cropped_image, list):
            # 直接使用列表中的每个tensor，不要合并
            cropped_images = []
            for img in cropped_image:
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                cropped_images.append(img)
        else:
            if len(cropped_image.shape) == 3:
                cropped_image = cropped_image.unsqueeze(0)
            cropped_images = [cropped_image[i:i+1] for i in range(len(cropped_image))]

        # 处理mask输入 - 现在应该是列表，但不要合并
        if isinstance(mask, list):
            # 直接使用列表中的每个tensor，不要合并
            mask_list = []
            for m in mask:
                if len(m.shape) == 2:
                    m = m.unsqueeze(0)
                mask_list.append(m)
        else:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            mask_list = [mask[i:i+1] for i in range(len(mask))]

        # 调试信息
        print(f"=== ImagePasteByMask Debug Info (Fixed List Input) ===")
        print(f"Original image type: {type(original_image)}")
        print(f"Cropped image type: {type(cropped_image)}")
        print(f"Mask type: {type(mask)}")
        print(f"Original image count: {len(original_image)}")
        print(f"Cropped images count: {len(cropped_images)}")
        print(f"Masks count: {len(mask_list)}")
        print(f"Original image shape: {original_image.shape}")
        
        # 显示每个crop image的形状
        for i, img in enumerate(cropped_images):
            print(f"Cropped image {i} shape: {img.shape}")
        for i, m in enumerate(mask_list):
            print(f"Mask {i} shape: {m.shape}")
        print("=" * 52)

        # 使用第一张原图作为基础
        base_img = tensor2pil(original_image[0])
        result_img = base_img.copy()

        # 确定要处理的数量（crop image和mask的较小值）
        num_cropped = len(cropped_images)
        num_masks = len(mask_list)
        process_count = min(num_cropped, num_masks)
        
        print(f"Will process {process_count} paste operations on single output image")
        
        # 依次将每个crop image粘贴到对应的mask位置
        for i in range(process_count):
            print(f"\n--- Processing paste operation {i+1}/{process_count} ---")
            
            # 获取当前的crop image和mask
            crop_img = tensor2pil(cropped_images[i][0])
            msk = tensor2pil(mask_list[i][0])

            print(f"  Cropped image size: {crop_img.size}")
            print(f"  Mask size: {msk.size}")

            # 找到mask的边界框
            non_zero = np.array(msk) > 0
            if not non_zero.any():
                print(f"  Mask {i} is empty, skipping...")
                continue

            rows = np.any(non_zero, axis=1)
            cols = np.any(non_zero, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            print(f"  Mask bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            print(f"  Expected crop size: {x_max-x_min+1} x {y_max-y_min+1}")
            print(f"  Actual crop size: {crop_img.size[0]} x {crop_img.size[1]}")

            # 检查尺寸是否匹配
            expected_width = x_max - x_min + 1
            expected_height = y_max - y_min + 1
            if crop_img.size != (expected_width, expected_height):
                print(f"  Warning: Size mismatch! Expected ({expected_width}, {expected_height}), got {crop_img.size}")
                # 可以选择调整大小或跳过
                crop_img = crop_img.resize((expected_width, expected_height), Image.LANCZOS)
                print(f"  Resized crop image to match expected size")

            # 创建羽化遮罩
            feather_mask = Image.new('L', crop_img.size, 0)
            feather_mask_array = np.array(feather_mask)
            
            # 计算羽化渐变
            h, w = feather_mask_array.shape
            for y in range(h):
                for x in range(w):
                    # 计算到边缘的最小距离
                    dist_to_edge = min(x, y, w-1-x, h-1-y)
                    if dist_to_edge < feather_amount:
                        # 创建从边缘向内的渐变
                        alpha = (dist_to_edge / feather_amount) * 255
                        feather_mask_array[y, x] = alpha
                    else:
                        feather_mask_array[y, x] = 255

            feather_mask = Image.fromarray(feather_mask_array.astype(np.uint8))
            
            # 将裁剪的图片粘贴到结果图片上（累积粘贴）
            result_img.paste(crop_img, (x_min, y_min), feather_mask)
            print(f"  Successfully pasted crop image {i+1} at position ({x_min}, {y_min})")

        print(f"\n*** FINAL COMPOSITE IMAGE created with {process_count} paste operations ***")
        
        # 返回最终的合成图片
        result_tensor = pil2tensor(result_img)
        return (result_tensor,)

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
    CATEGORY = "hhy"

    def crop_by_ratio(self, image, width, height, crop_position):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        result_images = []
        result_masks = []
        resized_originals = []
        
        for i in range(len(image)):
            img = tensor2pil(image[i])
            orig_width, orig_height = img.size
            
            # Calculate target aspect ratio
            target_ratio = width / height
            img_ratio = orig_width / orig_height
            
            # Determine how to scale the original image to match target dimensions
            # If target is wider than original, scale original width to match target width
            # If target is taller than original, scale original height to match target height
            if target_ratio > img_ratio:
                # Target is wider than original, scale width to match target width
                scale_height = orig_height * (width / orig_width)
                new_size = (width, int(scale_height))
            else:
                # Target is taller than original, scale height to match target height
                scale_width = orig_width * (height / orig_height)
                new_size = (int(scale_width), height)
            
            # Resize the original image to the new size
            resized_img = img.resize(new_size, Image.LANCZOS)
            resized_width, resized_height = resized_img.size
            
            # Save the resized original
            resized_originals.append(pil2tensor(resized_img))
            
            # Now determine crop dimensions based on target ratio
            if resized_width / resized_height > target_ratio:
                # Resized image is wider than target ratio, crop width
                crop_width = int(resized_height * target_ratio)
                crop_height = resized_height
            else:
                # Resized image is taller than target ratio, crop height
                crop_width = resized_width
                crop_height = int(resized_width / target_ratio)
            
            # Calculate crop coordinates based on crop_position
            left = (resized_width - crop_width) // 2  # 水平居中
            
            if crop_position == "center":
                top = (resized_height - crop_height) // 2  # 垂直居中
            elif crop_position == "top":
                top = 0  # 从顶部开始
            elif crop_position == "bottom":
                top = resized_height - crop_height  # 从底部开始
            
            right = left + crop_width
            bottom = top + crop_height
            
            # Crop the image
            cropped_img = resized_img.crop((left, top, right, bottom))
            
            # Resize to exact target dimensions
            final_img = cropped_img.resize((width, height), Image.LANCZOS)
            
            # Create a mask for the cropped region in the resized original image space
            mask = Image.new('L', (resized_width, resized_height), 0)
            mask_array = np.array(mask)
            mask_array[top:bottom, left:right] = 255
            mask = Image.fromarray(mask_array)
            
            result_images.append(pil2tensor(final_img))
            result_masks.append(pil2tensor(mask))

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
    CATEGORY = "hhy"

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
        
        # 获取裁剪图片的尺寸
        crop_img = tensor2pil(single_cropped[0])
        crop_width, crop_height = crop_img.size
        crop_ratio = crop_width / crop_height
        
        print(f"=== ImageMatchCropToMask Debug Info ===")
        print(f"Input processing - Original list length: {len(original_image) if isinstance(original_image, list) else 'single'}")
        print(f"Input processing - Cropped list length: {len(cropped_image) if isinstance(cropped_image, list) else 'single'}")  
        print(f"Input processing - Total masks to check: {len(masks)}")
        print(f"Final shapes - Original: {single_original.shape}, Cropped: {single_cropped.shape}, Masks: {masks.shape}")
        print(f"Cropped image size: {crop_width} x {crop_height}")
        print(f"Cropped image ratio: {crop_ratio:.4f}")
        
        best_match_idx = -1
        best_ratio_diff = float('inf')
        
        # 遍历所有mask，找到比例最匹配的
        for i in range(len(masks)):
            mask = tensor2pil(masks[i])
            mask_array = np.array(mask)
            
            # 找到mask的边界框
            non_zero = mask_array > 0
            if not non_zero.any():
                print(f"Mask {i} is empty, skipping...")
                continue
                
            rows = np.any(non_zero, axis=1)
            cols = np.any(non_zero, axis=0)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Mask {i} has no valid boundaries, skipping...")
                continue
                
            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]
            
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

NODE_CLASS_MAPPINGS = {
    "Image Crop By Mask": ImageCropByMask,
    "Image Paste By Mask": ImagePasteByMask,
    "Image Crop By Ratio": ImageCropByRatio,
    "Image Match Crop To Mask": ImageMatchCropToMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Crop By Mask": "Image Crop By Mask",
    "Image Paste By Mask": "Image Paste By Mask",
    "Image Crop By Ratio": "Image Crop By Ratio",
    "Image Match Crop To Mask": "Image Match Crop To Mask"
} 