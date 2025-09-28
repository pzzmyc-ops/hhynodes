from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Use alpha if available
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]
    # Convert RGB to grayscale
    return TF.rgb_to_grayscale(t.permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

class image_difference_mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 0.09, "min": 0.01, "max": 0.5, "step": 0.01}),
                "min_area": ("INT", {"default": 3000, "min": 100, "max": 10000, "step": 100}),
                "edge_smooth": ("INT", {"default": 51, "min": 3, "max": 51, "step": 2}),
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("diff_image", "diff_mask")
    FUNCTION = "detect_ai_edit_difference"
    CATEGORY = "hhy/image"

    def detect_ai_edit_difference(self, image1, image2, sensitivity=0.08, min_area=1000, 
                                edge_smooth=15, invert_mask=False):
        processed_images = []
        masks = []
        
        # Ensure we process the same number of images
        min_len = min(len(image1), len(image2))
        
        for idx in range(min_len):
            # Convert tensors to PIL images
            pil_image1 = tensor2pil(image1[idx])
            pil_image2 = tensor2pil(image2[idx])
            
            # Resize image2 to match image1 dimensions if they differ
            if pil_image1.size != pil_image2.size:
                pil_image2 = pil_image2.resize(pil_image1.size, Image.Resampling.LANCZOS)
            
            # Convert PIL images to OpenCV format
            cv_image1 = cv2.cvtColor(np.array(pil_image1), cv2.COLOR_RGB2BGR)
            cv_image2 = cv2.cvtColor(np.array(pil_image2), cv2.COLOR_RGB2BGR)
            
            # Detect significant changes for AI-edited images
            binary_mask = self._detect_ai_edits(cv_image1, cv_image2, sensitivity)
            
            # Filter out small noise regions
            binary_mask = self._filter_small_regions(binary_mask, min_area)
            
            # Apply edge smoothing and curve fitting
            binary_mask = self._smooth_edges(binary_mask, edge_smooth)
            
            # Create visualization
            diff_image = self._create_smooth_visualization(cv_image1, cv_image2, binary_mask)
            
            # Convert back to PIL and then to tensor
            pil_final = Image.fromarray(diff_image)
            final_tensor = pil2tensor(pil_final)
            processed_images.append(final_tensor)
            
            # Convert binary mask to tensor
            mask_tensor = torch.from_numpy(binary_mask.astype(np.float32) / 255.0).unsqueeze(0)
            
            # Invert mask if requested
            if invert_mask:
                mask_tensor = 1.0 - mask_tensor
                
            masks.append(mask_tensor)
            
        processed_images = torch.cat(processed_images, dim=0)
        masks = torch.cat(masks, dim=0)
        
        print(f"Processed {len(processed_images)} image pairs for AI edit detection")
        print("Edit mask shape:", masks.shape)
        print("Edit mask value range:", torch.min(masks).item(), torch.max(masks).item())
        
        return (processed_images, masks)
    
    def _detect_ai_edits(self, img1, img2, sensitivity):
        """Detect significant changes in AI-edited images, filtering out minor pixel variations"""
        # Convert to different color spaces for better detection
        # RGB差异
        diff_rgb = cv2.absdiff(img1, img2)
        diff_rgb_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY)
        
        # LAB色彩空间差异 (对颜色变化更敏感)
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        diff_lab = cv2.absdiff(lab1, lab2)
        diff_lab_gray = cv2.cvtColor(diff_lab, cv2.COLOR_LAB2BGR)
        diff_lab_gray = cv2.cvtColor(diff_lab_gray, cv2.COLOR_BGR2GRAY)
        
        # 结合RGB和LAB差异
        combined_diff = cv2.addWeighted(diff_rgb_gray, 0.7, diff_lab_gray, 0.3, 0)
        
        # 使用双边滤波减少噪声但保留边缘
        filtered_diff = cv2.bilateralFilter(combined_diff, 9, 75, 75)
        
        # 简单阈值
        threshold_value = int(sensitivity * 255)
        _, binary_mask = cv2.threshold(filtered_diff, threshold_value, 255, cv2.THRESH_BINARY)
        
        return binary_mask
    
    def _filter_small_regions(self, mask, min_area):
        """Filter out small noise regions using connected component analysis"""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create filtered mask
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # Keep this component
                component_mask = (labels == i).astype(np.uint8) * 255
                filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
        
        return filtered_mask
    
    def _smooth_edges(self, mask, smooth_kernel):
        """Apply edge smoothing and curve fitting for natural-looking edges"""
        if smooth_kernel < 3:
            return mask
            
        # 首先进行形态学操作来连接相近区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 闭运算：填充小孔洞，连接相近区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 开运算：去除小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 查找轮廓进行曲线拟合
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建平滑的蒙版
        smooth_mask = np.zeros_like(mask)
        
        for contour in contours:
            # 轮廓近似和平滑
            epsilon = 0.005 * cv2.arcLength(contour, True)  # 更小的epsilon保持更多细节
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果轮廓点太少，使用原始轮廓
            if len(approx) < 4:
                approx = contour
            
            # 填充平滑的轮廓
            cv2.fillPoly(smooth_mask, [approx], 255)
        
        # 使用高斯模糊进一步平滑边缘
        smooth_mask = cv2.GaussianBlur(smooth_mask, (smooth_kernel, smooth_kernel), 0)
        
        # 重新二值化以保持蒙版特性，但边缘更平滑
        _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
        
        return smooth_mask
    
    def _create_smooth_visualization(self, img1, img2, mask):
        """Create smooth visualization of differences"""
        # Convert to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        # Create smooth overlay
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # 创建彩色差异显示
        diff_rgb = cv2.absdiff(img1, img2)
        diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2RGB)
        
        # 增强差异显示
        diff_enhanced = cv2.convertScaleAbs(diff_rgb, alpha=2.0, beta=0)
        
        # 创建平滑的红色覆盖层
        red_overlay = np.zeros_like(img1_rgb)
        red_overlay[:, :, 0] = mask  # Red channel
        
        # 多层混合创建更好的可视化效果
        mask_3d = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=2)
        
        # 基础图像 + 差异增强 + 红色高亮
        result = (img1_rgb.astype(np.float32) * (1 - mask_3d * 0.3) + 
                 diff_enhanced.astype(np.float32) * mask_3d * 0.4 + 
                 red_overlay.astype(np.float32) * 0.3)
        
        return np.clip(result, 0, 255).astype(np.uint8)

NODE_CLASS_MAPPINGS = {
    "image difference mask": image_difference_mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image difference mask": "Image Difference Detector"
} 