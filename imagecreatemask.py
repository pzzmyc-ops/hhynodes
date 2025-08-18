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

class image_concat_mask:
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
    CATEGORY = "hhy"

    def image_concat_mask(self, image1, left_ratio=0.5, concat_direction="horizontal", image2=None, mask=None):
        processed_images = []
        masks = []
        
        for idx, img1 in enumerate(image1):
            # Convert tensor to PIL
            pil_image1 = tensor2pil(img1)
            
            # Get first image dimensions
            width1, height1 = pil_image1.size
            
            if concat_direction == "horizontal":
                # Horizontal concatenation (left-right)
                if image2 is not None and idx < len(image2):
                    # Use provided second image
                    pil_image2 = tensor2pil(image2[idx])
                    width2, height2 = pil_image2.size
                    
                    # Special case: when left_ratio is 0, directly concatenate without resizing
                    if left_ratio == 0.0:
                        # No adjustment, directly concatenate
                        # Use the taller image's height as the base height
                        final_height = max(height1, height2)
                        
                        # Resize both images to match the final height while maintaining aspect ratio
                        # Resize image1
                        scale1 = final_height / height1
                        new_width1 = int(width1 * scale1)
                        pil_image1_final = pil_image1.resize((new_width1, final_height), Image.Resampling.LANCZOS)
                        
                        # Resize image2
                        scale2 = final_height / height2
                        new_width2 = int(width2 * scale2)
                        pil_image2 = pil_image2.resize((new_width2, final_height), Image.Resampling.LANCZOS)
                        
                        total_width = new_width1 + new_width2
                        target_left_width = new_width1
                        target_right_width = new_width2
                    else:
                        # Calculate target dimensions based on ratio
                        # If left_ratio is 1/3, then right_ratio is 2/3
                        right_ratio = 1.0 - left_ratio
                        
                        # Calculate the total width needed
                        # left_width / total_width = left_ratio
                        # right_width / total_width = right_ratio
                        # We want to maintain the aspect ratio of the right image
                        target_right_width = width2
                        target_right_height = height2
                        
                        # Calculate total width based on right image and ratios
                        total_width = int(target_right_width / right_ratio)
                        target_left_width = int(total_width * left_ratio)
                        
                        # Use the right image's height as the base height
                        final_height = target_right_height
                        
                        # Resize right image to fit its allocated space
                        pil_image2 = pil_image2.resize((target_right_width, final_height), Image.Resampling.LANCZOS)
                        
                        # Calculate how to fit left image in its allocated space while maintaining aspect ratio
                        # Calculate scale factors for both width and height
                        width_scale = target_left_width / width1
                        height_scale = final_height / height1
                        
                        # Use the smaller scale factor to maintain aspect ratio
                        scale_factor = min(width_scale, height_scale)
                        
                        # Calculate new dimensions that maintain aspect ratio
                        new_left_width = int(width1 * scale_factor)
                        new_left_height = int(height1 * scale_factor)
                        
                        # Resize left image maintaining aspect ratio
                        pil_image1_resized = pil_image1.resize((new_left_width, new_left_height), Image.Resampling.LANCZOS)
                        
                        # Create left image with white padding to fill the allocated space
                        pil_image1_final = Image.new('RGB', (target_left_width, final_height), 'white')
                        
                        # Calculate centering position
                        paste_x = (target_left_width - new_left_width) // 2
                        paste_y = (final_height - new_left_height) // 2
                        
                        # Paste the resized image centered in the allocated space
                        pil_image1_final.paste(pil_image1_resized, (paste_x, paste_y))
                        
                else:
                    # When no second image is provided
                    if left_ratio == 0.0:
                        # When ratio is 0, just use the original image dimensions - no concatenation needed
                        target_left_width = width1
                        total_width = width1
                        target_right_width = 0
                        final_height = height1
                        
                        pil_image1_final = pil_image1
                        pil_image2 = None  # No second image needed
                    else:
                        # Create white image with same dimensions as image1
                        target_left_width = int(width1 / left_ratio * left_ratio)
                        total_width = int(width1 / left_ratio)
                        target_right_width = total_width - target_left_width
                        final_height = height1
                        
                        pil_image1_final = pil_image1
                        pil_image2 = Image.new('RGB', (target_right_width, final_height), 'white')
                
                # Create new image to hold both images side by side
                combined_image = Image.new('RGB', (total_width, final_height))
                
                # Paste both images
                combined_image.paste(pil_image1_final, (0, 0))
                if pil_image2 is not None and target_right_width > 0:
                    combined_image.paste(pil_image2, (target_left_width, 0))
                
                # Convert combined image to tensor
                combined_tensor = pil2tensor(combined_image)
                processed_images.append(combined_tensor)
                
                # Create mask (0 for left image area, 1 for right image area)
                final_mask = torch.zeros((1, final_height, total_width))
                if target_right_width > 0:
                    final_mask[:, :, target_left_width:] = 1.0  # Set right half to 1
                
                # If mask is provided, subtract it from the right side
                if mask is not None and idx < len(mask) and target_right_width > 0:
                    input_mask = mask[idx]
                    # Resize input mask to match right image dimensions
                    pil_input_mask = tensor2pil(input_mask)
                    pil_input_mask = pil_input_mask.resize((target_right_width, final_height), Image.Resampling.LANCZOS)
                    resized_input_mask = pil2tensor(pil_input_mask)
                    
                    # Subtract input mask from the right side
                    final_mask[:, :, target_left_width:] *= (1.0 - resized_input_mask)
                
                masks.append(final_mask)
                
            else:  # vertical concatenation (top-bottom)
                if image2 is not None and idx < len(image2):
                    # Use provided second image
                    pil_image2 = tensor2pil(image2[idx])
                    width2, height2 = pil_image2.size
                    
                    # Special case: when left_ratio is 0, directly concatenate without resizing
                    if left_ratio == 0.0:
                        # No adjustment, directly concatenate
                        # Use the wider image's width as the base width
                        final_width = max(width1, width2)
                        
                        # Resize both images to match the final width while maintaining aspect ratio
                        # Resize image1
                        scale1 = final_width / width1
                        new_height1 = int(height1 * scale1)
                        pil_image1_final = pil_image1.resize((final_width, new_height1), Image.Resampling.LANCZOS)
                        
                        # Resize image2
                        scale2 = final_width / width2
                        new_height2 = int(height2 * scale2)
                        pil_image2 = pil_image2.resize((final_width, new_height2), Image.Resampling.LANCZOS)
                        
                        total_height = new_height1 + new_height2
                        target_top_height = new_height1
                        target_bottom_height = new_height2
                    else:
                        # Calculate target dimensions based on ratio
                        # left_ratio now means top_ratio
                        bottom_ratio = 1.0 - left_ratio
                        
                        # Calculate the total height needed
                        # We want to maintain the aspect ratio of the bottom image
                        target_bottom_width = width2
                        target_bottom_height = height2
                        
                        # Calculate total height based on bottom image and ratios
                        total_height = int(target_bottom_height / bottom_ratio)
                        target_top_height = int(total_height * left_ratio)
                        
                        # Use the bottom image's width as the base width
                        final_width = target_bottom_width
                        
                        # Resize bottom image to fit its allocated space
                        pil_image2 = pil_image2.resize((final_width, target_bottom_height), Image.Resampling.LANCZOS)
                        
                        # Calculate how to fit top image in its allocated space while maintaining aspect ratio
                        # Calculate scale factors for both width and height
                        width_scale = final_width / width1
                        height_scale = target_top_height / height1
                        
                        # Use the smaller scale factor to maintain aspect ratio
                        scale_factor = min(width_scale, height_scale)
                        
                        # Calculate new dimensions that maintain aspect ratio
                        new_top_width = int(width1 * scale_factor)
                        new_top_height = int(height1 * scale_factor)
                        
                        # Resize top image maintaining aspect ratio
                        pil_image1_resized = pil_image1.resize((new_top_width, new_top_height), Image.Resampling.LANCZOS)
                        
                        # Create top image with white padding to fill the allocated space
                        pil_image1_final = Image.new('RGB', (final_width, target_top_height), 'white')
                        
                        # Calculate centering position
                        paste_x = (final_width - new_top_width) // 2
                        paste_y = (target_top_height - new_top_height) // 2
                        
                        # Paste the resized image centered in the allocated space
                        pil_image1_final.paste(pil_image1_resized, (paste_x, paste_y))
                        
                else:
                    # When no second image is provided
                    if left_ratio == 0.0:
                        # When ratio is 0, just use the original image dimensions - no concatenation needed
                        target_top_height = height1
                        total_height = height1
                        target_bottom_height = 0
                        final_width = width1
                        
                        pil_image1_final = pil_image1
                        pil_image2 = None  # No second image needed
                    else:
                        # Create white image with same dimensions as image1
                        target_top_height = int(height1 / left_ratio * left_ratio)
                        total_height = int(height1 / left_ratio)
                        target_bottom_height = total_height - target_top_height
                        final_width = width1
                        
                        pil_image1_final = pil_image1
                        pil_image2 = Image.new('RGB', (final_width, target_bottom_height), 'white')
                
                # Create new image to hold both images top and bottom
                combined_image = Image.new('RGB', (final_width, total_height))
                
                # Paste both images
                combined_image.paste(pil_image1_final, (0, 0))
                if pil_image2 is not None and target_bottom_height > 0:
                    combined_image.paste(pil_image2, (0, target_top_height))
                
                # Convert combined image to tensor
                combined_tensor = pil2tensor(combined_image)
                processed_images.append(combined_tensor)
                
                # Create mask (0 for top image area, 1 for bottom image area)
                final_mask = torch.zeros((1, total_height, final_width))
                if target_bottom_height > 0:
                    final_mask[:, target_top_height:, :] = 1.0  # Set bottom half to 1
                
                # If mask is provided, subtract it from the bottom side
                if mask is not None and idx < len(mask) and target_bottom_height > 0:
                    input_mask = mask[idx]
                    # Resize input mask to match bottom image dimensions
                    pil_input_mask = tensor2pil(input_mask)
                    pil_input_mask = pil_input_mask.resize((final_width, target_bottom_height), Image.Resampling.LANCZOS)
                    resized_input_mask = pil2tensor(pil_input_mask)
                    
                    # Subtract input mask from the bottom side
                    final_mask[:, target_top_height:, :] *= (1.0 - resized_input_mask)
                
                masks.append(final_mask)
            
        processed_images = torch.cat(processed_images, dim=0)
        masks = torch.cat(masks, dim=0)
        
        print("Mask shape:", masks.shape)
        print("Mask value range:", torch.min(masks).item(), torch.max(masks).item())
        print("Unique mask values:", torch.unique(masks))
        if concat_direction == "horizontal":
            print(f"Left ratio: {left_ratio}, Total width: {total_width}, Left width: {target_left_width}, Right width: {target_right_width}")
        else:
            print(f"Top ratio: {left_ratio}, Total height: {total_height}, Top height: {target_top_height}, Bottom height: {target_bottom_height}")
        
        return (processed_images, masks)

NODE_CLASS_MAPPINGS = {
    "image concat mask": image_concat_mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image concat mask": "Image Concat with Mask"
}