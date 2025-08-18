from PIL import Image
import numpy as np
import torch

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class image_resize:
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
    CATEGORY = "hhy"

    def image_resize(self, image, resampling="lanczos", target_size=1024, multiple_of=64, reference_image=None):
        resized_images = []
        
        # If reference image is provided, use its dimensions exactly
        if reference_image is not None:
            ref_img = tensor2pil(reference_image[0])
            ref_width, ref_height = ref_img.size
            
            for img in image:
                pil_img = tensor2pil(img)
                resized = pil_img.resize((ref_width, ref_height), resample=Image.Resampling(self._get_resample_filter(resampling)))
                resized_images.append(pil2tensor(resized))
        else:
            # Original resize logic
            for img in image:
                resized_images.append(pil2tensor(self.apply_resize_image(tensor2pil(img), target_size, resampling, multiple_of)))
        
        resized_images = torch.cat(resized_images, dim=0)
        return (resized_images, )
        
    def _get_resample_filter(self, resample):
        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }
        return resample_filters[resample]
        
    def apply_resize_image(self, image: Image.Image, target_size: int = 1024, resample='bicubic', multiple_of: int = 64):
        current_width, current_height = image.size
        
        # Calculate scaling ratio based on target size
        ratio = min(target_size / current_width, target_size / current_height)
        new_width = round(current_width * ratio)
        new_height = round(current_height * ratio)

        # Adjust to be multiple of multiple_of
        new_width = new_width - (new_width % multiple_of)
        new_height = new_height - (new_height % multiple_of)

        # Ensure minimum size
        new_width = max(multiple_of, new_width)
        new_height = max(multiple_of, new_height)

        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }

        resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))

        return resized_image


class ImageResizeProportional:
    def __init__(self):
        pass
    
    RESIZE_MODES = {
        "Lanczos": Image.LANCZOS,
        "Nearest": Image.NEAREST,
        "Box": Image.BOX,
        "Bilinear": Image.BILINEAR,
        "Bicubic": Image.BICUBIC,
        "Hamming": Image.HAMMING,
    }

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
                "resize_mode": (list(s.RESIZE_MODES.keys()),),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "hhy"

    def resize_image(self, image, targetsize, step, resize_mode):
        batch_size, height, width, _ = image.shape
        result = []
        resampling_method = self.RESIZE_MODES[resize_mode]
        
        for i in range(batch_size):
            img = Image.fromarray(np.clip(255. * image[i].cpu().numpy(), 0, 255).astype(np.uint8))
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            # Calculate target pixels
            target_pixels = targetsize ** 2
            
            # 1. Calculate ideal dimensions (maintaining aspect ratio and close to target pixels)
            scale = (target_pixels / (original_width * original_height)) ** 0.5
            ideal_width = original_width * scale
            ideal_height = original_height * scale
            
            # 2. Generate candidate dimensions (width rounding schemes)
            width_candidates = []
            width_candidates.append(round(ideal_width / step) * step)  # Round to nearest
            width_candidates.append((int(ideal_width) // step) * step)  # Round down
            width_candidates.append(((int(ideal_width) + step - 1) // step) * step)  # Round up
            
            # 3. Evaluate each candidate dimension
            best_size = None
            best_diff = float('inf')
            
            for w in set(width_candidates):  # Remove duplicates
                w = max(step, w)  # Ensure not smaller than step
                h = w / aspect_ratio
                # Round height to step multiples (three rounding methods)
                h_candidates = [
                    round(h / step) * step,
                    (int(h) // step) * step,
                    ((int(h) + step - 1) // step) * step
                ]
                
                for h_candidate in set(h_candidates):
                    h_candidate = max(step, h_candidate)
                    current_pixels = w * h_candidate
                    # Calculate difference from target pixels
                    diff = abs(current_pixels - target_pixels)
                    
                    # Prefer solutions closer to target pixels
                    if diff < best_diff or (diff == best_diff and current_pixels < best_size[0]*best_size[1]):
                        best_diff = diff
                        best_size = (int(w), int(h_candidate))
            
            # 4. Resize image using best dimensions
            new_width, new_height = best_size
            img = img.resize((new_width, new_height), resampling_method)
            img = np.array(img).astype(np.float32) / 255.0
            result.append(torch.from_numpy(img))
        
        return (torch.stack(result),)

class ImageResizeToReferencePixels:
    def __init__(self):
        pass
    
    RESIZE_MODES = {
        "Lanczos": Image.LANCZOS,
        "Nearest": Image.NEAREST,
        "Box": Image.BOX,
        "Bilinear": Image.BILINEAR,
        "Bicubic": Image.BICUBIC,
        "Hamming": Image.HAMMING,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "resize_mode": (list(cls.RESIZE_MODES.keys()),),
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
    CATEGORY = "hhy"

    def resize_to_reference_pixels(self, image, reference_image, resize_mode, step, ratio):
        # Get reference image dimensions and calculate total pixels
        ref_img = tensor2pil(reference_image[0])
        ref_width, ref_height = ref_img.size
        target_pixels = ref_width * ref_height
        
        # Apply ratio to target pixels
        target_pixels = int(target_pixels * ratio)
        
        batch_size, height, width, _ = image.shape
        result = []
        resampling_method = self.RESIZE_MODES[resize_mode]
        
        for i in range(batch_size):
            img = tensor2pil(image[i])
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            # Calculate scale factor to match target pixels
            scale = (target_pixels / (original_width * original_height)) ** 0.5
            ideal_width = original_width * scale
            ideal_height = original_height * scale
            
            # Generate candidate dimensions (width rounding schemes)
            width_candidates = []
            width_candidates.append(round(ideal_width / step) * step)  # Round to nearest
            width_candidates.append((int(ideal_width) // step) * step)  # Round down
            width_candidates.append(((int(ideal_width) + step - 1) // step) * step)  # Round up
            
            # Evaluate each candidate dimension
            best_size = None
            best_diff = float('inf')
            
            for w in set(width_candidates):  # Remove duplicates
                w = max(step, w)  # Ensure not smaller than step
                h = w / aspect_ratio
                # Round height to step multiples (three rounding methods)
                h_candidates = [
                    round(h / step) * step,
                    (int(h) // step) * step,
                    ((int(h) + step - 1) // step) * step
                ]
                
                for h_candidate in set(h_candidates):
                    h_candidate = max(step, h_candidate)
                    current_pixels = w * h_candidate
                    # Calculate difference from target pixels
                    diff = abs(current_pixels - target_pixels)
                    
                    # Prefer solutions closer to target pixels
                    if diff < best_diff or (diff == best_diff and current_pixels < best_size[0]*best_size[1]):
                        best_diff = diff
                        best_size = (int(w), int(h_candidate))
            
            # Resize image using best dimensions
            new_width, new_height = best_size
            img = img.resize((new_width, new_height), resampling_method)
            result.append(pil2tensor(img))
        
        return (torch.cat(result, dim=0),)

NODE_CLASS_MAPPINGS = {
    "image resize": image_resize,
    "ImageResizeProportional": ImageResizeProportional,
    "ImageResizeToReferencePixels": ImageResizeToReferencePixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image resize": "Image Resize",
    "ImageResizeProportional": "Proportional Image Resizer",
    "ImageResizeToReferencePixels": "Resize to Reference Pixels"
}
