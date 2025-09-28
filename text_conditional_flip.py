from PIL import Image
import numpy as np
import torch

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class TextConditionalFlip:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "conditional_flip"
    CATEGORY = "hhy/image"

    def conditional_flip(self, image, text):
        # 检查文字输入中是否包含"no"
        should_flip = "no" in text.lower()
        
        result_images = []
        
        for img in image:
            pil_img = tensor2pil(img)
            
            if should_flip:
                # 水平翻转（左右镜像）
                flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                result_images.append(pil2tensor(flipped_img))
            else:
                # 不翻转，保持原图
                result_images.append(pil2tensor(pil_img))
        
        result_tensor = torch.cat(result_images, dim=0)
        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "TextConditionalFlip": TextConditionalFlip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextConditionalFlip": "Text Conditional Flip",
}
