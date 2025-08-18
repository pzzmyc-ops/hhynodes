import torch
import numpy as np
from PIL import Image


def tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor to a PIL Image.
    Handles tensors shaped as (H, W, C), (1, H, W, C), or (B, H, W, C) with B==1.
    """
    array_uint8 = np.clip(255.0 * image_tensor.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8)


def mask_tensor_to_np(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert a ComfyUI MASK tensor to a numpy 2D array (uint8, 0-255)."""
    mask = mask_tensor.detach().cpu().numpy().squeeze()
    mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return mask_uint8


def pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a ComfyUI IMAGE tensor with shape (1, H, W, C) float32 in [0,1]."""
    array_float = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array_float).unsqueeze(0)


class PasteImageIntoMaskRectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "origin": ("IMAGE",),
                "anchor": (["top_left", "top_center", "center", "top_right"], {"default": "top_left"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "paste_image"
    CATEGORY = "hhy"

    def _find_white_rectangle_bbox(self, mask_np: np.ndarray):
        """Find the bounding box of the white region in a binary-like mask.
        Returns (x1, y1, x2, y2) inclusive coordinates or None if not found.
        """
        ys, xs = np.where(mask_np > 127)
        if ys.size == 0 or xs.size == 0:
            return None
        y1 = int(ys.min())
        y2 = int(ys.max())
        x1 = int(xs.min())
        x2 = int(xs.max())
        return x1, y1, x2, y2

    def _calculate_paste_position(self, bbox, img_size, anchor):
        """Calculate the paste position based on anchor point.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box of mask rectangle
            img_size: (width, height) of image to paste
            anchor: anchor point string
            
        Returns:
            (x, y) position to paste image
        """
        x1, y1, x2, y2 = bbox
        img_width, img_height = img_size
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

    def paste_image(self, image: torch.Tensor, mask: torch.Tensor, origin: torch.Tensor, anchor: str):
        # Determine batch sizes
        image_batch = image
        mask_batch = mask
        origin_batch = origin

        num_images = image_batch.shape[0] if image_batch.ndim == 4 else 1
        num_masks = mask_batch.shape[0] if mask_batch.ndim >= 3 else 1
        num_origins = origin_batch.shape[0] if origin_batch.ndim == 4 else 1
        batch_size = max(num_images, num_masks, num_origins)

        outputs = []

        for i in range(batch_size):
            # Select indices
            img_idx = i if i < num_images else num_images - 1
            msk_idx = i if i < num_masks else num_masks - 1
            org_idx = i if i < num_origins else num_origins - 1

            img_tensor = image_batch[img_idx:img_idx + 1] if num_images > 1 else image_batch
            mask_tensor = mask_batch[msk_idx:msk_idx + 1] if num_masks > 1 else mask_batch
            origin_tensor = origin_batch[org_idx:org_idx + 1] if num_origins > 1 else origin_batch

            # Convert to PIL / numpy
            pil_img = tensor_to_pil_image(img_tensor)
            mask_np = mask_tensor_to_np(mask_tensor)
            origin_pil = tensor_to_pil_image(origin_tensor)

            # Find white rectangle bbox
            bbox = self._find_white_rectangle_bbox(mask_np)

            # Start from origin image as base
            base = origin_pil.copy()

            if bbox is not None:
                # Calculate paste position based on anchor
                paste_x, paste_y = self._calculate_paste_position(bbox, pil_img.size, anchor)
                base.paste(pil_img, (paste_x, paste_y))
            else:
                # If no white region found, return origin unchanged for this item
                pass

            outputs.append(pil_to_image_tensor(base))

        # Concatenate along batch dimension
        result = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        return (result,)


NODE_CLASS_MAPPINGS = {
    "PasteImageIntoMaskRect": PasteImageIntoMaskRectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PasteImageIntoMaskRect": "Paste Image Into Mask Rect",
} 