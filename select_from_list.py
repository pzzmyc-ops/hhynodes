import torch
import hashlib


class SelectFromImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "select_mode": (["first", "last", "biggest"], {"default": "last"}),
            },
        }

    CATEGORY = "hhy/image"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "selected_flat_index", "info")
    FUNCTION = "select"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False)

    def _normalize_single_image(self, tensor_image):
        # Ensure a single image tensor in HWC format, float32, [0,1]
        if not isinstance(tensor_image, torch.Tensor):
            raise ValueError("Invalid image type; expected torch.Tensor")
        if tensor_image.dim() == 4:
            tensor_image = tensor_image[0]
        if tensor_image.dim() != 3:
            raise ValueError(f"Unexpected tensor dims: {tensor_image.shape}")
        # Convert CHW -> HWC if needed
        if tensor_image.shape[0] in (1, 3) and tensor_image.shape[0] < tensor_image.shape[1]:
            # Likely CHW
            tensor_image = tensor_image.permute(1, 2, 0).contiguous()
        # Ensure dtype
        if tensor_image.dtype != torch.float32:
            tensor_image = tensor_image.to(torch.float32)
        return tensor_image

    def _height_width(self, tensor_image):
        if tensor_image.dim() != 3:
            raise ValueError(f"Unexpected tensor dims: {tensor_image.shape}")
        # Assume HWC after normalization
        height, width = tensor_image.shape[0], tensor_image.shape[1]
        return int(height), int(width)

    def select(self, images, select_mode="last"):
        # ComfyUI list mode may pass select_mode as a list like ['first']
        if isinstance(select_mode, (list, tuple)):
            select_mode = select_mode[0] if len(select_mode) > 0 else "last"
        try:
            print(f"[SelectFromImageList] inputs={len(images)}, mode={select_mode}")
        except Exception:
            print("[SelectFromImageList] inputs=? (non-list), mode=", select_mode)
        # images is a list of IMAGE inputs (each may be a batch or a single)
        candidates = []  # list of (normalized_hwc, flat_index, info)
        flat_index = 0
        for batch_idx, img_batch in enumerate(images):
            if img_batch is None:
                continue
            # Debug: show each list element type/shape
            if isinstance(img_batch, torch.Tensor):
                try:
                    print(f"[SelectFromImageList] b{batch_idx+1}: Tensor dim={img_batch.dim()} shape={tuple(img_batch.shape)}")
                except Exception:
                    print(f"[SelectFromImageList] b{batch_idx+1}: Tensor")
            elif isinstance(img_batch, (list, tuple)):
                print(f"[SelectFromImageList] b{batch_idx+1}: List len={len(img_batch)}")
                for j, entry in enumerate(img_batch):
                    if isinstance(entry, torch.Tensor):
                        try:
                            print(f"  - entry {j+1}: Tensor dim={entry.dim()} shape={tuple(entry.shape)}")
                        except Exception:
                            print(f"  - entry {j+1}: Tensor")
                    else:
                        print(f"  - entry {j+1}: type={type(entry)}")
            else:
                print(f"[SelectFromImageList] b{batch_idx+1}: type={type(img_batch)}")
            # Support nested list/tuple entries
            entries = []
            if isinstance(img_batch, (list, tuple)):
                entries = list(img_batch)
            else:
                entries = [img_batch]

            local_idx = 0
            for entry in entries:
                if not isinstance(entry, torch.Tensor):
                    continue
                # Normalize batch: accept (H,W,C) or (B,H,W,C) or (C,H,W) or (B,C,H,W)
                if entry.dim() == 3:
                    single = self._normalize_single_image(entry)
                    h, w = self._height_width(single)
                    info = f"b{batch_idx+1}/i{local_idx+1}: {h}x{w}"
                    candidates.append((single, flat_index, info))
                    flat_index += 1
                    local_idx += 1
                elif entry.dim() == 4:
                    b = entry.shape[0]
                    for i in range(b):
                        single = self._normalize_single_image(entry[i])
                        h, w = self._height_width(single)
                        info = f"b{batch_idx+1}/i{local_idx+1}: {h}x{w}"
                        candidates.append((single, flat_index, info))
                        flat_index += 1
                        local_idx += 1
                else:
                    # Unsupported dims; skip
                    continue

        if not candidates:
            # Return a minimal placeholder image
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, 0, "No valid images provided")

        # Debug: list candidates
        try:
            cand_info = ", ".join([f"{i+1}:{c[0].shape[0]}x{c[0].shape[1]}" for i, c in enumerate(candidates)])
            print(f"[SelectFromImageList] candidates={len(candidates)} -> [{cand_info}]")
        except Exception:
            print(f"[SelectFromImageList] candidates={len(candidates)}")

        if select_mode == "first":
            chosen = candidates[0]
        elif select_mode == "last":
            chosen = candidates[-1]
        elif select_mode == "biggest":
            # Choose by area H*W
            chosen = max(candidates, key=lambda x: (x[0].shape[0] * x[0].shape[1]))
        else:
            chosen = candidates[-1]

        selected_image_hwc, selected_flat_index, selected_info = chosen
        # Return as a 1-image batch in HWC format
        output = selected_image_hwc.unsqueeze(0)
        info = f"Selected {selected_info} | mode={select_mode} | flat_index={selected_flat_index+1}"
        print(f"[SelectFromImageList] {info}")
        return (output, selected_flat_index + 1, info)

    @classmethod
    def IS_CHANGED(cls, images, select_mode, **kwargs):
        # Hash list length and per-item shapes
        try:
            lens = len(images)
            shape_sig = ",".join([str(getattr(img, "shape", None)) for img in images])
        except Exception:
            lens = 0
            shape_sig = "unknown"
        return hashlib.md5(f"{lens}:{shape_sig}:{select_mode}".encode()).hexdigest()[:16]


NODE_CLASS_MAPPINGS = {
    "SelectFromImageList": SelectFromImageList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectFromImageList": "Select From Image List",
}


