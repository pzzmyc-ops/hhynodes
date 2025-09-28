import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths

yolo_model_dir = os.path.join(folder_paths.models_dir, "yolo")
folder_paths.add_model_folder_path("yolo", yolo_model_dir)

if not os.path.exists(yolo_model_dir):
    os.makedirs(yolo_model_dir, exist_ok=True)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def add_mask(mask1, mask2):
    return torch.clamp(mask1 + mask2, 0.0, 1.0)

def np2pil(np_image):
    return Image.fromarray(np_image.astype(np.uint8))

def fix_torch_load_for_ultralytics():
    if hasattr(torch.load, '_original_torch_load'):
        return
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        import traceback
        stack = traceback.extract_stack()
        is_ultralytics_call = any('ultralytics' in frame.filename.lower() for frame in stack)
        if is_ultralytics_call:
            kwargs['weights_only'] = False
        try:
            return original_torch_load(*args, **kwargs)
        except Exception as e:
            if 'weights_only' in str(e) or 'WeightsUnpickler' in str(e):
                print(f"[YOLO Detection] 检测到PyTorch weights_only问题，使用兼容模式重试...")
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            else:
                raise e
    torch.load = patched_torch_load
    torch.load._original_torch_load = original_torch_load
    print("[YOLO Detection] PyTorch load 补丁已应用")

class YOLODetection:
    def __init__(self):
        self.model = None
        self.patch_applied = False
        
    @classmethod
    def INPUT_TYPES(cls):
        try:
            yolo_models = folder_paths.get_filename_list("yolo")
        except:
            yolo_models = []
        if not yolo_models:
            yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (yolo_models, {
                    "default": yolo_models[0]
                }),
                "confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "mask_ids": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "留空=输出所有mask，或指定ID如: 1 或 1,3"
                }),
                "mask_merge": ("BOOLEAN", {
                    "default": False,
                    "label_on": "合并mask",
                    "label_off": "分别输出"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("detection_image", "mask", "yolo_masks", "masked_images")
    FUNCTION = "detect"
    CATEGORY = "hhy/yolo"
    
    def apply_compatibility_patch(self):
        if not self.patch_applied:
            fix_torch_load_for_ultralytics()
            self.patch_applied = True
    
    def load_model(self, model_name):
        try:
            self.apply_compatibility_patch()
            from ultralytics import YOLO
            if self.model is None or getattr(self.model, 'model_name', None) != model_name:
                print(f"Loading YOLO model: {model_name}")
                model_path = folder_paths.get_full_path("yolo", model_name)
                if model_path is None:
                    print(f"Model {model_name} not found in {yolo_model_dir}, downloading...")
                    model_path = model_name
                else:
                    print(f"Using model from: {model_path}")
                self.model = YOLO(model_path)
                self.model.model_name = model_name
            return self.model
        except ImportError:
            raise ImportError("请安装ultralytics库: pip install ultralytics")
    
    def parse_mask_ids(self, mask_ids_str):
        if not mask_ids_str.strip():
            return "all"
        
        mask_ids_str = mask_ids_str.strip().lower()
        if mask_ids_str == "all":
            return "all"
        
        try:
            ids = []
            for id_str in mask_ids_str.split(','):
                id_str = id_str.strip()
                if id_str.isdigit():
                    ids.append(int(id_str))
            return ids if ids else "all"
        except:
            print(f"警告: mask_ids参数格式错误: {mask_ids_str}，使用默认值all")
            return "all"
    
    def detect(self, image, model_name="yolov8n.pt", confidence=0.5, iou_threshold=0.45, 
               mask_ids="1", mask_merge=False):
        model = self.load_model(model_name)
        
        requested_ids = self.parse_mask_ids(mask_ids)
        
        detection_images = []
        output_masks = []
        all_yolo_masks = []
        masked_images = []
        
        for img_tensor in image:
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            
            results = model(pil_image, conf=confidence, iou=iou_threshold, retina_masks=True)
            
            for result in results:
                yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                detection_image_pil = Image.fromarray(yolo_plot_image)
                detection_images.append(pil2tensor(detection_image_pil))
                
                individual_masks = []
                has_detection = False
                
                if result.masks is not None and len(result.masks) > 0:
                    print(f"检测到 {len(result.masks)} 个分割mask")
                    has_detection = True
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask_pil = np2pil(_mask).convert("L")
                        mask_tensor = image2mask(_mask_pil)
                        individual_masks.append(mask_tensor)
                        all_yolo_masks.append(mask_tensor)
                elif result.boxes is not None and len(result.boxes.xyxy) > 0:
                    print(f"检测到 {len(result.boxes)} 个检测框，创建方形mask")
                    has_detection = True
                    white_image = Image.new('L', pil_image.size, "white")
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        _mask = Image.new('L', pil_image.size, "black")
                        _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                        mask_tensor = image2mask(_mask)
                        individual_masks.append(mask_tensor)
                        all_yolo_masks.append(mask_tensor)
                
                if individual_masks:
                    if requested_ids == "all":
                        if mask_merge:
                            final_mask = individual_masks[0]
                            for i in range(1, len(individual_masks)):
                                final_mask = add_mask(final_mask, individual_masks[i])
                            print(f"合并了所有 {len(individual_masks)} 个mask")
                            output_masks.append(final_mask)
                        else:
                            print(f"分别输出所有 {len(individual_masks)} 个mask")
                            output_masks.extend(individual_masks)
                    else:
                        selected_masks = []
                        for mask_id in requested_ids:
                            mask_index = mask_id - 1
                            if mask_index < len(individual_masks):
                                selected_masks.append(individual_masks[mask_index])
                                print(f"选择了第 {mask_id} 个mask")
                            else:
                                print(f"请求的mask ID {mask_id} 超出范围（共{len(individual_masks)}个）")
                        
                        if selected_masks:
                            if mask_merge and len(selected_masks) > 1:
                                final_mask = selected_masks[0]
                                for i in range(1, len(selected_masks)):
                                    final_mask = add_mask(final_mask, selected_masks[i])
                                print(f"合并了 {len(selected_masks)} 个指定的mask")
                                output_masks.append(final_mask)
                            else:
                                print(f"分别输出 {len(selected_masks)} 个指定的mask")
                                output_masks.extend(selected_masks)
                        else:
                            print("没有有效的mask ID，输出空mask")
                            empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                            output_masks.append(empty_mask)
                else:
                    print("未检测到任何对象，输出空mask")
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    output_masks.append(empty_mask)
                    all_yolo_masks.append(empty_mask)

                if has_detection and individual_masks:

                    combined_mask = individual_masks[0]
                    for i in range(1, len(individual_masks)):
                        combined_mask = add_mask(combined_mask, individual_masks[i])
                    

                    img_array = np.array(pil_image)
                    mask_array = combined_mask.cpu().numpy().squeeze()
                    

                    masked_img_array = img_array.copy()
                    mask_3d = np.stack([mask_array, mask_array, mask_array], axis=-1)
                    masked_img_array = masked_img_array * mask_3d
                    
                    masked_img_pil = Image.fromarray(masked_img_array.astype(np.uint8))
                    masked_images.append(pil2tensor(masked_img_pil))
                else:

                    black_img = Image.new('RGB', pil_image.size, (0, 0, 0))
                    masked_images.append(pil2tensor(black_img))
        
        if not detection_images:
            for img_tensor in image:
                img_tensor_single = torch.unsqueeze(img_tensor, 0)
                pil_image = tensor2pil(img_tensor_single)
                detection_images.append(pil2tensor(pil_image))
                empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                output_masks.append(empty_mask)
                all_yolo_masks.append(empty_mask)

                black_img = Image.new('RGB', pil_image.size, (0, 0, 0))
                masked_images.append(pil2tensor(black_img))
        
        detection_images_tensor = torch.cat(detection_images, dim=0)
        output_masks_tensor = torch.cat(output_masks, dim=0) if output_masks else torch.zeros((1, 512, 512), dtype=torch.float32)
        all_yolo_masks_tensor = torch.cat(all_yolo_masks, dim=0) if all_yolo_masks else torch.zeros((1, 512, 512), dtype=torch.float32)
        masked_images_tensor = torch.cat(masked_images, dim=0) if masked_images else torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
        merge_status = "合并" if mask_merge else "分别输出"
        print(f"处理完成: {len(detection_images)} 张检测图像, {len(masked_images)} 张遮罩图像, mask_ids: {mask_ids}, {merge_status}")
        
        return (detection_images_tensor, output_masks_tensor, all_yolo_masks_tensor, masked_images_tensor)

NODE_CLASS_MAPPINGS = {
    "YOLODetection": YOLODetection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLODetection": "YOLO Detection by HHY"
} 