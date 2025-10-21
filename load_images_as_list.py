import os
import io
import base64
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import node_helpers


class LoadImagesAsList:
    """选择多张图片，执行时上传并加载为列表"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False,
                }),
            },
        }
    
    CATEGORY = "hhy/image"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "load_images"
    
    def load_images(self, images_data=""):
        """
        执行时上传图片并加载为列表
        images_data: JSON 字符串，包含 base64 编码的图片数据
        """
        import json
        
        output_images = []
        
        if not images_data or images_data.strip() == "":
            print("[LoadImagesAsList] No images data provided, returning placeholder")
            placeholder = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return ([placeholder], 1)
        
        try:
            # 解析 JSON 数据
            images_list = json.loads(images_data)
            print(f"[LoadImagesAsList] Processing {len(images_list)} images")
            
            input_dir = folder_paths.get_input_directory()
            
            for idx, img_data in enumerate(images_list):
                try:
                    filename = img_data.get("name", f"uploaded_{idx}.png")
                    base64_data = img_data.get("data", "")
                    
                    if not base64_data:
                        print(f"  ⚠ {idx+1}/{len(images_list)}: No data for {filename}")
                        continue
                    
                    # 解码 base64
                    # 移除 data:image/xxx;base64, 前缀（如果有）
                    if "," in base64_data:
                        base64_data = base64_data.split(",", 1)[1]
                    
                    img_bytes = base64.b64decode(base64_data)
                    
                    # 保存到 input 目录（如果还不存在）
                    filepath = os.path.join(input_dir, filename)
                    
                    # 如果文件已存在，添加数字后缀
                    if os.path.exists(filepath):
                        base_name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(filepath):
                            filename = f"{base_name}_{counter}{ext}"
                            filepath = os.path.join(input_dir, filename)
                            counter += 1
                    
                    # 写入文件
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)
                    
                    print(f"  💾 {idx+1}/{len(images_list)}: Saved {filename}")
                    
                    # 加载图片
                    img = node_helpers.pillow(Image.open, filepath)
                    img = node_helpers.pillow(ImageOps.exif_transpose, img)
                    
                    if img.mode == 'I':
                        img = img.point(lambda i: i * (1 / 255))
                    
                    image_rgb = img.convert("RGB")
                    image_np = np.array(image_rgb).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    
                    output_images.append(image_tensor)
                    print(f"  ✓ {idx+1}/{len(images_list)}: Loaded {filename} ({image_rgb.size[0]}x{image_rgb.size[1]})")
                    
                except Exception as e:
                    print(f"  ✗ {idx+1}/{len(images_list)}: Error - {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
        except json.JSONDecodeError as e:
            print(f"[LoadImagesAsList] ✗ JSON decode error: {e}")
            placeholder = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return ([placeholder], 1)
        except Exception as e:
            print(f"[LoadImagesAsList] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            placeholder = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return ([placeholder], 1)
        
        if not output_images:
            print("[LoadImagesAsList] No images loaded, returning placeholder")
            placeholder = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            output_images.append(placeholder)
        
        print(f"[LoadImagesAsList] Total loaded: {len(output_images)} images")
        return (output_images, len(output_images))


NODE_CLASS_MAPPINGS = {
    "LoadImagesAsList": LoadImagesAsList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesAsList": "Load Images as List",
}
