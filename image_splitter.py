

import numpy as np
import torch
from PIL import Image
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageSplitter:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_mode": (["three_equal_parts", "six_grid", "nine_grid", "left_right_half"], {"default": "three_equal_parts"}),
                "split_direction": (["horizontal", "vertical"], {"default": "vertical"}),
                "name_prefix": ("STRING", {"default": "bob"}),
                "save_to_folder": ("BOOLEAN", {"default": False}),
                "output_folder": ("STRING", {"default": "output/split_images"}),
            },
            "optional": {
                "pos_1_name": ("STRING", {"default": "headf"}),
                "pos_2_name": ("STRING", {"default": "headb"}),
                "pos_3_name": ("STRING", {"default": "heads"}),
                "pos_4_name": ("STRING", {"default": "head34l"}),
                "pos_5_name": ("STRING", {"default": "front"}),
                "pos_6_name": ("STRING", {"default": "head34r"}),
                "pos_7_name": ("STRING", {"default": "headul"}),
                "pos_8_name": ("STRING", {"default": "headll"}),
                "pos_9_name": ("STRING", {"default": "headlr"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "split_info")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_image"
    CATEGORY = "hhy/image"

    def split_three_equal_parts(self, image_array, direction="vertical"):
        height, width = image_array.shape[:2]
        if direction == "vertical":
            part_width = width // 3
            parts = [
                image_array[:, :part_width],
                image_array[:, part_width:2*part_width],
                image_array[:, 2*part_width:]
            ]
        else:
            part_height = height // 3
            parts = [
                image_array[:part_height, :],
                image_array[part_height:2*part_height, :],
                image_array[2*part_height:, :]
            ]
        return parts

    def split_nine_grid(self, image_array):
        height, width = image_array.shape[:2]
        grid_height = height // 3
        grid_width = width // 3
        parts = []
        for row in range(3):
            for col in range(3):
                start_y = row * grid_height
                end_y = (row + 1) * grid_height if row < 2 else height
                start_x = col * grid_width
                end_x = (col + 1) * grid_width if col < 2 else width
                part = image_array[start_y:end_y, start_x:end_x]
                parts.append(part)
        return parts

    def split_six_grid(self, image_array):
        height, width = image_array.shape[:2]
        grid_height = height // 2
        grid_width = width // 3
        parts = []
        for row in range(2):
            for col in range(3):
                start_y = row * grid_height
                end_y = (row + 1) * grid_height if row < 1 else height
                start_x = col * grid_width
                end_x = (col + 1) * grid_width if col < 2 else width
                part = image_array[start_y:end_y, start_x:end_x]
                parts.append(part)
        return parts

    def split_left_right_half(self, image_array):
        height, width = image_array.shape[:2]
        mid_width = width // 2
        left_part = image_array[:, :mid_width]
        right_part = image_array[:, mid_width:]
        return [left_part, right_part]

    def get_image_suffixes(self, split_mode, split_direction, num_parts, **kwargs):
        if split_mode == "three_equal_parts":
            return [
                kwargs.get("pos_1_name", "headf"),
                kwargs.get("pos_2_name", "headb"), 
                kwargs.get("pos_3_name", "heads")
            ]
        elif split_mode == "six_grid":
            return [
                kwargs.get("pos_1_name", "headf"),
                kwargs.get("pos_2_name", "headb"),
                kwargs.get("pos_3_name", "heads"),
                kwargs.get("pos_4_name", "head34l"),
                kwargs.get("pos_5_name", "front"),
                kwargs.get("pos_6_name", "head34r"),
            ]
        elif split_mode == "nine_grid":
            return [
                kwargs.get("pos_1_name", "headf"),
                kwargs.get("pos_2_name", "headb"),
                kwargs.get("pos_3_name", "heads"),
                kwargs.get("pos_4_name", "head34l"),
                kwargs.get("pos_5_name", "front"),
                kwargs.get("pos_6_name", "head34r"),
                kwargs.get("pos_7_name", "headul"),
                kwargs.get("pos_8_name", "headll"),
                kwargs.get("pos_9_name", "headlr"),
            ]
        elif split_mode == "left_right_half":
            return [
                kwargs.get("pos_1_name", "left"),
                kwargs.get("pos_2_name", "right"),
            ]
        else:
            return [f"part_{i+1}" for i in range(num_parts)]

    def save_images_to_folder(self, images, name_prefix, output_folder, suffixes):
        saved_paths = []
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"📁 创建主输出文件夹: {output_folder}")
        sub_folder = os.path.join(output_folder, name_prefix)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder, exist_ok=True)
            print(f"📁 创建子文件夹: {sub_folder}")
        for i, (img_tensor, suffix) in enumerate(zip(images, suffixes)):
            pil_img = tensor2pil(img_tensor)
            filename = f"{suffix}.png"
            filepath = os.path.join(sub_folder, filename)
            pil_img.save(filepath, "PNG")
            saved_paths.append(filepath)
            print(f"💾 保存: {filename}")
        return saved_paths

    def split_image(self, image, split_mode="three_equal_parts", split_direction="vertical", name_prefix="bob", save_to_folder=False, output_folder="output/split_images", **kwargs):
        try:
            if len(image.shape) == 4:
                image_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                image_array = (image.cpu().numpy() * 255).astype(np.uint8)
            print(f"🔄 开始分割图片...")
            print(f"📏 图片尺寸: {image_array.shape}")
            print(f"🎯 分割模式: {split_mode}")
            print(f"↔️ 分割方向: {split_direction}")
            if split_mode == "three_equal_parts":
                parts = self.split_three_equal_parts(image_array, direction=split_direction)
                print(f"📐 三等分拆分完成")
            elif split_mode == "six_grid":
                parts = self.split_six_grid(image_array)
                print(f"🔳 六宫格拆分完成")
            elif split_mode == "nine_grid":
                parts = self.split_nine_grid(image_array)
                print(f"🔲 九宫格拆分完成")
            elif split_mode == "left_right_half":
                parts = self.split_left_right_half(image_array)
                print(f"↔️ 左右对半拆分完成")
            else:
                raise ValueError(f"不支持的分割模式: {split_mode}")
            if split_mode == "nine_grid":
                expected_parts = 9
            elif split_mode == "six_grid":
                expected_parts = 6
            elif split_mode == "left_right_half":
                expected_parts = 2
            else:
                expected_parts = 3
            if len(parts) != expected_parts:
                error_msg = f"❌ 分割失败：应该得到{expected_parts}个部分，实际得到{len(parts)}个"
                print(error_msg)
                original_tensor = pil2tensor(Image.fromarray(image_array))
                return ([original_tensor], error_msg)
            result_images = []
            for i, part in enumerate(parts):
                if len(part.shape) == 2:
                    part = np.stack([part, part, part], axis=-1)
                pil_part = Image.fromarray(part)
                tensor_part = pil2tensor(pil_part)
                result_images.append(tensor_part)
                print(f"✅ 部分 {i+1} 尺寸: {part.shape}")
            suffixes = self.get_image_suffixes(split_mode, split_direction, len(parts), **kwargs)
            saved_paths = []
            if save_to_folder:
                try:
                    saved_paths = self.save_images_to_folder(result_images, name_prefix, output_folder, suffixes)
                    save_info = f"\n💾 保存了 {len(saved_paths)} 个文件到: {output_folder}"
                except Exception as e:
                    save_info = f"\n⚠️ 保存失败: {str(e)}"
                    print(f"⚠️ 保存图片时出错: {e}")
            else:
                save_info = ""
            if split_mode == "three_equal_parts":
                success_msg = f"✅ 图片分割完成！使用三等分模式成功分割为{len(parts)}个部分"
            elif split_mode == "six_grid":
                success_msg = f"✅ 图片分割完成！使用六宫格模式成功分割为{len(parts)}个部分"
            elif split_mode == "nine_grid":
                success_msg = f"✅ 图片分割完成！使用九宫格模式成功分割为{len(parts)}个部分"
            elif split_mode == "left_right_half":
                success_msg = f"✅ 图片分割完成！使用左右对半模式成功分割为{len(parts)}个部分"
            naming_info = f"\n🏷️ 命名规则: {name_prefix}_[{', '.join(suffixes)}]"
            final_msg = success_msg + naming_info + save_info
            print(final_msg)
            return (result_images, final_msg)
        except Exception as e:
            error_msg = f"❌ 分割过程中发生错误: {str(e)}"
            print(error_msg)
            logger.error(f"Error in split_image: {e}", exc_info=True)
            try:
                if len(image.shape) == 4:
                    original_tensor = image[0:1]
                else:
                    original_tensor = image.unsqueeze(0)
                return ([original_tensor], error_msg)
            except:
                empty_image = torch.zeros((1, 512, 512, 3))
                return ([empty_image], error_msg)

NODE_CLASS_MAPPINGS = {
    "ImageSplitter": ImageSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSplitter": "Image Splitter (Grid)"
} 