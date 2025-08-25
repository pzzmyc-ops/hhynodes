#!/usr/bin/env python3
"""
Image Splitter Node - ComfyUI Node

This ComfyUI node splits images using simple grid methods:
- Three equal parts: Splits image into 3 equal parts (horizontal or vertical)
- Nine grid: Splits image into 3x3 grid (9 parts)
"""

import numpy as np
import torch
from PIL import Image
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tensor2pil(image):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageSplitter:
    """ComfyUI Node for splitting images using different methods"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_mode": (["three_equal_parts", "six_grid", "nine_grid"], {"default": "three_equal_parts"}),
                "split_direction": (["horizontal", "vertical"], {"default": "vertical"}),
                "name_prefix": ("STRING", {"default": "bob"}),
                "save_to_folder": ("BOOLEAN", {"default": False}),
                "output_folder": ("STRING", {"default": "output/split_images"}),
            },
            "optional": {
                # 通用位置命名 (三等分和九宫格共用)
                "pos_1_name": ("STRING", {"default": "headf"}),      # 位置1: 左上/第一个
                "pos_2_name": ("STRING", {"default": "headb"}),      # 位置2: 中上/第二个  
                "pos_3_name": ("STRING", {"default": "heads"}),      # 位置3: 右上/第三个
                "pos_4_name": ("STRING", {"default": "head34l"}),    # 位置4: 左中
                "pos_5_name": ("STRING", {"default": "front"}),      # 位置5: 正中
                "pos_6_name": ("STRING", {"default": "head34r"}),    # 位置6: 右中
                "pos_7_name": ("STRING", {"default": "headul"}),     # 位置7: 左下
                "pos_8_name": ("STRING", {"default": "headll"}),     # 位置8: 下中
                "pos_9_name": ("STRING", {"default": "headlr"}),     # 位置9: 右下
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "split_info")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_image"
    CATEGORY = "hhy/image_processing"





    def split_three_equal_parts(self, image_array, direction="vertical"):
        """
        三等分拆分图片
        
        Args:
            image_array: numpy array of the image
            direction: 分割方向 ("vertical" 或 "horizontal")
            
        Returns:
            list of 3 image parts
        """
        height, width = image_array.shape[:2]
        
        if direction == "vertical":
            # 垂直方向三等分
            part_width = width // 3
            parts = [
                image_array[:, :part_width],
                image_array[:, part_width:2*part_width],
                image_array[:, 2*part_width:]
            ]
        else:
            # 水平方向三等分
            part_height = height // 3
            parts = [
                image_array[:part_height, :],
                image_array[part_height:2*part_height, :],
                image_array[2*part_height:, :]
            ]
        
        return parts

    def split_nine_grid(self, image_array):
        """
        九宫格拆分图片
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            list of 9 image parts (从左上到右下，按行排列)
        """
        height, width = image_array.shape[:2]
        
        # 计算每个格子的尺寸
        grid_height = height // 3
        grid_width = width // 3
        
        parts = []
        
        # 按行遍历，从左上到右下
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
        """
        六宫格拆分图片 (2行×3列)
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            list of 6 image parts (从左上到右下，按行排列)
        """
        height, width = image_array.shape[:2]
        
        # 计算每个格子的尺寸 (2行3列)
        grid_height = height // 2
        grid_width = width // 3
        
        parts = []
        
        # 按行遍历，从左上到右下
        for row in range(2):  # 2行
            for col in range(3):  # 3列
                start_y = row * grid_height
                end_y = (row + 1) * grid_height if row < 1 else height
                start_x = col * grid_width
                end_x = (col + 1) * grid_width if col < 2 else width
                
                part = image_array[start_y:end_y, start_x:end_x]
                parts.append(part)
        
        return parts

    def get_image_suffixes(self, split_mode, split_direction, num_parts, **kwargs):
        """
        获取图片名称后缀
        """
        if split_mode == "three_equal_parts":
            # 三等分使用前3个位置的命名
            return [
                kwargs.get("pos_1_name", "headf"),
                kwargs.get("pos_2_name", "headb"), 
                kwargs.get("pos_3_name", "heads")
            ]
        elif split_mode == "six_grid":
            # 六宫格使用前6个位置的命名 (2行×3列)
            return [
                kwargs.get("pos_1_name", "headf"),      # 第一行左: headf
                kwargs.get("pos_2_name", "headb"),      # 第一行中: headb
                kwargs.get("pos_3_name", "heads"),      # 第一行右: heads
                kwargs.get("pos_4_name", "head34l"),    # 第二行左: head34l
                kwargs.get("pos_5_name", "front"),      # 第二行中: front
                kwargs.get("pos_6_name", "head34r"),    # 第二行右: head34r  
            ]
        elif split_mode == "nine_grid":
            # 九宫格使用所有9个位置的命名
            return [
                kwargs.get("pos_1_name", "headf"),      # 左上: headf
                kwargs.get("pos_2_name", "headb"),      # 中上: headb
                kwargs.get("pos_3_name", "heads"),      # 右上: heads
                kwargs.get("pos_4_name", "head34l"),    # 左中: head34l
                kwargs.get("pos_5_name", "front"),      # 正中: front
                kwargs.get("pos_6_name", "head34r"),    # 右中: head34r  
                kwargs.get("pos_7_name", "headul"),     # 左下: headul
                kwargs.get("pos_8_name", "headll"),     # 下中: headll
                kwargs.get("pos_9_name", "headlr"),     # 右下: headlr
            ]
        else:
            # 如果没有定义，使用数字编号
            return [f"part_{i+1}" for i in range(num_parts)]

    def save_images_to_folder(self, images, name_prefix, output_folder, suffixes):
        """
        保存图片到文件夹
        """
        saved_paths = []
        
        # 创建主输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"📁 创建主输出文件夹: {output_folder}")
        
        # 使用name_prefix创建子文件夹
        sub_folder = os.path.join(output_folder, name_prefix)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder, exist_ok=True)
            print(f"📁 创建子文件夹: {sub_folder}")
        
        for i, (img_tensor, suffix) in enumerate(zip(images, suffixes)):
            # 转换为PIL图像
            pil_img = tensor2pil(img_tensor)
            
            # 生成文件名（不包含时间戳和前缀，因为已经在子文件夹中）
            filename = f"{suffix}.png"
            filepath = os.path.join(sub_folder, filename)
            
            # 保存图片
            pil_img.save(filepath, "PNG")
            saved_paths.append(filepath)
            print(f"💾 保存: {filename}")
        
        return saved_paths

    def split_image(self, image, split_mode="three_equal_parts", split_direction="vertical", name_prefix="bob", save_to_folder=False, output_folder="output/split_images", **kwargs):
        """分割图片的主函数"""
        
        try:
            # 转换tensor到numpy
            if len(image.shape) == 4:
                # 批量图片，取第一张
                image_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                image_array = (image.cpu().numpy() * 255).astype(np.uint8)
            
            print(f"🔄 开始分割图片...")
            print(f"📏 图片尺寸: {image_array.shape}")
            print(f"🎯 分割模式: {split_mode}")
            print(f"↔️ 分割方向: {split_direction}")
            
            # 根据分割模式选择不同的处理方法
            if split_mode == "three_equal_parts":
                # 三等分拆分
                parts = self.split_three_equal_parts(image_array, direction=split_direction)
                print(f"📐 三等分拆分完成")
            elif split_mode == "six_grid":
                # 六宫格拆分
                parts = self.split_six_grid(image_array)
                print(f"🔳 六宫格拆分完成")
            elif split_mode == "nine_grid":
                # 九宫格拆分
                parts = self.split_nine_grid(image_array)
                print(f"🔲 九宫格拆分完成")
            else:
                raise ValueError(f"不支持的分割模式: {split_mode}")
            
            # 根据分割模式检查部分数量
            if split_mode == "nine_grid":
                expected_parts = 9
            elif split_mode == "six_grid":
                expected_parts = 6
            else:
                expected_parts = 3
            if len(parts) != expected_parts:
                error_msg = f"❌ 分割失败：应该得到{expected_parts}个部分，实际得到{len(parts)}个"
                print(error_msg)
                # 返回原图作为输出
                original_tensor = pil2tensor(Image.fromarray(image_array))
                return ([original_tensor], error_msg)
            
            # 转换为tensor列表
            result_images = []
            for i, part in enumerate(parts):
                if len(part.shape) == 2:
                    # 灰度图转RGB
                    part = np.stack([part, part, part], axis=-1)
                
                pil_part = Image.fromarray(part)
                tensor_part = pil2tensor(pil_part)
                result_images.append(tensor_part)
                print(f"✅ 部分 {i+1} 尺寸: {part.shape}")
            
            # 获取图片名称后缀（传入用户输入的所有命名参数）
            suffixes = self.get_image_suffixes(split_mode, split_direction, len(parts), **kwargs)
            
            # 如果需要保存到文件夹
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
            
            # 根据分割模式生成不同的成功消息
            if split_mode == "three_equal_parts":
                success_msg = f"✅ 图片分割完成！使用三等分模式成功分割为{len(parts)}个部分"
            elif split_mode == "six_grid":
                success_msg = f"✅ 图片分割完成！使用六宫格模式成功分割为{len(parts)}个部分"
            elif split_mode == "nine_grid":
                success_msg = f"✅ 图片分割完成！使用九宫格模式成功分割为{len(parts)}个部分"
            
            # 添加命名信息
            naming_info = f"\n🏷️ 命名规则: {name_prefix}_[{', '.join(suffixes)}]"
            
            final_msg = success_msg + naming_info + save_info
            print(final_msg)
            
            return (result_images, final_msg)
            
        except Exception as e:
            error_msg = f"❌ 分割过程中发生错误: {str(e)}"
            print(error_msg)
            logger.error(f"Error in split_image: {e}", exc_info=True)
            
            # 返回原图作为错误处理
            try:
                if len(image.shape) == 4:
                    original_tensor = image[0:1]
                else:
                    original_tensor = image.unsqueeze(0)
                return ([original_tensor], error_msg)
            except:
                # 最后的错误处理
                empty_image = torch.zeros((1, 512, 512, 3))
                return ([empty_image], error_msg)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ImageSplitter": ImageSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSplitter": "Image Splitter (Grid)"
} 