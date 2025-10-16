"""
智能漫画拼接 ComfyUI 节点
功能：接收图片列表，按顺序拼接，自动识别分镜边界

特性：
1. 颜色相似度合并（避免JPEG压缩伪影）
2. 主色占比检测
3. 颜色方差检测
4. 连续性检测
5. 支持多种图片格式（jpg, png, bmp, webp, tiff）
"""
import numpy as np
from PIL import Image
import torch
from collections import Counter
import comfy.utils

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 配置类
class Config:
    def __init__(self):
        # === 边界检测参数 ===
        self.scan_lines = 6                     # 扫描首尾行数
        self.color_similarity = 10              # 颜色相似度阈值（合并相似颜色）
        self.merged_color_threshold = 15        # 合并后颜色种类阈值
        self.variance_threshold = 30            # 颜色方差阈值
        self.dominant_ratio_threshold = 0.5     # 主色占比阈值（50%）
        
        self.continuity_threshold = 20          # 连续性判定阈值（平均像素差）
        self.pure_color_diff = 5                # 纯色差异阈值（<5认为是分割线）

def merge_similar_colors(pixels, similarity_threshold=10):
    """
    合并相似颜色，避免JPEG压缩导致的颜色碎片化
    
    参数:
        pixels: 像素列表 [(R,G,B), ...]
        similarity_threshold: 相似度阈值，RGB差值小于此值的认为是同一种颜色
    
    返回:
        合并后的唯一颜色数量
    """
    if not pixels:
        return 0
    
    # 将相似颜色映射到代表色
    color_groups = {}
    
    for pixel in pixels:
        # 量化颜色：将RGB值按阈值分组
        quantized = tuple((c // similarity_threshold) * similarity_threshold for c in pixel)
        color_groups[quantized] = color_groups.get(quantized, 0) + 1
    
    return len(color_groups)

def calculate_color_variance(pixels):
    """
    计算颜色方差（标准差）
    方差小表示颜色变化小，可能是边界
    
    参数:
        pixels: 像素numpy数组 (N, 3)
    
    返回:
        RGB三通道的平均标准差
    """
    if len(pixels) == 0:
        return 0
    
    # 计算每个通道的标准差
    std_r = np.std(pixels[:, 0])
    std_g = np.std(pixels[:, 1])
    std_b = np.std(pixels[:, 2])
    
    # 返回平均标准差
    return (std_r + std_g + std_b) / 3

def get_dominant_color_ratio(pixels):
    """
    获取主色占比
    
    参数:
        pixels: 像素列表 [(R,G,B), ...]
    
    返回:
        主色占比（0-1之间）
    """
    if not pixels:
        return 0
    
    color_counter = Counter(pixels)
    most_common_color, count = color_counter.most_common(1)[0]
    return count / len(pixels)

def is_panel_boundary_v2(img_array, config, check_top=True, check_bottom=True):
    """
    改进的分镜边界检测（V2）
    
    参数:
        img_array: 图片的numpy数组
        config: 配置对象
        check_top: 是否检查顶部
        check_bottom: 是否检查底部
    
    返回:
        (top_is_boundary, bottom_is_boundary, top_info, bottom_info)
    """
    height = img_array.shape[0]
    top_is_boundary = False
    bottom_is_boundary = False
    top_info = {}
    bottom_info = {}
    
    if check_top and height > config.scan_lines:
        region = img_array[0:min(config.scan_lines, height)]
        pixels = region.reshape(-1, region.shape[-1])
        pixel_tuples = [tuple(p) for p in pixels]
        
        # 方法1：合并相似颜色后统计
        merged_colors = merge_similar_colors(pixel_tuples, config.color_similarity)
        
        # 方法2：颜色方差
        variance = calculate_color_variance(pixels)
        
        # 方法3：主色占比
        dominant_ratio = get_dominant_color_ratio(pixel_tuples)
        
        # 综合判断：满足任一条件即为边界
        top_is_boundary = (
            merged_colors <= config.merged_color_threshold or
            variance <= config.variance_threshold or
            dominant_ratio >= config.dominant_ratio_threshold
        )
        
        top_info = {
            'merged_colors': merged_colors,
            'variance': variance,
            'dominant_ratio': dominant_ratio,
            'is_boundary': top_is_boundary
        }
    
    if check_bottom and height > config.scan_lines:
        region = img_array[max(0, height - config.scan_lines):height]
        pixels = region.reshape(-1, region.shape[-1])
        pixel_tuples = [tuple(p) for p in pixels]
        
        # 方法1：合并相似颜色后统计
        merged_colors = merge_similar_colors(pixel_tuples, config.color_similarity)
        
        # 方法2：颜色方差
        variance = calculate_color_variance(pixels)
        
        # 方法3：主色占比
        dominant_ratio = get_dominant_color_ratio(pixel_tuples)
        
        # 综合判断
        bottom_is_boundary = (
            merged_colors <= config.merged_color_threshold or
            variance <= config.variance_threshold or
            dominant_ratio >= config.dominant_ratio_threshold
        )
        
        bottom_info = {
            'merged_colors': merged_colors,
            'variance': variance,
            'dominant_ratio': dominant_ratio,
            'is_boundary': bottom_is_boundary
        }
    
    return top_is_boundary, bottom_is_boundary, top_info, bottom_info

def check_content_alignment(prev_img_array, curr_img_array, config):
    """
    检查两张图片边界处的内容特征相似度
    
    新策略：
    1. 识别非背景像素（深色内容：线条、文字等）
    2. 检查两边是否都有相似的内容分布模式
    3. 排除纯色情况（全黑或全白的分割线）
    4. 如果两边都有内容且分布相似，认为是同一对象的延续（如对话框边框）
    
    参数:
        prev_img_array: 前一张图片的numpy数组
        curr_img_array: 当前图片的numpy数组
        config: 配置对象
    
    返回:
        has_similar_content: 是否有相似的内容特征
        content_overlap_ratio: 内容重叠占比
    """
    # 获取边界区域
    prev_height = prev_img_array.shape[0]
    prev_bottom = prev_img_array[max(0, prev_height - config.scan_lines):prev_height]
    curr_top = curr_img_array[0:min(config.scan_lines, curr_img_array.shape[0])]
    
    # 确保宽度一致
    min_width = min(prev_bottom.shape[1], curr_top.shape[1])
    prev_bottom = prev_bottom[:, :min_width]
    curr_top = curr_top[:, :min_width]
    
    # 检查是否是纯色（避免将两个黑色分割线判定为连续）
    prev_variance = np.std(prev_bottom)
    curr_variance = np.std(curr_top)
    
    if prev_variance < 10 and curr_variance < 10:
        return False, 0.0
    
    # 识别非背景像素（深色像素：RGB < 200）
    prev_has_content = np.any(prev_bottom < 200, axis=(0, 2))
    curr_has_content = np.any(curr_top < 200, axis=(0, 2))
    
    # 统计有内容的列
    prev_content_count = np.sum(prev_has_content)
    curr_content_count = np.sum(curr_has_content)
    
    # 如果两边都没有内容，无法判断
    if prev_content_count == 0 and curr_content_count == 0:
        return False, 0.0
    
    # 如果只有一边有内容，说明不连续
    if prev_content_count == 0 or curr_content_count == 0:
        return False, 0.0
    
    # 计算内容重叠度：两边都有内容的列数
    content_overlap = np.sum(prev_has_content & curr_has_content)
    content_union = np.sum(prev_has_content | curr_has_content)
    
    if content_union == 0:
        return False, 0.0
    
    overlap_ratio = content_overlap / content_union
    
    # 策略：如果两边都有内容，且重叠度 > 50%，认为是连续的
    has_similar_content = overlap_ratio > 0.5
    
    return has_similar_content, overlap_ratio

def check_boundary_continuity(prev_img_array, curr_img_array, config):
    """
    检查两张图片的边界是否连续（改进版）
    
    综合三个维度：
    1. 整体像素差异
    2. 内容对齐度
    3. 有内容的列占比
    
    参数:
        prev_img_array: 前一张图片的numpy数组
        curr_img_array: 当前图片的numpy数组
        config: 配置对象
    
    返回:
        is_continuous: 是否连续（True表示连续，不应断开）
        avg_diff: 平均像素差异
    """
    # 获取前一张图的底部区域
    prev_height = prev_img_array.shape[0]
    prev_bottom = prev_img_array[max(0, prev_height - config.scan_lines):prev_height]
    
    # 获取当前图的顶部区域
    curr_top = curr_img_array[0:min(config.scan_lines, curr_img_array.shape[0])]
    
    # 确保宽度一致
    min_width = min(prev_bottom.shape[1], curr_top.shape[1])
    prev_bottom = prev_bottom[:, :min_width]
    curr_top = curr_top[:, :min_width]
    
    # 维度1：整体像素差异
    diff = np.abs(prev_bottom.astype(int) - curr_top.astype(int))
    avg_diff = np.mean(diff)
    
    # 维度2：内容特征相似度检测
    has_similar_content, overlap_ratio = check_content_alignment(prev_img_array, curr_img_array, config)
    
    # 综合判断
    # 特殊情况：如果像素差异极小（<5）且没有相似内容特征，可能是两个纯色分割线
    if avg_diff < config.pure_color_diff and not has_similar_content:
        # 两个纯色分割线（如纯黑、纯白），应该断开
        is_continuous = False
    # 如果两边有相似的内容特征（如对话框边框、暗色场景），认为连续
    elif has_similar_content:
        is_continuous = True
    # 否则使用像素差异判断
    elif avg_diff < config.continuity_threshold:
        is_continuous = True
    else:
        is_continuous = False
    
    return is_continuous, avg_diff

def identify_panels(image_arrays, config, pbar=None):
    """识别哪些图片属于同一个分镜"""
    print(f"  识别分镜中 (扫描 {config.scan_lines} 行)...")
    
    panels = []
    current_panel = []
    prev_img_array = None
    
    for idx, img_array in enumerate(image_arrays):
        # 边界检测
        top_is_boundary, _, top_info, _ = is_panel_boundary_v2(
            img_array, config,
            check_top=True, 
            check_bottom=False
        )
        
        if idx == 0:
            # 第一张图，开始第一个分镜
            current_panel.append(idx)
            prev_img_array = img_array
        else:
            # 检查是否应该开始新分镜
            should_break = False
            
            if top_is_boundary:
                # 顶部有边界特征，检查是否应该断开
                # 检查连续性（内部已包含纯色检测）
                is_continuous, avg_diff = check_boundary_continuity(prev_img_array, img_array, config)
                
                if is_continuous:
                    # 判定为连续，不断开
                    current_panel.append(idx)
                    print(f"  [{idx+1:3d}/{len(image_arrays)}] 图片{idx}: 连续(差异{avg_diff:.1f}) 合并色{top_info['merged_colors']} 方差{top_info['variance']:.1f} 主色{top_info['dominant_ratio']:.0%}")
                else:
                    # 判定为不连续，断开
                    should_break = True
                    print(f"  [{idx+1:3d}/{len(image_arrays)}] 图片{idx}: 断开(差异{avg_diff:.1f}) → 分镜#{len(panels)+1} ({len(current_panel)}张) | 合并色{top_info['merged_colors']} 方差{top_info['variance']:.1f} 主色{top_info['dominant_ratio']:.0%}")
            else:
                # 顶部不是边界，继续当前分镜
                current_panel.append(idx)
            
            if should_break:
                # 保存上一个分镜，开始新分镜
                if current_panel:
                    panels.append(current_panel)
                current_panel = [idx]
            
            prev_img_array = img_array
        
        # 更新进度条
        if pbar is not None:
            pbar.update(1)
        
        # 最后一张图，保存当前分镜
        if idx == len(image_arrays) - 1 and current_panel:
            panels.append(current_panel)
    
    print(f"  识别完成！共 {len(panels)} 个分镜")
    return panels

def merge_panel_images(image_arrays, panel_indices):
    """将多张图片垂直拼接成一个分镜"""
    if not panel_indices:
        return None
    
    if len(panel_indices) == 1:
        return image_arrays[panel_indices[0]]
    
    img_arrays = [image_arrays[i] for i in panel_indices]
    merged = np.vstack(img_arrays)
    return merged

class SmartComicMerge:
    """智能漫画拼接节点"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scan_lines": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "color_similarity": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "merged_color_threshold": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "variance_threshold": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "dominant_ratio_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "continuity_threshold": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "pure_color_diff": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "merge_comics"
    CATEGORY = "hhy/image"
    
    def merge_comics(self, images, scan_lines, color_similarity, 
                     merged_color_threshold, variance_threshold,
                     dominant_ratio_threshold, continuity_threshold, 
                     pure_color_diff):
        
        print("="*70)
        print("智能漫画拼接系统 - ComfyUI节点")
        print("="*70)
        
        # 处理参数（当 INPUT_IS_LIST=True 时，参数都是列表）
        scan_lines = scan_lines[0] if isinstance(scan_lines, list) else scan_lines
        color_similarity = color_similarity[0] if isinstance(color_similarity, list) else color_similarity
        merged_color_threshold = merged_color_threshold[0] if isinstance(merged_color_threshold, list) else merged_color_threshold
        variance_threshold = variance_threshold[0] if isinstance(variance_threshold, list) else variance_threshold
        dominant_ratio_threshold = dominant_ratio_threshold[0] if isinstance(dominant_ratio_threshold, list) else dominant_ratio_threshold
        continuity_threshold = continuity_threshold[0] if isinstance(continuity_threshold, list) else continuity_threshold
        pure_color_diff = pure_color_diff[0] if isinstance(pure_color_diff, list) else pure_color_diff
        
        # 处理图片输入（将 tensor 转换为 numpy array）
        print("处理输入图片...")
        image_arrays = []
        
        if isinstance(images, list):
            # 如果是列表，处理每个元素
            for img_item in images:
                if img_item is None:
                    continue
                
                # 确保是批量格式
                if len(img_item.shape) == 3:
                    img_item = img_item.unsqueeze(0)
                
                # 处理批次中的每张图片
                for i in range(img_item.shape[0]):
                    single_img = img_item[i]  # [H, W, C]
                    # 转换为 PIL 再转为 numpy array (0-255 RGB)
                    img_pil = tensor2pil(single_img)
                    img_array = np.array(img_pil)
                    image_arrays.append(img_array)
        else:
            # 单个 tensor
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            for i in range(images.shape[0]):
                single_img = images[i]
                img_pil = tensor2pil(single_img)
                img_array = np.array(img_pil)
                image_arrays.append(img_array)
        
        if not image_arrays:
            print("没有有效的图片输入")
            return ([],)
        
        print(f"收到 {len(image_arrays)} 张图片")
        
        # 计算总步骤数：识别分镜
        total_steps = len(image_arrays)
        pbar = comfy.utils.ProgressBar(total_steps)
        
        # 配置参数
        config = Config()
        config.scan_lines = scan_lines
        config.color_similarity = color_similarity
        config.merged_color_threshold = merged_color_threshold
        config.variance_threshold = variance_threshold
        config.dominant_ratio_threshold = dominant_ratio_threshold
        config.continuity_threshold = continuity_threshold
        config.pure_color_diff = pure_color_diff
        
        print("\n算法配置:")
        print(f"  边界检测: 扫描{config.scan_lines}行 | 合并色≤{config.merged_color_threshold} 或 方差≤{config.variance_threshold} 或 主色≥{config.dominant_ratio_threshold:.0%}")
        print(f"  连续性: 像素差异<{config.continuity_threshold} | 纯色差异<{config.pure_color_diff}")
        print("="*70)
        
        # 识别分镜（带进度条）
        panels = identify_panels(image_arrays, config, pbar=pbar)
        
        if not panels:
            print("未识别到分镜!")
            # 确保进度条完成
            pbar.update_absolute(total_steps)
            return ([],)
        
        # 拼接分镜
        print(f"\n拼接 {len(panels)} 个分镜...")
        result_images = []
        
        for panel_idx, panel_indices in enumerate(panels, 1):
            merged_array = merge_panel_images(image_arrays, panel_indices)
            if merged_array is None:
                continue
            
            height, width = merged_array.shape[:2]
            print(f"  分镜{panel_idx:03d}: {len(panel_indices)}张图 {width}x{height}px (比例{height/width:.1f})")
            
            # 转换为tensor格式
            merged_pil = Image.fromarray(merged_array)
            merged_tensor = pil2tensor(merged_pil)
            result_images.append(merged_tensor)
        
        # 确保进度条完成
        pbar.update_absolute(total_steps)
        
        print(f"\n✓ 完成! 输出 {len(result_images)} 个分镜")
        print("="*70)
        
        return (result_images,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SmartComicMerge": SmartComicMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartComicMerge": "Smart Comic Merge",
}

