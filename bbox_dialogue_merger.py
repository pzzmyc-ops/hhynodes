import json
import ast
from typing import List, Dict, Any, Tuple


def parse_json_input(json_str: str) -> Any:
    """解析 JSON 字符串"""
    json_str = json_str.strip()
    
    # 移除代码块标记
    if "```json" in json_str:
        json_str = json_str.split("```json", 1)[1]
        json_str = json_str.split("```", 1)[0]
    elif "```" in json_str:
        json_str = json_str.split("```", 1)[1]
        json_str = json_str.split("```", 1)[0]
    
    try:
        return json.loads(json_str)
    except Exception:
        try:
            return ast.literal_eval(json_str)
        except Exception:
            return []


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """计算两个 bbox 的交并比 (IoU)
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU 值 (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def calculate_overlap_ratio(bbox1: List[int], bbox2: List[int]) -> float:
    """计算 bbox1 与 bbox2 的重叠比例（基于 bbox1 的面积）
    
    Args:
        bbox1: [x1, y1, x2, y2] - 参考 bbox
        bbox2: [x1, y1, x2, y2] - 目标 bbox
    
    Returns:
        float: 重叠比例 (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if bbox1_area == 0:
        return 0.0
    
    return intersection_area / bbox1_area


def get_bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """获取 bbox 的中心点"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def calculate_distance(bbox1: List[int], bbox2: List[int]) -> float:
    """计算两个 bbox 中心点的欧氏距离"""
    center1 = get_bbox_center(bbox1)
    center2 = get_bbox_center(bbox2)
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5


def find_best_match(dialogue_bbox: List[int], image_bboxes: List[Dict], 
                     iou_threshold: float = 0.1, overlap_threshold: float = 0.3,
                     log_details: List[str] = None) -> int:
    """为对话 bbox 找到最佳匹配的图片 bbox
    
    Args:
        dialogue_bbox: 对话框的 bbox
        image_bboxes: 图片 bbox 列表
        iou_threshold: IoU 阈值
        overlap_threshold: 重叠比例阈值
        log_details: 日志详情列表
    
    Returns:
        int: 最佳匹配的图片 bbox 索引，-1 表示使用距离匹配
    """
    best_match_idx = -1
    best_iou = 0.0
    best_overlap = 0.0
    best_iou_idx = -1
    best_overlap_idx = -1
    
    if log_details is not None:
        log_details.append(f"    Dialogue bbox: {dialogue_bbox}")
        log_details.append(f"    Trying to match with {len(image_bboxes)} image bboxes...")
    
    for idx, img_item in enumerate(image_bboxes):
        img_bbox = img_item["bbox_2d"]
        
        # 计算 IoU
        iou = calculate_iou(dialogue_bbox, img_bbox)
        if iou > best_iou:
            best_iou = iou
            best_iou_idx = idx
        
        # 计算重叠比例
        overlap = calculate_overlap_ratio(dialogue_bbox, img_bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best_overlap_idx = idx
        
        if log_details is not None:
            log_details.append(f"      Image bbox {idx+1}: {img_bbox}")
            log_details.append(f"        → IoU: {iou:.4f}, Overlap: {overlap:.4f}")
    
    # 如果 IoU 或重叠比例超过阈值，认为匹配成功
    iou_passed = best_iou >= iou_threshold
    overlap_passed = best_overlap >= overlap_threshold
    
    if log_details is not None:
        log_details.append(f"    Best IoU: {best_iou:.4f} (threshold: {iou_threshold:.4f}) → {'PASS' if iou_passed else 'FAIL'}")
        log_details.append(f"    Best Overlap: {best_overlap:.4f} (threshold: {overlap_threshold:.4f}) → {'PASS' if overlap_passed else 'FAIL'}")
    
    if iou_passed or overlap_passed:
        # 选择 IoU 更大的那个作为最佳匹配
        best_match_idx = best_iou_idx if best_iou >= best_overlap else best_overlap_idx
        if log_details is not None:
            match_reason = "IoU" if best_iou >= best_overlap else "Overlap"
            log_details.append(f"    ✓ Matched to image bbox {best_match_idx+1} (by {match_reason})")
        return best_match_idx
    
    # 否则返回 -1，使用距离匹配
    if log_details is not None:
        log_details.append(f"    ✗ No match found, will use distance fallback")
    return -1


def find_nearest_bbox(dialogue_bbox: List[int], image_bboxes: List[Dict],
                       log_details: List[str] = None) -> int:
    """找到距离最近的图片 bbox（用于纯对话框）"""
    min_distance = float('inf')
    nearest_idx = 0
    
    if log_details is not None:
        log_details.append(f"    Finding nearest bbox by distance...")
    
    for idx, img_item in enumerate(image_bboxes):
        img_bbox = img_item["bbox_2d"]
        distance = calculate_distance(dialogue_bbox, img_bbox)
        
        if log_details is not None:
            log_details.append(f"      Distance to image bbox {idx+1}: {distance:.2f}")
        
        if distance < min_distance:
            min_distance = distance
            nearest_idx = idx
    
    if log_details is not None:
        log_details.append(f"    ✓ Nearest is image bbox {nearest_idx+1} (distance: {min_distance:.2f})")
    
    return nearest_idx


def merge_dialogue_to_bboxes(long_json: List[Dict], short_json: List[Dict],
                              iou_threshold: float = 0.1, 
                              overlap_threshold: float = 0.3,
                              use_distance_fallback: bool = True,
                              log_details: List[str] = None) -> List[Dict]:
    """将长 JSON 的对话内容合并到短 JSON 的 bbox 中
    
    Args:
        long_json: 包含 bbox_2d 和 dialogue 的列表
        short_json: 只包含精确 bbox_2d 的列表
        iou_threshold: IoU 匹配阈值
        overlap_threshold: 重叠比例阈值
        use_distance_fallback: 如果没有匹配，是否使用距离最近的 bbox
        log_details: 日志详情列表
    
    Returns:
        List[Dict]: 合并后的结果
    """
    # 初始化结果，复制短 JSON 的 bbox，添加空的 dialogue 列表
    result = []
    for item in short_json:
        result.append({
            "bbox_2d": item["bbox_2d"],
            "dialogue": []
        })
    
    # 统计信息
    matched_count = 0
    distance_matched_count = 0
    skipped_count = 0
    
    if log_details is not None:
        log_details.append(f"\n=== Matching Process (Threshold: IoU={iou_threshold:.2f}, Overlap={overlap_threshold:.2f}) ===\n")
    
    # 遍历长 JSON 中的每个对话
    for idx, long_item in enumerate(long_json):
        long_bbox = long_item.get("bbox_2d")
        dialogues = long_item.get("dialogue", [])
        
        if not long_bbox or not dialogues:
            continue
        
        if log_details is not None:
            dialogue_preview = dialogues[0].get("dialogue", "")[:50] if dialogues else ""
            log_details.append(f"  Long JSON item {idx+1}/{len(long_json)}: \"{dialogue_preview}...\"")
        
        # 找到最佳匹配的短 JSON bbox
        match_idx = find_best_match(long_bbox, result, iou_threshold, overlap_threshold, log_details)
        
        if match_idx >= 0:
            # 有匹配的 bbox
            result[match_idx]["dialogue"].extend(dialogues)
            matched_count += 1
        elif use_distance_fallback:
            # 没有匹配，使用距离最近的 bbox
            nearest_idx = find_nearest_bbox(long_bbox, result, log_details)
            result[nearest_idx]["dialogue"].extend(dialogues)
            distance_matched_count += 1
        else:
            # 不使用距离回退，跳过此对话
            skipped_count += 1
            if log_details is not None:
                log_details.append(f"    ⚠ Skipped (distance fallback disabled)")
        
        if log_details is not None:
            log_details.append("")
    
    return result, matched_count, distance_matched_count, skipped_count


class BboxDialogueMergerNode:
    """合并两个 JSON 的对话和 bbox"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "long_json": ("STRING", {
                    "multiline": True, 
                    "default": "[]",
                    "tooltip": "包含 bbox_2d 和 dialogue 的 JSON"
                }),
                "short_json": ("STRING", {
                    "multiline": True, 
                    "default": "[]",
                    "tooltip": "只包含精确 bbox_2d 的 JSON"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "IoU 匹配阈值，越大越严格"
                }),
                "overlap_threshold": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "重叠比例阈值，越大越严格"
                }),
                "use_distance_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "如果没有匹配的 bbox，是否使用距离最近的 bbox"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("merged_json", "log")
    FUNCTION = "merge_dialogues"
    CATEGORY = "hhy/qwen3"
    
    def merge_dialogues(self, long_json, short_json, iou_threshold=0.1, 
                       overlap_threshold=0.3, use_distance_fallback=True):
        """合并对话到 bbox"""
        
        log_messages = []
        
        # 解析输入的 JSON
        try:
            long_data = parse_json_input(long_json)
            if not isinstance(long_data, list):
                long_data = [long_data] if long_data else []
        except Exception as e:
            return (json.dumps([], indent=2), f"Error parsing long_json: {str(e)}")
        
        try:
            short_data = parse_json_input(short_json)
            if not isinstance(short_data, list):
                short_data = [short_data] if short_data else []
        except Exception as e:
            return (json.dumps([], indent=2), f"Error parsing short_json: {str(e)}")
        
        log_messages.append(f"=== Bbox Dialogue Merger ===")
        log_messages.append(f"Long JSON items: {len(long_data)}")
        log_messages.append(f"Short JSON items (image bboxes): {len(short_data)}")
        log_messages.append(f"IoU threshold: {iou_threshold}")
        log_messages.append(f"Overlap threshold: {overlap_threshold}")
        log_messages.append(f"Use distance fallback: {use_distance_fallback}")
        log_messages.append("")
        
        # 统计长 JSON 中的对话数量
        total_dialogues = sum(len(item.get("dialogue", [])) for item in long_data)
        log_messages.append(f"Total dialogues in long JSON: {total_dialogues}")
        
        # 创建详细日志列表
        detail_logs = []
        
        # 执行合并
        result, matched_count, distance_matched_count, skipped_count = merge_dialogue_to_bboxes(
            long_data, short_data, iou_threshold, overlap_threshold, use_distance_fallback, detail_logs
        )
        
        # 添加详细日志到主日志
        log_messages.extend(detail_logs)
        
        # 统计结果
        log_messages.append(f"\n=== Merge Results Summary ===")
        log_messages.append(f"Matched by overlap (IoU/Overlap): {matched_count}/{len(long_data)}")
        log_messages.append(f"Matched by distance: {distance_matched_count}/{len(long_data)}")
        if skipped_count > 0:
            log_messages.append(f"Skipped (no fallback): {skipped_count}/{len(long_data)}")
        log_messages.append("")
        
        # 详细信息
        log_messages.append(f"=== Detailed Results ===")
        for idx, item in enumerate(result):
            bbox = item["bbox_2d"]
            dialogues = item["dialogue"]
            dialogue_count = len(dialogues)
            
            log_messages.append(f"Bbox {idx + 1}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            log_messages.append(f"  Dialogues: {dialogue_count}")
            
            for d_idx, dialogue in enumerate(dialogues):
                speaker = dialogue.get("speaker", "Unknown")
                text = dialogue.get("dialogue", "")
                log_messages.append(f"    {d_idx + 1}. {speaker}: {text[:50]}{'...' if len(text) > 50 else ''}")
            log_messages.append("")
        
        # 验证：确保所有对话都被分配
        result_dialogue_count = sum(len(item.get("dialogue", [])) for item in result)
        if result_dialogue_count == total_dialogues:
            log_messages.append(f"✓ All {total_dialogues} dialogues have been assigned successfully!")
        else:
            log_messages.append(f"⚠ Warning: {total_dialogues - result_dialogue_count} dialogues may be missing!")
        
        # 生成 JSON 输出
        output_json = json.dumps(result, ensure_ascii=False, indent=2)
        final_log = "\n".join(log_messages)
        
        return (output_json, final_log)


NODE_CLASS_MAPPINGS = {
    "BboxDialogueMerger": BboxDialogueMergerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDialogueMerger": "Bbox Dialogue Merger",
}

