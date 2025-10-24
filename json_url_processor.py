import json
import re
from typing import List, Dict, Any, Optional

class JSONURLProcessor:
    """
    JSON URL处理器节点
    用于处理包含bbox2d字段的JSON数据，将其替换为image字段并填入对应的URL链接
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "输入的JSON数据，包含bbox2d字段"
                }),
                "url_list": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "URL链接列表，每行一个URL或用分号分隔"
                }),
            },
            "optional": {
                "field_name": ("STRING", {
                    "default": "bbox_2d",
                    "tooltip": "要替换的字段名（默认：bbox_2d）"
                }),
                "new_field_name": ("STRING", {
                    "default": "image",
                    "tooltip": "新字段名（默认：image）"
                }),
                "preserve_original": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否保留原始字段"
                }),
                "url_separator": ("STRING", {
                    "default": "\n",
                    "tooltip": "URL分隔符（默认：换行符）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("processed_json", "process_log", "processed_count", "url_count")
    FUNCTION = "process_json_with_urls"
    CATEGORY = "JSON Processing"
    OUTPUT_NODE = True

    def _parse_urls(self, url_input: str, separator: str = "\n") -> List[str]:
        """解析URL输入，使用正则表达式搜索所有https链接"""
        if not url_input.strip():
            return []
        
        import re
        
        # 使用正则表达式搜索所有https链接
        # 匹配https://开头的URL，直到遇到空格、换行或其他分隔符
        url_pattern = r'https://[^\s\n\r\t;]+'
        urls = re.findall(url_pattern, url_input)
        
        # 过滤和清理URL
        cleaned_urls = []
        for url in urls:
            # 移除末尾可能的标点符号
            url = url.rstrip('.,;!?')
            if url and len(url) > 10:  # 确保是有效的URL长度
                cleaned_urls.append(url)
        
        return cleaned_urls

    def _parse_json_data(self, json_input: str) -> List[Dict[str, Any]]:
        """解析JSON数据"""
        try:
            # 尝试解析为JSON
            data = json.loads(json_input)
            
            # 确保是列表格式
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON数据必须是对象或对象数组")
            
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"数据处理失败: {str(e)}")

    def _validate_bbox_field(self, item: Dict[str, Any], field_name: str) -> bool:
        """验证bbox字段是否存在且格式正确"""
        if field_name not in item:
            return False
        
        bbox_value = item[field_name]
        
        # 检查是否是列表且包含4个数字
        if isinstance(bbox_value, list) and len(bbox_value) == 4:
            return all(isinstance(x, (int, float)) for x in bbox_value)
        
        return False

    def process_json_with_urls(self, json_data: str, url_list: str, field_name: str = "bbox_2d", 
                             new_field_name: str = "image", preserve_original: bool = False, 
                             url_separator: str = "\n") -> tuple:
        """
        处理JSON数据，将bbox2d字段替换为image字段并填入URL
        """
        try:
            # 解析输入数据
            data = self._parse_json_data(json_data)
            urls = self._parse_urls(url_list, url_separator)
            
            process_logs = []
            processed_count = 0
            
            # 验证输入
            if not data:
                return ("", "错误: 没有有效的JSON数据", 0, 0)
            
            if not urls:
                return ("", "错误: 没有找到有效的https链接", 0, 0)
            
            # 统计包含目标字段的项目
            valid_items = [item for item in data if self._validate_bbox_field(item, field_name)]
            
            process_logs.append(f"找到 {len(urls)} 个https链接")
            process_logs.append(f"找到 {len(valid_items)} 个有效的{field_name}字段")
            
            if len(valid_items) != len(urls):
                process_logs.append(f"警告: bbox项目数量({len(valid_items)})与URL数量({len(urls)})不匹配")
            
            # 处理每个项目
            for i, item in enumerate(data):
                if self._validate_bbox_field(item, field_name):
                    if i < len(urls):
                        # 创建新的有序字典来保持字段顺序
                        new_item = {}
                        
                        # 先添加image字段
                        new_item[new_field_name] = urls[i]
                        
                        # 然后添加其他字段（除了要替换的字段）
                        for key, value in item.items():
                            if key != field_name:
                                new_item[key] = value
                        
                        # 如果保留原始字段，在最后添加
                        if preserve_original:
                            new_item[field_name] = item[field_name]
                        
                        # 替换原项目
                        data[i] = new_item
                        
                        processed_count += 1
                        process_logs.append(f"项目{i+1}: {field_name} -> {new_field_name} = {urls[i]}")
                    else:
                        process_logs.append(f"项目{i+1}: 跳过（没有对应的URL）")
                else:
                    process_logs.append(f"项目{i+1}: 跳过（没有有效的{field_name}字段）")
            
            # 生成处理后的JSON
            processed_json = json.dumps(data, indent=2, ensure_ascii=False)
            
            # 生成处理日志
            summary = f"处理完成: {processed_count}/{len(data)} 个项目，{len(urls)} 个URL"
            process_logs.insert(0, summary)
            log_text = "\n".join(process_logs)
            
            return (processed_json, log_text, processed_count, len(urls))
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            return ("", error_msg, 0, 0)

    @classmethod
    def IS_CHANGED(cls, json_data, url_list, field_name, new_field_name, preserve_original, url_separator):
        """检测输入是否发生变化"""
        return f"{json_data}_{url_list}_{field_name}_{new_field_name}_{preserve_original}_{url_separator}"


class JSONURLBatchProcessor:
    """
    JSON URL批量处理器节点
    支持批量处理多个JSON文件和URL列表
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_files": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "JSON文件路径列表，每行一个文件路径"
                }),
                "url_files": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "URL文件路径列表，每行一个文件路径"
                }),
            },
            "optional": {
                "field_name": ("STRING", {
                    "default": "bbox_2d",
                    "tooltip": "要替换的字段名"
                }),
                "new_field_name": ("STRING", {
                    "default": "image",
                    "tooltip": "新字段名"
                }),
                "output_prefix": ("STRING", {
                    "default": "processed",
                    "tooltip": "输出文件前缀"
                }),
                "preserve_original": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否保留原始字段"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("output_files", "process_log", "processed_files", "error_log")
    FUNCTION = "batch_process_json"
    CATEGORY = "JSON Processing"

    def batch_process_json(self, json_files: str, url_files: str, field_name: str = "bbox_2d",
                          new_field_name: str = "image", output_prefix: str = "processed",
                          preserve_original: bool = False) -> tuple:
        """
        批量处理JSON文件
        """
        import os
        
        try:
            # 解析文件路径
            json_paths = [path.strip() for path in json_files.split('\n') if path.strip()]
            url_paths = [path.strip() for path in url_files.split('\n') if path.strip()]
            
            if len(json_paths) != len(url_paths):
                return ("", f"错误: JSON文件数量({len(json_paths)})与URL文件数量({len(url_paths)})不匹配", 0, "")
            
            processor = JSONURLProcessor()
            output_files = []
            process_logs = []
            processed_count = 0
            errors = []
            
            for i, (json_path, url_path) in enumerate(zip(json_paths, url_paths)):
                try:
                    # 读取JSON文件
                    if not os.path.exists(json_path):
                        errors.append(f"JSON文件不存在: {json_path}")
                        continue
                    
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                    
                    # 读取URL文件
                    if not os.path.exists(url_path):
                        errors.append(f"URL文件不存在: {url_path}")
                        continue
                    
                    with open(url_path, 'r', encoding='utf-8') as f:
                        url_content = f.read()
                    
                    # 处理JSON
                    processed_json, log, count, url_count = processor.process_json_with_urls(
                        json_content, url_content, field_name, new_field_name, preserve_original
                    )
                    
                    if count > 0:
                        # 保存处理后的文件
                        output_path = f"{output_prefix}_{i+1}.json"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(processed_json)
                        
                        output_files.append(output_path)
                        processed_count += 1
                        process_logs.append(f"文件{i+1}: {os.path.basename(json_path)} -> {output_path} ({count}个项目)")
                    else:
                        errors.append(f"文件{i+1}: 没有处理任何项目")
                
                except Exception as e:
                    errors.append(f"文件{i+1}: 处理失败 - {str(e)}")
            
            # 生成结果
            output_files_str = "\n".join(output_files)
            process_log_str = "\n".join(process_logs)
            error_log_str = "\n".join(errors)
            
            summary = f"批量处理完成: {processed_count}/{len(json_paths)} 个文件"
            process_log_str = f"{summary}\n{process_log_str}"
            
            return (output_files_str, process_log_str, processed_count, error_log_str)
            
        except Exception as e:
            error_msg = f"批量处理失败: {str(e)}"
            return ("", error_msg, 0, error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "JSONURLProcessor": JSONURLProcessor,
    "JSONURLBatchProcessor": JSONURLBatchProcessor,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONURLProcessor": "JSON URL Processor",
    "JSONURLBatchProcessor": "JSON URL Batch Processor",
}
