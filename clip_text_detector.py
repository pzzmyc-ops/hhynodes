import numpy as np
from PIL import Image
import torch
import comfy.utils
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import requests
import os
import tempfile
import shutil
import subprocess
import sys

# ModelScope 相关导入
try:
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("警告: ModelScope 未安装，请使用 'pip install modelscope' 安装以启用 ModelScope 功能")

# HuggingFace Hub 相关导入
try:
    from huggingface_hub import hf_hub_download, HfApi, hf_hub_url
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("警告: huggingface_hub 未安装，请使用 'pip install huggingface_hub' 安装")

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    # ComfyUI标准图像tensor格式: BHWC (batch, height, width, channels)
    
    if len(image.shape) == 4:
        # 4D tensor BHWC: [B, H, W, C]
        if image.shape[3] == 3 or image.shape[3] == 4:
            # BHWC格式: [B, H, W, C] - 这是ComfyUI标准格式
            image = image.squeeze(0)  # 去掉batch维度，变成HWC
        elif image.shape[1] == 3 or image.shape[1] == 4:
            # BCHW格式: [B, C, H, W] - 需要转换
            image = image.squeeze(0)  # 去掉batch维度
            image = image.permute(1, 2, 0)  # CHW -> HWC
        else:
            # 其他4D格式，假设是BHWC
            image = image.squeeze(0)  # 去掉batch维度
    elif len(image.shape) == 3:
        # 3D tensor，可能是CHW或HWC
        if image.shape[0] == 3 or image.shape[0] == 4:
            # CHW格式: [C, H, W]
            image = image.permute(1, 2, 0)  # CHW -> HWC
        # 否则已经是HWC格式
    
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """将PIL图像转换为tensor"""
    img_array = np.array(image).astype(np.float32) / 255.0
    # ComfyUI标准格式: BHWC (batch, height, width, channels)
    # img_array已经是HWC格式，只需要添加batch维度
    return torch.from_numpy(img_array).unsqueeze(0)

def get_model_required_files(model_name):
    """使用transformers库获取模型需要的文件列表"""
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoProcessor
        
        # 获取模型配置
        config = AutoConfig.from_pretrained(model_name)
        
        # 获取模型需要的文件
        required_files = []
        
        # 基础文件
        required_files.extend([
            "config.json",
            "tokenizer_config.json", 
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json"
        ])
        
        # 根据模型类型添加特定文件
        if hasattr(config, 'model_type'):
            if config.model_type == 'clip':
                # CLIP模型特定文件
                required_files.extend([
                    "preprocessor_config.json",
                    "tokenizer.json"
                ])
        
        # 检查权重文件格式偏好
        try:
            # 尝试获取safetensors文件
            from huggingface_hub import list_repo_files
            repo_files = list_repo_files(model_name)
            
            # 优先选择safetensors文件
            safetensors_files = [f for f in repo_files if f.endswith('.safetensors')]
            bin_files = [f for f in repo_files if f.endswith('.bin')]
            
            if safetensors_files:
                required_files.extend(safetensors_files)
                print(f"找到 {len(safetensors_files)} 个 .safetensors 文件")
            elif bin_files:
                required_files.extend(bin_files)
                print(f"找到 {len(bin_files)} 个 .bin 文件")
                
        except Exception as e:
            print(f"获取权重文件列表失败: {e}")
        
        # 过滤掉不存在的文件
        try:
            from huggingface_hub import list_repo_files
            available_files = list_repo_files(model_name)
            required_files = [f for f in required_files if f in available_files]
        except:
            pass
            
        print(f"模型 {model_name} 需要的文件:")
        for f in required_files:
            print(f"  - {f}")
            
        return required_files
        
    except Exception as e:
        print(f"获取模型文件列表失败: {e}")
        return []

def download_required_files(model_name, required_files, cache_dir=None):
    """下载模型需要的文件"""
    if not required_files:
        print("没有需要下载的文件")
        return None
        
    # 设置HuggingFace Mirror镜像地址
    hf_endpoint = "https://hf-mirror.com"
    
    # 尝试使用ModelScope下载
    if MODELSCOPE_AVAILABLE:
        try:
            print("正在尝试从 ModelScope 下载文件...")
            local_dir = cache_dir if cache_dir else f"./{model_name.split('/')[-1]}"
            os.makedirs(local_dir, exist_ok=True)
            
            for file_name in required_files:
                print(f"正在下载: {file_name}")
                try:
                    file_path = model_file_download(
                        model_id=model_name,
                        file_path=file_name,
                        cache_dir=local_dir
                    )
                    print(f"下载完成: {file_name}")
                except Exception as file_error:
                    print(f"下载文件 {file_name} 失败: {file_error}")
                    continue
            
            print("ModelScope 下载完成")
            return local_dir
            
        except Exception as e:
            print(f"ModelScope 下载失败: {e}")
            print("回退到 HuggingFace Mirror...")
    
    # 回退到HuggingFace Mirror
    if HF_HUB_AVAILABLE:
        try:
            print("正在从 HuggingFace Mirror 下载文件...")
            local_dir = cache_dir if cache_dir else f"./{model_name.split('/')[-1]}"
            os.makedirs(local_dir, exist_ok=True)
            
            for file_name in required_files:
                print(f"正在下载: {file_name}")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file_name,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        endpoint=hf_endpoint,
                        repo_type="model"
                    )
                    print(f"下载完成: {file_name}")
                except Exception as file_error:
                    print(f"下载文件 {file_name} 失败: {file_error}")
                    continue
            
            print("HuggingFace Mirror 下载完成")
            return local_dir
            
        except Exception as e:
            print(f"HuggingFace Mirror 下载失败: {e}")
    
    # 如果都失败了，抛出异常
    raise Exception("所有下载方式都失败了，请检查网络连接和模型名称")

def show_download_progress():
    """显示下载进度信息"""
    print("=" * 60)
    print("正在下载CLIP模型文件...")
    print("使用transformers库智能识别需要的文件")
    print("优先使用 ModelScope，失败时回退到 HuggingFace Mirror")
    print("只下载必要的文件，避免下载多余文件")
    print("=" * 60)

class CLIPFilter:
    """使用Hugging Face CLIP模型进行图像-文本对比"""
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "positive_prompts": ("STRING", {
                    "multiline": True,
                    "default": "text\ndialog\nspeech bubble\nwords\nletters",
                    "display": "multiline"
                }),
                "negative_prompts": ("STRING", {
                    "multiline": True,
                    "default": "comic panel\ncharacter\nperson\nmanga frame",
                    "display": "multiline"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "number"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": True,
                    "display": "boolean"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("filtered_images", "excluded_images", "all_images")
    OUTPUT_IS_LIST = (True, True, True)
    INPUT_IS_LIST = True
    FUNCTION = "filter_images"
    CATEGORY = "hhy/clip"
    
    def _load_model(self):
        """加载CLIP模型"""
        if self.model is None or self.processor is None:
            if hasattr(self, '_model_loading'):
                return False
            
            self._model_loading = True
            try:
                model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                print("正在加载Hugging Face CLIP模型...")
                print(f"模型名称: {model_name}")
                
                # 显示下载进度信息
                show_download_progress()
                
                # 首先尝试使用精确下载方法
                try:
                    # 获取模型需要的文件列表
                    required_files = get_model_required_files(model_name)
                    
                    if required_files:
                        # 下载需要的文件
                        model_path = download_required_files(model_name, required_files)
                        print(f"模型文件下载完成，路径: {model_path}")
                        
                        # 从本地路径加载模型
                        self.model = CLIPModel.from_pretrained(
                            model_path,
                            local_files_only=True
                        )
                        print("模型加载完成，正在加载处理器...")
                        
                        self.processor = CLIPProcessor.from_pretrained(
                            model_path,
                            local_files_only=True
                        )
                    else:
                        raise Exception("无法获取模型文件列表")
                    
                except Exception as download_error:
                    print(f"精确下载失败: {download_error}")
                    print("回退到标准transformers下载...")
                    
                    # 回退到标准方法
                    self.model = CLIPModel.from_pretrained(
                        model_name,
                        resume_download=True,
                        local_files_only=False
                    )
                    print("模型下载完成，正在加载处理器...")
                    
                    self.processor = CLIPProcessor.from_pretrained(
                        model_name,
                        resume_download=True,
                        local_files_only=False
                    )
                
                print("CLIP模型加载完成!")
                return True
            except Exception as e:
                print(f"CLIP模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                self._model_loading = False
        return True
    
    def filter_images(self, images, positive_prompts, negative_prompts, confidence_threshold, debug_output):
        
        # 处理列表输入，取第一个元素（除了images保持列表）
        positive_prompts = positive_prompts[0] if isinstance(positive_prompts, list) else positive_prompts
        negative_prompts = negative_prompts[0] if isinstance(negative_prompts, list) else negative_prompts
        confidence_threshold = confidence_threshold[0] if isinstance(confidence_threshold, list) else confidence_threshold
        debug_output = debug_output[0] if isinstance(debug_output, list) else debug_output
        
        # 解析正面和负面提示词
        if isinstance(positive_prompts, str):
            # 先按换行符分割，然后按逗号分割
            positive_list = []
            for line in positive_prompts.split('\n'):
                if line.strip():
                    # 如果行中包含逗号，按逗号分割
                    if ',' in line:
                        positive_list.extend([item.strip() for item in line.split(',') if item.strip()])
                    else:
                        positive_list.append(line.strip())
        else:
            positive_list = positive_prompts if isinstance(positive_prompts, list) else []
        
        if isinstance(negative_prompts, str):
            # 先按换行符分割，然后按逗号分割
            negative_list = []
            for line in negative_prompts.split('\n'):
                if line.strip():
                    # 如果行中包含逗号，按逗号分割
                    if ',' in line:
                        negative_list.extend([item.strip() for item in line.split(',') if item.strip()])
                    else:
                        negative_list.append(line.strip())
        else:
            negative_list = negative_prompts if isinstance(negative_prompts, list) else []
        
        if debug_output:
            print("=" * 60)
            print("Hugging Face CLIP图片筛选器开始工作")
            print(f"正面提示词: {positive_list}")
            print(f"负面提示词: {negative_list}")
            print(f"置信度阈值: {confidence_threshold}")
            print(f"调试输出: {'开启' if debug_output else '关闭'}")
            print("=" * 60)
        
        # 加载CLIP模型
        if not self._load_model():
            print("CLIP模型加载失败，返回所有图像")
            return self._fallback_filter(images)
        
        # 处理图像
        image_tensors = []
        
        if isinstance(images, list):
            for img_item in images:
                if img_item is None:
                    continue
                
                # ComfyUI图像tensor格式: BHWC (batch, height, width, channels)
                if len(img_item.shape) == 3:
                    # 单个图像 HWC，添加batch维度变成 BHWC
                    if debug_output:
                        print(f"处理列表项: 原始形状 {img_item.shape} -> 添加batch维度")
                    img_item = img_item.unsqueeze(0)
                
                # 现在img_item是BHWC格式
                if debug_output:
                    print(f"batch处理后形状: {img_item.shape}")
                
                for i in range(img_item.shape[0]):
                    single_img = img_item[i]  # 获取单个图像 HWC
                    if debug_output:
                        print(f"  提取图像 {i+1}: tensor形状 {single_img.shape}")
                    
                    # 添加batch维度用于CLIP处理
                    img_tensor = single_img.unsqueeze(0)  # HWC -> BHWC
                    if debug_output:
                        print(f"  准备CLIP处理: tensor形状 {img_tensor.shape}")
                    
                    image_tensors.append(img_tensor)
        else:
            # 单个图像tensor
            if len(images.shape) == 3:
                # HWC格式，添加batch维度
                if debug_output:
                    print(f"处理单个tensor: 原始形状 {images.shape} -> 添加batch维度")
                images = images.unsqueeze(0)
            
            # 现在images是BHWC格式
            if debug_output:
                print(f"batch处理后形状: {images.shape}")
            
            for i in range(images.shape[0]):
                single_img = images[i]  # 获取单个图像 HWC
                if debug_output:
                    print(f"  提取图像 {i+1}: tensor形状 {single_img.shape}")
                
                # 添加batch维度用于CLIP处理
                img_tensor = single_img.unsqueeze(0)  # HWC -> BHWC
                if debug_output:
                    print(f"  准备CLIP处理: tensor形状 {img_tensor.shape}")
                
                image_tensors.append(img_tensor)
        
        if not image_tensors:
            if debug_output:
                print("没有找到图像，返回空结果")
            return ([], [], [])
        
        if debug_output:
            print("-" * 60)
            print(f"图像处理完成:")
            print(f"  总处理图像数: {len(image_tensors)}")
            print("-" * 60)
        
        # 使用Hugging Face CLIP检测图像
        filtered_images = []
        excluded_images = []
        all_images = []
        
        matched_count = 0
        excluded_count = 0
        
        for i, img_tensor in enumerate(image_tensors):
            if debug_output:
                print(f"处理第 {i+1} 张图像:")
            
            matches_criteria, confidence = self._huggingface_clip_detection(
                img_tensor, positive_list, negative_list, confidence_threshold, debug_output
            )
            
            # 添加到对应的列表
            all_images.append(img_tensor)
            
            if matches_criteria:
                filtered_images.append(img_tensor)
                matched_count += 1
            else:
                excluded_images.append(img_tensor)
                excluded_count += 1
        
        if debug_output:
            print("=" * 60)
            print("筛选统计:")
            print(f"总图像数: {len(image_tensors)}")
            print(f"符合条件的图像数: {matched_count}")
            print(f"排除的图像数: {excluded_count}")
            print(f"符合条件比例: {matched_count/len(image_tensors)*100:.1f}%")
            print("=" * 60)
            print("输出结果:")
            print(f"  filtered_images: {len(filtered_images)} 张")
            print(f"  excluded_images: {len(excluded_images)} 张")
            print(f"  all_images: {len(all_images)} 张")
            print("=" * 60)
        
        return (filtered_images, excluded_images, all_images)
    
    def _huggingface_clip_detection(self, img_tensor, positive_list, negative_list, confidence_threshold, debug_output):
        """使用Hugging Face CLIP模型进行图像检测"""
        try:
            # 将tensor转换为PIL图像
            img_pil = tensor2pil(img_tensor.squeeze(0))
            
            if debug_output:
                print(f"图像尺寸: {img_pil.size}")
            
            # 合并所有提示词
            all_prompts = positive_list + negative_list
            
            if not all_prompts:
                if debug_output:
                    print("警告: 没有提供任何提示词，使用默认提示词")
                all_prompts = [
                    "a photo containing text",
                    "a photo with words",
                    "a photo with letters",
                    "a photo with writing",
                    "a photo without text",
                    "a photo without words",
                    "a photo without letters",
                    "a photo without writing"
                ]
                positive_list = all_prompts[:4]
                negative_list = all_prompts[4:]
            
            if debug_output:
                print(f"正面提示词: {positive_list}")
                print(f"负面提示词: {negative_list}")
            
            # 使用CLIP模型进行图像-文本对比
            inputs = self.processor(
                text=all_prompts, 
                images=img_pil, 
                return_tensors="pt", 
                padding=True
            )
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # 图像-文本相似度分数
                probs = logits_per_image.softmax(dim=1)  # 转换为概率
            
            if debug_output:
                print(f"CLIP输出形状: {logits_per_image.shape}")
                print(f"概率形状: {probs.shape}")
                print(f"原始相似度分数: {logits_per_image[0].tolist()}")
                print(f"概率分数: {probs[0].tolist()}")
            
            # 计算正面和负面概率
            positive_probs = probs[0][:len(positive_list)].sum().item()
            negative_probs = probs[0][len(positive_list):].sum().item()
            
            # 判断是否符合条件
            # 新的逻辑：只要正面概率足够高，且负面概率足够低，就保留
            positive_threshold = 0.3  # 正面概率阈值
            negative_threshold = 0.7  # 负面概率阈值（越低越好）
            
            matches_criteria = (positive_probs >= positive_threshold) and (negative_probs <= negative_threshold)
            confidence = positive_probs  # 使用正面概率作为置信度
            
            if debug_output:
                print(f"正面概率: {positive_probs:.4f} (阈值: {positive_threshold})")
                print(f"负面概率: {negative_probs:.4f} (阈值: {negative_threshold})")
                print(f"正面条件: {positive_probs >= positive_threshold}")
                print(f"负面条件: {negative_probs <= negative_threshold}")
                print(f"检测结果: {'符合条件' if matches_criteria else '不符合条件'}")
                print(f"置信度: {confidence:.4f}")
                print("-" * 50)
            
            return matches_criteria, confidence
            
        except Exception as e:
            print(f"Hugging Face CLIP检测错误: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.5
    
    def _fallback_filter(self, images):
        """当CLIP模型不可用时的备用方法"""
        print("Using fallback method - returning all images as filtered")
        if isinstance(images, list):
            return (images, [], images)
        else:
            return ([images], [], [images])

NODE_CLASS_MAPPINGS = {
    "HuggingFaceCLIPFilter": CLIPFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceCLIPFilter": "CLIP Filter",
}