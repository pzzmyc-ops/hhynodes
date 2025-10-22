import numpy as np
from PIL import Image
import torch
import comfy.utils
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

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

def resize_to_width(img_array, target_width):
    h, w = img_array.shape[:2]
    if w == target_width:
        return img_array
    
    scale = target_width / w
    new_height = int(h * scale)
    
    img_pil = Image.fromarray(img_array)
    resized_pil = img_pil.resize((target_width, new_height), Image.Resampling.LANCZOS)
    return np.array(resized_pil)

def merge_images_vertically(img1_array, img2_array):
    if img1_array is None or img2_array is None:
        return None
    
    h1, w1 = img1_array.shape[:2]
    h2, w2 = img2_array.shape[:2]
    
    target_width = max(w1, w2)
    
    if w1 != target_width:
        img1_array = resize_to_width(img1_array, target_width)
    
    if w2 != target_width:
        img2_array = resize_to_width(img2_array, target_width)
    
    merged = np.vstack([img1_array, img2_array])
    return merged

def calculate_image_features_clip(img_array, clip_model, clip_processor, positive_prompts, negative_prompts, debug_output=False):
    """使用CLIP模型检测图像特征"""
    try:
        # 将numpy数组转换为PIL图像
        img_pil = Image.fromarray(img_array)
        
        # 合并正面和负面提示词
        all_prompts = positive_prompts + negative_prompts
        
        if not all_prompts or len(all_prompts) < 2:
            if debug_output:
                print("警告: 没有提供足够的提示词，使用默认提示词")
            # 使用默认提示词
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
            positive_prompts = all_prompts[:4]
            negative_prompts = all_prompts[4:]
        
        if debug_output:
            print(f"正面提示词数量: {len(positive_prompts)}")
            print(f"正面提示词: {positive_prompts}")
            print(f"负面提示词数量: {len(negative_prompts)}")
            print(f"负面提示词: {negative_prompts}")
        
        # 使用CLIP处理图像和文本
        inputs = clip_processor(text=all_prompts, images=img_pil, return_tensors="pt", padding=True)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = F.softmax(logits_per_image, dim=1)
        
        # 计算正面和负面概率
        positive_probs = probs[0][:len(positive_prompts)].sum().item()
        negative_probs = probs[0][len(positive_prompts):].sum().item()
        
        # 判断是否符合条件
        matches_criteria = positive_probs > negative_probs
        confidence = max(positive_probs, negative_probs)
        
        if debug_output:
            print(f"正面概率: {positive_probs:.4f}")
            print(f"负面概率: {negative_probs:.4f}")
            print(f"检测结果: {'符合条件' if matches_criteria else '不符合条件'}")
            print(f"置信度: {confidence:.4f}")
            print("-" * 50)
        
        return {
            'matches_criteria': matches_criteria,
            'confidence': confidence,
            'positive_probability': positive_probs,
            'negative_probability': negative_probs
        }
        
    except Exception as e:
        print(f"CLIP image detection error: {e}")
        # 如果CLIP检测失败，返回默认值
        return {
            'matches_criteria': False,
            'confidence': 0.5,
            'positive_probability': 0.5,
            'negative_probability': 0.5
        }

def check_image_criteria_clip(img_array, clip_model, clip_processor, positive_prompts, negative_prompts, confidence_threshold=0.6, debug_output=False):
    """使用CLIP模型检测图像是否符合条件"""
    features = calculate_image_features_clip(img_array, clip_model, clip_processor, positive_prompts, negative_prompts, debug_output)
    
    # 基于CLIP的置信度判断
    matches_criteria = features['matches_criteria'] and features['confidence'] > confidence_threshold
    
    if debug_output:
        print(f"最终判断: {'符合条件' if matches_criteria else '不符合条件'} (阈值: {confidence_threshold})")
        print("=" * 60)
    
    return matches_criteria, features

class CLIPImageFilter:
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """加载CLIP模型"""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
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
    CATEGORY = "hhy/image"
    
    def filter_images(self, images, positive_prompts, negative_prompts, confidence_threshold, debug_output):
        
        confidence_threshold = confidence_threshold[0] if isinstance(confidence_threshold, list) else confidence_threshold
        positive_prompts = positive_prompts[0] if isinstance(positive_prompts, list) else positive_prompts
        negative_prompts = negative_prompts[0] if isinstance(negative_prompts, list) else negative_prompts
        debug_output = debug_output[0] if isinstance(debug_output, list) else debug_output
        
        # 解析正面和负面提示词
        if isinstance(positive_prompts, str):
            positive_list = [line.strip() for line in positive_prompts.split('\n') if line.strip()]
        else:
            positive_list = positive_prompts if isinstance(positive_prompts, list) else []
        
        if isinstance(negative_prompts, str):
            negative_list = [line.strip() for line in negative_prompts.split('\n') if line.strip()]
        else:
            negative_list = negative_prompts if isinstance(negative_prompts, list) else []
        
        if debug_output:
            print("=" * 60)
            print("CLIP图片筛选器开始工作")
            print(f"正面提示词: {positive_list}")
            print(f"负面提示词: {negative_list}")
            print(f"置信度阈值: {confidence_threshold}")
            print(f"调试输出: {'开启' if debug_output else '关闭'}")
            print("=" * 60)
            print(f"输入类型: {type(images)}")
            if isinstance(images, list):
                print(f"输入列表长度: {len(images)}")
                for idx, img_item in enumerate(images):
                    if img_item is not None:
                        print(f"  列表项 {idx}: tensor形状 {img_item.shape}")
                    else:
                        print(f"  列表项 {idx}: None")
            else:
                print(f"输入tensor形状: {images.shape}")
            print("=" * 60)
        
        # 检查CLIP模型是否加载成功
        if self.clip_model is None or self.clip_processor is None:
            print("CLIP model not loaded, falling back to returning all images")
            return self._fallback_filter(images)
        
        image_arrays = []
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
                    
                    img_pil = tensor2pil(single_img)
                    if debug_output:
                        print(f"  转换PIL: 尺寸 {img_pil.size}")
                    
                    img_array = np.array(img_pil)
                    if debug_output:
                        print(f"  转换numpy: 形状 {img_array.shape}")
                    
                    # 转换回正确的tensor格式 (BHWC)
                    img_tensor = pil2tensor(img_pil)
                    if debug_output:
                        print(f"  转换tensor: 形状 {img_tensor.shape}")
                    
                    image_arrays.append(img_array)
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
                
                img_pil = tensor2pil(single_img)
                if debug_output:
                    print(f"  转换PIL: 尺寸 {img_pil.size}")
                
                img_array = np.array(img_pil)
                if debug_output:
                    print(f"  转换numpy: 形状 {img_array.shape}")
                
                # 转换回正确的tensor格式 (BHWC)
                img_tensor = pil2tensor(img_pil)
                if debug_output:
                    print(f"  转换tensor: 形状 {img_tensor.shape}")
                
                image_arrays.append(img_array)
                image_tensors.append(img_tensor)
        
        if not image_arrays:
            if debug_output:
                print("没有找到图像，返回空结果")
            return ([], [], [])
        
        if debug_output:
            print("-" * 60)
            print(f"图像处理完成:")
            print(f"  总处理图像数: {len(image_arrays)}")
            print(f"  图像tensor数: {len(image_tensors)}")
            print("-" * 60)
        
        # 使用CLIP检测图像
        filtered_images = []
        excluded_images = []
        all_images = []
        
        matched_count = 0
        excluded_count = 0
        
        for i, img_array in enumerate(image_arrays):
            if debug_output:
                print(f"处理第 {i+1} 张图像:")
            
            matches_criteria, features = check_image_criteria_clip(
                img_array, self.clip_model, self.clip_processor, 
                positive_list, negative_list, confidence_threshold, debug_output
            )
            
            # 添加到对应的列表
            all_images.append(image_tensors[i])
            
            if matches_criteria:
                filtered_images.append(image_tensors[i])
                matched_count += 1
            else:
                excluded_images.append(image_tensors[i])
                excluded_count += 1
        
        if debug_output:
            print("=" * 60)
            print("筛选统计:")
            print(f"总图像数: {len(image_arrays)}")
            print(f"符合条件的图像数: {matched_count}")
            print(f"排除的图像数: {excluded_count}")
            print(f"符合条件比例: {matched_count/len(image_arrays)*100:.1f}%")
            print("=" * 60)
            print("输出结果:")
            print(f"  filtered_images: {len(filtered_images)} 张")
            print(f"  excluded_images: {len(excluded_images)} 张")
            print(f"  all_images: {len(all_images)} 张")
            print("=" * 60)
        
        return (filtered_images, excluded_images, all_images)
    
    def _fallback_filter(self, images):
        """当CLIP模型不可用时的备用方法"""
        print("Using fallback method - returning all images as filtered")
        if isinstance(images, list):
            return (images, [], images)
        else:
            return ([images], [], [images])

class AdjacentImageMerger:
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """加载CLIP模型"""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded successfully for AdjacentImageMerger")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
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
                "min_group_size": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": True,
                    "display": "boolean"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_images",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "merge_adjacent_images"
    CATEGORY = "hhy/image"
    
    def merge_adjacent_images(self, images, positive_prompts, negative_prompts, confidence_threshold, min_group_size, debug_output):
        
        confidence_threshold = confidence_threshold[0] if isinstance(confidence_threshold, list) else confidence_threshold
        positive_prompts = positive_prompts[0] if isinstance(positive_prompts, list) else positive_prompts
        negative_prompts = negative_prompts[0] if isinstance(negative_prompts, list) else negative_prompts
        min_group_size = min_group_size[0] if isinstance(min_group_size, list) else min_group_size
        debug_output = debug_output[0] if isinstance(debug_output, list) else debug_output
        
        # 解析正面和负面提示词
        if isinstance(positive_prompts, str):
            positive_list = [line.strip() for line in positive_prompts.split('\n') if line.strip()]
        else:
            positive_list = positive_prompts if isinstance(positive_prompts, list) else []
        
        if isinstance(negative_prompts, str):
            negative_list = [line.strip() for line in negative_prompts.split('\n') if line.strip()]
        else:
            negative_list = negative_prompts if isinstance(negative_prompts, list) else []
        
        if debug_output:
            print("=" * 60)
            print("CLIP临近图像拼接器开始工作")
            print(f"正面提示词: {positive_list}")
            print(f"负面提示词: {negative_list}")
            print(f"置信度阈值: {confidence_threshold}")
            print(f"最小组合大小: {min_group_size}")
            print(f"调试输出: {'开启' if debug_output else '关闭'}")
            print("=" * 60)
            print(f"输入类型: {type(images)}")
            if isinstance(images, list):
                print(f"输入列表长度: {len(images)}")
                for idx, img_item in enumerate(images):
                    if img_item is not None:
                        print(f"  列表项 {idx}: tensor形状 {img_item.shape}")
                    else:
                        print(f"  列表项 {idx}: None")
            else:
                print(f"输入tensor形状: {images.shape}")
            print("=" * 60)
        
        # 检查CLIP模型是否加载成功
        if self.clip_model is None or self.clip_processor is None:
            print("CLIP model not loaded, falling back to returning original images")
            return self._fallback_merge(images)
        
        image_arrays = []
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
                    
                    img_pil = tensor2pil(single_img)
                    if debug_output:
                        print(f"  转换PIL: 尺寸 {img_pil.size}")
                    
                    img_array = np.array(img_pil)
                    if debug_output:
                        print(f"  转换numpy: 形状 {img_array.shape}")
                    
                    # 转换回正确的tensor格式 (BHWC)
                    img_tensor = pil2tensor(img_pil)
                    if debug_output:
                        print(f"  转换tensor: 形状 {img_tensor.shape}")
                    
                    image_arrays.append(img_array)
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
                
                img_pil = tensor2pil(single_img)
                if debug_output:
                    print(f"  转换PIL: 尺寸 {img_pil.size}")
                
                img_array = np.array(img_pil)
                if debug_output:
                    print(f"  转换numpy: 形状 {img_array.shape}")
                
                # 转换回正确的tensor格式 (BHWC)
                img_tensor = pil2tensor(img_pil)
                if debug_output:
                    print(f"  转换tensor: 形状 {img_tensor.shape}")
                
                image_arrays.append(img_array)
                image_tensors.append(img_tensor)
        
        if not image_arrays:
            if debug_output:
                print("没有找到图像，返回空结果")
            return ([],)
        
        if debug_output:
            print(f"处理图像数量: {len(image_arrays)}")
            print("-" * 60)
        
        # 使用CLIP检测图像，标记哪些是文本图像
        text_flags = []
        text_count = 0
        
        for i, img_array in enumerate(image_arrays):
            if debug_output:
                print(f"检测第 {i+1} 张图像:")
            
            matches_criteria, features = check_image_criteria_clip(
                img_array, self.clip_model, self.clip_processor, 
                positive_list, negative_list, confidence_threshold, debug_output
            )
            
            text_flags.append(matches_criteria)
            if matches_criteria:
                text_count += 1
        
        if debug_output:
            print("=" * 60)
            print("文本检测统计:")
            print(f"总图像数: {len(image_arrays)}")
            print(f"文本图像数: {text_count}")
            print(f"非文本图像数: {len(image_arrays) - text_count}")
            print("=" * 60)
        
        # 执行临近拼接逻辑
        merged_images = []
        i = 0
        
        while i < len(image_arrays):
            if text_flags[i]:
                # 找到连续的文本图像组
                text_group = [image_tensors[i]]
                j = i + 1
                
                while j < len(image_arrays) and text_flags[j]:
                    text_group.append(image_tensors[j])
                    j += 1
                
                if len(text_group) >= min_group_size:
                    # 如果文本组足够大，垂直拼接
                    merged_array = None
                    for k, tensor in enumerate(text_group):
                        img_pil = tensor2pil(tensor)
                        img_array = np.array(img_pil)
                        
                        if k == 0:
                            merged_array = img_array
                        else:
                            merged_array = merge_images_vertically(merged_array, img_array)
                    
                    if merged_array is not None:
                        merged_tensor = pil2tensor(Image.fromarray(merged_array))
                        merged_images.append(merged_tensor)
                        if debug_output:
                            print(f"拼接了 {len(text_group)} 张文本图像")
                    
                    i = j
                else:
                    # 如果文本组不够大，尝试与下一个非文本图像合并
                    if j < len(image_arrays):
                        next_non_text = image_tensors[j]
                        merged_array = None
                        
                        # 拼接文本组
                        for k, tensor in enumerate(text_group):
                            img_pil = tensor2pil(tensor)
                            img_array = np.array(img_pil)
                            
                            if k == 0:
                                merged_array = img_array
                            else:
                                merged_array = merge_images_vertically(merged_array, img_array)
                        
                        # 与非文本图像合并
                        if merged_array is not None:
                            next_img_pil = tensor2pil(next_non_text)
                            next_img_array = np.array(next_img_pil)
                            final_merged = merge_images_vertically(merged_array, next_img_array)
                            
                            if final_merged is not None:
                                merged_tensor = pil2tensor(Image.fromarray(final_merged))
                                merged_images.append(merged_tensor)
                                if debug_output:
                                    print(f"拼接了 {len(text_group)} 张文本图像 + 1 张非文本图像")
                        
                        i = j + 1
                    else:
                        # 如果后面没有非文本图像，直接输出文本组
                        if len(text_group) == 1:
                            merged_images.append(text_group[0])
                        else:
                            merged_array = None
                            for k, tensor in enumerate(text_group):
                                img_pil = tensor2pil(tensor)
                                img_array = np.array(img_pil)
                                
                                if k == 0:
                                    merged_array = img_array
                                else:
                                    merged_array = merge_images_vertically(merged_array, img_array)
                            
                            if merged_array is not None:
                                merged_tensor = pil2tensor(Image.fromarray(merged_array))
                                merged_images.append(merged_tensor)
                                if debug_output:
                                    print(f"拼接了 {len(text_group)} 张文本图像")
                        
                        i = j
            else:
                # 非文本图像直接输出
                merged_images.append(image_tensors[i])
                i += 1
        
        if debug_output:
            print("=" * 60)
            print("拼接完成统计:")
            print(f"最终输出图像数量: {len(merged_images)}")
            print(f"压缩比例: {len(image_arrays)/len(merged_images):.2f}x")
            print("=" * 60)
            print("输出结果:")
            print(f"  merged_images: {len(merged_images)} 张")
            for i, img_tensor in enumerate(merged_images):
                print(f"    图像 {i+1}: tensor形状 {img_tensor.shape}")
            print("=" * 60)
        
        return (merged_images,)
    
    def _fallback_merge(self, images):
        """当CLIP模型不可用时的备用方法"""
        print("Using fallback method - returning original images")
        if isinstance(images, list):
            return (images,)
        else:
            return ([images],)

NODE_CLASS_MAPPINGS = {
    "SmartTextDetector": CLIPImageFilter,
    "AdjacentImageMerger": AdjacentImageMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartTextDetector": "CLIP Image Filter",
    "AdjacentImageMerger": "CLIP Adjacent Image Merger",
}
