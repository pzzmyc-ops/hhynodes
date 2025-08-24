import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths
from typing import Dict, List, Tuple, Optional

# 添加模型路径
insightface_model_dir = os.path.join(folder_paths.models_dir, "insightface")
folder_paths.add_model_folder_path("insightface", insightface_model_dir)

yolo_model_dir = os.path.join(folder_paths.models_dir, "yolo")
folder_paths.add_model_folder_path("yolo", yolo_model_dir)

for dir_path in [yolo_model_dir, insightface_model_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

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
                print(f"[YOLO InsightFace Detection] 检测到PyTorch weights_only问题，使用兼容模式重试...")
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            else:
                raise e
    torch.load = patched_torch_load
    torch.load._original_torch_load = original_torch_load

class YOLOInsightFaceDetection:
    def __init__(self):
        self.yolo_model = None
        self.insightface_app = None
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
                "yolo_model": (yolo_models, {
                    "default": yolo_models[0] if yolo_models else "yolov8n.pt"
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
                "enable_face_analysis": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用人脸分析",
                    "label_off": "仅YOLO检测"
                }),
                "analysis_confidence_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("detection_image", "male_only_image", "female_only_image", "yolo_mask", "male_mask", "female_mask", "detection_info")
    FUNCTION = "detect_and_analyze_faces"
    CATEGORY = "hhy"
    
    def apply_compatibility_patch(self):
        if not self.patch_applied:
            fix_torch_load_for_ultralytics()
            self.patch_applied = True
    
    def load_yolo_model(self, model_name):
        try:
            self.apply_compatibility_patch()
            from ultralytics import YOLO
            if self.yolo_model is None or getattr(self.yolo_model, 'model_name', None) != model_name:
                print(f"Loading YOLO model: {model_name}")
                model_path = folder_paths.get_full_path("yolo", model_name)
                if model_path is None:
                    print(f"Model {model_name} not found, downloading...")
                    model_path = model_name
                else:
                    print(f"Using YOLO model from: {model_path}")
                self.yolo_model = YOLO(model_path)
                self.yolo_model.model_name = model_name
            return self.yolo_model
        except ImportError:
            raise ImportError("请安装ultralytics库: pip install ultralytics")
    
    def load_insightface_model(self):
        """加载InsightFace人脸分析模型"""
        if self.insightface_app is not None:
            return self.insightface_app
            
        try:
            import insightface
        except ImportError:
            raise ImportError("请安装insightface库: pip install insightface")
        
        try:
            print("正在初始化InsightFace模型...")
            # 创建FaceAnalysis应用，指定需要的任务
            self.insightface_app = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider'],  # 可以改为 'CUDAExecutionProvider' 如果有GPU
                root=insightface_model_dir,
                name='buffalo_l'  # 使用buffalo_l模型，包含年龄和性别识别
            )
            
            # 准备模型，设置输入尺寸
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # 打印InsightFace版本信息
            print(f"InsightFace版本: {insightface.__version__}")
            print("InsightFace模型加载成功")
            return self.insightface_app
            
        except Exception as e:
            print(f"InsightFace模型加载失败: {e}")
            print("正在尝试下载默认模型...")
            try:
                # 如果模型不存在，insightface会自动下载
                self.insightface_app = insightface.app.FaceAnalysis(
                    providers=['CPUExecutionProvider'],
                    root=insightface_model_dir,
                    name='buffalo_l'
                )
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                print("InsightFace模型下载并加载成功")
                return self.insightface_app
            except Exception as e2:
                raise ValueError(f"InsightFace模型加载失败: {e2}")
    
    def analyze_face_with_insightface(self, face_crop):
        """使用InsightFace分析人脸"""
        try:
            # InsightFace需要BGR格式的图像
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                # 从RGB转换为BGR
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            else:
                face_bgr = face_crop
            
            print(f"InsightFace分析图像尺寸: {face_bgr.shape}")
            
            # 使用InsightFace进行人脸分析
            faces = self.insightface_app.get(face_bgr)
            
            if len(faces) == 0:
                print("InsightFace未检测到人脸")
                return {
                    "gender": "Unknown",
                    "age": 0,
                    "confidence": 0.0,
                    "embedding": None
                }
            
            # 使用第一个检测到的人脸
            face = faces[0]
            
            # 调试：打印face对象的所有属性
            print(f"Face对象属性: {dir(face)}")
            
            # 提取性别信息 - 先检查可能的属性名
            gender = "Unknown"
            if hasattr(face, 'sex'):
                sex_value = face.sex
                print(f"face.sex = {sex_value} (类型: {type(sex_value)})")
                
                # 处理不同类型的性别值
                if isinstance(sex_value, str):
                    # 字符串类型：'M' = 男性, 'F' = 女性
                    gender = "Man" if sex_value.upper() == 'M' else "Woman"
                elif isinstance(sex_value, (int, float)):
                    # 数字类型：1 = 男性, 0 = 女性
                    gender = "Man" if sex_value == 1 else "Woman"
                else:
                    print(f"未知的性别值类型: {type(sex_value)}")
                    
            elif hasattr(face, 'gender'):
                gender_value = face.gender
                print(f"face.gender = {gender_value} (类型: {type(gender_value)})")
                
                # 处理不同类型的性别值
                if isinstance(gender_value, str):
                    # 字符串类型：'M' = 男性, 'F' = 女性
                    gender = "Man" if gender_value.upper() == 'M' else "Woman"
                elif isinstance(gender_value, (int, float)):
                    # 数字类型：1 = 男性, 0 = 女性
                    gender = "Man" if gender_value == 1 else "Woman"
                else:
                    print(f"未知的性别值类型: {type(gender_value)}")
            else:
                print("未找到性别属性")
            
            # 提取年龄信息
            age = 0
            if hasattr(face, 'age'):
                age = int(face.age)
                print(f"face.age = {age}")
            else:
                print("未找到年龄属性")
            
            # InsightFace没有直接的置信度，我们使用检测置信度
            confidence = 1.0
            if hasattr(face, 'det_score'):
                confidence = float(face.det_score)
                print(f"face.det_score = {confidence}")
            elif hasattr(face, 'score'):
                confidence = float(face.score)
                print(f"face.score = {confidence}")
            else:
                print("未找到置信度属性")
            
            # 获取人脸embedding（可选）
            embedding = face.normed_embedding if hasattr(face, 'normed_embedding') else None
            
            result = {
                "gender": gender,
                "age": age,
                "confidence": confidence,
                "embedding": embedding,
                "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else None,
                "landmarks": face.kps.tolist() if hasattr(face, 'kps') else None
            }
            
            print(f"InsightFace分析结果: {gender}, 年龄: {age}, 置信度: {confidence:.3f}")
            return result
            
        except Exception as e:
            print(f"InsightFace分析失败: {e}")
            return {
                "gender": "Unknown",
                "age": 0,
                "confidence": 0.0,
                "embedding": None
            }
    
    def add_analysis_labels_to_yolo_image(self, yolo_image_pil, boxes, analysis_results, confidence_threshold):
        """在YOLO检测结果图像上添加人脸分析标签"""
        draw = ImageDraw.Draw(yolo_image_pil)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        
        detection_info = []
        
        for i, (box, analysis_result) in enumerate(zip(boxes, analysis_results)):
            x1, y1, x2, y2 = box
            
            # 确定性别标签颜色
            if analysis_result["gender"] == "Man":
                gender_color = (0, 100, 255)  # 蓝色
            elif analysis_result["gender"] == "Woman":
                gender_color = (255, 20, 147)  # 深粉色
            else:
                gender_color = (128, 128, 128)  # 灰色
            
            # 准备分析标签文本
            if analysis_result["confidence"] >= confidence_threshold:
                analysis_label = f"{analysis_result['gender']}: {analysis_result['confidence']:.2f}"
                if analysis_result["age"] > 0:
                    analysis_label += f" (Age: {analysis_result['age']})"
            else:
                analysis_label = f"Low Conf: {analysis_result['confidence']:.2f}"
            
            # 绘制分析标签背景 (在框的左下角)
            text_bbox = draw.textbbox((0, 0), analysis_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 分析标签位置 (框内左下角)
            analysis_x = x1 + 5
            analysis_y = y2 - text_height - 10
            
            # 确保标签不超出图像边界
            if analysis_y < 0:
                analysis_y = y1 + 5
            
            # 绘制分析标签背景
            draw.rectangle([analysis_x - 2, analysis_y - 2, analysis_x + text_width + 4, analysis_y + text_height + 2], 
                         fill=gender_color)
            
            # 绘制分析标签文本
            draw.text((analysis_x, analysis_y), analysis_label, fill=(255, 255, 255), font=font)
            
            # 在框内上方添加详细信息
            if analysis_result["age"] > 0:
                detail_text = f"Gender: {analysis_result['gender']}, Age: {analysis_result['age']}"
            else:
                detail_text = f"Gender: {analysis_result['gender']}"
            
            detail_x = x1 + 5
            detail_y = y1 + 5
            
            # 绘制详细信息背景
            detail_bbox = draw.textbbox((0, 0), detail_text, font=small_font)
            detail_width = detail_bbox[2] - detail_bbox[0]
            detail_height = detail_bbox[3] - detail_bbox[1]
            
            draw.rectangle([detail_x - 2, detail_y - 2, detail_x + detail_width + 4, detail_y + detail_height + 2], 
                         fill=(0, 0, 0, 180))
            
            # 绘制详细信息文本
            draw.text((detail_x, detail_y), detail_text, fill=gender_color, font=small_font)
            
            # 记录检测信息
            detection_info.append({
                "person_id": i + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "gender": analysis_result["gender"],
                "age": analysis_result["age"],
                "confidence": analysis_result["confidence"],
                "has_embedding": analysis_result["embedding"] is not None
            })
        
        return yolo_image_pil, detection_info
    
    def create_masked_image(self, original_image_pil, individual_masks, analysis_results, mask_gender):
        """创建遮盖特定性别人脸的图像
        
        Args:
            original_image_pil: 原始PIL图像
            individual_masks: YOLO检测的mask列表
            analysis_results: 人脸分析结果列表
            mask_gender: 要遮盖的性别 ("Man" 或 "Woman")
        
        Returns:
            遮盖后的PIL图像
        """
        # 创建反向mask（只保留非指定性别的部分）
        img_array = np.array(original_image_pil)
        final_mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
        
        for mask_tensor, analysis_result in zip(individual_masks, analysis_results):
            # 如果检测到的性别与要遮盖的性别匹配，从final_mask中移除这部分
            if analysis_result["gender"] == mask_gender:
                mask_array = mask_tensor.cpu().numpy().squeeze()
                # 从final_mask中减去当前mask（即遮盖掉这部分）
                final_mask = final_mask * (1.0 - mask_array)
                print(f"使用YOLO mask遮盖 {mask_gender}")
        
        # 应用mask到图像
        mask_3d = np.stack([final_mask, final_mask, final_mask], axis=-1)
        masked_img_array = img_array * mask_3d
        
        return Image.fromarray(masked_img_array.astype(np.uint8))
    
    def detect_and_analyze_faces(self, image, yolo_model="yolov8n.pt", confidence=0.5, 
                                iou_threshold=0.45, enable_face_analysis=True, 
                                analysis_confidence_threshold=0.6):
        
        # 加载YOLO模型
        yolo = self.load_yolo_model(yolo_model)
        
        # 只有在启用人脸分析时才加载InsightFace模型
        if enable_face_analysis:
            insightface_app = self.load_insightface_model()
        
        result_images = []
        male_only_images = []
        female_only_images = []
        all_yolo_masks = []
        all_male_masks = []
        all_female_masks = []
        all_detection_info = []
        
        for img_tensor in image:
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            
            # YOLO检测（启用mask）
            results = yolo(pil_image, conf=confidence, iou=iou_threshold, retina_masks=True)
            
            for result in results:
                # 首先获取YOLO的原始绘制结果
                yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                result_image_pil = Image.fromarray(yolo_plot_image)
                
                boxes = []
                analysis_results = []
                individual_masks = []
                
                # 首先处理YOLO masks
                if result.masks is not None and len(result.masks) > 0:
                    print(f"检测到 {len(result.masks)} 个分割mask")
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask_pil = np2pil(_mask).convert("L")
                        mask_tensor = image2mask(_mask_pil)
                        individual_masks.append(mask_tensor)
                        all_yolo_masks.append(mask_tensor)
                
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"YOLO检测到 {len(result.boxes)} 个对象")
                    
                    # 如果没有分割mask，创建方形mask
                    if result.masks is None or len(result.masks) == 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            _mask = Image.new('L', pil_image.size, "black")
                            white_image = Image.new('L', pil_image.size, "white")
                            _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                            mask_tensor = image2mask(_mask)
                            individual_masks.append(mask_tensor)
                            all_yolo_masks.append(mask_tensor)
                    
                    for i, box_data in enumerate(result.boxes):
                        class_id = int(box_data.cls[0])
                        class_name = yolo.names[class_id]
                        box_confidence = float(box_data.conf[0])
                        
                        # 直接处理所有YOLO检测到的对象
                        print(f"检测到 {class_name} (置信度: {box_confidence:.2f})")
                        
                        # 获取边界框
                        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 确保边界框在图像范围内
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(pil_image.width, x2)
                        y2 = min(pil_image.height, y2)
                        
                        # 如果框太小，跳过
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        
                        boxes.append([x1, y1, x2, y2])
                        
                        # 确保有对应的mask（索引对应）
                        if i < len(individual_masks):
                            # 如果启用人脸分析，进行InsightFace分析
                            if enable_face_analysis:
                                # 提取检测区域进行人脸分析
                                crop_img = np.array(pil_image)[y1:y2, x1:x2]
                                if crop_img.size > 0:
                                    analysis_result = self.analyze_face_with_insightface(crop_img)
                                    analysis_results.append(analysis_result)
                                    print(f"检测到 {class_name}: {analysis_result['gender']} "
                                          f"(置信度: {analysis_result['confidence']:.2f})")
                                else:
                                    # 如果裁剪区域为空，添加默认结果
                                    analysis_results.append({
                                        "gender": "Unknown",
                                        "age": 0,
                                        "confidence": 0.0,
                                        "embedding": None
                                    })
                        else:
                            # 如果没有对应的mask，跳过这个检测
                            print(f"警告：检测 {i} 没有对应的mask，跳过")
                            continue
                
                # 如果启用人脸分析且有检测结果，在YOLO图像上添加分析标签
                if enable_face_analysis and boxes and analysis_results:
                    result_image_pil, detection_info = self.add_analysis_labels_to_yolo_image(
                        result_image_pil, boxes, analysis_results, analysis_confidence_threshold
                    )
                    all_detection_info.extend(detection_info)
                elif not enable_face_analysis and boxes:
                    # 如果未启用人脸分析，仅记录YOLO检测信息
                    for i, box in enumerate(boxes):
                        all_detection_info.append({
                            "person_id": i + 1,
                            "bbox": [int(x) for x in box],
                            "yolo_class": yolo.names[int(result.boxes[i].cls[0])],
                            "yolo_confidence": float(result.boxes[i].conf[0])
                        })
                    print(f"YOLO检测完成，未启用人脸分析")
                else:
                    print("未检测到符合条件的对象")
                
                # 添加检测结果图像（原有输出）
                result_images.append(pil2tensor(result_image_pil))
                
                # 如果启用人脸分析且有检测结果，生成性别专用图像
                if enable_face_analysis and individual_masks and analysis_results:
                    # 生成男性专用图像（遮盖女性）
                    male_only_image = self.create_masked_image(pil_image, individual_masks, analysis_results, "Woman")
                    male_only_images.append(pil2tensor(male_only_image))
                    
                    # 生成女性专用图像（遮盖男性）
                    female_only_image = self.create_masked_image(pil_image, individual_masks, analysis_results, "Man")
                    female_only_images.append(pil2tensor(female_only_image))
                    
                    # 生成性别专用mask
                    male_masks = []
                    female_masks = []
                    for mask_tensor, analysis_result in zip(individual_masks, analysis_results):
                        if analysis_result["gender"] == "Man":
                            male_masks.append(mask_tensor)
                        elif analysis_result["gender"] == "Woman":
                            female_masks.append(mask_tensor)
                    
                    # 处理男性mask
                    if male_masks:
                        combined_male_mask = male_masks[0]
                        for i in range(1, len(male_masks)):
                            combined_male_mask = add_mask(combined_male_mask, male_masks[i])
                        all_male_masks.append(combined_male_mask)
                    else:
                        # 如果当前图像没有男性，添加空mask
                        empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                        all_male_masks.append(empty_mask)
                    
                    # 处理女性mask
                    if female_masks:
                        combined_female_mask = female_masks[0]
                        for i in range(1, len(female_masks)):
                            combined_female_mask = add_mask(combined_female_mask, female_masks[i])
                        all_female_masks.append(combined_female_mask)
                    else:
                        # 如果当前图像没有女性，添加空mask
                        empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                        all_female_masks.append(empty_mask)
                    
                    print("生成性别专用图像和mask完成")
                else:
                    # 如果没有人脸分析结果，返回原图和空mask
                    male_only_images.append(pil2tensor(pil_image))
                    female_only_images.append(pil2tensor(pil_image))
                    
                    # 添加空的性别mask
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_male_masks.append(empty_mask)
                    all_female_masks.append(empty_mask)
                    
                    print("未进行人脸分析，性别专用图像使用原图，mask为空")
                
                # 如果没有检测到任何对象，添加空mask
                if not individual_masks:
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_yolo_masks.append(empty_mask)
        
        # 合并结果
        if result_images:
            result_tensor = torch.cat(result_images, dim=0)
        else:
            # 如果没有结果，返回原图
            result_tensor = image
        
        # 合并男性专用图像
        if male_only_images:
            male_only_tensor = torch.cat(male_only_images, dim=0)
        else:
            # 如果没有结果，返回原图
            male_only_tensor = image
        
        # 合并女性专用图像
        if female_only_images:
            female_only_tensor = torch.cat(female_only_images, dim=0)
        else:
            # 如果没有结果，返回原图
            female_only_tensor = image
        
        # 合并YOLO masks
        if all_yolo_masks:
            yolo_masks_tensor = torch.cat(all_yolo_masks, dim=0)
        else:
            # 如果没有mask，创建空mask
            yolo_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        
        # 合并男性masks
        if all_male_masks:
            male_masks_tensor = torch.cat(all_male_masks, dim=0)
        else:
            # 如果没有男性mask，创建空mask
            male_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        
        # 合并女性masks
        if all_female_masks:
            female_masks_tensor = torch.cat(all_female_masks, dim=0)
        else:
            # 如果没有女性mask，创建空mask
            female_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        
        # 生成检测信息字符串
        if enable_face_analysis:
            info_text = f"YOLO+InsightFace分析结果 - 检测到 {len(all_detection_info)} 个对象:\n"
            for info in all_detection_info:
                info_text += f"Person {info['person_id']}: {info['gender']}"
                if info.get('age', 0) > 0:
                    info_text += f", Age: {info['age']}"
                info_text += f" (置信度: {info['confidence']:.2f})\n"
        else:
            info_text = f"YOLO检测结果 - 检测到 {len(all_detection_info)} 个对象:\n"
            for info in all_detection_info:
                info_text += f"Object {info['person_id']}: {info['yolo_class']} "
                info_text += f"(YOLO置信度: {info['yolo_confidence']:.2f})\n"
        
        print(f"处理完成: {len(result_images)} 张图像，{len(all_detection_info)} 个检测结果，{len(all_yolo_masks)} 个YOLO mask，{len(all_male_masks)} 个男性mask，{len(all_female_masks)} 个女性mask")
        
        return (result_tensor, male_only_tensor, female_only_tensor, yolo_masks_tensor, male_masks_tensor, female_masks_tensor, info_text)

NODE_CLASS_MAPPINGS = {
    "YOLOInsightFaceDetection": YOLOInsightFaceDetection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOInsightFaceDetection": "YOLO InsightFace Detection by HHY"
}
