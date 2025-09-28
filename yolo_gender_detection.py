import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths
import urllib.request
from typing import Dict, List, Tuple, Optional

# 添加性别检测模型路径
gender_model_dir = os.path.join(folder_paths.models_dir, "gender")
folder_paths.add_model_folder_path("gender", gender_model_dir)

yolo_model_dir = os.path.join(folder_paths.models_dir, "yolo")
folder_paths.add_model_folder_path("yolo", yolo_model_dir)

for dir_path in [yolo_model_dir, gender_model_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def imread_unicode(image_path: str) -> np.ndarray:
    """支持中文路径的图像读取函数"""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("解码图像失败")
        return img
    except Exception as e:
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img
            else:
                raise ValueError(f"无法读取图像: {image_path}")
        except:
            raise ValueError(f"读取图像失败: {image_path}, 错误: {str(e)}")

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
                print(f"[YOLO Gender Detection] 检测到PyTorch weights_only问题，使用兼容模式重试...")
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            else:
                raise e
    torch.load = patched_torch_load
    torch.load._original_torch_load = original_torch_load

class YOLOGenderDetection:
    def __init__(self):
        self.yolo_model = None
        self.gender_model = None
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

                "enable_gender_detection": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用性别识别",
                    "label_off": "仅YOLO检测"
                }),
                "gender_confidence_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("detection_image", "detection_info")
    FUNCTION = "detect_and_classify_gender"
    CATEGORY = "hhy/yolo"
    
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
    
    def load_gender_model(self):
        """加载性别识别模型"""
        if self.gender_model is not None:
            return self.gender_model
            
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model, Sequential
            from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
        except ImportError:
            raise ImportError("请安装tensorflow: pip install tensorflow")
        
        # 权重文件路径
        weights_file = os.path.join(gender_model_dir, "gender_model_weights.h5")
        
        # 如果权重不存在，下载
        if not os.path.exists(weights_file):
            print("正在下载性别识别模型权重...")
            weights_url = "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"
            try:
                urllib.request.urlretrieve(weights_url, weights_file)
                print(f"权重下载完成: {weights_file}")
            except Exception as e:
                raise ValueError(f"下载权重失败: {e}")
        
        # 创建VGG基础模型
        base_model = self.create_vgg_base_model()
        
        # 添加性别分类层
        classes = 2
        gender_output = Convolution2D(classes, (1, 1), name="predictions")(base_model.layers[-4].output)
        gender_output = Flatten()(gender_output)
        gender_output = Activation("softmax")(gender_output)
        
        # 创建性别模型
        gender_model = Model(inputs=base_model.inputs, outputs=gender_output)
        
        # 加载权重
        try:
            gender_model.load_weights(weights_file)
            print("性别识别模型加载成功")
            self.gender_model = gender_model
            return gender_model
        except Exception as e:
            raise ValueError(f"加载性别模型权重失败: {e}")
    
    def create_vgg_base_model(self):
        """创建VGG-Face基础模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
        
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))

        return model
    
    def preprocess_face_for_gender(self, face_crop):
        """预处理人脸图像用于性别识别"""
        # face_crop 已经是从 PIL 图像转换的 RGB 格式，不需要颜色空间转换
        print(f"裁剪区域形状: {face_crop.shape}")
        
        # 直接调整大小（face_crop 已经是 RGB 格式）
        face_resized = cv2.resize(face_crop, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        print(f"预处理后形状: {face_batch.shape}, 数值范围: [{face_batch.min():.3f}, {face_batch.max():.3f}]")
        return face_batch
    
    def predict_gender(self, face_crop):
        """预测单个人脸的性别"""
        try:
            processed_face = self.preprocess_face_for_gender(face_crop)
            predictions = self.gender_model.predict(processed_face, verbose=0)[0]
            
            # 性别标签
            gender_labels = ["Woman", "Man"]
            
            woman_conf = float(predictions[0])
            man_conf = float(predictions[1])
            
            dominant_gender = gender_labels[np.argmax(predictions)]
            confidence = max(woman_conf, man_conf)
            
            return {
                "dominant_gender": dominant_gender,
                "confidence": confidence,
                "woman_confidence": woman_conf,
                "man_confidence": man_conf
            }
        except Exception as e:
            print(f"性别预测失败: {e}")
            return {
                "dominant_gender": "Unknown",
                "confidence": 0.0,
                "woman_confidence": 0.0,
                "man_confidence": 0.0
            }
    
    def add_gender_labels_to_yolo_image(self, yolo_image_pil, boxes, gender_results, confidence_threshold):
        """在YOLO检测结果图像上添加性别标签"""
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
        
        for i, (box, gender_result) in enumerate(zip(boxes, gender_results)):
            x1, y1, x2, y2 = box
            
            # 确定性别标签颜色
            if gender_result["dominant_gender"] == "Man":
                gender_color = (0, 100, 255)  # 蓝色
            elif gender_result["dominant_gender"] == "Woman":
                gender_color = (255, 20, 147)  # 深粉色
            else:
                gender_color = (128, 128, 128)  # 灰色
            
            # 准备性别标签文本
            if gender_result["confidence"] >= confidence_threshold:
                gender_label = f"{gender_result['dominant_gender']}: {gender_result['confidence']:.2f}"
            else:
                gender_label = f"Low Conf: {gender_result['confidence']:.2f}"
            
            # 绘制性别标签背景 (在框的左下角)
            text_bbox = draw.textbbox((0, 0), gender_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 性别标签位置 (框内左下角)
            gender_x = x1 + 5
            gender_y = y2 - text_height - 10
            
            # 确保标签不超出图像边界
            if gender_y < 0:
                gender_y = y1 + 5
            
            # 绘制性别标签背景
            draw.rectangle([gender_x - 2, gender_y - 2, gender_x + text_width + 4, gender_y + text_height + 2], 
                         fill=gender_color)
            
            # 绘制性别标签文本
            draw.text((gender_x, gender_y), gender_label, fill=(255, 255, 255), font=font)
            
            # 在框内上方添加详细概率信息
            detail_text = f"W:{gender_result['woman_confidence']:.1f}% M:{gender_result['man_confidence']:.1f}%"
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
                "gender": gender_result["dominant_gender"],
                "confidence": gender_result["confidence"],
                "woman_confidence": gender_result["woman_confidence"],
                "man_confidence": gender_result["man_confidence"]
            })
        
        return yolo_image_pil, detection_info
    
    def detect_and_classify_gender(self, image, yolo_model="yolov8n.pt", confidence=0.5, 
                                 iou_threshold=0.45, enable_gender_detection=True, 
                                 gender_confidence_threshold=0.6):
        
        # 加载YOLO模型
        yolo = self.load_yolo_model(yolo_model)
        
        # 只有在启用性别检测时才加载性别模型
        if enable_gender_detection:
            gender_model = self.load_gender_model()
        
        result_images = []
        all_detection_info = []
        
        for img_tensor in image:
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            
            # YOLO检测
            results = yolo(pil_image, conf=confidence, iou=iou_threshold)
            
            for result in results:
                # 首先获取YOLO的原始绘制结果
                yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                result_image_pil = Image.fromarray(yolo_plot_image)
                
                boxes = []
                gender_results = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"YOLO检测到 {len(result.boxes)} 个对象")
                    
                    for box_data in result.boxes:
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
                        
                        # 如果启用性别检测，进行性别识别
                        if enable_gender_detection:
                            # 提取检测区域进行性别识别
                            crop_img = np.array(pil_image)[y1:y2, x1:x2]
                            if crop_img.size > 0:
                                gender_result = self.predict_gender(crop_img)
                                gender_results.append(gender_result)
                                print(f"检测到 {class_name}: {gender_result['dominant_gender']} "
                                      f"(置信度: {gender_result['confidence']:.2f})")
                            else:
                                # 如果裁剪区域为空，添加默认结果
                                gender_results.append({
                                    "dominant_gender": "Unknown",
                                    "confidence": 0.0,
                                    "woman_confidence": 0.0,
                                    "man_confidence": 0.0
                                })
                
                # 如果启用性别检测且有检测结果，在YOLO图像上添加性别标签
                if enable_gender_detection and boxes and gender_results:
                    result_image_pil, detection_info = self.add_gender_labels_to_yolo_image(
                        result_image_pil, boxes, gender_results, gender_confidence_threshold
                    )
                    all_detection_info.extend(detection_info)
                elif not enable_gender_detection and boxes:
                    # 如果未启用性别检测，仅记录YOLO检测信息
                    for i, box in enumerate(boxes):
                        all_detection_info.append({
                            "person_id": i + 1,
                            "bbox": [int(x) for x in box],
                            "yolo_class": yolo.names[int(result.boxes[i].cls[0])],
                            "yolo_confidence": float(result.boxes[i].conf[0])
                        })
                    print(f"YOLO检测完成，未启用性别识别")
                else:
                    print("未检测到符合条件的对象")
                
                result_images.append(pil2tensor(result_image_pil))
        
        # 合并结果
        if result_images:
            result_tensor = torch.cat(result_images, dim=0)
        else:
            # 如果没有结果，返回原图
            result_tensor = image
        
        # 生成检测信息字符串
        if enable_gender_detection:
            info_text = f"YOLO+性别检测结果 - 检测到 {len(all_detection_info)} 个对象:\n"
            for info in all_detection_info:
                info_text += f"Person {info['person_id']}: {info['gender']} "
                info_text += f"(性别置信度: {info['confidence']:.2f})\n"
        else:
            info_text = f"YOLO检测结果 - 检测到 {len(all_detection_info)} 个对象:\n"
            for info in all_detection_info:
                info_text += f"Object {info['person_id']}: {info['yolo_class']} "
                info_text += f"(YOLO置信度: {info['yolo_confidence']:.2f})\n"
        
        print(f"处理完成: {len(result_images)} 张图像，{len(all_detection_info)} 个检测结果")
        
        return (result_tensor, info_text)

NODE_CLASS_MAPPINGS = {
    "YOLOGenderDetection": YOLOGenderDetection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOGenderDetection": "YOLO Gender Detection by HHY"
}
