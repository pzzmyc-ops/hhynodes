import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2cv2(image):
    """将tensor转换为opencv格式"""
    return cv2.cvtColor(np.array(tensor2pil(image)), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    """将opencv图像转换为PIL"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class AgeGenderPredictor:
    """年龄性别预测器类"""
    def __init__(self, model_name="abhilash88/age-gender-prediction"):
        self.model_name = model_name
        self.classifier = None
        self.load_model()

    def load_model(self):
        """加载Hugging Face模型"""
        try:
            from transformers import pipeline
            print(f"正在加载模型: {self.model_name}")
            self.classifier = pipeline(
                "image-classification", 
                model=self.model_name, 
                trust_remote_code=True
            )
            print(f"模型加载成功: {self.model_name}")
        except ImportError:
            raise ImportError("请安装transformers库: pip install transformers")
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")

    def predict(self, image):
        """预测年龄和性别"""
        if self.classifier is None:
            return {"age": "未知", "gender": "未知", "confidence": 0.0}
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif torch.is_tensor(image):
                image = tensor2pil(image)
            
            result = self.classifier(image)
            
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                if isinstance(prediction, dict):
                    age = prediction.get('age', '未知')
                    gender = prediction.get('gender', '未知')
                    confidence = prediction.get('score', 0.0)
                    return {
                        "age": str(age),
                        "gender": str(gender),
                        "confidence": float(confidence)
                    }
                else:
                    return {
                        "age": "需要检查模型输出格式",
                        "gender": "需要检查模型输出格式",
                        "confidence": 0.0
                    }
            else:
                return {"age": "预测失败", "gender": "预测失败", "confidence": 0.0}
                
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return {"age": f"错误: {str(e)}", "gender": "预测失败", "confidence": 0.0}

class AgeGenderPredictionNode:
    """ComfyUI年龄性别预测节点"""
    
    def __init__(self):
        self.predictor = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ([
                    "abhilash88/age-gender-prediction",
                    "dima806/facial_age_image_detection",
                    "nateraw/vit-age-classifier",
                ], {
                    "default": "abhilash88/age-gender-prediction"
                }),
                "show_confidence": ("BOOLEAN", {
                    "default": True,
                    "label_on": "显示置信度",
                    "label_off": "仅显示结果"
                }),
                "font_size": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "text_position": ([
                    "top_left",
                    "top_right", 
                    "bottom_left",
                    "bottom_right",
                    "center"
                ], {
                    "default": "top_left"
                }),
            },
            "optional": {
                "crop_box": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "可选: x1,y1,x2,y2 指定分析区域"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("annotated_image", "age", "gender", "confidence", "combined_result")
    FUNCTION = "predict_age_gender"
    CATEGORY = "hhy/yolo"
    
    def load_predictor(self, model_name):
        """加载预测器"""
        try:
            if self.predictor is None or getattr(self.predictor, 'model_name', None) != model_name:
                print(f"加载年龄性别预测模型: {model_name}")
                self.predictor = AgeGenderPredictor(model_name)
            return self.predictor
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")
    
    def parse_crop_box(self, crop_box_str):
        """解析裁剪框参数"""
        if not crop_box_str.strip():
            return None
        try:
            coords = [int(x.strip()) for x in crop_box_str.split(',')]
            if len(coords) == 4:
                return coords
            else:
                print("警告: 裁剪框参数格式错误，应为 x1,y1,x2,y2")
                return None
        except:
            print(f"警告: 解析裁剪框参数失败: {crop_box_str}")
            return None
    
    def get_text_position(self, image_shape, text_size, position_type):
        """计算文本位置"""
        h, w = image_shape[:2]
        text_w, text_h = text_size
        
        positions = {
            "top_left": (10, text_h + 20),
            "top_right": (w - text_w - 10, text_h + 20),
            "bottom_left": (10, h - 10),
            "bottom_right": (w - text_w - 10, h - 10),
            "center": ((w - text_w) // 2, (h + text_h) // 2)
        }
        
        return positions.get(position_type, positions["top_left"])
    
    def predict_age_gender(self, image, model_name="abhilash88/age-gender-prediction", 
                          show_confidence=True, font_size=30, text_position="top_left", 
                          crop_box=""):
        """年龄性别预测主函数"""
        
        # 加载预测器
        predictor = self.load_predictor(model_name)
        
        # 解析裁剪框
        crop_coords = self.parse_crop_box(crop_box)
        
        annotated_images = []
        ages = []
        genders = []
        confidences = []
        combined_results = []
        
        for img_tensor in image:
            # 转换为PIL图像
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            cv2_image = tensor2cv2(img_tensor_single)
            
            # 确定分析区域
            analysis_image = pil_image
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                # 确保坐标在图像范围内
                h, w = cv2_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                # 裁剪图像用于分析
                analysis_image = pil_image.crop((x1, y1, x2, y2))
            
            # 进行年龄性别预测
            prediction = predictor.predict(analysis_image)
            
            age = prediction["age"]
            gender = prediction["gender"]
            confidence = prediction["confidence"]
            
            # 创建标注图像
            annotated_image = cv2_image.copy()
            
            # 准备显示文本
            if show_confidence:
                display_text = f"Age: {age}, Gender: {gender} ({confidence:.2f})"
            else:
                display_text = f"Age: {age}, Gender: {gender}"
            
            # 设置字体和颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 30.0
            
            # 根据性别设置颜色
            if "female" in str(gender).lower() or "woman" in str(gender).lower():
                color = (0, 255, 0)  # 绿色
            elif "male" in str(gender).lower() or "man" in str(gender).lower():
                color = (255, 0, 0)  # 红色
            else:
                color = (255, 255, 0)  # 黄色（未知）
            
            thickness = max(1, int(font_scale * 2))
            
            # 获取文本尺寸
            text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
            text_w, text_h = text_size
            
            # 计算文本位置
            text_x, text_y = self.get_text_position(annotated_image.shape, (text_w, text_h), text_position)
            
            # 绘制裁剪框（如果指定了）
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                # 在裁剪框附近显示文本
                if text_position == "top_left":
                    text_x, text_y = x1, max(y1 - 10, text_h + 10)
            
            # 绘制文本背景
            cv2.rectangle(annotated_image, 
                        (text_x - 5, text_y - text_h - 5), 
                        (text_x + text_w + 5, text_y + 5), 
                        (0, 0, 0), -1)
            
            # 绘制文本
            cv2.putText(annotated_image, display_text, (text_x, text_y), 
                       font, font_scale, color, thickness, cv2.LINE_AA)
            
            # 转换回PIL并添加到结果列表
            annotated_pil = cv2_to_pil(annotated_image)
            annotated_images.append(pil2tensor(annotated_pil))
            ages.append(str(age))
            genders.append(str(gender))
            confidences.append(float(confidence))
            
            # 创建组合结果
            combined_result = f"年龄: {age}, 性别: {gender}, 置信度: {confidence:.3f}"
            combined_results.append(combined_result)
            
            print(f"年龄性别预测结果: {combined_result}")
        
        # 合并结果
        if annotated_images:
            annotated_images_tensor = torch.cat(annotated_images, dim=0)
            # 返回第一个结果（如果有多张图像，可以修改为返回列表）
            final_age = ages[0] if ages else "未知"
            final_gender = genders[0] if genders else "未知"
            final_confidence = confidences[0] if confidences else 0.0
            final_combined = combined_results[0] if combined_results else "处理失败"
        else:
            # 如果没有结果，返回原图像
            annotated_images_tensor = image
            final_age = "处理失败"
            final_gender = "处理失败"
            final_confidence = 0.0
            final_combined = "处理失败"
        
        return (annotated_images_tensor, final_age, final_gender, final_confidence, final_combined)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "AgeGenderPredictionNode": AgeGenderPredictionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgeGenderPredictionNode": "年龄性别预测 by HHY"
}
