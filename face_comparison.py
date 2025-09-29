import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths
import time
import json

# 添加模型路径
yolo_model_dir = os.path.join(folder_paths.models_dir, "yolo")
folder_paths.add_model_folder_path("yolo", yolo_model_dir)

if not os.path.exists(yolo_model_dir):
    os.makedirs(yolo_model_dir, exist_ok=True)

def tensor2pil(image):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def fix_torch_load_for_ultralytics():
    """修复YOLO模型加载的兼容性问题"""
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
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            else:
                raise e
    torch.load = patched_torch_load
    torch.load._original_torch_load = original_torch_load

def detect_faces_yolo(image, model_name, confidence=0.5):
    """使用YOLO模型检测人脸"""
    try:
        fix_torch_load_for_ultralytics()
        from ultralytics import YOLO
        
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            model_path = model_name
        
        yolo_model = YOLO(model_path)
        results = yolo_model(image, conf=confidence, retina_masks=True)
        
        faces = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence_score = float(box.conf[0])
                    faces.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence_score
                    })
        
        return faces
    except Exception as e:
        raise Exception(f"YOLO人脸检测失败: {e}")

def init_insightface(model_name="buffalo_l", det_size=640):
    """初始化InsightFace模型"""
    try:
        import insightface
        app = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        return app
    except Exception as e:
        raise Exception(f"InsightFace模型加载失败: {e}")

def extract_face_features(image, insightface_app):
    """使用InsightFace提取人脸特征"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        faces = insightface_app.get(image)
        
        if len(faces) > 0:
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return {
                'embedding': face.normed_embedding,
                'bbox': face.bbox,
                'confidence': face.det_score
            }
        else:
            raise Exception("未检测到人脸")
    except Exception as e:
        raise Exception(f"特征提取失败: {e}")

def calculate_similarity(face1, face2):
    """计算人脸相似度 - 纯特征向量"""
    embedding1 = face1['embedding']
    embedding2 = face2['embedding']
    
    # 纯特征向量相似度（不使用置信度权重）
    similarity = float(np.dot(embedding1, embedding2))
    
    return similarity

def resize_keep_ratio(image, target_size):
    """等比例缩放"""
    w, h = image.size
    tw, th = target_size
    scale = min(tw / w, th / h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    result = Image.new('RGB', target_size, color='white')
    x_offset = (tw - new_w) // 2
    y_offset = (th - new_h) // 2
    result.paste(resized, (x_offset, y_offset))
    
    return result

def safe_json_convert(obj):
    """安全的JSON类型转换"""
    if isinstance(obj, (np.integer, np.signedinteger)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_font(size):
    """加载字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            continue
    return ImageFont.load_default()

def create_result_image(ref_image, test_image, ref_faces, test_faces, similarity, is_same_person):
    """创建结果图像"""
    target_size = (300, 300)
    ref_resized = resize_keep_ratio(ref_image, target_size)
    test_resized = resize_keep_ratio(test_image, target_size)
    
    width = target_size[0] * 2 + 50
    height = target_size[1] + 100
    result = Image.new('RGB', (width, height), color='white')
    
    result.paste(ref_resized, (0, 0))
    result.paste(test_resized, (target_size[0] + 50, 0))
    
    draw = ImageDraw.Draw(result)
    font = load_font(16)
    
    # 绘制YOLO检测框
    if ref_faces:
        # 计算缩放比例
        ref_scale_x = target_size[0] / ref_image.size[0]
        ref_scale_y = target_size[1] / ref_image.size[1]
        scale = min(ref_scale_x, ref_scale_y)
        
        # 计算居中偏移
        scaled_w = int(ref_image.size[0] * scale)
        scaled_h = int(ref_image.size[1] * scale)
        x_offset = (target_size[0] - scaled_w) // 2
        y_offset = (target_size[1] - scaled_h) // 2
        
        # 绘制参考图像的检测框
        best_ref_face = max(ref_faces, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_ref_face['bbox']
        scaled_box = [
            x1 * scale + x_offset,
            y1 * scale + y_offset,
            x2 * scale + x_offset,
            y2 * scale + y_offset
        ]
        draw.rectangle(scaled_box, outline='red', width=2)
    
    if test_faces:
        # 计算缩放比例
        test_scale_x = target_size[0] / test_image.size[0]
        test_scale_y = target_size[1] / test_image.size[1]
        scale = min(test_scale_x, test_scale_y)
        
        # 计算居中偏移
        scaled_w = int(test_image.size[0] * scale)
        scaled_h = int(test_image.size[1] * scale)
        x_offset = (target_size[0] - scaled_w) // 2
        y_offset = (target_size[1] - scaled_h) // 2
        
        # 绘制测试图像的检测框
        best_test_face = max(test_faces, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_test_face['bbox']
        scaled_box = [
            x1 * scale + x_offset + target_size[0] + 50,
            y1 * scale + y_offset,
            x2 * scale + x_offset + target_size[0] + 50,
            y2 * scale + y_offset
        ]
        draw.rectangle(scaled_box, outline='blue', width=2)
    
    # 添加文字
    draw.text((10, target_size[1] + 10), f"Similarity: {similarity:.4f}", fill='black', font=font)
    
    result_text = "Same Person" if is_same_person else "Different Person"
    color = 'green' if is_same_person else 'red'
    draw.text((10, target_size[1] + 35), result_text, fill=color, font=font)
    
    return result

class FaceComparisonNode:
    """简化的人脸对比节点"""
    
    def __init__(self):
        self.insightface_app = None
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            yolo_models = folder_paths.get_filename_list("yolo")
        except:
            yolo_models = []
        
        if not yolo_models:
            yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "face_yolov8n.pt", "face_yolov8s.pt", "face_yolov8m.pt"]
        
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "test_image": ("IMAGE",),
                "yolo_model": (yolo_models, {"default": yolo_models[0]}),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "detection_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result_image", "result_json")
    FUNCTION = "compare_faces"
    CATEGORY = "hhy/yolo"
    
    def compare_faces(self, reference_image, test_image, yolo_model, similarity_threshold, detection_confidence):
        """人脸对比主函数"""
        try:
            start_time = time.time()
            
            # 初始化InsightFace
            if self.insightface_app is None:
                self.insightface_app = init_insightface()
            
            # 转换图像
            ref_pil = tensor2pil(reference_image)
            test_pil = tensor2pil(test_image)
            
            # YOLO检测人脸
            ref_faces = detect_faces_yolo(ref_pil, yolo_model, detection_confidence)
            test_faces = detect_faces_yolo(test_pil, yolo_model, detection_confidence)
            
            if len(ref_faces) == 0:
                raise Exception("参考图像未检测到人脸")
            if len(test_faces) == 0:
                raise Exception("测试图像未检测到人脸")
            
            # 提取特征
            ref_face = extract_face_features(ref_pil, self.insightface_app)
            test_face = extract_face_features(test_pil, self.insightface_app)
            
            # 计算相似度
            similarity = calculate_similarity(ref_face, test_face)
            is_same_person = similarity >= similarity_threshold
            
            total_time = time.time() - start_time
            
            # 创建结果图像
            result_image = create_result_image(ref_pil, test_pil, ref_faces, test_faces, similarity, is_same_person)
            result_tensor = pil2tensor(result_image)
            
            # 创建JSON结果
            result_json = {
                "is_same_person": safe_json_convert(is_same_person),
                "similarity": safe_json_convert(similarity),
                "threshold": safe_json_convert(similarity_threshold),
                "processing_time": safe_json_convert(total_time),
                "ref_face_count": safe_json_convert(len(ref_faces)),
                "test_face_count": safe_json_convert(len(test_faces)),
                "ref_confidence": safe_json_convert(ref_face['confidence']),
                "test_confidence": safe_json_convert(test_face['confidence']),
                "algorithm": "pure_embedding_similarity"
            }
            
            return (result_tensor, json.dumps(result_json, indent=2))
            
        except Exception as e:
            # 错误处理
            error_image = Image.new('RGB', (600, 400), color='white')
            draw = ImageDraw.Draw(error_image)
            font = load_font(20)
            draw.text((50, 150), f"Error: {str(e)}", fill='red', font=font)
            
            error_json = {
                "is_same_person": False,
                "similarity": 0.0,
                "threshold": safe_json_convert(similarity_threshold),
                "error": str(e)
            }
            
            return (pil2tensor(error_image), json.dumps(error_json, indent=2))

# 节点映射
NODE_CLASS_MAPPINGS = {
    "FaceComparisonNode": FaceComparisonNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceComparisonNode": "🔍 人脸对比识别",
}