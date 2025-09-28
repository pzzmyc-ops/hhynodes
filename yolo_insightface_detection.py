import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import folder_paths
from typing import Dict, List, Tuple, Optional

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
        self.device = self.get_device()
        self.check_onnxruntime_gpu()
        print(f"[YOLO InsightFace Detection] 使用设备: {self.device}")
    
    def get_device(self):
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[YOLO InsightFace Detection] 检测到CUDA，GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = "cpu"
            print("[YOLO InsightFace Detection] 未检测到CUDA，使用CPU")
        return device
    
    def check_onnxruntime_gpu(self):
        try:
            import onnxruntime as ort
            version = ort.__version__
            providers = ort.get_available_providers()
            print(f"[YOLO InsightFace Detection] ONNX Runtime版本: {version}")
            if 'CUDAExecutionProvider' in providers:
                print("[YOLO InsightFace Detection] ✓ ONNX Runtime GPU支持已安装")
            else:
                print("[YOLO InsightFace Detection] ⚠ ONNX Runtime GPU支持未安装")
                if self.device == "cuda":
                    print("[YOLO InsightFace Detection] 建议安装GPU版本:")
                    print("  pip uninstall onnxruntime")
                    print("  pip install onnxruntime-gpu")
                    print("  或者如果使用conda:")
                    print("  conda install onnxruntime-gpu -c conda-forge")
        except ImportError:
            print("[YOLO InsightFace Detection] ⚠ ONNX Runtime未安装")
            print("[YOLO InsightFace Detection] 建议安装:")
            if self.device == "cuda":
                print("  pip install onnxruntime-gpu")
            else:
                print("  pip install onnxruntime")
    
    def verify_gpu_usage(self):
        if self.device == "cuda" and self.yolo_model is not None:
            try:
                model_device = next(self.yolo_model.model.parameters()).device
                print(f"[YOLO InsightFace Detection] YOLO模型当前设备: {model_device}")
                return str(model_device).startswith('cuda')
            except:
                print("[YOLO InsightFace Detection] 无法验证YOLO模型设备")
                return False
        return self.device == "cpu"
    
    def verify_insightface_device(self, providers, ctx_id):
        try:
            if self.insightface_app is not None:
                if hasattr(self.insightface_app, 'models'):
                    for model_name, model in self.insightface_app.models.items():
                        if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                            actual_providers = model.session.get_providers()
                            print(f"[YOLO InsightFace Detection] {model_name} 实际使用提供者: {actual_providers}")
                            if 'CUDAExecutionProvider' in actual_providers:
                                print(f"[YOLO InsightFace Detection] ✓ {model_name} 成功使用GPU")
                            else:
                                print(f"[YOLO InsightFace Detection] ⚠ {model_name} 使用CPU")
                print(f"[YOLO InsightFace Detection] InsightFace Context ID: {ctx_id}")
                print(f"[YOLO InsightFace Detection] 请求的提供者: {providers}")
        except Exception as e:
            print(f"[YOLO InsightFace Detection] 无法验证InsightFace设备: {e}")
        
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
                "smooth_window_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 21,
                    "step": 2,
                    "display": "number"
                }),
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("detection_image", "male_only_image", "female_only_image", "yolo_mask", "male_mask", "female_mask", "detection_info")
    FUNCTION = "detect_and_analyze_faces"
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
                print(f"[YOLO InsightFace Detection] 加载YOLO模型: {model_name}")
                model_path = folder_paths.get_full_path("yolo", model_name)
                if model_path is None:
                    print(f"[YOLO InsightFace Detection] 模型 {model_name} 未找到，正在下载...")
                    model_path = model_name
                else:
                    print(f"[YOLO InsightFace Detection] 使用本地YOLO模型: {model_path}")
                self.yolo_model = YOLO(model_path)
                if self.device == "cuda" and torch.cuda.is_available():
                    print(f"[YOLO InsightFace Detection] 将YOLO模型移动到GPU")
                    self.yolo_model.to(self.device)
                else:
                    print(f"[YOLO InsightFace Detection] YOLO模型使用CPU")
                self.yolo_model.model_name = model_name
                print(f"[YOLO InsightFace Detection] YOLO模型加载完成，设备: {self.device}")
                if self.verify_gpu_usage():
                    print("[YOLO InsightFace Detection] ✓ YOLO模型已成功部署到GPU")
                else:
                    print("[YOLO InsightFace Detection] ⚠ YOLO模型未在GPU上运行")
            return self.yolo_model
        except ImportError:
            raise ImportError("请安装ultralytics库: pip install ultralytics")
    
    def load_insightface_model(self):
        if self.insightface_app is not None:
            return self.insightface_app
        try:
            import insightface
        except ImportError:
            raise ImportError("请安装insightface库: pip install insightface")
        try:
            print("[YOLO InsightFace Detection] 正在初始化InsightFace模型...")
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                print(f"[YOLO InsightFace Detection] ONNX Runtime可用提供者: {available_providers}")
                cuda_available = 'CUDAExecutionProvider' in available_providers
                print(f"[YOLO InsightFace Detection] CUDA执行提供者可用: {cuda_available}")
            except ImportError:
                print("[YOLO InsightFace Detection] 警告: 无法导入onnxruntime，可能影响GPU检测")
                cuda_available = False
            if self.device == "cuda" and torch.cuda.is_available() and cuda_available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
                print("[YOLO InsightFace Detection] InsightFace配置为使用GPU加速")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                if self.device == "cuda":
                    print("[YOLO InsightFace Detection] 警告: GPU可用但ONNX Runtime CUDA不可用，InsightFace使用CPU")
                else:
                    print("[YOLO InsightFace Detection] InsightFace使用CPU")
            self.insightface_app = insightface.app.FaceAnalysis(
                providers=providers,
                root=insightface_model_dir,
                name='buffalo_l'
            )
            self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.verify_insightface_device(providers, ctx_id)
            print(f"[YOLO InsightFace Detection] InsightFace版本: {insightface.__version__}")
            print("[YOLO InsightFace Detection] InsightFace模型加载成功")
            return self.insightface_app
        except Exception as e:
            print(f"InsightFace模型加载失败: {e}")
            print("正在尝试下载默认模型...")
            try:
                self.insightface_app = insightface.app.FaceAnalysis(
                    providers=providers,
                    root=insightface_model_dir,
                    name='buffalo_l'
                )
                self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                print("InsightFace模型下载并加载成功")
                return self.insightface_app
            except Exception as e2:
                raise ValueError(f"InsightFace模型加载失败: {e2}")
    
    def analyze_face_with_insightface(self, face_crop):
        try:
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            else:
                face_bgr = face_crop
            print(f"InsightFace分析图像尺寸: {face_bgr.shape}")
            faces = self.insightface_app.get(face_bgr)
            if len(faces) == 0:
                print("InsightFace未检测到人脸")
                return {
                    "gender": "Unknown",
                    "age": 0,
                    "confidence": 0.0,
                    "embedding": None
                }
            face = faces[0]
            print(f"Face对象属性: {dir(face)}")
            gender = "Unknown"
            if hasattr(face, 'sex'):
                sex_value = face.sex
                print(f"face.sex = {sex_value} (类型: {type(sex_value)})")
                if isinstance(sex_value, str):
                    gender = "Man" if sex_value.upper() == 'M' else "Woman"
                elif isinstance(sex_value, (int, float)):
                    gender = "Man" if sex_value == 1 else "Woman"
                else:
                    print(f"未知的性别值类型: {type(sex_value)}")
            elif hasattr(face, 'gender'):
                gender_value = face.gender
                print(f"face.gender = {gender_value} (类型: {type(gender_value)})")
                if isinstance(gender_value, str):
                    gender = "Man" if gender_value.upper() == 'M' else "Woman"
                elif isinstance(gender_value, (int, float)):
                    gender = "Man" if gender_value == 1 else "Woman"
                else:
                    print(f"未知的性别值类型: {type(gender_value)}")
            else:
                print("未找到性别属性")
            age = 0
            if hasattr(face, 'age'):
                age = int(face.age)
                print(f"face.age = {age}")
            else:
                print("未找到年龄属性")
            confidence = 1.0
            if hasattr(face, 'det_score'):
                confidence = float(face.det_score)
                print(f"face.det_score = {confidence}")
            elif hasattr(face, 'score'):
                confidence = float(face.score)
                print(f"face.score = {confidence}")
            else:
                print("未找到置信度属性")
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
        draw = ImageDraw.Draw(yolo_image_pil)
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
            if analysis_result["gender"] == "Man":
                gender_color = (0, 100, 255)
            elif analysis_result["gender"] == "Woman":
                gender_color = (255, 20, 147)
            else:
                gender_color = (128, 128, 128)
            if analysis_result["confidence"] >= confidence_threshold:
                analysis_label = f"{analysis_result['gender']}: {analysis_result['confidence']:.2f}"
                if analysis_result["age"] > 0:
                    analysis_label += f" (Age: {analysis_result['age']})"
            else:
                analysis_label = f"Low Conf: {analysis_result['confidence']:.2f}"
            text_bbox = draw.textbbox((0, 0), analysis_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            analysis_x = x1 + 5
            analysis_y = y2 - text_height - 10
            if analysis_y < 0:
                analysis_y = y1 + 5
            draw.rectangle([analysis_x - 2, analysis_y - 2, analysis_x + text_width + 4, analysis_y + text_height + 2], 
                         fill=gender_color)
            draw.text((analysis_x, analysis_y), analysis_label, fill=(255, 255, 255), font=font)
            if analysis_result["age"] > 0:
                detail_text = f"Gender: {analysis_result['gender']}, Age: {analysis_result['age']}"
            else:
                detail_text = f"Gender: {analysis_result['gender']}"
            detail_x = x1 + 5
            detail_y = y1 + 5
            detail_bbox = draw.textbbox((0, 0), detail_text, font=small_font)
            detail_width = detail_bbox[2] - detail_bbox[0]
            detail_height = detail_bbox[3] - detail_bbox[1]
            draw.rectangle([detail_x - 2, detail_y - 2, detail_x + detail_width + 4, detail_y + detail_height + 2], 
                         fill=(0, 0, 0, 180))
            draw.text((detail_x, detail_y), detail_text, fill=gender_color, font=small_font)
            detection_info.append({
                "person_id": i + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "gender": analysis_result["gender"],
                "age": analysis_result["age"],
                "confidence": analysis_result["confidence"],
                "has_embedding": analysis_result["embedding"] is not None
            })
        return yolo_image_pil, detection_info
    
    def create_cropped_image(self, original_image_pil, boxes, analysis_results, target_gender, reference_size=None, expand_pixels=0, smoothed_centers=None, img_idx=None):
        img_array = np.array(original_image_pil)
        target_boxes = []
        for box, analysis_result in zip(boxes, analysis_results):
            if analysis_result["gender"] == target_gender:
                target_boxes.append(box)
        
        if not target_boxes:
            if reference_size:
                # 计算外扩后的尺寸用于空白图像
                final_width = reference_size[0] + expand_pixels * 2
                final_height = reference_size[1] + expand_pixels * 2
                return Image.new('RGB', (final_width, final_height), (0, 0, 0))
            else:
                return Image.new('RGB', original_image_pil.size, (0, 0, 0))
        
        if reference_size is None:
            first_box = target_boxes[0]
            x1, y1, x2, y2 = first_box
            reference_size = (x2 - x1, y2 - y1)
            print(f"设置参考尺寸为第一个{target_gender}检测框: {reference_size}")
        
        ref_width, ref_height = reference_size
        # 计算外扩后的最终尺寸
        final_width = ref_width + expand_pixels * 2
        final_height = ref_height + expand_pixels * 2
        
        print(f"{target_gender}裁剪: 基础{ref_width}x{ref_height} + 外扩{expand_pixels}px = 最终{final_width}x{final_height}")
        
        cropped_faces = []
        
        for i, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box
            current_width = x2 - x1
            current_height = y2 - y1
            
            # 使用平滑后的中心点（如果提供）
            if smoothed_centers and img_idx is not None:
                # 查找当前帧对应的平滑中心点
                smoothed_center = None
                for center_data in smoothed_centers:
                    if center_data[2] == img_idx:  # 匹配帧索引
                        smoothed_center = center_data
                        break
                
                if smoothed_center:
                    center_x, center_y = smoothed_center[0], smoothed_center[1]
                    print(f"{target_gender}使用平滑中心点: ({center_x},{center_y}) vs 原始({(x1+x2)//2},{(y1+y2)//2})")
                else:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    print(f"{target_gender}未找到平滑中心点，使用原始中心点: ({center_x},{center_y})")
            else:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            
            # 智能位移算法：预检查外扩区域，必要时移动裁剪框
            ideal_x1 = center_x - final_width // 2
            ideal_y1 = center_y - final_height // 2
            ideal_x2 = ideal_x1 + final_width
            ideal_y2 = ideal_y1 + final_height
            
            # 计算需要的位移量
            shift_x = 0
            shift_y = 0
            
            # X轴位移计算
            if ideal_x1 < 0:
                shift_x = -ideal_x1  # 向右移动
                print(f"{target_gender}外扩超出左边界{ideal_x1}，需要向右移动{shift_x}像素")
            elif ideal_x2 > original_image_pil.width:
                shift_x = original_image_pil.width - ideal_x2  # 向左移动
                print(f"{target_gender}外扩超出右边界{ideal_x2-original_image_pil.width}，需要向左移动{-shift_x}像素")
            
            # Y轴位移计算
            if ideal_y1 < 0:
                shift_y = -ideal_y1  # 向下移动
                print(f"{target_gender}外扩超出上边界{ideal_y1}，需要向下移动{shift_y}像素")
            elif ideal_y2 > original_image_pil.height:
                shift_y = original_image_pil.height - ideal_y2  # 向上移动
                print(f"{target_gender}外扩超出下边界{ideal_y2-original_image_pil.height}，需要向上移动{-shift_y}像素")
            
            # 应用位移，计算最终裁剪区域
            final_x1 = ideal_x1 + shift_x
            final_y1 = ideal_y1 + shift_y
            final_x2 = final_x1 + final_width
            final_y2 = final_y1 + final_height
            
            # 最终安全检查（理论上不应该超出）
            final_x1 = max(0, min(final_x1, original_image_pil.width - final_width))
            final_y1 = max(0, min(final_y1, original_image_pil.height - final_height))
            final_x2 = final_x1 + final_width
            final_y2 = final_y1 + final_height
            
            # 检查最终区域是否仍然超出（这种情况说明图片太小，无法容纳外扩）
            if final_x2 > original_image_pil.width or final_y2 > original_image_pil.height:
                print(f"{target_gender}图片太小无法容纳外扩，回退到基础尺寸")
                # 回退到基础尺寸
                base_x1 = center_x - ref_width // 2
                base_y1 = center_y - ref_height // 2
                base_x2 = base_x1 + ref_width
                base_y2 = base_y1 + ref_height
                
                # 确保基础尺寸在边界内
                base_x1 = max(0, min(base_x1, original_image_pil.width - ref_width))
                base_y1 = max(0, min(base_y1, original_image_pil.height - ref_height))
                base_x2 = base_x1 + ref_width
                base_y2 = base_y1 + ref_height
                
                # 裁剪基础区域
                base_crop = img_array[base_y1:base_y2, base_x1:base_x2]
                
                # 创建最终尺寸的画布，将基础裁剪居中放置
                final_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                start_y = (final_height - ref_height) // 2
                start_x = (final_width - ref_width) // 2
                final_image[start_y:start_y+ref_height, start_x:start_x+ref_width] = base_crop
                
                cropped_faces.append(final_image)
                print(f"{target_gender}使用基础尺寸居中放置")
            else:
                # 成功进行智能位移外扩
                cropped_face = img_array[final_y1:final_y2, final_x1:final_x2]
                cropped_faces.append(cropped_face)
                
                if shift_x != 0 or shift_y != 0:
                    print(f"{target_gender}智能位移成功: 原中心({center_x},{center_y}) -> 调整后区域({final_x1},{final_y1})到({final_x2},{final_y2})")
                else:
                    print(f"{target_gender}外扩无需位移: 区域({final_x1},{final_y1})到({final_x2},{final_y2})")
            
            print(f"原始框({current_width}x{current_height}) -> 外扩后({final_width}x{final_height})")
        
        if len(cropped_faces) == 1:
            result_array = cropped_faces[0]
        else:
            result_array = np.concatenate(cropped_faces, axis=1)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def create_gender_mask_on_original(self, original_image_pil, boxes, analysis_results, target_gender, reference_size, expand_pixels=0, smoothed_centers=None, img_idx=None):
        """在原始图像尺寸上创建指定性别的智能裁剪区域mask"""
        # 创建与原始图像相同尺寸的黑色mask
        mask = Image.new('L', original_image_pil.size, 0)
        
        # 找到所有目标性别的检测框
        target_boxes = []
        for box, analysis_result in zip(boxes, analysis_results):
            if analysis_result["gender"] == target_gender:
                target_boxes.append(box)
        
        if target_boxes and reference_size:
            ref_width, ref_height = reference_size
            # 计算外扩后的尺寸
            final_width = ref_width + expand_pixels * 2
            final_height = ref_height + expand_pixels * 2
            
            draw = ImageDraw.Draw(mask)
            
            for i, box in enumerate(target_boxes):
                x1, y1, x2, y2 = box
                
                # 使用平滑后的中心点（如果提供）
                if smoothed_centers and img_idx is not None:
                    # 查找当前帧对应的平滑中心点
                    smoothed_center = None
                    for center_data in smoothed_centers:
                        if center_data[2] == img_idx:  # 匹配帧索引
                            smoothed_center = center_data
                            break
                    
                    if smoothed_center:
                        center_x, center_y = smoothed_center[0], smoothed_center[1]
                    else:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                else:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                
                # 智能位移算法：与create_cropped_image保持一致
                ideal_x1 = center_x - final_width // 2
                ideal_y1 = center_y - final_height // 2
                ideal_x2 = ideal_x1 + final_width
                ideal_y2 = ideal_y1 + final_height
                
                # 计算需要的位移量
                shift_x = 0
                shift_y = 0
                
                # X轴位移计算
                if ideal_x1 < 0:
                    shift_x = -ideal_x1
                elif ideal_x2 > original_image_pil.width:
                    shift_x = original_image_pil.width - ideal_x2
                
                # Y轴位移计算
                if ideal_y1 < 0:
                    shift_y = -ideal_y1
                elif ideal_y2 > original_image_pil.height:
                    shift_y = original_image_pil.height - ideal_y2
                
                # 应用位移，计算最终mask区域
                final_x1 = ideal_x1 + shift_x
                final_y1 = ideal_y1 + shift_y
                final_x2 = final_x1 + final_width
                final_y2 = final_y1 + final_height
                
                # 最终安全检查
                final_x1 = max(0, min(final_x1, original_image_pil.width - final_width))
                final_y1 = max(0, min(final_y1, original_image_pil.height - final_height))
                final_x2 = final_x1 + final_width
                final_y2 = final_y1 + final_height
                
                # 检查是否需要回退到基础尺寸
                if final_x2 > original_image_pil.width or final_y2 > original_image_pil.height:
                    # 回退到基础尺寸
                    base_x1 = max(0, min(center_x - ref_width // 2, original_image_pil.width - ref_width))
                    base_y1 = max(0, min(center_y - ref_height // 2, original_image_pil.height - ref_height))
                    base_x2 = base_x1 + ref_width
                    base_y2 = base_y1 + ref_height
                    
                    draw.rectangle([base_x1, base_y1, base_x2, base_y2], fill=255)
                    print(f"在原图上创建{target_gender}回退mask: ({base_x1},{base_y1})到({base_x2},{base_y2}), 基础尺寸{ref_width}x{ref_height}")
                else:
                    # 成功进行智能位移外扩
                    draw.rectangle([final_x1, final_y1, final_x2, final_y2], fill=255)
                    if shift_x != 0 or shift_y != 0:
                        print(f"在原图上创建{target_gender}位移外扩mask: ({final_x1},{final_y1})到({final_x2},{final_y2}), 位移({shift_x},{shift_y})")
                    else:
                        print(f"在原图上创建{target_gender}外扩mask: ({final_x1},{final_y1})到({final_x2},{final_y2}), 基础{ref_width}x{ref_height}+外扩{expand_pixels}px")
        
        return image2mask(mask)
    
    def detect_and_analyze_faces(self, image, yolo_model="yolov8n.pt", confidence=0.5, 
                                iou_threshold=0.45, enable_face_analysis=True, 
                                analysis_confidence_threshold=0.6, smooth_window_size=5, expand_pixels=0):
        yolo = self.load_yolo_model(yolo_model)
        if enable_face_analysis:
            insightface_app = self.load_insightface_model()
        print(f"[YOLO InsightFace Detection] 第一阶段：收集所有图片的检测结果... (设备: {self.device})")
        if self.device == "cuda":
            print(f"[YOLO InsightFace Detection] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        all_image_data = []
        for img_idx, img_tensor in enumerate(image):
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            results = yolo(pil_image, conf=confidence, iou=iou_threshold, retina_masks=True)
            image_data = {
                'img_idx': img_idx,
                'pil_image': pil_image,
                'img_tensor': img_tensor,
                'boxes': [],
                'analysis_results': [],
                'individual_masks': [],
                'yolo_result': None
            }
            for result in results:
                image_data['yolo_result'] = result
                boxes = []
                analysis_results = []
                individual_masks = []
                if result.masks is not None and len(result.masks) > 0:
                    print(f"图片{img_idx}: 检测到 {len(result.masks)} 个分割mask")
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask_pil = np2pil(_mask).convert("L")
                        mask_tensor = image2mask(_mask_pil)
                        individual_masks.append(mask_tensor)
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"图片{img_idx}: YOLO检测到 {len(result.boxes)} 个对象")
                    if result.masks is None or len(result.masks) == 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            _mask = Image.new('L', pil_image.size, "black")
                            white_image = Image.new('L', pil_image.size, "white")
                            _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                            mask_tensor = image2mask(_mask)
                            individual_masks.append(mask_tensor)
                    for i, box_data in enumerate(result.boxes):
                        class_id = int(box_data.cls[0])
                        class_name = yolo.names[class_id]
                        box_confidence = float(box_data.conf[0])
                        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(pil_image.width, x2)
                        y2 = min(pil_image.height, y2)
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        boxes.append([x1, y1, x2, y2])
                        if enable_face_analysis and i < len(individual_masks):
                            crop_img = np.array(pil_image)[y1:y2, x1:x2]
                            if crop_img.size > 0:
                                analysis_result = self.analyze_face_with_insightface(crop_img)
                                analysis_results.append(analysis_result)
                                print(f"图片{img_idx}: 检测到 {class_name}: {analysis_result['gender']} "
                                      f"(置信度: {analysis_result['confidence']:.2f})")
                            else:
                                analysis_results.append({
                                    "gender": "Unknown",
                                    "age": 0,
                                    "confidence": 0.0,
                                    "embedding": None
                                })
                image_data['boxes'] = boxes
                image_data['analysis_results'] = analysis_results
                image_data['individual_masks'] = individual_masks
            all_image_data.append(image_data)
        print("第二阶段：智能计算无黑边裁剪尺寸...")
        male_max_size = None
        female_max_size = None
        
        # 第一步：找到最大检测框尺寸 a
        for image_data in all_image_data:
            if enable_face_analysis:
                for box, analysis_result in zip(image_data['boxes'], image_data['analysis_results']):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    if analysis_result["gender"] == "Man":
                        if male_max_size is None or (width * height) > (male_max_size[0] * male_max_size[1]):
                            male_max_size = (width, height)
                            print(f"发现男性最大尺寸: {male_max_size}")
                    elif analysis_result["gender"] == "Woman":
                        if female_max_size is None or (width * height) > (female_max_size[0] * female_max_size[1]):
                            female_max_size = (width, height)
                            print(f"发现女性最大尺寸: {female_max_size}")

        # 第二步：智能计算无黑边的最优裁剪尺寸
        def calculate_no_blackbar_size(target_gender, max_size):
            if max_size is None:
                return None
            
            print(f"\n=== 开始智能计算 {target_gender} 的无黑边裁剪尺寸 ===")
            max_width, max_height = max_size
            print(f"原始最大尺寸: {max_width}x{max_height}")
            
            # 收集所有边界约束
            width_constraints = []
            height_constraints = []
            
            for image_data in all_image_data:
                img_width = image_data['pil_image'].width
                img_height = image_data['pil_image'].height
                
                if enable_face_analysis:
                    for box, analysis_result in zip(image_data['boxes'], image_data['analysis_results']):
                        if analysis_result["gender"] == target_gender:
                            x1, y1, x2, y2 = box
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # 计算从中心点能够裁剪的最大尺寸（不超出边界）
                            # 宽度限制：中心点到左右边界的距离 * 2
                            max_width_from_center = min(center_x, img_width - center_x) * 2
                            # 高度限制：中心点到上下边界的距离 * 2  
                            max_height_from_center = min(center_y, img_height - center_y) * 2
                            
                            # 如果原始最大尺寸会超出边界，记录约束
                            if max_width > max_width_from_center:
                                width_constraints.append(max_width_from_center)
                                print(f"宽度约束: 图片{image_data['img_idx']}, 中心({center_x},{center_y}), "
                                      f"图片宽{img_width}, 最大可裁剪宽度{max_width_from_center}")
                            
                            if max_height > max_height_from_center:
                                height_constraints.append(max_height_from_center)
                                print(f"高度约束: 图片{image_data['img_idx']}, 中心({center_x},{center_y}), "
                                      f"图片高{img_height}, 最大可裁剪高度{max_height_from_center}")
            
            # 第三步：计算修正后的尺寸
            # 对于宽度，无论是否有约束，都要考虑所有约束的最小值
            all_width_limits = [max_width]  # 包含原始最大尺寸
            if width_constraints:
                all_width_limits.extend(width_constraints)
            
            optimal_width = min(all_width_limits)
            print(f"宽度计算: 原始{max_width}, 约束{width_constraints}, 最终{optimal_width}")
            
            # 对于高度，同样处理
            all_height_limits = [max_height]  # 包含原始最大尺寸  
            if height_constraints:
                all_height_limits.extend(height_constraints)
            
            optimal_height = min(all_height_limits)
            print(f"高度计算: 原始{max_height}, 约束{height_constraints}, 最终{optimal_height}")
            
            # 调整为偶数
            optimal_width = optimal_width if optimal_width % 2 == 0 else optimal_width - 1
            optimal_height = optimal_height if optimal_height % 2 == 0 else optimal_height - 1
            
            print(f"{target_gender} 最终无黑边裁剪尺寸: {optimal_width}x{optimal_height}")
            print(f"=== {target_gender} 智能计算完成 ===\n")
            
            return (optimal_width, optimal_height)
        
        # 分别计算男性和女性的无黑边基础尺寸（不包含外扩）
        male_base_size = calculate_no_blackbar_size("Man", male_max_size)
        female_base_size = calculate_no_blackbar_size("Woman", female_max_size)
        
        print(f"智能算法计算的基础尺寸（不含外扩）:")
        print(f"  男性基础尺寸: {male_base_size}")
        print(f"  女性基础尺寸: {female_base_size}")
        print(f"  外扩像素: {expand_pixels}")
        
        # 收集所有帧的中心点数据用于平滑跟踪
        male_centers = []
        female_centers = []
        
        for image_data in all_image_data:
            if enable_face_analysis:
                for box, analysis_result in zip(image_data['boxes'], image_data['analysis_results']):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if analysis_result["gender"] == "Man":
                        male_centers.append((center_x, center_y, image_data['img_idx']))
                    elif analysis_result["gender"] == "Woman":
                        female_centers.append((center_x, center_y, image_data['img_idx']))
        
        # 应用平滑滤波
        def smooth_centers(centers, window_size=5):
            if len(centers) <= 1:
                return centers
            
            print(f"对{len(centers)}个中心点进行平滑处理，窗口大小:{window_size}")
            smoothed = []
            
            for i in range(len(centers)):
                # 计算滑动窗口范围
                start = max(0, i - window_size // 2)
                end = min(len(centers), i + window_size // 2 + 1)
                
                # 计算窗口内的平均值
                window_centers = centers[start:end]
                avg_x = sum(c[0] for c in window_centers) / len(window_centers)
                avg_y = sum(c[1] for c in window_centers) / len(window_centers)
                
                smoothed.append((int(avg_x), int(avg_y), centers[i][2]))  # 保留原始帧索引
            
            return smoothed
        
        # 对中心点进行平滑处理
        if male_centers:
            male_centers = smooth_centers(male_centers, smooth_window_size)
            print(f"男性中心点平滑完成: {len(male_centers)}个点，窗口大小:{smooth_window_size}")
        
        if female_centers:
            female_centers = smooth_centers(female_centers, smooth_window_size)
            print(f"女性中心点平滑完成: {len(female_centers)}个点，窗口大小:{smooth_window_size}")
        
        # 外扩将在裁剪阶段基于原图空间进行
        male_max_size = male_base_size
        female_max_size = female_base_size
        print("第三阶段：基于最大尺寸处理所有图片...")
        result_images = []
        male_only_images = []
        female_only_images = []
        all_yolo_masks = []
        all_male_masks = []
        all_female_masks = []
        all_detection_info = []
        for image_data in all_image_data:
            img_idx = image_data['img_idx']
            pil_image = image_data['pil_image']
            boxes = image_data['boxes']
            analysis_results = image_data['analysis_results']
            individual_masks = image_data['individual_masks']
            yolo_result = image_data['yolo_result']
            print(f"处理图片 {img_idx}: {len(boxes)} 个检测框")
            if yolo_result:
                yolo_plot_image = cv2.cvtColor(yolo_result.plot(), cv2.COLOR_BGR2RGB)
                result_image_pil = Image.fromarray(yolo_plot_image)
            else:
                result_image_pil = pil_image
            for mask_tensor in individual_masks:
                all_yolo_masks.append(mask_tensor)
            if not individual_masks:
                empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                all_yolo_masks.append(empty_mask)
                if enable_face_analysis and boxes and analysis_results:
                    result_image_pil, detection_info = self.add_analysis_labels_to_yolo_image(
                        result_image_pil, boxes, analysis_results, analysis_confidence_threshold
                    )
                    all_detection_info.extend(detection_info)
            elif not enable_face_analysis and boxes and yolo_result:
                for i, box in enumerate(boxes):
                    all_detection_info.append({
                        "person_id": i + 1,
                        "bbox": [int(x) for x in box],
                        "yolo_class": yolo.names[int(yolo_result.boxes[i].cls[0])],
                        "yolo_confidence": float(yolo_result.boxes[i].conf[0])
                    })
                print(f"YOLO检测完成，未启用人脸分析")
            else:
                print("未检测到符合条件的对象")
            result_images.append(pil2tensor(result_image_pil))
            if enable_face_analysis and boxes and analysis_results:
                male_only_image = self.create_cropped_image(pil_image, boxes, analysis_results, "Man", male_max_size, expand_pixels, male_centers, img_idx)
                male_only_images.append(pil2tensor(male_only_image))
                female_only_image = self.create_cropped_image(pil_image, boxes, analysis_results, "Woman", female_max_size, expand_pixels, female_centers, img_idx)
                female_only_images.append(pil2tensor(female_only_image))
                
                # 基于原始图像创建智能裁剪区域的性别mask
                male_mask = self.create_gender_mask_on_original(pil_image, boxes, analysis_results, "Man", male_max_size, expand_pixels, male_centers, img_idx)
                all_male_masks.append(male_mask)
                female_mask = self.create_gender_mask_on_original(pil_image, boxes, analysis_results, "Woman", female_max_size, expand_pixels, female_centers, img_idx)
                all_female_masks.append(female_mask)
                print("生成性别专用图像和基于原始图像的mask完成")
            else:
                if male_max_size:
                    # 计算外扩后的尺寸
                    male_final_width = male_max_size[0] + expand_pixels * 2
                    male_final_height = male_max_size[1] + expand_pixels * 2
                    male_blank = Image.new('RGB', (male_final_width, male_final_height), (0, 0, 0))
                    male_only_images.append(pil2tensor(male_blank))
                else:
                    male_only_images.append(pil2tensor(pil_image))
                
                if female_max_size:
                    # 计算外扩后的尺寸
                    female_final_width = female_max_size[0] + expand_pixels * 2
                    female_final_height = female_max_size[1] + expand_pixels * 2
                    female_blank = Image.new('RGB', (female_final_width, female_final_height), (0, 0, 0))
                    female_only_images.append(pil2tensor(female_blank))
                else:
                    female_only_images.append(pil2tensor(pil_image))
                
                # 创建基于原始图像尺寸的空mask（全黑）
                empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                all_male_masks.append(empty_mask)
                all_female_masks.append(empty_mask)
                print("未进行人脸分析，性别专用图像使用空白图像或原图，mask基于原始图像尺寸")
        if result_images:
            result_tensor = torch.cat(result_images, dim=0)
        else:
            result_tensor = image
        if male_only_images:
            male_only_tensor = torch.cat(male_only_images, dim=0)
        else:
            male_only_tensor = image
        if female_only_images:
            female_only_tensor = torch.cat(female_only_images, dim=0)
        else:
            female_only_tensor = image
        if all_yolo_masks:
            yolo_masks_tensor = torch.cat(all_yolo_masks, dim=0)
        else:
            yolo_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        if all_male_masks:
            male_masks_tensor = torch.cat(all_male_masks, dim=0)
        else:
            male_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        if all_female_masks:
            female_masks_tensor = torch.cat(all_female_masks, dim=0)
        else:
            female_masks_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
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

