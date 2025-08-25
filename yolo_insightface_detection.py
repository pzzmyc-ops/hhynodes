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
        """检测并返回最佳可用设备"""
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
        """检查ONNX Runtime GPU支持"""
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
        """验证模型是否真的在GPU上运行"""
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
        """验证InsightFace实际使用的设备"""
        try:
            if self.insightface_app is not None:
                # 检查模型的实际提供者
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
        """加载InsightFace人脸分析模型"""
        if self.insightface_app is not None:
            return self.insightface_app
            
        try:
            import insightface
        except ImportError:
            raise ImportError("请安装insightface库: pip install insightface")
        
        try:
            print("[YOLO InsightFace Detection] 正在初始化InsightFace模型...")
            
            # 检查ONNX Runtime的可用提供者
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                print(f"[YOLO InsightFace Detection] ONNX Runtime可用提供者: {available_providers}")
                
                # 检查CUDA提供者是否真的可用
                cuda_available = 'CUDAExecutionProvider' in available_providers
                print(f"[YOLO InsightFace Detection] CUDA执行提供者可用: {cuda_available}")
            except ImportError:
                print("[YOLO InsightFace Detection] 警告: 无法导入onnxruntime，可能影响GPU检测")
                cuda_available = False
            
            # 根据设备和CUDA可用性选择执行提供者
            if self.device == "cuda" and torch.cuda.is_available() and cuda_available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0  # GPU context
                print("[YOLO InsightFace Detection] InsightFace配置为使用GPU加速")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1  # CPU context
                if self.device == "cuda":
                    print("[YOLO InsightFace Detection] 警告: GPU可用但ONNX Runtime CUDA不可用，InsightFace使用CPU")
                else:
                    print("[YOLO InsightFace Detection] InsightFace使用CPU")
            
            # 创建FaceAnalysis应用，指定需要的任务
            self.insightface_app = insightface.app.FaceAnalysis(
                providers=providers,
                root=insightface_model_dir,
                name='buffalo_l'  # 使用buffalo_l模型，包含年龄和性别识别
            )
            
            # 准备模型，设置输入尺寸
            self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            # 验证InsightFace实际使用的设备
            self.verify_insightface_device(providers, ctx_id)
            
            # 打印InsightFace版本信息
            print(f"[YOLO InsightFace Detection] InsightFace版本: {insightface.__version__}")
            print("[YOLO InsightFace Detection] InsightFace模型加载成功")
            return self.insightface_app
            
        except Exception as e:
            print(f"InsightFace模型加载失败: {e}")
            print("正在尝试下载默认模型...")
            try:
                # 如果模型不存在，insightface会自动下载
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
    
    def create_cropped_image(self, original_image_pil, boxes, analysis_results, target_gender, reference_size=None):
        """创建裁剪特定性别人脸的图像
        
        Args:
            original_image_pil: 原始PIL图像
            boxes: YOLO检测的边界框列表
            analysis_results: 人脸分析结果列表
            target_gender: 要保留的性别 ("Man" 或 "Woman")
            reference_size: 参考尺寸 (width, height)，如果为None则使用第一个检测框的大小
        
        Returns:
            裁剪后的PIL图像
        """
        img_array = np.array(original_image_pil)
        
        # 找到目标性别的检测框
        target_boxes = []
        for box, analysis_result in zip(boxes, analysis_results):
            if analysis_result["gender"] == target_gender:
                target_boxes.append(box)
        
        if not target_boxes:
            # 如果没有找到目标性别，返回空白图像
            if reference_size:
                return Image.new('RGB', reference_size, (0, 0, 0))
            else:
                return Image.new('RGB', original_image_pil.size, (0, 0, 0))
        
        # 如果没有参考尺寸，使用第一个检测框的尺寸作为参考
        if reference_size is None:
            first_box = target_boxes[0]
            x1, y1, x2, y2 = first_box
            reference_size = (x2 - x1, y2 - y1)
            print(f"设置参考尺寸为第一个{target_gender}检测框: {reference_size}")
        
        ref_width, ref_height = reference_size
        
        # 裁剪所有目标性别的人脸并拼接
        cropped_faces = []
        for box in target_boxes:
            x1, y1, x2, y2 = box
            current_width = x2 - x1
            current_height = y2 - y1
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 基于参考尺寸和中心点计算新的边界框
            new_x1 = center_x - ref_width // 2
            new_y1 = center_y - ref_height // 2
            new_x2 = new_x1 + ref_width
            new_y2 = new_y1 + ref_height
            
            # 确保边界框在图像范围内
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(original_image_pil.width, new_x2)
            new_y2 = min(original_image_pil.height, new_y2)
            
            # 裁剪图像
            cropped_face = img_array[new_y1:new_y2, new_x1:new_x2]
            
            # 如果裁剪后的尺寸不等于参考尺寸，进行填充或调整
            if cropped_face.shape[:2] != (ref_height, ref_width):
                # 创建参考尺寸的空白图像
                padded_face = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)
                
                # 计算居中位置
                h, w = cropped_face.shape[:2]
                start_y = (ref_height - h) // 2
                start_x = (ref_width - w) // 2
                end_y = start_y + h
                end_x = start_x + w
                
                # 确保不超出边界
                end_y = min(end_y, ref_height)
                end_x = min(end_x, ref_width)
                actual_h = end_y - start_y
                actual_w = end_x - start_x
                
                # 放置裁剪的人脸到中心位置
                padded_face[start_y:end_y, start_x:end_x] = cropped_face[:actual_h, :actual_w]
                cropped_faces.append(padded_face)
            else:
                cropped_faces.append(cropped_face)
            
            print(f"裁剪{target_gender}人脸: 原始框({current_width}x{current_height}) -> 统一尺寸({ref_width}x{ref_height})")
        
        # 如果有多个人脸，水平拼接
        if len(cropped_faces) == 1:
            result_array = cropped_faces[0]
        else:
            result_array = np.concatenate(cropped_faces, axis=1)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def detect_and_analyze_faces(self, image, yolo_model="yolov8n.pt", confidence=0.5, 
                                iou_threshold=0.45, enable_face_analysis=True, 
                                analysis_confidence_threshold=0.6):
        
        # 加载YOLO模型
        yolo = self.load_yolo_model(yolo_model)
        
        # 只有在启用人脸分析时才加载InsightFace模型
        if enable_face_analysis:
            insightface_app = self.load_insightface_model()
        
        # 第一阶段：收集所有检测结果
        print(f"[YOLO InsightFace Detection] 第一阶段：收集所有图片的检测结果... (设备: {self.device})")
        if self.device == "cuda":
            print(f"[YOLO InsightFace Detection] GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        all_image_data = []  # 存储每张图片的所有数据
        
        for img_idx, img_tensor in enumerate(image):
            img_tensor_single = torch.unsqueeze(img_tensor, 0)
            pil_image = tensor2pil(img_tensor_single)
            
            # YOLO检测（启用mask）
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
                
                # 处理YOLO masks
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
                    
                    for i, box_data in enumerate(result.boxes):
                        class_id = int(box_data.cls[0])
                        class_name = yolo.names[class_id]
                        box_confidence = float(box_data.conf[0])
                        
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
                        
                        # 如果启用人脸分析，进行InsightFace分析
                        if enable_face_analysis and i < len(individual_masks):
                            # 提取检测区域进行人脸分析
                            crop_img = np.array(pil_image)[y1:y2, x1:x2]
                            if crop_img.size > 0:
                                analysis_result = self.analyze_face_with_insightface(crop_img)
                                analysis_results.append(analysis_result)
                                print(f"图片{img_idx}: 检测到 {class_name}: {analysis_result['gender']} "
                                      f"(置信度: {analysis_result['confidence']:.2f})")
                            else:
                                # 如果裁剪区域为空，添加默认结果
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
        
        # 第二阶段：找到最大的检测框尺寸
        print("第二阶段：计算最大检测框尺寸...")
        male_max_size = None
        female_max_size = None
        
        for image_data in all_image_data:
            if enable_face_analysis:
                for box, analysis_result in zip(image_data['boxes'], image_data['analysis_results']):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    if analysis_result["gender"] == "Man":
                        if male_max_size is None or (width * height) > (male_max_size[0] * male_max_size[1]):
                            male_max_size = (width, height)
                            print(f"更新男性最大尺寸: {male_max_size}")
                    elif analysis_result["gender"] == "Woman":
                        if female_max_size is None or (width * height) > (female_max_size[0] * female_max_size[1]):
                            female_max_size = (width, height)
                            print(f"更新女性最大尺寸: {female_max_size}")
        
        print(f"最终男性参考尺寸: {male_max_size}")
        print(f"最终女性参考尺寸: {female_max_size}")
        
        # 第三阶段：使用最大尺寸处理所有图片
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
            
            # 生成YOLO检测结果图像
            if yolo_result:
                yolo_plot_image = cv2.cvtColor(yolo_result.plot(), cv2.COLOR_BGR2RGB)
                result_image_pil = Image.fromarray(yolo_plot_image)
            else:
                result_image_pil = pil_image
            
            # 添加YOLO masks到总列表
            for mask_tensor in individual_masks:
                all_yolo_masks.append(mask_tensor)
            
            # 如果没有检测到任何对象，添加空mask
            if not individual_masks:
                empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                all_yolo_masks.append(empty_mask)
                
                # 如果启用人脸分析且有检测结果，在YOLO图像上添加分析标签
                if enable_face_analysis and boxes and analysis_results:
                    result_image_pil, detection_info = self.add_analysis_labels_to_yolo_image(
                        result_image_pil, boxes, analysis_results, analysis_confidence_threshold
                    )
                    all_detection_info.extend(detection_info)
            elif not enable_face_analysis and boxes and yolo_result:
                # 如果未启用人脸分析，仅记录YOLO检测信息
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
                
            # 添加检测结果图像（原有输出）
            result_images.append(pil2tensor(result_image_pil))
            
            # 如果启用人脸分析且有检测结果，生成性别专用图像
            if enable_face_analysis and boxes and analysis_results:
                # 生成男性专用图像（裁剪男性人脸）
                male_only_image = self.create_cropped_image(pil_image, boxes, analysis_results, "Man", male_max_size)
                male_only_images.append(pil2tensor(male_only_image))
                
                # 生成女性专用图像（裁剪女性人脸）
                female_only_image = self.create_cropped_image(pil_image, boxes, analysis_results, "Woman", female_max_size)
                female_only_images.append(pil2tensor(female_only_image))
                    
                # 为男性创建裁剪区域的mask
                if male_max_size:
                    male_mask = torch.ones((1, male_max_size[1], male_max_size[0]), dtype=torch.float32)
                    all_male_masks.append(male_mask)
                else:
                    # 如果没有男性参考尺寸，创建空mask
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_male_masks.append(empty_mask)
                    
                # 为女性创建裁剪区域的mask
                if female_max_size:
                    female_mask = torch.ones((1, female_max_size[1], female_max_size[0]), dtype=torch.float32)
                    all_female_masks.append(female_mask)
                else:
                    # 如果没有女性参考尺寸，创建空mask
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_female_masks.append(empty_mask)
                
                print("生成性别专用图像和mask完成")
            else:
                # 如果没有人脸分析结果，返回空白图像和空mask
                # 如果有参考尺寸，使用参考尺寸创建空白图像
                if male_max_size:
                    male_blank = Image.new('RGB', male_max_size, (0, 0, 0))
                    male_only_images.append(pil2tensor(male_blank))
                    male_mask = torch.zeros((1, male_max_size[1], male_max_size[0]), dtype=torch.float32)
                    all_male_masks.append(male_mask)
                else:
                    male_only_images.append(pil2tensor(pil_image))
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_male_masks.append(empty_mask)
                
                if female_max_size:
                    female_blank = Image.new('RGB', female_max_size, (0, 0, 0))
                    female_only_images.append(pil2tensor(female_blank))
                    female_mask = torch.zeros((1, female_max_size[1], female_max_size[0]), dtype=torch.float32)
                    all_female_masks.append(female_mask)
                else:
                    female_only_images.append(pil2tensor(pil_image))
                    empty_mask = torch.zeros((1, pil_image.size[1], pil_image.size[0]), dtype=torch.float32)
                    all_female_masks.append(empty_mask)
                
                print("未进行人脸分析，性别专用图像使用空白图像或原图")
        
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
