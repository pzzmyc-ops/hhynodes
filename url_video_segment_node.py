import os
import sys
import tempfile
import shutil
import subprocess
import urllib.request
import urllib.parse
from typing import Tuple, List

import numpy as np
import cv2
import torch
from PIL import Image
import folder_paths
import shutil
import logging
import time
import zipfile

class URLVideoSegmenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "min_scene_length": ("INT", {"default": 30, "min": 1, "max": 100000, "step": 1}),
                "batch_size": ("INT", {"default": 5000, "min": 100, "max": 20000, "step": 100}),
                "overlap": ("INT", {"default": 200, "min": 0, "max": 2000, "step": 10}),
                "output_mode": ("STRING", {"default": "zip", "choices": ["zip", "list"]}),
                "cut_video": ("BOOLEAN", {"default": False}),
                "crop_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "crop_height": ("INT", {"default": 1536, "min": 1, "max": 8192}),
                "crop_position": (["center", "top", "bottom"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "run"
    CATEGORY = "hhy/video"

    def __init__(self):
        self.logger = logging.getLogger("URLVideoSegmenter")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _ensure_output_dir(self) -> str:
        base_output = folder_paths.get_output_directory()
        base_dir = os.path.join(base_output, "segment")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _download_video(self, url: str, download_dir: str) -> str:
        parsed_name = os.path.basename(urllib.parse.urlparse(url).path) or "download.mp4"
        if not os.path.splitext(parsed_name)[1]:
            parsed_name = f"{parsed_name}.mp4"
        temp_path = os.path.join(download_dir, parsed_name)
        self.logger.info(f"开始下载: {url}")
        start_time = time.time()
        with urllib.request.urlopen(url) as response, open(temp_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        elapsed = time.time() - start_time
        size_mb = os.path.getsize(temp_path) / (1024*1024) if os.path.exists(temp_path) else 0
        self.logger.info(f"下载完成 -> {temp_path} ({size_mb:.2f} MB, 用时 {elapsed:.2f}s)")
        return temp_path

    def _ensure_model(self):
        try:
            import transnetv2_pytorch as transnetv2
        except Exception as exc:
            raise RuntimeError("Missing dependency: transnetv2-pytorch. Please install it.") from exc

        model_dir = os.path.join(folder_paths.models_dir, "VLM", "transnetv2-pytorch-weights")
        os.makedirs(model_dir, exist_ok=True)
        weights_path = os.path.join(model_dir, "transnetv2-pytorch-weights.pth")
        if not os.path.exists(weights_path):
            try:
                from huggingface_hub import hf_hub_download
                self.logger.info("正在从 Hugging Face 下载 TransNetV2 权重...")
                hf_hub_download(
                    repo_id="MiaoshouAI/transnetv2-pytorch-weights",
                    filename="transnetv2-pytorch-weights.pth",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Missing weights for TransNetV2. Install huggingface_hub or place the pth file at: "
                    + weights_path
                ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"

        import transnetv2_pytorch as transnetv2
        model = transnetv2.TransNetV2()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        self.logger.info(f"模型加载完成，使用设备: {device}")
        if torch.cuda.is_available() and device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU: {gpu_name}, 显存: {gpu_mem:.1f} GB")
            except Exception:
                pass
        return model, device

    def _resolve_ffmpeg(self) -> str:
        env_bin = os.environ.get("FFMPEG_BINARY", "").strip()
        if env_bin:
            resolved = shutil.which(env_bin) if not os.path.isabs(env_bin) else (env_bin if os.path.exists(env_bin) else None)
            if resolved:
                return resolved

        candidates = [
            "ffmpeg",
            "ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        for c in candidates:
            resolved = shutil.which(c) if not os.path.isabs(c) else (c if os.path.exists(c) else None)
            if resolved:
                return resolved

        raise RuntimeError("未找到 ffmpeg 可执行文件。请安装 ffmpeg 或设置环境变量 FFMPEG_BINARY 指向其路径。")

    def _crop_video(self, input_path: str, output_path: str, crop_width: int, crop_height: int, crop_position: str) -> str:
        ffmpeg_bin = self._resolve_ffmpeg()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {input_path}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取视频第一帧: {input_path}")
            
            orig_height, orig_width = frame.shape[:2]
            self.logger.info(f"检测到视频尺寸: {orig_width}x{orig_height}")
        finally:
            cap.release()

        target_ratio = crop_width / crop_height
        orig_ratio = orig_width / orig_height
        
        if target_ratio > orig_ratio:
            scale_width = orig_width
            scale_height = int(orig_width / target_ratio)
        else:
            scale_width = int(orig_height * target_ratio)
            scale_height = orig_height
        
        if crop_position == "center":
            crop_x = (orig_width - scale_width) // 2
            crop_y = (orig_height - scale_height) // 2
        elif crop_position == "top":
            crop_x = (orig_width - scale_width) // 2
            crop_y = 0
        elif crop_position == "bottom":
            crop_x = (orig_width - scale_width) // 2
            crop_y = orig_height - scale_height
        
        filter_complex = f"crop={scale_width}:{scale_height}:{crop_x}:{crop_y},scale={crop_width}:{crop_height}"
        
        cmd = [
            ffmpeg_bin,
            "-i", input_path,
            "-vf", filter_complex,
            "-c:a", "copy",
            "-y",
            output_path
        ]
        
        self.logger.info(f"开始裁剪视频: {orig_width}x{orig_height} -> {crop_width}x{crop_height}, 位置: {crop_position}")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed = time.time() - start_time
            self.logger.info(f"视频裁剪完成: {output_path} (用时 {elapsed:.2f}s)")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"视频裁剪失败: {e.stderr}")
            raise RuntimeError(f"FFmpeg裁剪失败: {e}")

    def _predict_batches(self, model, device: str, resized_frames: List[np.ndarray], batch_size: int = 2000) -> List[float]:
        predictions_all: List[float] = []
        if not resized_frames:
            return predictions_all

        start = 0
        total = len(resized_frames)
        while start < total:
            end = min(start + batch_size, total)
            batch = resized_frames[start:end]

            frames_array = np.stack(batch, axis=0)
            frames_array_batch = frames_array[np.newaxis, ...]
            frames_tensor = torch.from_numpy(frames_array_batch).to(dtype=torch.uint8).to(device)

            with torch.no_grad():
                preds = model(frames_tensor)
                if isinstance(preds, tuple):
                    single = preds[0]
                else:
                    single = preds

            batch_preds = single.detach().cpu().numpy().squeeze()
            if batch_preds.ndim == 0:
                predictions_all.append(float(batch_preds))
            else:
                predictions_all.extend(batch_preds.tolist())

            start = end

        return predictions_all

    def _find_scenes(self, predictions: np.ndarray, threshold: float, min_scene_length: int) -> List[Tuple[int, int]]:
        predictions_binary = (predictions > threshold).astype(np.uint8)

        scenes: List[Tuple[int, int]] = []
        t_prev, start = 0, 0
        t = 0
        for i, t in enumerate(predictions_binary):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t == 0:
            scenes.append([start, len(predictions_binary)])

        if len(scenes) == 0:
            return [(0, len(predictions_binary))]

        filtered: List[Tuple[int, int]] = []
        for s, e in scenes:
            if e - s >= min_scene_length:
                filtered.append((s, e))
            else:
                if filtered:
                    ps, pe = filtered[-1]
                    filtered[-1] = (ps, e)
                else:
                    filtered.append((s, e))

        return filtered if filtered else [(0, len(predictions_binary))]


    def _segment_single_video(self, video_path: str, output_dir: str, threshold: float = 0.5, min_scene_length: int = 30, batch_size: int = 5000, overlap: int = 200) -> Tuple[str, List[str]]:
        model, device = self._ensure_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"开始分镜处理: {video_path}")
        self.logger.info(f"视频属性: {total_frames} 帧, {fps:.2f} fps")
        predictions_merged: List[float] = []
        buffer_frames: List[np.ndarray] = []
        is_first_batch = True
        batch_idx = 0
        while True:
            while len(buffer_frames) < batch_size + (0 if is_first_batch else 0):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb.astype(np.uint8))
                resized_pil = pil_image.resize((48, 27), Image.BILINEAR)
                resized = np.array(resized_pil, dtype=np.uint8)
                buffer_frames.append(resized)
            if not buffer_frames:
                break
            current_batch = list(buffer_frames)
            batch_idx += 1
            self.logger.info(f"GPU开始处理批次 {batch_idx}: {len(current_batch)} 帧")
            t0 = time.time()
            batch_preds = self._predict_batches(model, device, current_batch, batch_size=len(current_batch))
            t1 = time.time()
            self.logger.info(f"GPU完成批次 {batch_idx}: 用时 {t1 - t0:.2f}s")
            if is_first_batch:
                predictions_merged.extend(batch_preds)
                is_first_batch = False
            else:
                trimmed = batch_preds[overlap:] if len(batch_preds) > overlap else []
                predictions_merged.extend(trimmed)
            if len(current_batch) > overlap:
                buffer_frames = current_batch[-overlap:]
            else:
                buffer_frames = []
            if not cap.isOpened():
                break
            if not cap.grab():
                break
            else:
                ret, frame = cap.retrieve()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb.astype(np.uint8))
                    resized_pil = pil_image.resize((48, 27), Image.BILINEAR)
                    resized = np.array(resized_pil, dtype=np.uint8)
                    buffer_frames.append(resized)
        cap.release()
        preds = np.array(predictions_merged, dtype=np.float32)
        self.logger.info(f"合并预测完成，总帧数: {len(preds)}")
        scenes = self._find_scenes(preds, threshold, min_scene_length)
        self.logger.info(f"检测到场景数量: {len(scenes)} (最小场景长度: {min_scene_length}, 阈值: {threshold})")
        os.makedirs(output_dir, exist_ok=True)
        segment_paths: List[str] = []
        ffmpeg_bin = self._resolve_ffmpeg()
        self.logger.info(f"使用 ffmpeg: {ffmpeg_bin}")
        for i, (start_frame, end_frame) in enumerate(scenes, start=1):
            seg_name = f"segment_{i:03d}.mp4"
            seg_path = os.path.join(output_dir, seg_name)
            start_time = start_frame / fps
            end_time = end_frame / fps
            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-crf", "23",
                seg_path,
            ]
            try:
                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                dt = time.time() - t0
                if os.path.exists(seg_path):
                    file_size = os.path.getsize(seg_path)
                    # 检查文件是否太小（小于1KB可能只有容器头部，没有实际数据）
                    if file_size == 0:
                        self.logger.error(f"⚠️ 切分后立即检查: {seg_name} 大小为 0 字节！ (帧范围: {start_frame}-{end_frame}, 时间: {start_time:.2f}s-{end_time:.2f}s)")
                    elif file_size < 1024:
                        self.logger.error(f"⚠️ 切分后立即检查: {seg_name} 大小异常小 ({file_size} 字节)，可能只有容器头没有数据！ (帧范围: {start_frame}-{end_frame}, 时间: {start_time:.2f}s-{end_time:.2f}s)")
                    else:
                        self.logger.info(f"创建分段成功: {seg_name} ({file_size / 1024:.2f} KB, 用时 {dt:.2f}s)")
                    segment_paths.append(os.path.abspath(seg_path))
                else:
                    self.logger.error(f"创建分段失败: {seg_name} (返回码 {result.returncode})")
            except Exception:
                self.logger.exception(f"创建分段异常: {seg_name}")
        # 打包前检查所有片段
        problematic_segments = []
        for seg_path in segment_paths:
            if os.path.exists(seg_path):
                size = os.path.getsize(seg_path)
                if size == 0:
                    problematic_segments.append(f"{os.path.basename(seg_path)} (0字节)")
                elif size < 1024:
                    problematic_segments.append(f"{os.path.basename(seg_path)} ({size}字节)")
        
        if problematic_segments:
            self.logger.error(f"❌ 打包前检查: 发现 {len(problematic_segments)} 个异常片段: {', '.join(problematic_segments)}")
        else:
            self.logger.info(f"✓ 打包前检查: 所有 {len(segment_paths)} 个片段大小正常")
        
        first_segment = segment_paths[0] if segment_paths else ""
        self.logger.info(f"分镜输出目录: {output_dir}")
        if first_segment:
            self.logger.info(f"首个分段: {first_segment}")
        else:
            self.logger.warning("未生成任何分段文件")
        return first_segment, segment_paths

    def run(self, url: str, threshold: float, min_scene_length: int, batch_size: int, overlap: int, output_mode: str, cut_video: bool, crop_width: int, crop_height: int, crop_position: str) -> Tuple[str]:
        try:
            if not url or not isinstance(url, str):
                return ("",)
            output_dir = self._ensure_output_dir()
            self.logger.info(f"输出目录: {output_dir}")
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded = self._download_video(url, tmpdir)
                
                if cut_video:
                    cropped_name = os.path.splitext(os.path.basename(downloaded))[0] + "_cropped.mp4"
                    cropped_path = os.path.join(tmpdir, cropped_name)
                    video_to_process = self._crop_video(downloaded, cropped_path, crop_width, crop_height, crop_position)
                else:
                    video_to_process = downloaded
                
                first_segment_path, all_segments = self._segment_single_video(
                    video_to_process,
                    output_dir,
                    threshold=threshold,
                    min_scene_length=min_scene_length,
                    batch_size=batch_size,
                    overlap=overlap,
                )
            if not first_segment_path:
                return ("",)
            if output_mode == "zip":
                base_name = os.path.splitext(os.path.basename(downloaded))[0]
                zip_name = f"{base_name}_segments.zip"
                zip_path = os.path.join(output_dir, zip_name)
                try:
                    # 打包前再次检查异常片段
                    problematic_before_zip = []
                    for seg in all_segments:
                        if os.path.exists(seg):
                            size = os.path.getsize(seg)
                            if size == 0:
                                problematic_before_zip.append(f"{os.path.basename(seg)} (0字节)")
                            elif size < 1024:
                                problematic_before_zip.append(f"{os.path.basename(seg)} ({size}字节)")
                    
                    if problematic_before_zip:
                        self.logger.error(f"❌ 打包时检查: 即将打包 {len(problematic_before_zip)} 个异常片段: {', '.join(problematic_before_zip)}")
                    
                    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for seg in all_segments:
                            if os.path.exists(seg):
                                zf.write(seg, arcname=os.path.basename(seg))
                    
                    # 打包后验证zip文件内容
                    if os.path.exists(zip_path):
                        zip_size = os.path.getsize(zip_path)
                        self.logger.info(f"✓ 打包完成: {zip_name} ({zip_size / 1024 / 1024:.2f} MB)")
                        
                        # 验证zip内容
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zf:
                                problematic_in_zip = []
                                for info in zf.infolist():
                                    if not info.is_dir():
                                        if info.file_size == 0:
                                            problematic_in_zip.append(f"{info.filename} (0字节)")
                                        elif info.file_size < 1024:
                                            problematic_in_zip.append(f"{info.filename} ({info.file_size}字节)")
                                
                                if problematic_in_zip:
                                    self.logger.error(f"❌ 打包后检查: ZIP内发现 {len(problematic_in_zip)} 个异常文件: {', '.join(problematic_in_zip)}")
                                else:
                                    self.logger.info(f"✓ 打包后检查: ZIP内所有 {len(zf.infolist())} 个文件大小正常")
                        except Exception as e:
                            self.logger.error(f"验证ZIP内容失败: {e}")
                    
                    return (zip_path.replace("\\", "/"),)
                except Exception:
                    self.logger.exception("压缩分段为 ZIP 失败")
                    return ("",)
            else:
                posix_paths = [p.replace("\\", "/") for p in all_segments]
                return ("\n".join(posix_paths),)
        except Exception:
            self.logger.exception("节点执行失败")
            return ("",)

NODE_CLASS_MAPPINGS = {
    "URLVideoSegmenter": URLVideoSegmenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "URLVideoSegmenter": "URL → Video Segments (All Paths)",
}
