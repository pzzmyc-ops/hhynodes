import os
import sys
import tempfile
import shutil
import subprocess
import urllib.request
import urllib.parse
from typing import Tuple, List
import uuid
from datetime import datetime
import json

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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "log")
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
        self.task_id = None
        self.task_folder = None
        self.segment_files = []

    def _ensure_output_dir(self) -> str:
        """为每个任务创建独特的输出目录"""
        base_output = folder_paths.get_output_directory()
        # 生成独特的任务ID（时间戳+短UUID）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        self.task_id = f"{timestamp}_{short_uuid}"
        
        # 创建独特的文件夹
        self.task_folder = os.path.join(base_output, f"video_task_{self.task_id}")
        os.makedirs(self.task_folder, exist_ok=True)
        self.logger.info(f"为任务创建独特文件夹: {self.task_folder}")
        return self.task_folder

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


    def _segment_single_video(self, video_path: str, output_dir: str, threshold: float = 0.5, min_scene_length: int = 30, batch_size: int = 5000, overlap: int = 200) -> Tuple[str, List[str], dict]:
        start_time = time.time()
        model, device = self._ensure_model()
        
        # 使用字典记录日志数据
        log_data = {
            "task_info": {
                "task_id": self.task_id,
                "task_folder": self.task_folder,
                "output_dir": output_dir
            },
            "video_info": {},
            "detection_params": {
                "threshold": threshold,
                "min_scene_length": min_scene_length,
                "batch_size": batch_size,
                "overlap": overlap
            },
            "segments": [],
            "processing_batches": [],
            "summary": {},
            "time": {}
        }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 记录视频信息
        log_data["video_info"] = {
            "path": video_path,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(total_frames / fps, 2)
        }
        
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
            batch_time = round(t1 - t0, 2)
            self.logger.info(f"GPU完成批次 {batch_idx}: 用时 {batch_time}s")
            
            # 记录批次处理信息
            log_data["processing_batches"].append({
                "batch_index": batch_idx,
                "frames_count": len(current_batch),
                "time_seconds": batch_time
            })
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
        
        # 记录场景检测结果
        log_data["detection_result"] = {
            "total_scenes": len(scenes),
            "predicted_frames": len(preds)
        }
        os.makedirs(output_dir, exist_ok=True)
        segment_paths: List[str] = []
        ffmpeg_bin = self._resolve_ffmpeg()
        self.logger.info(f"使用 ffmpeg: {ffmpeg_bin}")
        for i, (start_frame, end_frame) in enumerate(scenes, start=1):
            # 使用独特的视频名字（任务ID + 序号）
            seg_name = f"video_{self.task_id}_{i:03d}.mp4"
            seg_path = os.path.join(output_dir, seg_name)
            # 记录生成的文件
            self.segment_files.append(seg_path)
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
                    duration = end_time - start_time
                    frame_count = end_frame - start_frame
                    
                    # 确定状态
                    if file_size == 0:
                        status = "error"
                        status_msg = "大小为 0 字节"
                        self.logger.error(f"⚠️ {seg_name} 大小为 0 字节！")
                    elif file_size < 1024:
                        status = "warning"
                        status_msg = "大小异常小，可能只有容器头没有数据"
                        self.logger.error(f"⚠️ {seg_name} 大小异常小 ({file_size} 字节)")
                    else:
                        status = "success"
                        status_msg = "正常"
                        self.logger.info(f"创建分段成功: {seg_name} ({file_size / 1024:.2f} KB, 用时 {dt:.2f}s)")
                    
                    segment_info = {
                        "index": i,
                        "filename": seg_name,
                        "file_path": os.path.abspath(seg_path),
                        "file_size_bytes": file_size,
                        "file_size_kb": round(file_size / 1024, 2),
                        "file_size_mb": round(file_size / 1024 / 1024, 2) if file_size >= 1024 * 1024 else 0,
                        "time_range": {
                            "start_seconds": round(start_time, 2),
                            "end_seconds": round(end_time, 2),
                            "duration_seconds": round(duration, 2)
                        },
                        "frame_range": {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "frame_count": frame_count
                        },
                        "generation_time_seconds": round(dt, 2),
                        "status": status,
                        "status_message": status_msg
                    }
                    log_data["segments"].append(segment_info)
                    segment_paths.append(os.path.abspath(seg_path))
                else:
                    self.logger.error(f"创建分段失败: {seg_name} (返回码 {result.returncode})")
                    segment_info = {
                        "index": i,
                        "filename": seg_name,
                        "status": "failed",
                        "status_message": f"文件未生成 (返回码 {result.returncode})"
                    }
                    log_data["segments"].append(segment_info)
            except Exception as e:
                self.logger.exception(f"创建分段异常: {seg_name}")
                segment_info = {
                    "index": i,
                    "filename": seg_name,
                    "status": "error",
                    "status_message": f"异常: {str(e)}"
                }
                log_data["segments"].append(segment_info)
        # 统计汇总信息
        problematic_segments = []
        total_size = 0
        success_count = 0
        warning_count = 0
        error_count = 0
        
        for seg_info in log_data["segments"]:
            if seg_info.get("status") == "success":
                success_count += 1
                total_size += seg_info.get("file_size_bytes", 0)
            elif seg_info.get("status") == "warning":
                warning_count += 1
                total_size += seg_info.get("file_size_bytes", 0)
                problematic_segments.append({
                    "filename": seg_info.get("filename"),
                    "size_bytes": seg_info.get("file_size_bytes", 0),
                    "issue": "size_too_small"
                })
            elif seg_info.get("status") in ["error", "failed"]:
                error_count += 1
                problematic_segments.append({
                    "filename": seg_info.get("filename"),
                    "issue": seg_info.get("status_message", "unknown")
                })
        
        # 记录汇总信息
        log_data["summary"] = {
            "total_segments": len(segment_paths),
            "success_segments": success_count,
            "warning_segments": warning_count,
            "error_segments": error_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "average_size_kb": round(total_size / len(segment_paths) / 1024, 2) if segment_paths else 0,
            "problematic_segments": problematic_segments
        }
        
        # 记录处理时间
        end_time = time.time()
        log_data["time"]["segmentation_time_seconds"] = round(end_time - start_time, 2)
        
        # 输出到控制台
        if problematic_segments:
            self.logger.error(f"发现 {len(problematic_segments)} 个异常片段")
        else:
            self.logger.info("所有片段大小正常")
        
        first_segment = segment_paths[0] if segment_paths else ""
        self.logger.info(f"分镜输出目录: {output_dir}")
        if first_segment:
            self.logger.info(f"首个分段: {first_segment}")
        else:
            self.logger.warning("未生成任何分段文件")
        
        return first_segment, segment_paths, log_data

    def _cleanup_task_files(self) -> dict:
        """清理任务产生的临时文件和文件夹"""
        cleanup_data = {
            "status": "success",
            "task_folder": self.task_folder,
            "files_deleted": 0,
            "space_freed_mb": 0
        }
        
        if not self.task_folder or not os.path.exists(self.task_folder):
            self.logger.info("没有需要清理的任务文件夹")
            cleanup_data["status"] = "skipped"
            cleanup_data["message"] = "没有需要清理的任务文件夹"
            return cleanup_data
        
        try:
            # 统计清理信息
            file_count = 0
            total_size = 0
            
            # 删除所有记录的分段文件
            for seg_file in self.segment_files:
                if os.path.exists(seg_file):
                    size = os.path.getsize(seg_file)
                    total_size += size
                    file_count += 1
                    os.remove(seg_file)
                    self.logger.debug(f"已删除: {seg_file}")
            
            # 删除任务文件夹
            if os.path.exists(self.task_folder):
                shutil.rmtree(self.task_folder)
                self.logger.info(f"✓ 清理完成: 删除 {file_count} 个文件，释放 {total_size / 1024 / 1024:.2f} MB")
            
            # 清空记录
            self.segment_files = []
            
            cleanup_data["files_deleted"] = file_count
            cleanup_data["space_freed_mb"] = round(total_size / 1024 / 1024, 2)
            cleanup_data["message"] = "清理成功"
            
            return cleanup_data
            
        except Exception as e:
            self.logger.error(f"清理任务文件失败: {e}")
            cleanup_data["status"] = "error"
            cleanup_data["message"] = str(e)
            return cleanup_data

    def run(self, url: str, threshold: float, min_scene_length: int, batch_size: int, overlap: int, output_mode: str, cut_video: bool, crop_width: int, crop_height: int, crop_position: str) -> Tuple[str, str]:
        run_start_time = time.time()
        
        try:
            if not url or not isinstance(url, str):
                error_log = {
                    "status": "error",
                    "error": {"message": "输入URL无效", "type": "ValueError"}
                }
                return ("", "```log\n" + json.dumps(error_log, ensure_ascii=False, indent=2) + "\n```")
            
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
                
                first_segment_path, all_segments, segment_log_data = self._segment_single_video(
                    video_to_process,
                    output_dir,
                    threshold=threshold,
                    min_scene_length=min_scene_length,
                    batch_size=batch_size,
                    overlap=overlap,
                )
            
            if not first_segment_path:
                segment_log_data["status"] = "error"
                segment_log_data["error"] = {"message": "视频切分失败，未生成任何片段"}
                final_log = "```log\n" + json.dumps(segment_log_data, ensure_ascii=False, indent=2) + "\n```"
                return ("", final_log)
            if output_mode == "zip":
                # 使用独特的任务ID作为zip文件名
                zip_name = f"video_segments_{self.task_id}.zip"
                # ZIP文件保存到输出目录的根目录，而不是任务文件夹内
                base_output = folder_paths.get_output_directory()
                zip_path = os.path.join(base_output, zip_name)
                
                # 添加ZIP信息到日志数据
                segment_log_data["zip_info"] = {
                    "zip_filename": zip_name,
                    "zip_path": zip_path
                }
                
                try:
                    zip_start_time = time.time()
                    
                    # 打包
                    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for seg in all_segments:
                            if os.path.exists(seg):
                                zf.write(seg, arcname=os.path.basename(seg))
                    
                    # 打包后验证zip文件内容
                    if os.path.exists(zip_path):
                        zip_size = os.path.getsize(zip_path)
                        self.logger.info(f"✓ 打包完成: {zip_name} ({zip_size / 1024 / 1024:.2f} MB)")
                        
                        # 验证zip内容
                        zip_verification = {
                            "status": "success",
                            "total_files": 0,
                            "total_uncompressed_bytes": 0,
                            "compressed_size_bytes": zip_size,
                            "compressed_size_mb": round(zip_size / 1024 / 1024, 2),
                            "compression_ratio": 0,
                            "problematic_files": []
                        }
                        
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zf:
                                zip_verification["total_files"] = len(zf.infolist())
                                
                                problematic_in_zip = []
                                total_uncompressed = 0
                                for info in zf.infolist():
                                    if not info.is_dir():
                                        total_uncompressed += info.file_size
                                        if info.file_size == 0:
                                            problematic_in_zip.append({
                                                "filename": info.filename,
                                                "size_bytes": 0,
                                                "issue": "zero_size"
                                            })
                                        elif info.file_size < 1024:
                                            problematic_in_zip.append({
                                                "filename": info.filename,
                                                "size_bytes": info.file_size,
                                                "issue": "too_small"
                                            })
                                
                                zip_verification["total_uncompressed_bytes"] = total_uncompressed
                                zip_verification["total_uncompressed_mb"] = round(total_uncompressed / 1024 / 1024, 2)
                                if total_uncompressed > 0:
                                    zip_verification["compression_ratio"] = round((1 - zip_size / total_uncompressed) * 100, 1)
                                zip_verification["problematic_files"] = problematic_in_zip
                                
                                if problematic_in_zip:
                                    self.logger.error(f"ZIP内发现 {len(problematic_in_zip)} 个异常文件")
                                else:
                                    self.logger.info("ZIP内所有文件大小正常")
                                    
                        except Exception as e:
                            self.logger.error(f"验证ZIP内容失败: {e}")
                            zip_verification["status"] = "error"
                            zip_verification["error_message"] = str(e)
                        
                        segment_log_data["zip_info"]["verification"] = zip_verification
                        segment_log_data["zip_info"]["packaging_time_seconds"] = round(time.time() - zip_start_time, 2)
                    
                    # 打包完成后清理临时文件
                    cleanup_data = self._cleanup_task_files()
                    segment_log_data["cleanup"] = cleanup_data
                    
                    # 记录总时间
                    segment_log_data["time"]["total_time_seconds"] = round(time.time() - run_start_time, 2)
                    segment_log_data["status"] = "success"
                    segment_log_data["output_mode"] = "zip"
                    segment_log_data["output_path"] = zip_path
                    
                    # 转换为 ```log 格式
                    final_log = "```log\n" + json.dumps(segment_log_data, ensure_ascii=False, indent=2) + "\n```"
                    
                    return (zip_path.replace("\\", "/"), final_log)
                except Exception as e:
                    self.logger.exception("压缩分段为 ZIP 失败")
                    segment_log_data["status"] = "error"
                    segment_log_data["error"] = {
                        "message": f"ZIP打包失败: {str(e)}",
                        "type": type(e).__name__
                    }
                    
                    # 即使打包失败也要清理临时文件
                    cleanup_data = self._cleanup_task_files()
                    segment_log_data["cleanup"] = cleanup_data
                    segment_log_data["time"]["total_time_seconds"] = round(time.time() - run_start_time, 2)
                    
                    error_log = "```log\n" + json.dumps(segment_log_data, ensure_ascii=False, indent=2) + "\n```"
                    return ("", error_log)
            else:
                # list 模式
                segment_log_data["status"] = "success"
                segment_log_data["output_mode"] = "list"
                segment_log_data["time"]["total_time_seconds"] = round(time.time() - run_start_time, 2)
                
                posix_paths = [p.replace("\\", "/") for p in all_segments]
                final_log = "```log\n" + json.dumps(segment_log_data, ensure_ascii=False, indent=2) + "\n```"
                return ("\n".join(posix_paths), final_log)
                
        except Exception as e:
            self.logger.exception("节点执行失败")
            
            # 构建错误日志
            error_log_data = {
                "status": "error",
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                },
                "time": {
                    "total_time_seconds": round(time.time() - run_start_time, 2)
                }
            }
            
            # 节点执行失败时也尝试清理临时文件
            try:
                cleanup_data = self._cleanup_task_files()
                error_log_data["cleanup"] = cleanup_data
            except Exception:
                pass
            
            error_log = "```log\n" + json.dumps(error_log_data, ensure_ascii=False, indent=2) + "\n```"
            return ("", error_log)

NODE_CLASS_MAPPINGS = {
    "URLVideoSegmenter": URLVideoSegmenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "URLVideoSegmenter": "URL → Video Segments (All Paths)",
}
