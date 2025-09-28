import os
import cv2
import torch
import numpy as np
import folder_paths
import hashlib

# Get video file extensions
video_extensions = ["mp4", "webm", "mkv", "avi", "mov", "m4v", "wmv", "flv", "gif"]

class LoadFramesFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "last_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "selected_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "force_rate": ("FLOAT", {"default": 24, "min": 1, "max": 60, "step": 0.1}),
                "force_output_frames": ("INT", {"default": 213, "min": 1, "max": 1000, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            },
        }

    CATEGORY = "hhy"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("selected_frames", "all_frames", "frames_before_selected", "frames_after_selected")
    FUNCTION = "load_frames"

    def load_frames(self, video, first_frame, last_frame, selected_frame, force_rate, force_output_frames, custom_width, custom_height):
        # Convert parameters to ensure type consistency
        first_frame = int(first_frame)
        last_frame = int(last_frame)
        selected_frame = int(selected_frame)
        force_rate = float(force_rate)
        force_output_frames = int(force_output_frames)
        custom_width = int(custom_width)
        custom_height = int(custom_height)
        
        # Get the full path to the video file
        video_path = folder_paths.get_annotated_filepath(video)
        
        # Check if file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video metadata first
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0:
            raise ValueError(f"Could not determine frame count for video: {video_path}")
        if original_fps <= 0:
            original_fps = 30  # Default fallback FPS
        
        cap.release()
        
        # Force specific output parameters
        target_fps = force_rate  # Always use forced FPS (default 24)
        target_frame_time = 1 / target_fps
        base_frame_time = 1 / original_fps
        
        # Calculate frame expansion ratio if needed
        frame_expansion_ratio = force_output_frames / total_frames if total_frames > 0 else 1.0
        
        # Load all frames for the second output
        start_frame = 0
        if first_frame > 0 and last_frame == 0:
            # Special case: start from first_frame
            start_frame = first_frame - 1
        
        # Load all frames (or from start_frame) for the second output
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        all_frames_tensor = self.load_all_frames(cap, total_frames, custom_width, custom_height, force_output_frames, start_frame, target_frame_time, base_frame_time, frame_expansion_ratio)[0]
        
        # Reset video capture for the primary output
        cap.release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not reopen video: {video_path}")
        
        # Handle the primary output based on first_frame and last_frame parameters
        if first_frame > 0 and last_frame > 0:
            # Load both specified frames
            first_img = self.load_specified_frame(cap, first_frame - 1, total_frames, custom_width, custom_height)
            
            # Reset cap to load the last frame
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not reopen video: {video_path}")
                
            last_img = self.load_specified_frame(cap, total_frames - last_frame, total_frames, custom_width, custom_height)
            
            # Combine both frames into a batch
            selected_frames = torch.cat([first_img, last_img], dim=0)
        elif first_frame > 0:
            # Load only the first frame
            selected_frames = self.load_specified_frame(cap, first_frame - 1, total_frames, custom_width, custom_height)
        elif last_frame > 0:
            # Load only the last frame
            selected_frames = self.load_specified_frame(cap, total_frames - last_frame, total_frames, custom_width, custom_height)
        else:
            # If both are 0, use the first frame only for the primary output
            selected_frames = self.load_specified_frame(cap, 0, total_frames, custom_width, custom_height)
        
        # Handle selected_frame logic for new outputs
        frames_before_selected = torch.zeros((0, *all_frames_tensor.shape[1:]), dtype=all_frames_tensor.dtype)
        frames_after_selected = torch.zeros((0, *all_frames_tensor.shape[1:]), dtype=all_frames_tensor.dtype)
        
        if selected_frame > 0:
            # Ensure selected_frame is within valid range
            if selected_frame <= total_frames:
                # Calculate the split position in our all_frames_tensor
                # all_frames_tensor starts from start_frame
                selected_index_in_tensor = selected_frame - 1 - start_frame
                
                if selected_index_in_tensor >= 0 and selected_index_in_tensor < all_frames_tensor.shape[0]:
                    # Split frames before and after selected frame
                    if selected_index_in_tensor > 0:
                        frames_before_selected = all_frames_tensor[:selected_index_in_tensor]
                    if selected_index_in_tensor + 1 < all_frames_tensor.shape[0]:
                        frames_after_selected = all_frames_tensor[selected_index_in_tensor + 1:]
        
        cap.release()
        return (selected_frames, all_frames_tensor, frames_before_selected, frames_after_selected)
    
    def load_all_frames(self, cap, total_frames, custom_width, custom_height, force_output_frames, start_frame=0, target_frame_time=None, base_frame_time=None, frame_expansion_ratio=1.0):
        frames = []
        
        # Set position to starting frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # First, load all available frames from the video
        source_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            frame = self.resize_frame(frame, custom_width, custom_height)
            
            # Convert to tensor format
            frame = np.array(frame, dtype=np.float32) / 255.0
            source_frames.append(frame)
        
        if not source_frames:
            raise RuntimeError(f"Failed to read any frames from video")
        
        # Now generate exactly force_output_frames frames
        frames = []
        for i in range(force_output_frames):
            # Calculate which source frame to use
            if len(source_frames) >= force_output_frames:
                # If we have enough frames, sample them
                source_index = int(i * len(source_frames) / force_output_frames)
            else:
                # If we need to extend, repeat frames
                source_index = int(i * len(source_frames) / force_output_frames)
                if source_index >= len(source_frames):
                    source_index = len(source_frames) - 1
            
            frames.append(source_frames[source_index])
        
        # Combine all frames into a single batch tensor
        frames_tensor = torch.from_numpy(np.stack(frames))
        
        return (frames_tensor,)
    
    def load_specified_frame(self, cap, frame_index, total_frames, custom_width, custom_height):
        if frame_index >= total_frames or frame_index < 0:
            raise ValueError(f"Frame index {frame_index} out of range (0-{total_frames-1})")
        
        # Set position to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_index} from video")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        frame = self.resize_frame(frame, custom_width, custom_height)
        
        # Convert to tensor format expected by ComfyUI
        frame_tensor = self.convert_to_tensor(frame)
        
        return frame_tensor
    
    @staticmethod
    def resize_frame(frame, custom_width, custom_height):
        if custom_width > 0 and custom_height > 0:
            frame = cv2.resize(frame, (custom_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
        elif custom_width > 0:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(custom_width / aspect_ratio)
            frame = cv2.resize(frame, (custom_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        elif custom_height > 0:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = int(custom_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
        return frame

    @staticmethod
    def convert_to_tensor(frame):
        frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame)[None, ...]  # Add batch dimension
        return frame_tensor

    @classmethod
    def IS_CHANGED(s, video, first_frame, last_frame, selected_frame, force_rate, force_output_frames, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        if not folder_paths.exists_annotated_filepath(video):
            return "FILE_NOT_FOUND"
            
        # Return modification time and requested parameters as hash
        mtime = os.path.getmtime(video_path)
        return f"{mtime}_{first_frame}_{last_frame}_{selected_frame}_{force_rate}_{force_output_frames}"
        
    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

# New class for extracting the clearest frame from the last N frames
class ExtractClearestFrame:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "last_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "force_rate": ("FLOAT", {"default": 24, "min": 1, "max": 60, "step": 0.1}),
                "force_output_frames": ("INT", {"default": 213, "min": 1, "max": 1000, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            },
        }

    CATEGORY = "hhy"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clearest_frame", "previous_frames", "frame_position", "frame_info")
    FUNCTION = "extract_clearest_frame"

    def extract_clearest_frame(self, video, last_n_frames, force_rate, force_output_frames, custom_width, custom_height):
        # Convert parameters to ensure type consistency
        last_n_frames = int(last_n_frames)
        force_rate = float(force_rate)
        force_output_frames = int(force_output_frames)
        custom_width = int(custom_width)
        custom_height = int(custom_height)
        
        # Get the full path to the video file
        video_path = folder_paths.get_annotated_filepath(video)
        
        # Check if file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Could not determine frame count for video: {video_path}")
        
        # Force specific output parameters
        target_fps = force_rate  # Always use forced FPS (default 24)
        
        # Use force_output_frames as the target
        effective_total = force_output_frames
        
        # Start from beginning for frame extension logic
        video_start_frame = 0
        
        # Read all available frames first
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)
        
        # Store source frames
        source_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            source_frames.append(frame)
        
        if not source_frames:
            raise RuntimeError(f"Failed to read any frames from video")
        
        # Generate exactly force_output_frames frames through extension/sampling
        all_frames = []
        for i in range(force_output_frames):
            # Calculate which source frame to use
            if len(source_frames) >= force_output_frames:
                # If we have enough frames, sample them
                source_index = int(i * len(source_frames) / force_output_frames)
            else:
                # If we need to extend, repeat frames
                source_index = int(i * len(source_frames) / force_output_frames)
                if source_index >= len(source_frames):
                    source_index = len(source_frames) - 1
            
            all_frames.append(source_frames[source_index])
        
        # Analyze only the last N frames from our collected frames
        last_n = min(last_n_frames, len(all_frames))
        frames_to_analyze = all_frames[-last_n:]
        
        best_frame = None
        best_sharpness = -1
        best_frame_index = -1
        
        # Find the clearest frame among the last N frames
        for i, frame in enumerate(frames_to_analyze):
            # Calculate frame sharpness using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame.copy()  # Make a copy to ensure we keep it
                best_frame_index = len(all_frames) - last_n + i  # Index in all_frames
        
        if best_frame is None:
            raise RuntimeError(f"Failed to extract any frames from video")
        
        # Convert BGR to RGB
        best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        best_frame = self.resize_frame(best_frame, custom_width, custom_height)
        
        # Convert to tensor format expected by ComfyUI
        frame_tensor = self.convert_to_tensor(best_frame)
        
        # Get frames before the best frame (within our max_frames limit)
        previous_frames = []
        for i in range(best_frame_index):
            # Convert BGR to RGB
            frame = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            frame = self.resize_frame(frame, custom_width, custom_height)
            
            # Convert to float array
            frame = np.array(frame, dtype=np.float32) / 255.0
            previous_frames.append(frame)
        
        if previous_frames:
            # Combine all frames into a single batch tensor
            previous_frames_tensor = torch.from_numpy(np.stack(previous_frames))
        else:
            # If no previous frames, create an empty tensor with the right shape
            previous_frames_tensor = torch.zeros((0, *frame_tensor.shape[1:]), dtype=frame_tensor.dtype)
        
        # Calculate statistics about processed frames
        actual_frame_position = video_start_frame + best_frame_index
        
        # Create an informative string about the frames
        frame_info = f"Original video frames: {len(source_frames)}\n" \
                     f"Generated frames: {force_output_frames} at {force_rate}fps\n" \
                     f"Analyzed last {last_n} frames for clarity\n" \
                     f"Clearest frame position in generated sequence: {best_frame_index+1}/{force_output_frames}\n" \
                     f"Sharpness score: {best_sharpness:.2f}\n" \
                     f"Included previous frames: {len(previous_frames)}"
        
        # Print the information to console
        print(f"\n--- Video Frame Analysis ---")
        print(f"Video: {os.path.basename(video_path)}")
        print(frame_info)
        print(f"----------------------------\n")
        
        return (frame_tensor, previous_frames_tensor, actual_frame_position, frame_info)
    
    @staticmethod
    def resize_frame(frame, custom_width, custom_height):
        if custom_width > 0 and custom_height > 0:
            frame = cv2.resize(frame, (custom_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
        elif custom_width > 0:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(custom_width / aspect_ratio)
            frame = cv2.resize(frame, (custom_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        elif custom_height > 0:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = int(custom_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, custom_height), interpolation=cv2.INTER_LANCZOS4)
        return frame

    @staticmethod
    def convert_to_tensor(frame):
        frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame)[None, ...]  # Add batch dimension
        return frame_tensor

    @classmethod
    def IS_CHANGED(s, video, last_n_frames, force_rate, force_output_frames, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        if not folder_paths.exists_annotated_filepath(video):
            return "FILE_NOT_FOUND"
            
        # Return modification time and requested parameters as hash
        mtime = os.path.getmtime(video_path)
        return f"{mtime}_{last_n_frames}_{force_rate}_{force_output_frames}"
        
    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

# New class for frame interpolation and reduction on image lists
class FrameInterpolationProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "force_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "force_output_frames": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
            },
        }

    CATEGORY = "hhy"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("processed_images", "process_info", "total_frames", "fps")
    FUNCTION = "process_frames"

    def process_frames(self, images, force_rate, force_output_frames):
        # Convert parameters to ensure type consistency
        force_rate = float(force_rate)
        force_output_frames = int(force_output_frames)
        
        # Get input frame count
        input_frame_count = images.shape[0]
        
        if input_frame_count == 0:
            raise ValueError("No input images provided")
        
        # Determine processing mode
        if force_output_frames == input_frame_count:
            # No change needed
            processed_images = images
            process_mode = "unchanged"
        elif force_output_frames > input_frame_count:
            # Need to interpolate (add frames)
            processed_images = self.interpolate_frames(images, force_output_frames)
            process_mode = "interpolated"
        else:
            # Need to reduce frames
            processed_images = self.reduce_frames(images, force_output_frames)
            process_mode = "reduced"
        
        # Create process information
        process_info = f"Input frames: {input_frame_count}\n" \
                      f"Output frames: {force_output_frames}\n" \
                      f"Target FPS: {force_rate}\n" \
                      f"Process mode: {process_mode}\n" \
                      f"Frame ratio: {force_output_frames / input_frame_count:.2f}x"
        
        # Print processing information
        print(f"\n--- Frame Processing ---")
        print(f"Input frames: {input_frame_count} -> Output frames: {force_output_frames}")
        print(f"Target FPS: {force_rate}")
        print(f"Process mode: {process_mode}")
        print(f"Frame ratio: {force_output_frames / input_frame_count:.2f}x")
        print(f"----------------------\n")
        
        return (processed_images, process_info, force_output_frames, force_rate)
    
    def interpolate_frames(self, images, target_frames):
        """
        Interpolate frames to increase the total count
        Uses linear interpolation between adjacent frames
        """
        input_frames = images.shape[0]
        
        if target_frames <= input_frames:
            return images
        
        # Create indices for interpolation
        output_frames = []
        
        for i in range(target_frames):
            # Calculate the position in the original sequence
            pos = i * (input_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
            
            # Get the two frames to interpolate between
            frame_idx = int(pos)
            next_frame_idx = min(frame_idx + 1, input_frames - 1)
            
            # Calculate interpolation weight
            weight = pos - frame_idx
            
            if frame_idx == next_frame_idx or weight == 0:
                # Use the exact frame
                interpolated_frame = images[frame_idx]
            else:
                # Linear interpolation between two frames
                frame1 = images[frame_idx]
                frame2 = images[next_frame_idx]
                interpolated_frame = (1 - weight) * frame1 + weight * frame2
            
            output_frames.append(interpolated_frame)
        
        # Stack all frames into a tensor
        return torch.stack(output_frames)
    
    def reduce_frames(self, images, target_frames):
        """
        Reduce frames by sampling from the input sequence
        Uses uniform sampling to select the most representative frames
        """
        input_frames = images.shape[0]
        
        if target_frames >= input_frames:
            return images
        
        # Create indices for uniform sampling
        indices = []
        
        for i in range(target_frames):
            # Calculate the index in the original sequence
            idx = int(i * input_frames / target_frames)
            # Ensure we don't exceed bounds
            idx = min(idx, input_frames - 1)
            indices.append(idx)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        
        # If we have fewer unique indices than target, fill with evenly spaced frames
        if len(unique_indices) < target_frames:
            # Create evenly spaced indices
            step = max(1, input_frames // target_frames)
            unique_indices = list(range(0, input_frames, step))[:target_frames]
            
            # If still not enough, add remaining frames from the end
            while len(unique_indices) < target_frames:
                unique_indices.append(input_frames - 1)
        
        # Take only the first target_frames indices
        unique_indices = unique_indices[:target_frames]
        
        # Extract the selected frames
        selected_frames = [images[idx] for idx in unique_indices]
        
        return torch.stack(selected_frames)
    
    @classmethod
    def IS_CHANGED(s, images, force_rate, force_output_frames, **kwargs):
        # Generate hash based on input parameters and image tensor properties
        image_hash = hashlib.md5(str(images.shape).encode()).hexdigest()[:8]
        param_hash = hashlib.md5(f"{force_rate}_{force_output_frames}".encode()).hexdigest()[:8]
        return f"{image_hash}_{param_hash}"

class SelectFrameFromImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "select_mode": (["first", "last", "index_from_start", "index_from_end", "biggest"],),
                "index": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
            },
        }

    CATEGORY = "hhy"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "selected_index", "info")
    FUNCTION = "select_frame"

    def select_frame(self, images, select_mode, index):
        total_frames = images.shape[0]
        if total_frames == 0:
            raise ValueError("No input images provided")

        if select_mode == "first":
            idx0 = 0
        elif select_mode == "last":
            idx0 = total_frames - 1
        elif select_mode == "index_from_start":
            idx0 = max(0, min(total_frames - 1, int(index) - 1))
        elif select_mode == "index_from_end":
            idx0 = max(0, min(total_frames - 1, total_frames - int(index)))
        elif select_mode == "biggest":
            # Choose frame with the largest total pixel sum as a proxy for "biggest"
            # images shape: (N, H, W, C)
            flattened = images.view(images.shape[0], -1)
            sums = flattened.sum(dim=1)
            idx0 = int(torch.argmax(sums).item())
        else:
            idx0 = total_frames - 1

        selected = images[idx0:idx0 + 1]
        info = f"Total frames: {total_frames}, mode: {select_mode}, selected(0-based): {idx0}"
        return (selected, idx0 + 1, info)

    @classmethod
    def IS_CHANGED(s, images, select_mode, index, **kwargs):
        image_hash = hashlib.md5(str(images.shape).encode()).hexdigest()[:8]
        param_hash = hashlib.md5(f"{select_mode}_{index}".encode()).hexdigest()[:8]
        return f"{image_hash}_{param_hash}"

NODE_CLASS_MAPPINGS = {
    "LoadFramesFromVideo": LoadFramesFromVideo,
    "ExtractClearestFrame": ExtractClearestFrame,
    "FrameInterpolationProcessor": FrameInterpolationProcessor,
    "SelectFrameFromImages": SelectFrameFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFramesFromVideo": "Load Frames From Video",
    "ExtractClearestFrame": "Extract Clearest Frame",
    "FrameInterpolationProcessor": "Frame Interpolation Processor",
    "SelectFrameFromImages": "Select Frame From Images",
}
