#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import math
import time
from typing import NamedTuple, List, Optional, Tuple, Union
import comfy.model_management as model_management
import comfy.utils
import queue
import threading

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1

class BodyResult(NamedTuple):
    keypoints: List[Optional[Keypoint]]
    total_score: float = 0.0
    total_parts: int = 0

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Optional[HandResult]
    right_hand: Optional[HandResult]
    face: Optional[FaceResult]

def numpy2tensor(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)

def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)

def is_normalized(keypoints: List[Optional[Keypoint]]) -> bool:
    point_normalized = [
        0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1 
        for k in keypoints 
        if k is not None
    ]
    if not point_normalized:
        return False
    return all(point_normalized)

def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], xinsr_stick_scaling: bool = False) -> np.ndarray:
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    CH, CW, _ = canvas.shape
    stickwidth = 4
    max_side = max(CW, CH)
    if xinsr_stick_scaling:
        stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7)
    else:
        stick_scale = 1
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        if keypoint1 is None or keypoint2 is None:
            continue
        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth*stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    return canvas

def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01:
            cv2.line(canvas, (x1, y1), (x2, y2), [ie / float(len(edges)) * 255, 255, 255], thickness=2)
    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas

def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape
    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)
        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand)
            canvas = draw_handpose(canvas, pose.right_hand)
        if draw_face:
            canvas = draw_facepose(canvas, pose.face)
    return canvas

# YOLO检测相关函数
def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def yolo_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def inference_detector(model, oriImg, detect_classes=[0]):
    input_shape = (640, 640)
    img, ratio = yolo_preprocess(oriImg, input_shape)

    device = next(model.parameters()).device

    input_tensor = torch.from_numpy(img[None, :, :, :]).to(device, dtype=torch.float32)
    
    with torch.no_grad():

        output = model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]
        

        output = output.float().cpu().detach().numpy()
        

    predictions = demo_postprocess(output[0], input_shape)
    
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
        
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    isscore = final_scores > 0.3
    iscat = np.isin(final_cls_inds, detect_classes)
    isbbox = [i and j for (i, j) in zip(isscore, iscat)]
    final_boxes = final_boxes[isbbox]
    return final_boxes

def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale

def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, 
                   output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.), 
                   inv: bool = False) -> np.ndarray:
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat

def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    warp_size = (int(w), int(h))
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale

def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)
    return locs, vals

def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores

def preprocess(img: np.ndarray, out_bbox, input_size: Tuple[int, int] = (192, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)
    return out_img, out_center, out_scale

def postprocess(outputs: List[np.ndarray], model_input_size: Tuple[int, int], 
                center: Tuple[int, int], scale: Tuple[int, int], 
                simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])
    return np.array(all_key), np.array(all_score)

class SimpleDwposeDetector:
    def __init__(self, model_path: str, yolo_model_path: str = None, device: str = "cuda"):
        self.model_path = model_path
        self.yolo_model_path = yolo_model_path
        self.device = device
        self.model = None
        self.yolo_model = None
        self.pose_input_size = (288, 384)
        self.use_yolo = yolo_model_path is not None

        self.yolo_queue = queue.Queue(maxsize=50)  
        self.pose_queue = queue.Queue(maxsize=20)  
        self.result_queue = queue.Queue() 
        
        self.yolo_workers = 20
        self.pose_workers = 20
        self.pose_batch_size = 5
    def load_model(self):
        if self.model is None:
            self.model = torch.jit.load(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        
        if self.use_yolo and self.yolo_model is None:
            self.yolo_model = torch.jit.load(self.yolo_model_path, map_location=self.device)
            self.yolo_model.to(self.device)
            self.yolo_model.eval()
    
    def yolo_worker(self, worker_id, images_queue, results_queue):
        while True:
            try:
                item = images_queue.get(timeout=1.0)
                if item is None:
                    break
                
                img_idx, img = item
                
                if self.use_yolo:
                    det_result = inference_detector(self.yolo_model, img, detect_classes=[0])
                    if det_result is None or len(det_result) == 0:
                        det_result = []
                else:
                    det_result = []
                
                results_queue.put((img_idx, img, det_result))
                images_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                try:
                    images_queue.task_done()
                except:
                    pass
    
    def pose_batch_collector(self, yolo_results_queue, pose_tasks_queue):
        batch = []
        batch_count = 0
        last_batch_time = time.time()
        
        while True:
            try:
                item = yolo_results_queue.get(timeout=0.1)
                if item is None:
                    if batch:
                        batch_count += 1
                        pose_tasks_queue.put(batch)
                    break
                
                batch.append(item)
                
                if len(batch) >= self.pose_batch_size:
                    batch_count += 1
                    pose_tasks_queue.put(batch)
                    batch = []
                    last_batch_time = time.time()
                
                yolo_results_queue.task_done()
                
            except queue.Empty:
                current_time = time.time()
                if batch and (current_time - last_batch_time) > 0.1:
                    batch_count += 1
                    pose_tasks_queue.put(batch)
                    batch = []
                    last_batch_time = current_time
                continue
            except Exception as e:
                pass
    
    def pose_worker(self, worker_id, tasks_queue, results_queue):
        processed_batches = 0
        while True:
            try:
                batch = tasks_queue.get(timeout=1.0)
                if batch is None:
                    break
                
                start_time = time.time()
                
                for i, (img_idx, img, det_result) in enumerate(batch):
                    keypoints, scores = self._inference_pose(self.model, det_result, img, self.pose_input_size, self.pose_batch_size)
                    keypoints_info = self._process_keypoints(keypoints, scores)
                    result = self.format_result(keypoints_info)
                    
                    results_queue.put((img_idx, result))
                    
                    if hasattr(self, 'pbar') and self.pbar is not None:
                        self.pbar.update(1)
                
                processed_batches += 1
                tasks_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                tasks_queue.task_done()
                
    def detect_poses(self, batch_images) -> List[List[PoseResult]]:
        if self.model is None:
            self.load_model()
        
        num_images = len(batch_images)
        
        pbar = comfy.utils.ProgressBar(num_images)
        self.pbar = pbar
        
        self.yolo_queue.queue.clear()
        self.pose_queue.queue.clear()
        self.result_queue.queue.clear()
        
        try:
            return self._pipeline_pose_processing(batch_images, pbar)
        finally:
            self.pbar = None
    
    def _pipeline_pose_processing(self, batch_images, pbar) -> List[List[PoseResult]]:
        input_queue = queue.Queue()
        for i, img in enumerate(batch_images):
            input_queue.put((i, img))
        
        yolo_threads = []
        pose_threads = []
        
        for i in range(self.yolo_workers):
            t = threading.Thread(
                target=self.yolo_worker,
                args=(i, input_queue, self.yolo_queue)
            )
            t.daemon = True
            t.start()
            yolo_threads.append(t)
        
        collector_thread = threading.Thread(
            target=self.pose_batch_collector,
            args=(self.yolo_queue, self.pose_queue)
        )
        collector_thread.daemon = True
        collector_thread.start()
        
        for i in range(self.pose_workers):
            t = threading.Thread(
                target=self.pose_worker,
                args=(i, self.pose_queue, self.result_queue)
            )
            t.daemon = True
            t.start()
            pose_threads.append(t)
        
        input_queue.join()
        
        for _ in range(self.yolo_workers):
            input_queue.put(None)
        
        for t in yolo_threads:
            t.join()
        
        self.yolo_queue.put(None)
        collector_thread.join()
        
        for _ in range(self.pose_workers):
            self.pose_queue.put(None)
        
        for t in pose_threads:
            t.join()
        
        results = [None] * len(batch_images)
        while not self.result_queue.empty():
            try:
                img_idx, result = self.result_queue.get_nowait()
                results[img_idx] = result
            except queue.Empty:
                break
        
        for i in range(len(batch_images)):
            if results[i] is None:
                results[i] = []
        
        pbar.update_absolute(len(batch_images))
        return results
    
    
    def _inference_pose(self, model, out_bbox, oriImg, model_input_size=(288, 384), bs: int = 5):
        resized_img, center, scale = preprocess(oriImg, out_bbox, model_input_size)
        
        orig_img_count = len(resized_img)
        if bs > 0 and (orig_img_count % bs) != 0:
            for _ in range(bs - (orig_img_count % bs)):
                resized_img.append(np.zeros_like(resized_img[0]))
        
        input_tensor = np.stack(resized_img, axis=0).transpose(0, 3, 1, 2)
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        input_tensor = torch.from_numpy(input_tensor).to(device, dtype)
        
        with torch.no_grad():
            batched_out1, batched_out2 = [], []
            total_batches = (input_tensor.shape[0] // bs) if bs > 0 else 1
            if total_batches == 0:
                total_batches = 1
            for i in range(total_batches):
                start = i * bs
                end = (i + 1) * bs
                curr = input_tensor[start:end]
                outputs = model(curr)
                batched_out1.append(outputs[0].float())
                batched_out2.append(outputs[1].float())
            
            out1 = torch.cat(batched_out1, dim=0)[:orig_img_count].float().cpu().detach().numpy()
            out2 = torch.cat(batched_out2, dim=0)[:orig_img_count].float().cpu().detach().numpy()
        
        all_out = []
        for batch_idx in range(len(out1)):
            all_out.append([out1[batch_idx:batch_idx+1, ...], out2[batch_idx:batch_idx+1, ...]])
        
        keypoints, scores = postprocess(all_out, model_input_size, center, scale)
        return keypoints, scores
    
    def _process_keypoints(self, keypoints, scores):
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        return new_keypoints_info
    
    @staticmethod
    def format_result(keypoints_info: Optional[np.ndarray]) -> List[PoseResult]:
        def format_keypoint_part(part: np.ndarray) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return None if all(keypoint is None for keypoint in keypoints) else keypoints
        pose_results = []
        if keypoints_info is None:
            return pose_results
        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])
            if face is not None:
                face.append(body_keypoints[14])
                face.append(body_keypoints[15])
            body = BodyResult(body_keypoints, 0.0, len(body_keypoints))
            pose_results.append(PoseResult(body, left_hand, right_hand, face))
        return pose_results

def get_face_bbox_from_keypoints(face_keypoints: List[Keypoint], img_height: int, img_width: int, scale: float = 1.3, extra_padding: int = 0):
    """Extract face bounding box from face keypoints (WanAnimate compatible)
    Args:
        face_keypoints: List of face keypoints
        img_height: Image height
        img_width: Image width
        scale: Area scale factor (default 1.3)
        extra_padding: Additional padding in pixels (default 0)
    Returns: (x1, x2, y1, y2) - note the order!
    """
    if not face_keypoints or all(kp is None for kp in face_keypoints):
        # Fallback: use upper-center area
        fallback_size = int(min(img_height, img_width) * 0.3)
        x1 = (img_width - fallback_size) // 2
        x2 = x1 + fallback_size
        y1 = int(img_height * 0.1)
        y2 = y1 + fallback_size
        return (x1, x2, y1, y2)  # Note: x1, x2, y1, y2 order!
    
    # Get valid keypoints (skip first point like WanAnimate does)
    # Note: DWPose keypoints are ALREADY in pixel coordinates, not normalized!
    valid_points = [(kp.x, kp.y) for kp in face_keypoints[1:] if kp is not None]
    
    if not valid_points:
        # Fallback
        fallback_size = int(min(img_height, img_width) * 0.3)
        x1 = (img_width - fallback_size) // 2
        x2 = x1 + fallback_size
        y1 = int(img_height * 0.1)
        y2 = y1 + fallback_size
        return (x1, x2, y1, y2)  # Note: x1, x2, y1, y2 order!
    
    # Calculate initial bounding box
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    # Calculate dimensions
    initial_width = max_x - min_x
    initial_height = max_y - min_y
    initial_area = initial_width * initial_height
    
    # Expand area by scale
    expanded_area = initial_area * scale
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))
    
    # Calculate deltas
    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4  # Quarter for asymmetric expansion
    
    # Asymmetric expansion: more space at top (for forehead), less at bottom
    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, img_width)
    expanded_min_y = max(min_y - 3 * delta_height, 0)  # 3x expansion upward
    expanded_max_y = min(max_y + delta_height, img_height)  # 1x expansion downward
    
    # Apply extra padding (uniform on all sides)
    if extra_padding > 0:
        expanded_min_x = max(expanded_min_x - extra_padding, 0)
        expanded_max_x = min(expanded_max_x + extra_padding, img_width)
        expanded_min_y = max(expanded_min_y - extra_padding, 0)
        expanded_max_y = min(expanded_max_y + extra_padding, img_height)
    
    # Return in WanAnimate format: (x1, x2, y1, y2)
    return (int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y))

def convert_to_openpose_format(poses: List[PoseResult], img_height: int, img_width: int) -> dict:
    """Convert pose results to OpenPose JSON format"""
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d": compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': img_height,
        'canvas_width': img_width,
    }

class DWPoseDetectorNode:
    def __init__(self):
        self.detector = None
        self.device = model_management.get_torch_device()
    def get_model_path(self):
        # 统一使用 custom_nodes\comfyui_controlnet_aux\ckpts 文件夹
        # 情况1: 独立运行 - 使用__file__路径
        if '__file__' in globals() and os.path.exists(__file__):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # hhynodes和comfyui_controlnet_aux是同级的，都在custom_nodes下
            ckpt_dir = os.path.join(current_dir, "..", "comfyui_controlnet_aux", "ckpts")
            ckpt_dir = os.path.abspath(ckpt_dir)
        else:
            # 情况2: init动态生成 - 从ComfyUI根目录补齐路径
            ckpt_dir = os.path.join(os.getcwd(), "custom_nodes", "comfyui_controlnet_aux", "ckpts")
        
        model_path = os.path.join(ckpt_dir, "hr16", "DWPose-TorchScript-BatchSize5", "dw-ll_ucoco_384_bs5.torchscript.pt")
        yolo_path = os.path.join(ckpt_dir, "hr16", "yolox-onnx", "yolox_l.torchscript.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ DWPose模型文件不存在: {model_path}")
        
        if not os.path.exists(yolo_path):
            yolo_path = None
        
        return model_path, yolo_path
    def initialize_detector(self):
        if self.detector is None:
            try:
                model_path, yolo_path = self.get_model_path()
                self.detector = SimpleDwposeDetector(model_path, yolo_path, self.device)
            except Exception as e:
                raise
        else:
            self.detector.yolo_queue.queue.clear()
            self.detector.pose_queue.queue.clear()
            self.detector.result_queue.queue.clear()
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_body": ("BOOLEAN", {"default": True}),
                "detect_hand": ("BOOLEAN", {"default": True}),
                "detect_face": ("BOOLEAN", {"default": True}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "thick_lines": ("BOOLEAN", {"default": False}),
                "face_extra_padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 10, "tooltip": "Extra padding in pixels for face crop (0=standard, 100=include more head)"}),
                "smooth_window_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2, "tooltip": "Temporal smoothing window size (1=no smooth, 5=medium, 11=high smooth). Reduces face jitter in videos"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("original_images", "pose_images", "pose_keypoints", "face_images")
    OUTPUT_IS_LIST = (True, True, False, True)
    FUNCTION = "process"
    CATEGORY = "hhy"
    def process(self, image, detect_body=True, detect_hand=True, detect_face=True, resolution=512, thick_lines=False, face_extra_padding=0, smooth_window_size=5):
        try:
            self.initialize_detector()
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    batch_images = []
                    for i in range(image.shape[0]):
                        img_array = (image[i].cpu().numpy() * 255).astype(np.uint8)
                        batch_images.append(img_array)
                else:
                    batch_images = [(image.cpu().numpy() * 255).astype(np.uint8)]
            else:
                batch_images = [image] if not isinstance(image, list) else image
            resized_images = []
            for i, input_array in enumerate(batch_images):
                h, w = input_array.shape[:2]
                if max(h, w) != resolution:
                    if h > w:
                        new_h, new_w = resolution, int(w * resolution / h)
                    else:
                        new_h, new_w = int(h * resolution / w), resolution
                    input_array = cv2.resize(input_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                resized_images.append(input_array)
            
            batch_poses_results = self.detector.detect_poses(resized_images)
            
            # ===== PHASE 1: Collect all face bbox data for smoothing =====
            print(f"[DWPose Face Crop] Phase 1: Collecting face bbox data from {len(resized_images)} frames...")
            face_bbox_data = []  # List of (center_x, center_y, size, frame_idx)
            
            for i, (input_array, poses) in enumerate(zip(resized_images, batch_poses_results)):
                H, W = input_array.shape[:2]
                
                if poses and len(poses) > 0 and poses[0].face is not None:
                    face_bbox = get_face_bbox_from_keypoints(poses[0].face, H, W, scale=1.3, extra_padding=face_extra_padding)
                    x1_detected, x2_detected, y1_detected, y2_detected = face_bbox
                    
                    center_x = (x1_detected + x2_detected) / 2
                    center_y = (y1_detected + y2_detected) / 2
                    size = max(x2_detected - x1_detected, y2_detected - y1_detected)
                    
                    face_bbox_data.append((center_x, center_y, size, i))
                else:
                    # No face detected, use None as placeholder
                    face_bbox_data.append(None)
            
            # ===== PHASE 2: Apply sliding window smoothing =====
            def smooth_bbox_data(bbox_data, window_size=5):
                """Apply sliding window average to bbox center and size"""
                if window_size <= 1:
                    return bbox_data
                
                # Find valid indices (frames with detected faces)
                valid_indices = [i for i, data in enumerate(bbox_data) if data is not None]
                
                if len(valid_indices) <= 1:
                    print(f"[DWPose Face Crop] Only {len(valid_indices)} valid detections, skipping smoothing")
                    return bbox_data
                
                print(f"[DWPose Face Crop] Phase 2: Smoothing {len(valid_indices)} bbox detections with window size {window_size}")
                
                smoothed_data = bbox_data.copy()
                
                for i in valid_indices:
                    # Find neighboring valid detections within window
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(bbox_data), i + window_size // 2 + 1)
                    
                    valid_neighbors = []
                    for j in range(window_start, window_end):
                        if bbox_data[j] is not None:
                            valid_neighbors.append(bbox_data[j])
                    
                    if len(valid_neighbors) > 0:
                        # Calculate average
                        avg_center_x = sum(data[0] for data in valid_neighbors) / len(valid_neighbors)
                        avg_center_y = sum(data[1] for data in valid_neighbors) / len(valid_neighbors)
                        avg_size = sum(data[2] for data in valid_neighbors) / len(valid_neighbors)
                        
                        smoothed_data[i] = (avg_center_x, avg_center_y, avg_size, bbox_data[i][3])
                
                return smoothed_data
            
            smoothed_bbox_data = smooth_bbox_data(face_bbox_data, smooth_window_size)
            print(f"[DWPose Face Crop] Phase 3: Processing {len(resized_images)} frames with smoothed bbox...")
            
            # ===== PHASE 3: Process frames with smoothed bbox =====
            original_tensors = []
            pose_tensors = []
            all_pose_keypoints = []
            face_tensors = []
            
            for i, (input_array, poses) in enumerate(zip(resized_images, batch_poses_results)):
                pose_canvas = draw_poses(
                    poses, 
                    input_array.shape[0], 
                    input_array.shape[1], 
                    draw_body=detect_body, 
                    draw_hand=detect_hand, 
                    draw_face=detect_face,
                    xinsr_stick_scaling=thick_lines
                )
                
                # Convert pose data to OpenPose format
                openpose_dict = convert_to_openpose_format(poses, input_array.shape[0], input_array.shape[1])
                
                # Extract face image
                H, W = input_array.shape[:2]
                
                # Use smoothed bbox data if available
                smoothed_data = smoothed_bbox_data[i]
                
                if smoothed_data is not None:
                    # Get smoothed bbox and calculate coordinates
                    smooth_center_x, smooth_center_y, smooth_size, _ = smoothed_data
                    half_size = int(smooth_size / 2)
                    x1 = int(smooth_center_x - half_size)
                    x2 = int(smooth_center_x + half_size)
                    y1 = int(smooth_center_y - half_size)
                    y2 = int(smooth_center_y + half_size)
                    
                    # CRITICAL FIX: Make bbox square to prevent face distortion
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    if bbox_width != bbox_height:
                        # Expand to square using the larger dimension
                        target_size = max(bbox_width, bbox_height)
                        
                        # Calculate center of original bbox
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Calculate new square bbox centered on original center
                        half_size = target_size // 2
                        x1 = max(0, center_x - half_size)
                        x2 = min(W, center_x + half_size)
                        y1 = max(0, center_y - half_size)
                        y2 = min(H, center_y + half_size)
                        
                        # If we hit boundaries, adjust the opposite side to maintain square
                        actual_width = x2 - x1
                        actual_height = y2 - y1
                        if actual_width < target_size:
                            if x1 == 0:
                                x2 = min(W, x1 + target_size)
                            else:
                                x1 = max(0, x2 - target_size)
                        if actual_height < target_size:
                            if y1 == 0:
                                y2 = min(H, y1 + target_size)
                            else:
                                y1 = max(0, y2 - target_size)
                    
                    # Crop the region first
                    face_image = input_array[y1:y2, x1:x2]
                    
                    # CRITICAL FIX PART 2: If cropped region is still not square (due to boundary limits),
                    # crop a square from the center using the shorter dimension
                    crop_h, crop_w = face_image.shape[:2]
                    if crop_w != crop_h:
                        square_size = min(crop_w, crop_h)
                        start_x = (crop_w - square_size) // 2
                        start_y = (crop_h - square_size) // 2
                        face_image = face_image[start_y:start_y+square_size, start_x:start_x+square_size]
                    
                    # Check if valid
                    if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                        # Create fallback
                        fallback_size = int(min(H, W) * 0.3)
                        fallback_x1 = (W - fallback_size) // 2
                        fallback_x2 = fallback_x1 + fallback_size
                        fallback_y1 = int(H * 0.1)
                        fallback_y2 = fallback_y1 + fallback_size
                        face_image = input_array[fallback_y1:fallback_y2, fallback_x1:fallback_x2]
                        
                        if face_image.size == 0:
                            face_image = np.zeros((fallback_size, fallback_size, 3), dtype=input_array.dtype)
                else:
                    # No face detected, create fallback
                    fallback_size = int(min(H, W) * 0.3)
                    fallback_x1 = (W - fallback_size) // 2
                    fallback_x2 = fallback_x1 + fallback_size
                    fallback_y1 = int(H * 0.1)
                    fallback_y2 = fallback_y1 + fallback_size
                    face_image = input_array[fallback_y1:fallback_y2, fallback_x1:fallback_x2]
                    
                    if face_image.size == 0:
                        face_image = np.zeros((fallback_size, fallback_size, 3), dtype=input_array.dtype)
                
                # Resize to 512x512
                face_image = cv2.resize(face_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                face_tensor = numpy2tensor(face_image)
                
                original_tensor = numpy2tensor(input_array)
                pose_tensor = numpy2tensor(pose_canvas)
                original_tensors.append(original_tensor)
                pose_tensors.append(pose_tensor)
                all_pose_keypoints.append(openpose_dict)
                face_tensors.append(face_tensor)
            
            return (original_tensors, pose_tensors, all_pose_keypoints, face_tensors)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    blank_images = []
                    blank_keypoints = []
                    blank_faces = []
                    for i in range(image.shape[0]):
                        blank = torch.zeros_like(image[i])
                        blank_images.append(blank.unsqueeze(0))
                        blank_keypoints.append({"people": [], "canvas_height": 512, "canvas_width": 512})
                        blank_face = torch.zeros((512, 512, 3), dtype=torch.float32)
                        blank_faces.append(blank_face.unsqueeze(0))
                    return (blank_images, blank_images, blank_keypoints, blank_faces)
                else:
                    blank = torch.zeros_like(image)
                    blank_keypoints = [{"people": [], "canvas_height": 512, "canvas_width": 512}]
                    blank_face = torch.zeros((512, 512, 3), dtype=torch.float32).unsqueeze(0)
                    return ([blank.unsqueeze(0)], [blank.unsqueeze(0)], blank_keypoints, [blank_face])
            else:
                blank_tensor = numpy2tensor(np.zeros((512, 512, 3), dtype=np.uint8))
                blank_keypoints = [{"people": [], "canvas_height": 512, "canvas_width": 512}]
                blank_face = numpy2tensor(np.zeros((512, 512, 3), dtype=np.uint8))
                return ([blank_tensor], [blank_tensor], blank_keypoints, [blank_face])

NODE_CLASS_MAPPINGS = {
    "DWPoseDetectorNode": DWPoseDetectorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseDetectorNode": "Multi-Thread DWPose Detector by hhy"
}
