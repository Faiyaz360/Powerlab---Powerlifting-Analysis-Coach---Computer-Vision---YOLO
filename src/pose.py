"""Pose estimation — the swappable MODEL INTERFACE.

Two backends, both returning the SAME ``PoseResult`` (landmarks indexed by the MediaPipe slot
numbers below), so the rest of the pipeline never changes:

- ``"mediapipe"`` — BlazePose Tasks, CPU, 33 landmarks (default fallback).
- ``"yolo"``      — YOLO11-pose on the GPU, COCO-17 keypoints remapped onto the same slots.
                    Sharper joint placement; needs torch + ultralytics.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --- landmark slot indices (MediaPipe BlazePose numbering; the pipeline speaks these) ---
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

NUM_LANDMARKS = 33

KP_CONF_MIN = 0.3  # YOLO keypoints below this confidence are dropped (NaN) so they get interpolated

# COCO-17 keypoint index -> our landmark slot. (COCO has no foot tip/heel, so 29-32 stay NaN.)
_COCO_TO_SLOT = {
    5: L_SHOULDER, 6: R_SHOULDER,
    7: L_ELBOW, 8: R_ELBOW,
    9: L_WRIST, 10: R_WRIST,
    11: L_HIP, 12: R_HIP,
    13: L_KNEE, 14: R_KNEE,
    15: L_ANKLE, 16: R_ANKLE,
}

# MediaPipe Tasks model download (complexity 0/1/2 -> lite/full/heavy)
_MODEL_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class PoseResult:
    """Per-frame landmarks plus video metadata.

    ``landmarks`` shape (num_frames, 33, 3); last axis [x_px, y_px, visibility]. NaN if missing.
    """

    landmarks: np.ndarray
    fps: float
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return int(self.landmarks.shape[0])


def _video_meta(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return float(fps), width, height


_CACHE_DIR = Path(__file__).resolve().parent.parent / ".posecache"


def estimate_pose(video_path, backend: str = "mediapipe", model_complexity: int = 2,
                  yolo_model: str = "yolo26m-pose.pt", progress=None, use_cache: bool = True) -> PoseResult:
    """Run pose estimation over a video. ``backend``: "mediapipe" or "yolo".

    Results are cached to ``.posecache/`` (keyed by video + mtime + backend) so re-analysing the
    same clip skips the slow pose pass — useful when iterating on downstream metrics.
    """
    cache = _cache_path(video_path, backend, model_complexity, yolo_model)
    if use_cache and cache and cache.exists():
        d = np.load(cache)
        return PoseResult(d["landmarks"], float(d["fps"]), int(d["width"]), int(d["height"]))

    if backend == "yolo":
        result = _estimate_pose_yolo(video_path, yolo_model, progress)
    elif backend == "rtmpose":
        result = _estimate_pose_rtmpose(video_path, progress)
    elif backend == "mediapipe":
        result = _estimate_pose_mediapipe(video_path, model_complexity, progress)
    else:
        raise ValueError(f"Unknown pose backend: {backend!r}")

    if use_cache and cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, landmarks=result.landmarks, fps=result.fps,
                 width=result.width, height=result.height)
    return result


def _cache_path(video_path, backend, complexity, yolo_model):
    p = Path(video_path)
    try:
        mtime = int(p.stat().st_mtime)
    except OSError:
        return None
    if backend == "yolo":
        tag = yolo_model.replace(".pt", "")
    elif backend == "rtmpose":
        tag = "rtmpose"
    else:
        tag = f"mp{complexity}"
    return _CACHE_DIR / f"{p.stem}_{backend}_{tag}_{mtime}.npz"


# ---------------------------------------------------------------- MediaPipe backend

def _ensure_model(complexity: int) -> Path:
    variant = _MODEL_VARIANTS.get(complexity, "full")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODELS_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        print(f"Downloading pose model ({variant}) — one time only ...")
        urllib.request.urlretrieve(_MODEL_URL.format(v=variant), path)
    return path


def _estimate_pose_mediapipe(video_path, model_complexity, progress) -> PoseResult:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = _ensure_model(model_complexity)
    fps, width, height = _video_meta(video_path)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    last_ts = -1
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int(round(idx * 1000.0 / fps)))
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)
            arr = np.full((NUM_LANDMARKS, 3), np.nan, dtype=float)
            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks[0]):
                    arr[i] = (lm.x * width, lm.y * height, lm.visibility)
            frames.append(arr)
            idx += 1
            if progress:
                progress(idx)
    cap.release()
    landmarks = np.stack(frames) if frames else np.empty((0, NUM_LANDMARKS, 3))
    return PoseResult(landmarks=landmarks, fps=fps, width=width, height=height)


# ---------------------------------------------------------------- YOLO11-pose (GPU) backend

def _estimate_pose_yolo(video_path, yolo_model, progress) -> PoseResult:
    import torch
    from ultralytics import YOLO

    fps, width, height = _video_meta(video_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(yolo_model)  # weights auto-download on first use

    frames = []
    idx = 0
    for res in model.predict(source=str(video_path), stream=True, device=device, verbose=False):
        arr = np.full((NUM_LANDMARKS, 3), np.nan, dtype=float)
        kps = res.keypoints
        if kps is not None and kps.data is not None and len(kps.data) > 0:
            person = _pick_person(res)
            coco = kps.data[person].cpu().numpy()  # (17, 3) -> x, y, conf
            for coco_idx, slot in _COCO_TO_SLOT.items():
                x, y, conf = coco[coco_idx]
                if conf >= KP_CONF_MIN:  # else leave NaN -> interpolated downstream
                    arr[slot] = (x, y, conf)
        frames.append(arr)
        idx += 1
        if progress:
            progress(idx)
    landmarks = np.stack(frames) if frames else np.empty((0, NUM_LANDMARKS, 3))
    return PoseResult(landmarks=landmarks, fps=fps, width=width, height=height)


def _pick_person(res) -> int:
    """When several people are detected, take the largest bounding box (the lifter)."""
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return 0
    xywh = boxes.xywh.cpu().numpy()
    areas = xywh[:, 2] * xywh[:, 3]
    return int(np.argmax(areas))


# ---------------------------------------------------------------- RTMPose (Apache) backend

def _estimate_pose_rtmpose(video_path, progress) -> PoseResult:
    """RTMPose via rtmlib (RTMDet person-detector + RTMPose), COCO-17 -> our slots.

    Apache-2.0, runs through onnxruntime — GPU on Colab/Linux, CPU locally. Needs:
        pip install rtmlib onnxruntime-gpu   (or plain onnxruntime for CPU)
    """
    import onnxruntime as ort
    from rtmlib import Body

    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    body = Body(mode="performance", backend="onnxruntime", device=device)

    fps, width, height = _video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keypoints, scores = body(frame)  # (P, 17, 2), (P, 17) in pixels
        arr = np.full((NUM_LANDMARKS, 3), np.nan, dtype=float)
        if len(keypoints) > 0:
            person = int(np.argmax(scores.mean(axis=1)))  # most-confident detection
            kp, sc = keypoints[person], scores[person]
            for coco_idx, slot in _COCO_TO_SLOT.items():
                if sc[coco_idx] >= KP_CONF_MIN:
                    arr[slot] = (kp[coco_idx][0], kp[coco_idx][1], sc[coco_idx])
        frames.append(arr)
        idx += 1
        if progress:
            progress(idx)
    cap.release()
    landmarks = np.stack(frames) if frames else np.empty((0, NUM_LANDMARKS, 3))
    return PoseResult(landmarks=landmarks, fps=fps, width=width, height=height)
