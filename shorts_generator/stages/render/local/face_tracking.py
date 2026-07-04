"""Face-tracking sub-stage — speaker-aware vertical reframing.

Runs MediaPipe Tasks BlazeFace over every frame, tracks the dominant face with a
velocity-smoothed selector (so the crop follows the active speaker without
jitter), and writes a lossless silent intermediate cropped to the target aspect
ratio. Audio is muxed back on afterwards by the renderer.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .encoding import has_nvidia_gpu, ratio

# ---------------------------------------------------------------------------
# Tracker tunables
# ---------------------------------------------------------------------------
_FACE_LOSS_GRACE_FRAMES = 15    # coast on last velocity before recentring
_FACE_SMOOTHING = 0.12          # lower = smoother, laggier crop motion
_CENTER_RETURN_SPEED = 0.025    # how fast the crop drifts back to centre when lost
_FACE_MIN_CONFIDENCE = float(os.getenv("LOCAL_FACE_MIN_CONFIDENCE", "0.5"))
_FACE_MODEL_URL = os.getenv(
    "LOCAL_FACE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
)
_FACE_MODEL_DOWNLOAD_RETRIES = 3


@dataclass
class _TrackingState:
    """Mutable temporal state for the dominant-face tracker."""

    center: Optional[Tuple[float, float]] = None
    box: Optional[Tuple[float, float, float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0


# ---------------------------------------------------------------------------
# Model download / cache
# ---------------------------------------------------------------------------
def _model_cache_dir() -> Path:
    """Return the on-disk cache directory for downloaded MediaPipe models."""
    override = os.getenv("LOCAL_MODEL_CACHE_DIR")
    if override:
        return Path(override)
    base = os.getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "shorts_generator" / "mediapipe"


def _download_face_model(url: str, destination: Path) -> None:
    """Atomically download the face model, retrying transient failures."""
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    last_error: Optional[BaseException] = None
    for attempt in range(1, _FACE_MODEL_DOWNLOAD_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "shorts-generator"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            if partial.stat().st_size <= 0:
                raise RuntimeError("downloaded model file is empty")
            os.replace(partial, destination)
            return
        except Exception as exc:  # retried below, re-raised after the last attempt
            last_error = exc
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
            if attempt < _FACE_MODEL_DOWNLOAD_RETRIES:
                time.sleep(0.75 * attempt)
    raise RuntimeError(f"Could not download the face detection model from {url}: {last_error}")


@lru_cache(maxsize=1)
def _ensure_face_model() -> str:
    """Resolve a local path to the BlazeFace ``.tflite`` model, downloading once."""
    override = os.getenv("LOCAL_FACE_MODEL_PATH")
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"LOCAL_FACE_MODEL_PATH points to a missing file: {override}")
        return override

    model_path = _model_cache_dir() / "blaze_face_short_range.tflite"
    if model_path.is_file() and model_path.stat().st_size > 0:
        return str(model_path)

    print("[clip/local] downloading face detection model (first run)", flush=True)
    _download_face_model(_FACE_MODEL_URL, model_path)
    return str(model_path)


# ---------------------------------------------------------------------------
# Detection + tracking
# ---------------------------------------------------------------------------
class _FaceDetector:
    """MediaPipe Tasks Vision face detector running in VIDEO mode.

    Wraps :class:`mediapipe.tasks.python.vision.FaceDetector` so the tracker
    keeps consuming plain pixel-space detection dictionaries. Owns a native
    detector handle that must be released via :meth:`close`.
    """

    def __init__(self, min_confidence: float = _FACE_MIN_CONFIDENCE) -> None:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_tasks  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        model_path = _ensure_face_model()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._mp = mp
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame: Any, timestamp_ms: int) -> List[Dict[str, float]]:
        """Detect faces in one BGR frame at a strictly increasing timestamp."""
        import cv2  # type: ignore

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        detections: List[Dict[str, float]] = []
        for detection in result.detections or []:
            box = detection.bounding_box
            x = max(0.0, float(box.origin_x))
            y = max(0.0, float(box.origin_y))
            w = min(float(width) - x, float(box.width))
            h = min(float(height) - y, float(box.height))
            if w <= 1.0 or h <= 1.0:
                continue
            confidence = float(detection.categories[0].score) if detection.categories else 0.0
            detections.append(
                {
                    "x": x, "y": y, "w": w, "h": h,
                    "cx": x + w / 2.0, "cy": y + h / 2.0,
                    "confidence": confidence,
                }
            )
        return detections

    def close(self) -> None:
        """Release the native detector; safe to call more than once."""
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None


def _box_iou(
    first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    overlap = overlap_w * overlap_h
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def _track_faces(
    detections: List[Dict[str, float]],
    state: _TrackingState,
    frame_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Select and smoothly track the largest stable face across frames."""
    width, height = frame_size
    frame_center = (width / 2.0, height / 2.0)
    chosen: Optional[Dict[str, float]] = None

    if detections:
        max_area = max(item["w"] * item["h"] for item in detections) or 1.0

        def score(item: Dict[str, float]) -> float:
            area_score = (item["w"] * item["h"]) / max_area
            if state.box is None or state.center is None:
                continuity = 0.0
            else:
                box = (item["x"], item["y"], item["w"], item["h"])
                iou = _box_iou(state.box, box)
                dx = (item["cx"] - state.center[0]) / max(width, 1)
                dy = (item["cy"] - state.center[1]) / max(height, 1)
                proximity = max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5 * 2.0)
                continuity = 0.6 * iou + 0.4 * proximity
            return 0.45 * area_score + 0.45 * continuity + 0.10 * item["confidence"]

        chosen = max(detections, key=score)

    if chosen is not None:
        target = (chosen["cx"], chosen["cy"])
        new_box = (chosen["x"], chosen["y"], chosen["w"], chosen["h"])
        state.missed_frames = 0
        if state.center is None:
            state.center = target
        else:
            old_x, old_y = state.center
            predicted_x = old_x + state.velocity[0]
            predicted_y = old_y + state.velocity[1]
            new_x = predicted_x + (target[0] - predicted_x) * _FACE_SMOOTHING
            new_y = predicted_y + (target[1] - predicted_y) * _FACE_SMOOTHING
            state.velocity = (
                0.75 * state.velocity[0] + 0.25 * (new_x - old_x),
                0.75 * state.velocity[1] + 0.25 * (new_y - old_y),
            )
            state.center = (new_x, new_y)
        state.box = new_box
    else:
        state.missed_frames += 1
        if state.center is None:
            state.center = frame_center
        elif state.missed_frames <= _FACE_LOSS_GRACE_FRAMES:
            state.center = (
                state.center[0] + state.velocity[0],
                state.center[1] + state.velocity[1],
            )
            state.velocity = (state.velocity[0] * 0.8, state.velocity[1] * 0.8)
        else:
            state.center = (
                state.center[0] + (frame_center[0] - state.center[0]) * _CENTER_RETURN_SPEED,
                state.center[1] + (frame_center[1] - state.center[1]) * _CENTER_RETURN_SPEED,
            )
            state.velocity = (0.0, 0.0)
            state.box = None

    return state.center


def _open_capture(path: str, cv2: Any) -> Any:
    """Open a video, requesting hardware decoding when OpenCV supports it."""
    cap = None
    if has_nvidia_gpu() and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        try:
            cap = cv2.VideoCapture(
                path,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, getattr(cv2, "VIDEO_ACCELERATION_ANY", 1)],
            )
        except (TypeError, cv2.error):
            cap = None
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(path)
    return cap


def crop_frames(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Track faces, crop every frame, and write a lossless silent intermediate."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from exc
    try:
        import mediapipe  # type: ignore  # noqa: F401 - presence check only
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for local face tracking. Install it with:\n"
            "    pip install mediapipe"
        ) from exc

    target_ratio = ratio(aspect_ratio)
    cap = _open_capture(in_path, cv2)
    writer = None
    detector = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {in_path}")
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError(f"Invalid video dimensions in {in_path}")
        if not fps or fps != fps or fps <= 0:  # guard against NaN / 0 fps
            fps = 30.0

        source_ratio = source_width / source_height
        if target_ratio < source_ratio:
            crop_height = source_height
            crop_width = int(round(crop_height * target_ratio))
        else:
            crop_width = source_width
            crop_height = int(round(crop_width / target_ratio))
        crop_width = max(2, min(source_width, crop_width)) & ~1
        crop_height = max(2, min(source_height, crop_height)) & ~1

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (crop_width, crop_height)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the lossless intermediate video")

        detector = _FaceDetector(_FACE_MIN_CONFIDENCE)
        state = _TrackingState()
        frame_index = 0
        last_timestamp = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1  # timestamps must be strictly increasing
            last_timestamp = timestamp_ms
            frame_index += 1
            detections = detector.detect(frame, timestamp_ms)
            center_x, center_y = _track_faces(detections, state, (source_width, source_height))
            x0 = int(round(center_x - crop_width / 2))
            y0 = int(round(center_y - crop_height / 2))
            x0 = max(0, min(source_width - crop_width, x0))
            y0 = max(0, min(source_height - crop_height, y0))
            cropped = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
            if cropped.shape[1] != crop_width or cropped.shape[0] != crop_height:
                raise RuntimeError("Unexpected frame dimensions while cropping")
            writer.write(cropped)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if detector is not None:
            detector.close()
    return out_path
