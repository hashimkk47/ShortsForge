"""Encoder sub-stage — the FFmpeg layer of the local renderer.

Owns everything that shells out to FFmpeg: hardware-encoder detection and
selection (NVENC with a CPU fallback), accurate cutting, audio muxing, the final
platform-ready encode, and subtitle burning. Geometry helpers for the supported
delivery aspect ratios live here too, since the encode depends on them.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import List, Sequence, Tuple

#: Social-delivery pixel dimensions per supported aspect ratio.
_OUTPUT_SIZES: dict[str, Tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "16:9": (1920, 1080),
}


def ratio(aspect_ratio: str) -> float:
    """Parse and validate a supported output aspect ratio into a float."""
    if aspect_ratio not in _OUTPUT_SIZES:
        raise ValueError(
            f"Unsupported aspect ratio {aspect_ratio!r}; use 9:16, 1:1, 4:5, or 16:9"
        )
    width, height = aspect_ratio.split(":")
    return float(width) / float(height)


def output_size(aspect_ratio: str) -> Tuple[int, int]:
    """Return social-platform delivery dimensions for an aspect ratio."""
    ratio(aspect_ratio)  # validate
    return _OUTPUT_SIZES[aspect_ratio]


def _run_ffmpeg(cmd: Sequence[str], operation: str) -> None:
    """Run FFmpeg synchronously and surface useful diagnostics on failure."""
    try:
        completed = subprocess.run(
            list(cmd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if completed.returncode:
        details = completed.stderr.strip()[-3000:]
        raise RuntimeError(f"FFmpeg failed while {operation}: {details}")


def _run_video_ffmpeg(cmd: Sequence[str], operation: str) -> None:
    """Run a video encode, retrying on CPU if a runtime NVENC error occurs."""
    command = list(cmd)
    try:
        _run_ffmpeg(command, operation)
        return
    except RuntimeError:
        if "h264_nvenc" not in command:
            raise

    # Rebuild the command with a libx264 encoder and no hardware acceleration.
    fallback = command.copy()
    if "-hwaccel" in fallback:
        index = fallback.index("-hwaccel")
        del fallback[index : index + 2]
    encoder_index = fallback.index("h264_nvenc") - 1
    terminators = [
        index
        for index in (
            fallback.index("-c:a") if "-c:a" in fallback else len(fallback),
            fallback.index("-pix_fmt") if "-pix_fmt" in fallback else len(fallback),
        )
        if index > encoder_index
    ]
    encoder_end = min(terminators, default=len(fallback))
    fallback[encoder_index:encoder_end] = ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    _run_ffmpeg(fallback, f"{operation} with CPU encoding")


@lru_cache(maxsize=1)
def has_nvidia_gpu() -> bool:
    """Return whether FFmpeg can complete a real NVENC encode (cached)."""
    null_device = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
        "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", null_device,
    ]
    try:
        return subprocess.run(
            cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _video_encoder_args() -> List[str]:
    """Select a high-quality GPU or CPU H.264 encoder configuration."""
    if has_nvidia_gpu():
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p6", "-tune", "hq",
            "-rc", "vbr", "-cq", "19", "-b:v", "0",
        ]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]


def cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """Accurately cut and re-encode a source interval, preserving audio."""
    if start < 0 or end <= start:
        raise ValueError(f"Invalid clip interval: start={start}, end={end}")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}"]
    if has_nvidia_gpu():
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend(
        [
            "-i", source_path,
            "-t", f"{end - start:.6f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            *_video_encoder_args(),
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
    )
    _run_video_ffmpeg(cmd, "cutting the source clip")
    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """Mux source audio onto the lossless cropped video without re-encoding video."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    _run_ffmpeg(cmd, "muxing audio")
    return out_path


def encode_video(
    in_path: str, out_path: str, aspect_ratio: str, video_filter: str | None = None
) -> str:
    """Perform the final platform-ready H.264 encode with Lanczos scaling."""
    width, height = output_size(aspect_ratio)
    filters = []
    if video_filter:
        filters.append(video_filter)
    filters.append(f"scale={width}:{height}:flags=lanczos")
    filters.append("setsar=1")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", ",".join(filters),
        *_video_encoder_args(),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-vsync", "0",
        out_path,
    ]
    _run_video_ffmpeg(cmd, "encoding the final video")
    return out_path
