"""Local-mode transcriber — faster-whisper (CPU or CUDA).

Reads a local media file and returns the
:class:`~shorts_generator.types.Transcript` shape the highlight engine expects.
Results are cached next to the media as a sidecar ``.srt`` so re-runs skip
Whisper entirely, and so the local renderer can burn the same subtitles.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from ...config import LOCAL_OUTPUT_DIR, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
from ...types import Segment, Transcript


def _transcript_cache_path(media_path: str) -> Path:
    """Return the ``.srt`` cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Transcript) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.extend([str(idx), f"{start} --> {end}", text, ""])

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Transcript:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("﻿") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]  # skip the numeric index line
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = (part.strip() for part in lines[0].split("-->", 1))
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": "\n".join(lines[1:]).strip(),
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    """Return the compute device, verifying CUDA actually initialises."""
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # Touch the device to catch missing cuBLAS/cuDNN libraries early.
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def transcribe_local(media: str, language: Optional[str] = None) -> Transcript:
    """Run faster-whisper on a local file path, caching the result as ``.srt``."""
    cache_path = _transcript_cache_path(media)
    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(media):
        print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
        cached = _load_srt_cache(cache_path)
        # A cache with no segments is likely from a failed run — drop and redo.
        if cached["segments"] and cached["duration"] > 0.0:
            print(
                f"[transcribe/local] {len(cached['segments'])} cached segments, "
                f"{cached['duration']:.0f}s of audio",
                flush=True,
            )
            return cached
        print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
        cache_path.unlink(missing_ok=True)

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    from ...config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}",
        flush=True,
    )

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)
    transcribe_kwargs = {
        "audio": media,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": LOCAL_WHISPER_VAD_FILTER,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments: list[Segment] = [
        {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
        for s in segments_iter
    ]

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    transcript: Transcript = {"duration": duration, "segments": segments}

    written = _write_srt_cache(media, transcript)
    print(f"[transcribe/local] wrote cache: {written}", flush=True)
    return transcript
