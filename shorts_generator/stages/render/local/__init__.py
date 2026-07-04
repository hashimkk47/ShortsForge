"""Local renderer — cut, face-track, subtitle, and encode each clip on-device.

This composite stage wires together the sub-stages in this package:

    cut (encoding) -> face_tracking -> mux (encoding) -> subtitles -> encode

Each clip is rendered into an isolated temp directory and atomically moved into
place, so a crash or a per-clip failure never leaves a half-written mp4 behind.
The public surface is deliberately small: :func:`crop_clip_local` renders one
clip and :func:`crop_highlights_local` renders a highlight list.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from ....config import LOCAL_OUTPUT_DIR
from ....types import Highlight, Short
from . import encoding, face_tracking, subtitles

_DELETE_RETRIES = 8
#: Temp paths registered for best-effort cleanup at interpreter exit.
_active_temp_dirs: set[str] = set()


def _cleanup(*paths: Optional[str]) -> None:
    """Remove temp files/dirs, retrying Windows sharing violations."""
    for raw_path in paths:
        if not raw_path:
            continue
        path = os.path.abspath(raw_path)
        for attempt in range(_DELETE_RETRIES):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.remove(path)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == _DELETE_RETRIES - 1:
                    raise
                time.sleep(0.08 * (attempt + 1))
        _active_temp_dirs.discard(path)


def _cleanup_orphan_temp_dirs() -> None:
    for path in tuple(_active_temp_dirs):
        try:
            _cleanup(path)
        except OSError:
            pass


atexit.register(_cleanup_orphan_temp_dirs)


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
) -> str:
    """Cut, face-track, reframe, subtitle, and encode one local clip."""
    requested_out_path = out_path
    source_path = os.path.abspath(source_path)
    out_path = os.path.abspath(out_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    encoding.ratio(aspect_ratio)  # validate early
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="shorts_clip_")
    _active_temp_dirs.add(os.path.abspath(temp_dir))
    cut_path = os.path.join(temp_dir, "cut.mp4")
    cropped_path = os.path.join(temp_dir, "cropped.mkv")
    muxed_path = os.path.join(temp_dir, "muxed.mkv")
    clipped_srt = os.path.join(temp_dir, "subtitles.srt")
    partial_output = os.path.join(
        os.path.dirname(out_path), f".{Path(out_path).name}.{os.getpid()}.partial.mp4"
    )
    _active_temp_dirs.add(os.path.abspath(partial_output))
    try:
        encoding.cut_subclip(source_path, start_time, end_time, cut_path)
        face_tracking.crop_frames(cut_path, cropped_path, aspect_ratio)
        encoding.mux_audio(cropped_path, cut_path, muxed_path)

        source_srt = subtitles.find_source_subtitles(source_path)
        subtitle_path = None
        if source_srt:
            subtitle_path = subtitles.offset_subtitles(
                source_srt, clipped_srt, start_time, end_time
            )
        if subtitle_path:
            subtitles.burn_subtitles(muxed_path, subtitle_path, partial_output, aspect_ratio)
        else:
            encoding.encode_video(muxed_path, partial_output, aspect_ratio)
        os.replace(partial_output, out_path)
        return requested_out_path
    finally:
        _cleanup(partial_output, temp_dir)


def crop_highlights_local(
    source_path: str,
    highlights: List[Highlight],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
) -> List[Short]:
    """Render local clips for highlights, keeping per-item failures isolated."""
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Short] = []
    for index, highlight in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{index:02d}.mp4")
        title = highlight.get("title", "(untitled)")
        print(f"[clip/local] {index}/{len(highlights)}: {title}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(highlight["start_time"]),
                float(highlight["end_time"]),
                aspect_ratio,
                out_path,
            )
            results.append({**highlight, "clip_url": out_path})
        except Exception as exc:
            print(f"[clip/local] {index} failed: {exc}", flush=True)
            results.append({**highlight, "clip_url": None, "error": str(exc)})
    return results
