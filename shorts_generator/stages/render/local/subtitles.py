"""Subtitle sub-stage — discover, retime, and burn captions for a clip.

Subtitles come from the sidecar ``.srt`` the transcriber caches next to the
source video. For each clip the overlapping cues are extracted and shifted onto
the clip's local timeline (0 = clip start), then burned into the final encode.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from ....config import LOCAL_OUTPUT_DIR
from .encoding import encode_video

#: Default ASS style string passed to FFmpeg's ``subtitles`` filter.
_SUBTITLE_STYLE = os.getenv(
    "LOCAL_SUBTITLE_STYLE",
    (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=55"
    ),
)

_SRT_TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)


def _srt_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def find_source_subtitles(source_path: str) -> Optional[str]:
    """Find the sidecar SRT next to the source, or in the output cache dir."""
    filename = Path(source_path).with_suffix(".srt").name
    candidates = (
        Path(source_path).with_suffix(".srt"),
        Path(LOCAL_OUTPUT_DIR) / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def offset_subtitles(
    source_srt: str, output_srt: str, clip_start: float, clip_end: float
) -> Optional[str]:
    """Extract overlapping SRT cues and shift them onto the clip timeline.

    Returns the written path, or ``None`` if no cue overlaps the clip window.
    """
    text = Path(source_srt).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    output: List[str] = []
    for block in blocks:
        match = _SRT_TIME_RE.search(block)
        if not match:
            continue
        cue_start = _srt_seconds(match.group("start"))
        cue_end = _srt_seconds(match.group("end"))
        if cue_end <= clip_start or cue_start >= clip_end:
            continue
        shifted_start = max(0.0, cue_start - clip_start)
        shifted_end = min(clip_end, cue_end) - clip_start
        timing = f"{_srt_timestamp(shifted_start)} --> {_srt_timestamp(shifted_end)}"
        body = block[match.end():].lstrip("\r\n")
        output.append(f"{len(output) + 1}\n{timing}\n{body}")
    if not output:
        return None
    Path(output_srt).write_text("\n\n".join(output) + "\n", encoding="utf-8")
    return output_srt


def _escape_subtitle_filter_path(path: str) -> str:
    """Escape a path for FFmpeg's subtitles filter, including Windows drives."""
    value = Path(path).resolve().as_posix()
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_subtitles(
    in_path: str,
    subtitle_path: str,
    out_path: str,
    aspect_ratio: str,
    style: Optional[str] = None,
) -> str:
    """Burn styled subtitles into the final delivery encode."""
    escaped_path = _escape_subtitle_filter_path(subtitle_path)
    force_style = (style or _SUBTITLE_STYLE).replace("'", "\\'")
    subtitle_filter = f"subtitles='{escaped_path}':force_style='{force_style}'"
    return encode_video(in_path, out_path, aspect_ratio, subtitle_filter)
