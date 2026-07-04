"""API-mode renderer — per-clip vertical reframing via MuAPI ``/autocrop``."""
from __future__ import annotations

from typing import List

from ...clients import muapi
from ...types import Highlight, Short
from ..download.muapi import extract_video_url


def crop_clip(
    source_video_url: str, start_time: float, end_time: float, aspect_ratio: str = "9:16"
) -> str:
    """Submit one autocrop job and return the URL of the rendered short."""
    payload = {
        "video_url": source_video_url,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "aspect_ratio": aspect_ratio,
    }
    print(f"[clip] {start_time:.1f}s -> {end_time:.1f}s @ {aspect_ratio}", flush=True)
    result = muapi.run("autocrop", payload, label=f"autocrop({start_time:.0f}-{end_time:.0f})")
    return extract_video_url(result)


def crop_highlights(
    source_video_url: str, highlights: List[Highlight], aspect_ratio: str = "9:16"
) -> List[Short]:
    """Crop every highlight, attaching the resulting URL back onto each item."""
    out: List[Short] = []
    for i, h in enumerate(highlights, 1):
        print(f"[clip] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            url = crop_clip(source_video_url, h["start_time"], h["end_time"], aspect_ratio=aspect_ratio)
            out.append({**h, "clip_url": url})
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            out.append({**h, "clip_url": None, "error": str(e)})
    return out
