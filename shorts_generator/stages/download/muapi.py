"""API-mode downloader — YouTube fetch via MuAPI ``/youtube-download``."""
from __future__ import annotations

from typing import Dict

from ...clients import muapi


def extract_video_url(result: Dict) -> str:
    """Pull a downloaded/rendered mp4 URL out of a MuAPI result.

    MuAPI result shapes vary by endpoint, so probe the common keys and the
    nested ``output``/``outputs``/``result`` containers.
    """
    for key in ("video_url", "url", "output_url", "result_url"):
        v = result.get(key)
        if isinstance(v, str) and v.startswith("http"):
            return v

    output = result.get("outputs") or result.get("output") or result.get("result") or {}
    if isinstance(output, dict):
        for key in ("video_url", "url", "output_url"):
            v = output.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
    if isinstance(output, list) and output and isinstance(output[0], str) and output[0].startswith("http"):
        return output[0]

    raise RuntimeError(f"Could not find downloaded video URL in MuAPI response: {result}")


def download_youtube(source: str, fmt: str = "720") -> str:
    """Hand a YouTube URL to MuAPI; return a hosted mp4 URL we can read from."""
    print(f"[download] requesting {source} @ {fmt}p", flush=True)
    result = muapi.run(
        "youtube-download",
        {"video_url": source, "format": fmt},
        label="youtube-download",
    )
    out = extract_video_url(result)
    print(f"[download] ready: {out}", flush=True)
    return out
