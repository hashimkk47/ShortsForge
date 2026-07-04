"""API-mode transcriber — MuAPI ``/openai-whisper``.

Sends a hosted media URL to MuAPI's Whisper endpoint and returns the
:class:`~shorts_generator.types.Transcript` shape the highlight engine expects.
The endpoint runs ``verbose_json`` server-side, so per-segment timestamps come
back for free.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from ...clients import muapi
from ...types import Segment, Transcript


def _coerce_verbose(raw) -> Dict:
    """Normalise a verbose_json blob (dict or JSON string) into a dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_verbose_payload(result: Dict) -> Dict:
    """Locate the verbose_json blob (with ``segments`` + ``duration``).

    MuAPI wraps results inconsistently across endpoints, so probe each container.
    """
    for key in ("output", "result", "outputs"):
        v = result.get(key)
        if isinstance(v, dict) and "segments" in v:
            return v
        if isinstance(v, list) and v:
            decoded = _coerce_verbose(v[0])
            if "segments" in decoded:
                return decoded
        if isinstance(v, str):
            decoded = _coerce_verbose(v)
            if "segments" in decoded:
                return decoded

    if "segments" in result:
        return result

    raise RuntimeError(f"Could not find Whisper segments in MuAPI response: {result}")


def transcribe(media: str, language: Optional[str] = None) -> Transcript:
    """Run MuAPI ``/openai-whisper`` on a hosted media URL."""
    print(f"[transcribe] muapi /openai-whisper on {media}", flush=True)
    payload = {"audio_url": media, "response_format": "verbose_json"}
    if language:
        payload["language"] = language

    result = muapi.run("openai-whisper", payload, label="openai-whisper")
    verbose = _extract_verbose_payload(result)

    segments: list[Segment] = []
    for s in verbose.get("segments") or []:
        segments.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip(),
            }
        )

    duration = float(verbose.get("duration") or (segments[-1]["end"] if segments else 0.0))
    print(f"[transcribe] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments}
