"""Pipeline orchestrator — compose stages and run them end to end.

A :class:`Pipeline` is just four interchangeable callables (download,
transcribe, rank, render). The orchestration below is mode-agnostic: it never
references MuAPI or faster-whisper directly, so a stage can be replaced without
editing this file. See :mod:`shorts_generator.stages` for how the concrete
API-mode and local-mode line-ups are wired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .protocols import Downloader, HighlightEngine, Renderer, Transcriber
from .types import Highlight, PipelineResult, Short


@dataclass
class Pipeline:
    """An interchangeable chain of the four core pipeline stages."""

    mode: str
    download: Downloader
    transcribe: Transcriber
    rank: HighlightEngine
    render: Renderer

    def run(
        self,
        source: str,
        num_clips: int = 3,
        aspect_ratio: str = "9:16",
        download_format: str = "720",
        language: Optional[str] = None,
    ) -> PipelineResult:
        """Execute download -> transcribe -> rank -> render and collect results."""
        source_ref = self.download(source, download_format)

        transcript = self.transcribe(source_ref, language)
        if not transcript["segments"]:
            raise RuntimeError(
                "Whisper produced no segments. The video may have no detectable speech."
            )

        highlights: List[Highlight] = self.rank(transcript, num_clips)
        if not highlights:
            raise RuntimeError("Highlight generator returned zero clips.")

        top = sorted(highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
        print(
            f"[pipeline/{self.mode}] rendering {len(top)} of {len(highlights)} candidates",
            flush=True,
        )
        shorts: List[Short] = self.render(source_ref, top, aspect_ratio)

        return {
            "mode": self.mode,
            "source_video_url": source_ref,
            "transcript": transcript,
            "highlights": highlights,
            "shorts": shorts,
        }


def generate_shorts(
    source: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
) -> PipelineResult:
    """Run the full pipeline and return a structured result.

    Args:
        source: YouTube URL, ``file://`` URL, or local path (local mode only).
        num_clips: how many shorts to render.
        aspect_ratio: e.g. ``"9:16"``, ``"1:1"``, ``"4:5"``, ``"16:9"``.
        download_format: source resolution (``"360"`` / ``"480"`` / ``"720"`` / ``"1080"``).
        language: ISO-639-1 code to force the transcriber's language detection.
        mode: ``"api"`` (MuAPI, hosted) or ``"local"`` (on-device).

    Returns:
        A :class:`~shorts_generator.types.PipelineResult` with the transcript,
        every ranked highlight, and the top ``num_clips`` rendered shorts.
    """
    from .stages import build_pipeline

    return build_pipeline(mode).run(
        source,
        num_clips=num_clips,
        aspect_ratio=aspect_ratio,
        download_format=download_format,
        language=language,
    )
