"""Shared data structures that flow between pipeline stages.

Everything is a plain ``TypedDict`` so the objects stay JSON-serialisable and
carry zero runtime overhead (a ``TypedDict`` *is* a ``dict``). The types exist
purely to document the contract each stage produces and consumes, which is what
makes the stages independently replaceable — a new implementation only has to
honour these shapes.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict


class Segment(TypedDict):
    """One timestamped span of transcribed speech."""

    start: float  # seconds from the start of the source video
    end: float    # seconds from the start of the source video
    text: str


class Transcript(TypedDict):
    """Full transcription of a source video."""

    duration: float          # total media duration in seconds
    segments: List[Segment]


class Highlight(TypedDict):
    """A candidate viral moment ranked by the highlight engine."""

    title: str
    start_time: float        # seconds from the start of the source video
    end_time: float          # seconds from the start of the source video
    score: int               # viral potential, 0-100
    hook_sentence: str
    virality_reason: str


class Short(Highlight, total=False):
    """A rendered short — a :class:`Highlight` plus its output location.

    ``clip_url`` is a hosted URL in API mode or a local file path in local mode.
    It is ``None`` when rendering failed, in which case ``error`` explains why.
    """

    clip_url: Optional[str]
    error: str


class PipelineResult(TypedDict):
    """The complete result returned by :func:`shorts_generator.generate_shorts`."""

    mode: str                    # "api" or "local"
    source_video_url: str        # hosted URL (api) or local path (local)
    transcript: Transcript
    highlights: List[Highlight]  # every ranked candidate
    shorts: List[Short]          # the top ``num_clips`` that were rendered
