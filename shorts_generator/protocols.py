"""Stage contracts for the shorts-generation pipeline.

The pipeline is a linear chain of independently replaceable stages::

    Downloader -> Transcriber -> HighlightEngine -> SubtitleEngine
        -> FaceTracking -> EffectsEngine -> Encoder -> Uploader

Each stage is defined here as a :class:`typing.Protocol` — a structural
interface. Any object (function or class instance) that matches the signature
satisfies the contract, so a stage can be swapped for an alternative
implementation without touching the orchestrator or its neighbours.

Concrete pipelines compose these into a smaller number of *composite* stages
for convenience (see :class:`Pipeline` in :mod:`shorts_generator.pipeline`):

* ``Downloader``      -> ``download`` step
* ``Transcriber``     -> ``transcribe`` step
* ``HighlightEngine`` -> ``rank`` step
* ``SubtitleEngine`` + ``FaceTracking`` + ``EffectsEngine`` + ``Encoder``
                      -> ``render`` step (the local renderer implements these as
                         internal sub-stages; see ``stages/render/local/``)
* ``Uploader``        -> not yet implemented (see the roadmap)

The fine-grained protocols below document the target seams so future
implementations — and the planned plugin system — have a stable contract to
build against.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from .types import Highlight, Short, Transcript


@runtime_checkable
class Downloader(Protocol):
    """Resolve a source reference to something the transcriber can read.

    Returns a hosted URL (API mode) or a local file path (local mode).
    """

    def __call__(self, source: str, fmt: str = "720") -> str: ...


@runtime_checkable
class Transcriber(Protocol):
    """Turn a media reference into a timestamped :class:`Transcript`."""

    def __call__(self, media: str, language: Optional[str] = None) -> Transcript: ...


@runtime_checkable
class HighlightEngine(Protocol):
    """Rank the most viral-worthy moments in a transcript."""

    def __call__(self, transcript: Transcript, num_clips: int) -> List[Highlight]: ...


@runtime_checkable
class Renderer(Protocol):
    """Turn ranked highlights into rendered vertical shorts.

    Composite stage: an implementation is responsible for cutting each clip and
    (in local mode) applying subtitles, face-aware reframing, effects, and the
    final encode.
    """

    def __call__(
        self, source: str, highlights: List[Highlight], aspect_ratio: str = "9:16"
    ) -> List[Short]: ...


# ---------------------------------------------------------------------------
# Fine-grained sub-stage contracts (seams inside the local renderer / roadmap)
# ---------------------------------------------------------------------------
@runtime_checkable
class SubtitleEngine(Protocol):
    """Produce a subtitle file for a clip's local timeline (0 = clip start)."""

    def __call__(
        self, source_srt: str, clip_start: float, clip_end: float, out_srt: str
    ) -> Optional[str]: ...


@runtime_checkable
class FaceTracker(Protocol):
    """Reframe a clip to ``aspect_ratio`` while keeping the speaker in frame."""

    def __call__(self, in_path: str, out_path: str, aspect_ratio: str) -> str: ...


@runtime_checkable
class Encoder(Protocol):
    """Perform the final platform-ready video encode."""

    def __call__(self, in_path: str, out_path: str, aspect_ratio: str) -> str: ...


@runtime_checkable
class Uploader(Protocol):
    """Publish a rendered short to a destination and return its URL/id.

    Not yet implemented — reserved for the upload-automation roadmap item.
    """

    def __call__(self, short: Short) -> str: ...
