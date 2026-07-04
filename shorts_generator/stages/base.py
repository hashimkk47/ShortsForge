"""Stage contracts (re-exported).

The stage :class:`~typing.Protocol` interfaces live in
:mod:`shorts_generator.protocols` so the pipeline can depend on them without a
circular import. They are re-exported here because ``stages.base`` is the
natural place to look for "what does a stage have to implement?".
"""
from ..protocols import (
    Downloader,
    Encoder,
    FaceTracker,
    HighlightEngine,
    Renderer,
    SubtitleEngine,
    Transcriber,
    Uploader,
)

__all__ = [
    "Downloader",
    "Transcriber",
    "HighlightEngine",
    "Renderer",
    "SubtitleEngine",
    "FaceTracker",
    "Encoder",
    "Uploader",
]
