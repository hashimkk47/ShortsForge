"""Stage registry — wire interchangeable stage implementations into a pipeline.

``build_pipeline(mode)`` returns a :class:`~shorts_generator.pipeline.Pipeline`
composed of the stage implementations for that mode. Swapping a stage means
passing a different callable here (or overriding one via ``**overrides``); the
orchestrator never needs to change. This is the seam the roadmap's plugin system
will extend.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from ..pipeline import Pipeline
from .download import download_youtube, download_youtube_local
from .highlights import get_highlights
from .llm import call_local_llm, call_muapi_llm
from .render import crop_highlights, crop_highlights_local
from .transcribe import transcribe, transcribe_local

#: Default stage line-up per mode. Each entry is a fully-wired pipeline factory.
_MODES = {
    "api": lambda: Pipeline(
        mode="api",
        download=download_youtube,
        transcribe=transcribe,
        rank=partial(get_highlights, llm_fn=call_muapi_llm),
        render=crop_highlights,
    ),
    "local": lambda: Pipeline(
        mode="local",
        download=download_youtube_local,
        transcribe=transcribe_local,
        rank=partial(get_highlights, llm_fn=call_local_llm),
        render=crop_highlights_local,
    ),
}


def build_pipeline(mode: str = "api", **overrides: Any) -> Pipeline:
    """Return the wired :class:`Pipeline` for ``mode`` (``"api"`` or ``"local"``).

    Any stage can be replaced without touching the orchestrator, e.g.::

        build_pipeline("local", render=my_custom_renderer)
    """
    mode = (mode or "api").lower()
    try:
        pipeline = _MODES[mode]()
    except KeyError:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.") from None
    for stage, impl in overrides.items():
        if not hasattr(pipeline, stage):
            raise ValueError(f"Unknown stage override: {stage!r}")
        setattr(pipeline, stage, impl)
    return pipeline


__all__ = ["Pipeline", "build_pipeline"]
