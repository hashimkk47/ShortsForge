"""Renderer stage — ranked highlights in, rendered vertical shorts out.

* :mod:`.muapi` — API mode: MuAPI ``/autocrop`` reframes each clip server-side.
* :mod:`.local` — local mode: a composite renderer that cuts, face-tracks,
  subtitles, and encodes each clip on your machine. Its sub-stages
  (``face_tracking``, ``subtitles``, ``encoding``) map onto the fine-grained
  stage contracts in :mod:`shorts_generator.stages.base`.
"""
from .local import crop_highlights_local
from .muapi import crop_highlights

__all__ = ["crop_highlights", "crop_highlights_local"]
