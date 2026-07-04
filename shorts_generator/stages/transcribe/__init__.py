"""Transcriber stage — media in, timestamped transcript out.

* :mod:`.muapi`   — API mode: MuAPI ``/openai-whisper`` (server-side Whisper).
* :mod:`.whisper` — local mode: faster-whisper (CPU or CUDA), SRT-cached.
"""
from .muapi import transcribe
from .whisper import transcribe_local

__all__ = ["transcribe", "transcribe_local"]
