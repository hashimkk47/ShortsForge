"""Central configuration, loaded once from the environment / ``.env`` file.

Settings are grouped by the stage that consumes them. Every value can be
overridden with an environment variable so the same code runs unchanged across
a laptop, a CUDA workstation, and CI.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# API mode — MuAPI (hosted download / transcription / LLM / autocrop)
# ---------------------------------------------------------------------------
MUAPI_API_KEY = os.getenv("MUAPI_API_KEY", "").strip()
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# ---------------------------------------------------------------------------
# Local mode — highlight LLM provider
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()  # openai / gemini / ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# ---------------------------------------------------------------------------
# Local mode — Whisper transcription (faster-whisper)
# ---------------------------------------------------------------------------
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto / cpu / cuda

# Voice Activity Detection (VAD) for faster-whisper. Disabled by default because
# it is too aggressive on mixed speech/music content, clipping real speech.
#   threshold                0.5   lower = more sensitive
#   min_speech_duration_ms   250   drop shorter blips
#   min_silence_duration_ms  2000  raise to avoid splitting mid-sentence
LOCAL_WHISPER_VAD_FILTER = _env_bool("LOCAL_WHISPER_VAD_FILTER", False)
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
LOCAL_WHISPER_VAD_PARAMETERS: Dict[str, Any] = (
    json.loads(_vad_params_env)
    if _vad_params_env
    else {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }
)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")


def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Local mode needs an OpenAI key when "
            "LLM_PROVIDER=openai. Add it to your .env, or switch LLM_PROVIDER to "
            "'ollama' to run the highlight model entirely offline."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Local mode needs a Gemini key when "
            "LLM_PROVIDER=gemini. Add it to your .env, or switch LLM_PROVIDER to "
            "'ollama' to run the highlight model entirely offline."
        )
    return GEMINI_API_KEY
