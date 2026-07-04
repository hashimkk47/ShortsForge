"""Highlight LLM backends.

The highlight engine is text-in/text-out, so its LLM is fully pluggable:

* :mod:`.muapi` — hosted ``gpt-5-mini`` via MuAPI (API mode).
* :mod:`.local` — OpenAI, Gemini, or Ollama, selected by ``LLM_PROVIDER`` (local mode).

Each backend is a plain ``Callable[[str], str]`` (see ``LLMFn``).
"""
from typing import Callable

from .local import call_local_llm
from .muapi import call_muapi_llm

#: A highlight LLM backend: takes a prompt, returns the raw completion text.
LLMFn = Callable[[str], str]

__all__ = ["LLMFn", "call_local_llm", "call_muapi_llm"]
