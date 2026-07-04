"""API-mode highlight LLM — MuAPI ``gpt-5-mini``."""
from __future__ import annotations

from ...clients import muapi

# Cap LLM polls at 5 min — a wedged call should fail fast rather than block the run.
GPT_CALL_TIMEOUT_SECONDS = 300


def call_muapi_llm(prompt: str) -> str:
    """Run a single prompt through MuAPI ``gpt-5-mini`` and return the text."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")
