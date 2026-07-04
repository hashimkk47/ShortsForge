# Contributing to ShortsForge

Thanks for your interest in improving ShortsForge! Contributions of all kinds are
welcome — bug reports, features, docs, and new stage implementations.

## Getting started

```bash
git clone https://github.com/hashimkk47/ShortsForge.git
cd ShortsForge
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements-local.txt
pip install ruff black          # dev tooling
cp .env.example .env            # then fill in what you need
```

## Project layout

ShortsForge is a linear chain of independently replaceable **stages**. The
contracts live in [`shorts_generator/protocols.py`](shorts_generator/protocols.py),
and each mode is wired together in
[`shorts_generator/stages/__init__.py`](shorts_generator/stages/__init__.py).

```
shorts_generator/
├── protocols.py        # stage contracts (Protocol interfaces)
├── pipeline.py         # mode-agnostic orchestrator
├── stages/
│   ├── __init__.py     # registry: build_pipeline(mode)
│   ├── download/       # Downloader stage (muapi, ytdlp)
│   ├── transcribe/     # Transcriber stage (muapi, whisper)
│   ├── highlights/     # Highlight engine (LLM-agnostic)
│   ├── llm/            # Highlight LLM backends (muapi, local)
│   └── render/         # Renderer stage (muapi, local: face/subs/encode)
└── clients/            # thin service clients (MuAPI)
```

## Adding or swapping a stage

1. Implement a callable that matches the relevant `Protocol` in `protocols.py`.
2. Register it in `stages/__init__.py`, or pass it as an override:
   `build_pipeline("local", render=my_renderer)`.
3. No other stage or the orchestrator should need to change.

## Code style

- Format with **black** and lint with **ruff** (config in `pyproject.toml`):
  ```bash
  black . && ruff check .
  ```
- Keep functions typed and add a short docstring for anything non-obvious.
- Match the surrounding style; keep the public API (`generate_shorts`) stable.

## Pull requests

1. Create a feature branch from your copy of the repository.
2. Keep changes focused; describe what and why.
3. Note any new dependencies and whether they belong to a specific mode.
4. Make sure `python -c "import shorts_generator"` still succeeds.

## Reporting bugs

Open an issue with your OS, Python version, the exact command, the mode
(`api`/`local`), and the full error output. For local mode, include your GPU and
whether `ffmpeg` is on your PATH.
