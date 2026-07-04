<div align="center">

# 🔥 ShortsForge

**Turn long-form video into ranked, vertical, subtitle-ready short-form clips — on your own machine.**

Drop in a YouTube link (or a local file) and get back the most viral-worthy moments,
auto-cropped to 9:16, face-tracked, and captioned — ready for TikTok, Reels, and Shorts.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Local first](https://img.shields.io/badge/inference-local%20%7C%20offline-8A2BE2?style=flat-square)](#-architecture)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-000000?style=flat-square)](#-whisper-setup)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-ffffff?style=flat-square&logo=ollama&logoColor=black)](#-ollama-setup)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20%2F%20NVENC-76B900?style=flat-square&logo=nvidia&logoColor=white)](#gpu-acceleration)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

</div>

---

## 📖 Overview

**ShortsForge** is an open-source pipeline that finds the best moments in a
long video and renders them as polished vertical clips. It runs **fully on your
machine** in local mode — local transcription (Whisper), local LLM ranking
(Ollama), local face tracking (MediaPipe), and local encoding (FFmpeg/NVENC) —
with **no per-clip credits, no watermarks, and no uploads of your footage**. A
hosted **API mode** is also available for a zero-setup path.

The whole system is built as a chain of **interchangeable stages**, so any part
— the downloader, the transcriber, the highlight engine, the renderer — can be
swapped without touching the rest.

## ✨ Features

- **🎬 Long-form in, vertical out** — any YouTube URL or local file → N ranked 9:16 clips.
- **🖥️ Local & offline** — transcription, LLM ranking, face tracking, and encoding all run on-device.
- **🔀 Two modes** — `local` (self-hosted, private) or `api` (hosted MuAPI, zero setup).
- **🧠 Virality-aware ranking** — clips scored on hooks, emotional peaks, opinion bombs, revelations, conflict, quotables, story peaks, and practical value.
- **📈 Score + hook + reason** — every clip ships with a 0–100 viral score, an opening hook line, and a one-line rationale.
- **🎤 Whisper transcription** — `faster-whisper` (CPU or CUDA), up to **large-v3**, with SRT caching.
- **🤖 Pluggable highlight LLM** — Ollama (offline), OpenAI, or Gemini via a single env var.
- **🎯 Speaker-aware reframing** — MediaPipe Tasks BlazeFace + a velocity-smoothed dominant-face tracker keep the talker centred.
- **💬 Burned-in subtitles** — styled captions auto-timed to each clip.
- **⚡ GPU acceleration** — automatic CUDA decode + NVENC encode, with a seamless CPU fallback.
- **🧩 Long-video aware** — videos over 30 min are auto-chunked with overlap so nothing is missed.
- **♻️ Smart dedupe** — overlapping highlights collapse to the highest score.
- **🧰 CLI + Python library** — run from the shell or `import generate_shorts(...)`.
- **📦 JSON output** — `--output-json` dumps the transcript, every candidate, and final clip paths.

## 🧱 Architecture

ShortsForge is a linear pipeline of independently replaceable stages. Each stage
has a contract (a `Protocol` in [`shorts_generator/protocols.py`](shorts_generator/protocols.py))
and is wired per mode in [`shorts_generator/stages/`](shorts_generator/stages/).

```
        ┌────────────┐   ┌──────────────┐   ┌────────────────┐   ┌────────────────┐
 source │ Downloader │──▶│ Transcriber  │──▶│ Highlight      │──▶│ Renderer       │──▶ shorts
  URL / │            │   │              │   │ Engine (LLM)   │   │  ├─ Subtitles   │
  file  └────────────┘   └──────────────┘   └────────────────┘   │  ├─ Face Track  │
                                                                  │  ├─ Effects*    │
                                                                  │  ├─ Encoder     │
                                                                  │  └─ Uploader*   │
                                                                  └────────────────┘
                                                        (* planned — see the roadmap)
```

| Stage | `--mode local` | `--mode api` |
|---|---|---|
| **Downloader** | `yt-dlp` (or a local file path) | MuAPI `/youtube-download` |
| **Transcriber** | `faster-whisper` (CPU/CUDA) | MuAPI `/openai-whisper` |
| **Highlight Engine** | Ollama / OpenAI / Gemini | MuAPI `gpt-5-mini` |
| **Renderer** | FFmpeg cut → MediaPipe face-track → subtitle burn → NVENC/x264 encode | MuAPI `/autocrop` |

```
shorts_generator/
├── protocols.py        # stage contracts (interchangeable-module seams)
├── pipeline.py         # mode-agnostic orchestrator
├── types.py            # Transcript / Highlight / Short / PipelineResult
├── cli.py              # command-line interface
├── config.py           # env-driven settings
├── clients/muapi.py    # MuAPI submit + poll client
└── stages/
    ├── __init__.py     # registry → build_pipeline(mode)
    ├── download/       # muapi.py · ytdlp.py
    ├── transcribe/     # muapi.py · whisper.py
    ├── highlights/     # engine.py  (LLM-agnostic ranking)
    ├── llm/            # muapi.py · local.py (openai/gemini/ollama)
    └── render/         # muapi.py · local/{encoding,face_tracking,subtitles}.py
```

## 🚀 Quick start

```bash
git clone https://github.com/hashimkk47/ShortsForge.git
cd ShortsForge
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt                 # local mode deps
cp .env.example .env                                  # then edit .env

# Fully local: yt-dlp + Whisper + Ollama + FFmpeg
python main.py "https://youtu.be/VIDEO_ID" --mode local --num-clips 3
```

Rendered clips land in `./output/short_01.mp4`, `short_02.mp4`, …

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **Local mode:** [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH`, plus a highlight LLM ([Ollama](#-ollama-setup) recommended).
- **API mode:** a [MuAPI](https://muapi.ai) key (no local deps required).
- **Optional:** an NVIDIA GPU + drivers for CUDA transcription and NVENC encoding.

### 🪟 Windows setup

```powershell
git clone https://github.com/hashimkk47/ShortsForge.git
cd ShortsForge
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt
copy .env.example .env
```

Install FFmpeg and confirm it's on your `PATH`:

```powershell
winget install Gyan.FFmpeg      # or: choco install ffmpeg
ffmpeg -version
```

### 🐧 Linux setup

```bash
git clone https://github.com/hashimkk47/ShortsForge.git
cd ShortsForge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
cp .env.example .env

sudo apt update && sudo apt install -y ffmpeg   # Debian/Ubuntu
ffmpeg -version
```

### 🤖 Ollama setup

Ollama runs the highlight-ranking LLM **fully offline**.

1. Install Ollama from [ollama.com](https://ollama.com/download).
2. Pull a model and make sure the server is running:
   ```bash
   ollama pull qwen3:8b
   ollama serve            # usually already running as a service
   ```
3. Point ShortsForge at it in `.env`:
   ```ini
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=qwen3:8b
   OLLAMA_HOST=http://localhost:11434
   ```

Prefer a hosted LLM instead? Set `LLM_PROVIDER=openai` (`OPENAI_API_KEY`) or
`LLM_PROVIDER=gemini` (`GEMINI_API_KEY`).

### 🎤 Whisper setup

Local transcription uses [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
and is configured entirely through `.env`:

```ini
LOCAL_WHISPER_MODEL=base       # tiny · base · small · medium · large-v3
LOCAL_WHISPER_DEVICE=auto      # auto · cpu · cuda
```

- **CPU:** `base` or `small` are fast and accurate enough for most talking-head content.
- **GPU:** set `large-v3` for best accuracy (needs a capable NVIDIA GPU).
- Transcripts are cached as a sidecar `.srt` next to the video and reused on re-runs.

<a id="gpu-acceleration"></a>
#### GPU acceleration

With `LOCAL_WHISPER_DEVICE=auto`, ShortsForge detects a working CUDA device and
uses it for transcription (and NVENC for encoding), falling back to CPU/x264
automatically if CUDA/NVENC aren't available. For CUDA Whisper, install a CUDA
build of `torch` (see the note in `requirements-local.txt`).

## 💡 Usage examples

**Local mode (private, offline except an optional hosted LLM):**
```bash
python main.py "https://youtu.be/VIDEO_ID" --mode local --num-clips 5
```

**Straight from a local file — skip YouTube entirely:**
```bash
python main.py "/videos/podcast.mp4" --mode local
python main.py "file:///videos/podcast.mp4" --mode local
```

**API mode (zero setup):**
```bash
python main.py "https://youtu.be/VIDEO_ID" --mode api --output-json result.json
```

**As a Python library:**
```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "/videos/podcast.mp4",
    num_clips=5,
    aspect_ratio="9:16",
    mode="local",
)
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

**Swap a stage without touching the pipeline:**
```python
from shorts_generator.stages import build_pipeline

def my_uploader_renderer(source, highlights, aspect_ratio="9:16"):
    ...  # your own renderer/uploader implementation

pipe = build_pipeline("local", render=my_uploader_renderer)
result = pipe.run("https://youtu.be/VIDEO_ID", num_clips=3)
```

**Batch a list of URLs:**
```bash
xargs -a urls.txt -I{} python main.py "{}" --mode local
```

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `api` | `local` (on-device) or `api` (hosted MuAPI) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | `9:16`, `1:1`, `4:5`, or `16:9` |
| `--format` | `720` | Source download resolution: `360`/`480`/`720`/`1080` |
| `--language` | auto | Force transcription language code (e.g. `en`) |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |

### Output

```
========================================================================
Mode:          local
Highlights:    7 candidates -> kept top 3
========================================================================

#1  score=92  124.3s -> 187.6s
     title:  The one mistake that cost me $50K
     hook:   "Nobody talks about this, but it killed my first startup..."
     clip:   output/short_01.mp4
```

`--output-json` additionally writes `{mode, source_video_url, transcript,
highlights[], shorts[]}` for downstream automation.

## 🗺️ Roadmap

Planned work, roughly in priority order:

- [ ] **TikTok / CapCut-style animated subtitles**
- [ ] **Karaoke word-by-word highlighting**
- [ ] **ASS subtitle rendering** (rich styling beyond SRT)
- [ ] **Active speaker detection** (audio-driven face selection)
- [ ] **Multi-face tracking** (split/switch framing for conversations)
- [ ] **Better clip ranking** (retrieval + reranking, per-platform tuning)
- [ ] **B-roll insertion**
- [ ] **Sound effects & transitions**
- [ ] **Motion graphics** (lower-thirds, progress bars, zoom punches)
- [ ] **Thumbnail generation**
- [ ] **Upload automation** (the planned `Uploader` stage — TikTok/YouTube/Reels)
- [ ] **Plugin system** (register third-party stage implementations)

## ❓ FAQ

**Is my video uploaded anywhere in local mode?**
No. In `local` mode, download, transcription, ranking, face tracking, and
encoding all run on your machine. The only optional network call is a hosted LLM
if you choose `LLM_PROVIDER=openai`/`gemini` instead of Ollama.

**Do I need a GPU?**
No — everything works on CPU. A GPU makes transcription (CUDA) and encoding
(NVENC) much faster; ShortsForge auto-detects and falls back gracefully.

**"Whisper produced no segments."**
The clip may have no detectable speech, or a hard-to-detect language. Pass
`--language en` (or the correct ISO-639-1 code) to skip auto-detection.

**"ffmpeg was not found on PATH."**
Install FFmpeg (see the OS setup sections) and confirm `ffmpeg -version` works in
the same shell you run ShortsForge from.

**Which Whisper model should I use?**
`base`/`small` on CPU for speed, `large-v3` on GPU for accuracy. Set it with
`LOCAL_WHISPER_MODEL`.

**Can I change what counts as a "highlight"?**
Yes — edit the virality framework and prompts in
[`shorts_generator/stages/highlights/engine.py`](shorts_generator/stages/highlights/engine.py).

**Which aspect ratios are supported?**
`9:16`, `1:1`, `4:5`, and `16:9`.

## 🤝 Contributing

Contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)**. The
stage-based design makes it easy to add a new downloader, transcriber, highlight
LLM, or renderer without disturbing the rest of the pipeline.

## 📄 License

[MIT](LICENSE)

> ⚠️ ShortsForge is under active development. Interfaces, stage APIs, and rendering features may change before the first stable release.
