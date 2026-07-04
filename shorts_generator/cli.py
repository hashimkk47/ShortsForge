"""Command-line interface for ShortsForge.

Usage::

    python -m shorts_generator "https://youtu.be/VIDEO_ID" --mode local --num-clips 3
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, generate_shorts


def _configure_stdio() -> None:
    """Force UTF-8 stdio so arrows/emoji don't crash on Windows 'charmap'."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shorts_generator",
        description="ShortsForge — AI long-form-to-shorts generator",
    )
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (MuAPI, hosted) or local (yt-dlp + faster-whisper + LLM + ffmpeg)",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360/480/720/1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force transcription language code, e.g. 'en' (default: auto)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _print_summary(result: dict, fallback_mode: str) -> None:
    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', fallback_mode)}")
    print(f"Source video:  {result['source_video_url']}")
    print(f"Highlights:    {len(result['highlights'])} candidates -> kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, short in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={short.get('score')}  "
              f"{short.get('start_time'):.1f}s -> {short.get('end_time'):.1f}s")
        print(f"     title:  {short.get('title')}")
        print(f"     hook:   {short.get('hook_sentence')}")
        if short.get("clip_url"):
            print(f"     clip:   {short['clip_url']}")
        else:
            print(f"     clip:   FAILED ({short.get('error')})")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    try:
        result = generate_shorts(
            source=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, args.mode)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
