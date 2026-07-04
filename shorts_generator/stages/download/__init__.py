"""Downloader stage — resolve a source reference to readable media.

* :mod:`.muapi`  — API mode: hand the URL to MuAPI, get back a hosted mp4 URL.
* :mod:`.ytdlp`  — local mode: download with yt-dlp, or pass through a local path.
"""
from .muapi import download_youtube
from .ytdlp import download_youtube_local

__all__ = ["download_youtube", "download_youtube_local"]
