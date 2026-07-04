"""ShortsForge — turn long-form video into ranked, vertical short-form clips.

Public API::

    from shorts_generator import generate_shorts

    result = generate_shorts("https://youtu.be/VIDEO_ID", num_clips=5, mode="local")
    for short in result["shorts"]:
        print(short["score"], short["title"], short["clip_url"])
"""
from .pipeline import Pipeline, generate_shorts

__version__ = "0.1.0"
__all__ = ["Pipeline", "generate_shorts", "__version__"]
