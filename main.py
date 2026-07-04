"""Convenience entry point: ``python main.py <url> [options]``.

Equivalent to ``python -m shorts_generator``. See ``--help`` for all options.
"""
import sys

from shorts_generator.cli import main

if __name__ == "__main__":
    sys.exit(main())
