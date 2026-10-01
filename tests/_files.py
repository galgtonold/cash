"""Edit a file so that every reader can tell it changed.

A file's mtime comes from a clock that may not tick between two writes
(Windows stamps files from a clock of about 16 ms), so an edit that keeps
the size can look like no edit at all to a check by mtime and size.
"""

from __future__ import annotations

import os
from pathlib import Path


def rewrite(path: str | Path, text: str) -> None:
    """Write *text* to *path*; its mtime ends up later than before the write."""
    path = Path(path)
    before = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text, encoding="utf-8")
    if before is not None and path.stat().st_mtime_ns <= before:
        os.utime(path, ns=(before + 1_000_000, before + 1_000_000))
