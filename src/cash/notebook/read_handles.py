"""An open file read by name, as a value a lineage can name.

``f = open(path)`` binds a new handle on every run, and a handle has no
content to hash, so its lineage was a fresh one each run. Everything read
from it then missed too: ``lines = f.readlines()`` re-ran (it always does: it
moves the handle), and so did every cell built on ``lines``, though the file
had not changed. ``with open(path) as f`` did not have the problem, which
made the two spellings of the same read behave differently (student 2 of the
JuNE notebooks: a 25 s parse re-ran after every restart).

A handle opened read-only on a regular file is named here by what decides
what reading it gives: the file (its real path and its stat), how it decodes,
and where it stands. The statement that bound it reads the file, so the
file's content is already in its lineage; the stat here covers a handle bound
by a call that opened the file out of sight.

Nothing restores a handle: the statements that read from it, or look at it,
run every time (see ``reads_open_handle``), and what they bind carries a
digest of its value. Only a value built from those is ever served from the
cache, and only when it would come out the same.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Any

__all__ = ["read_handle_digest", "is_read_handle"]

_FILE_TYPES = (io.TextIOWrapper, io.BufferedReader, io.FileIO)


def is_read_handle(value: Any) -> bool:
    """An open handle on a regular file, opened for reading only."""
    if type(value) not in _FILE_TYPES:
        return False
    try:
        if value.closed or not value.readable() or value.writable():
            return False
    except (ValueError, OSError):
        return False
    name = getattr(value, "name", None)
    return isinstance(name, str) and os.path.isfile(name)


def read_handle_digest(value: Any) -> str | None:
    """A digest of what reading *value* from here on gives, or ``None`` when
    it is not a read handle on a regular file."""
    if not is_read_handle(value):
        return None
    try:
        position = value.tell()
        stat = os.stat(value.name)
    except (OSError, ValueError):
        return None
    text = isinstance(value, io.TextIOWrapper)
    parts = (
        type(value).__name__,
        os.path.realpath(value.name),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        str(stat.st_ino),
        str(getattr(value, "mode", "")),
        str(value.encoding if text else ""),
        str(value.errors if text else ""),
        repr(getattr(value, "_CHUNK_SIZE", "")),
        str(position),
    )
    return "read-handle:" + hashlib.sha256("\0".join(parts).encode("utf-8", "surrogateescape")).hexdigest()
