"""A local file as a tracked cache dependency."""

from __future__ import annotations

import glob
import hashlib
import os

from .data_source import DataSource
from .tracking.file_dep_snapshot import file_content_hash

__all__ = ["FileDataSource"]


class FileDataSource(DataSource):
    """Tracks a local file by its content.

    The token is the file's content digest, the fingerprint a file the function
    reads itself is checked by (:func:`cash.tracking.file_dep_snapshot.file_content_hash`):
    a ``touch`` that leaves the bytes alone keeps the entry, and an edit that
    leaves the mtime where it was does not. The digest is memoized on the
    file's stat, so an unchanged file costs one ``stat``.

    A directory is every file under it by content and every directory in it
    by its listing; a glob pattern is each match so, and the list of
    matches -- as ``file_depends_on=`` checks them. The directory's own
    mtime does not move when a file in it is edited, and a pattern names no
    file that exists, so either alone never moved.
    """

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)

    def get_id(self) -> str:
        return f"file:{self.filepath}"

    def state_token(self) -> str:
        if glob.has_magic(self.filepath):
            matches = sorted(glob.glob(self.filepath, recursive=True))
            return _digest(f"{match}\x00{_path_token(match)}" for match in matches)
        return _path_token(self.filepath)


def _path_token(path: str) -> str:
    """The token of one file or directory (see `FileDataSource`)."""
    if os.path.isdir(path):
        parts = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            rel = os.path.relpath(root, path)
            parts.append(f"dir:{rel}:{sorted(dirs)!r}")
            parts.extend(
                f"{os.path.join(rel, name)}\x00{_file_token(os.path.join(root, name))}" for name in sorted(files)
            )
        return _digest(parts)
    return _file_token(path)


def _file_token(path: str) -> str:
    digest = file_content_hash(path)
    if digest is not None:
        return digest
    # A file that exists but cannot be read still moves with its mtime.
    try:
        return f"unreadable:{os.path.getmtime(path)}"
    except OSError:
        return "absent"


def _digest(parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "surrogateescape"))
        h.update(b"\x01")
    return "tree:" + h.hexdigest()
