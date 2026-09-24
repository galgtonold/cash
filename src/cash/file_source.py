"""A local file as a tracked cache dependency."""

from __future__ import annotations

import os

from .data_source import DataSource
from .tracking.file_dep_snapshot import file_content_hash

__all__ = ["FileDataSource"]


class FileDataSource(DataSource):
    """Tracks a local file by its content.

    The token is the file's content digest, the fingerprint a file the function
    reads itself is checked by (:func:`cash.tracking.file_dep_snapshot.file_content_hash`):
    a ``touch`` that leaves the bytes alone keeps the entry, and an edit that
    leaves the mtime where it was does not. Files above ``full_hash_max_bytes``
    are hashed by sampling, and the digest is memoized on the file's stat, so
    an unchanged file costs one ``stat``.
    """

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)

    def get_id(self) -> str:
        return f"file:{self.filepath}"

    def state_token(self) -> str:
        digest = file_content_hash(self.filepath)
        if digest is not None:
            return digest
        # A file that exists but cannot be read still moves with its mtime.
        try:
            return f"unreadable:{os.path.getmtime(self.filepath)}"
        except OSError:
            return "absent"
