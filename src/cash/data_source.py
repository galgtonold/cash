"""Abstract data source interface for cache-aware data loading."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .diagnostics import warn_diagnostic
from .exceptions import CashCacheIneffectiveWarning
from .tracking.file_dep_snapshot import file_content_hash

__all__ = ["DataSource", "FileDataSource"]

_warned_bool_token_sources: set[str] = set()


class DataSource(ABC):
    """Abstract base class for a tracked cache dependency.

    Cash folds a *token* representing the source's current state into the cache
    key (see :meth:`state_token`). The cached entry invalidates when that token
    changes - so the token must be a **value that changes with the data** (an
    mtime, a version string, a content digest), NOT a boolean.
    """

    @abstractmethod
    def get_id(self) -> str:
        """Unique identifier for the data source."""

    @abstractmethod
    def state_token(self) -> Any:
        """The value folded into the cache key for this source.

        Return something that *changes* when the underlying data changes (a
        version, a digest, an mtime). A plain ``bool`` cannot track changes and
        the cache would never invalidate - Cash warns if it sees one.
        """


def state_token_of(source: DataSource) -> str:
    """*source*'s token as it goes into the key, warning once per source type
    when the token is a ``bool``, which cannot track a change."""
    # Local: import cycle data_source -> dependency_state -> data_source.
    from .dependency_state import EXPLAINING

    token = source.state_token()
    if isinstance(token, bool) and not EXPLAINING.get():
        name = type(source).__qualname__
        if name not in _warned_bool_token_sources:
            _warned_bool_token_sources.add(name)

            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-BOOL-STATE-TOKEN",
                f"{name}.state_token() returned a bool, which cannot track "
                f"changes, so the cache will NOT invalidate when the source "
                f"changes.",
                "return something that moves with the data -- a version, a digest, an mtime -- from state_token().",
            )
    return str(token)


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
