"""Abstract data source interface for cache-aware data loading."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

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
    def has_changed(self) -> Any:
        """Return a value representing the source's current state.

        Despite the name, this is the **state token** folded into the cache key,
        not a yes/no flag: return something that *changes* when the underlying
        data changes (a version, a digest, an mtime). Returning a plain ``bool``
        cannot track changes and the cache will never invalidate - Cash warns if
        it sees one. ``FileDataSource`` exposes ``_get_mtime``, so the base
        :meth:`state_token` uses the file mtime and its ``has_changed`` (a real
        bool) is never used as a token.
        """

    @abstractmethod
    def update_state(self) -> None:
        """Update the internal state to current (mark as seen)."""

    def state_token(self) -> Any:
        """The value folded into the cache key for this source.

        Default: the file mtime (when the source exposes ``_get_mtime``),
        otherwise :meth:`has_changed`. Override this to track anything else.
        """
        getter = getattr(self, "_get_mtime", None)
        if getter is not None:
            return getter()
        token = self.has_changed()
        if isinstance(token, bool):
            name = type(self).__qualname__
            if name not in _warned_bool_token_sources:
                _warned_bool_token_sources.add(name)
                from .diagnostics import warn_diagnostic
                from .exceptions import CashCacheIneffectiveWarning
                warn_diagnostic(
                    CashCacheIneffectiveWarning,
                    "KEY-BOOL-STATE-TOKEN",
                    f"{name}.has_changed() returned a bool, which cannot track "
                    f"changes, so the cache will NOT invalidate when the source "
                    f"changes.",
                    "return something that moves with the data -- a version, a "
                    "digest, an mtime -- from has_changed(), or override "
                    "state_token() to return that token.",
                )
        return token

class FileDataSource(DataSource):
    """Tracks a file for changes using modification time."""

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        self._last_mtime = self._get_mtime()

    def _get_mtime(self) -> float:
        try:
            return os.path.getmtime(self.filepath)
        except OSError:
            return 0.0

    def get_id(self) -> str:
        return f"file:{self.filepath}"

    def has_changed(self) -> bool:
        current_mtime = self._get_mtime()
        return current_mtime != self._last_mtime

    def update_state(self) -> None:
        self._last_mtime = self._get_mtime()
