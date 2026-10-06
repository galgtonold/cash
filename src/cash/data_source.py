"""Abstract data source interface for cache-aware data loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._active import EXPLAINING
from .diagnostics import warn_diagnostic
from .exceptions import CashCacheIneffectiveWarning

__all__ = ["DataSource"]

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
    return _token_text(token)


def _token_text(token: Any) -> str:
    """*token* as text that changes whenever it does. A frame, array or
    table -- or a collection holding one -- is its content digest: its
    printed form elides the middle of a long array and rounds floats to 8
    digits, so a change there kept the key."""
    if isinstance(token, str):
        return token
    if not isinstance(token, (int, float, bytes, type(None))):
        from .value_hash import is_bulky, compute_hash

        if is_bulky(token) or (
            isinstance(token, (list, tuple, dict, set, frozenset))
            and any(is_bulky(v) for v in (token.values() if isinstance(token, dict) else token))
        ):
            return f"{type(token).__name__}:{compute_hash(token)}"
    return str(token)
