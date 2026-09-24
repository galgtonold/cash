"""Which file tracker records the reads made on this task or thread.

Everything that reports a read looks the tracker up here at call time; a
`FileAccessTracker` sets it for the duration of its block, and `untracked`
clears it for cash's own I/O.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Optional

from cash.effect_observer import active_observer as _active_effect_observer

if TYPE_CHECKING:
    from cash.tracking.file_tracker import FileAccessTracker

__all__ = ["active_tracker", "untracked"]

# Active tracker for the current asyncio task / thread.
# Read by the patched I/O dispatchers to decide whether to record the
# access. Isolated per task/thread by contextvars semantics.
active_tracker: contextvars.ContextVar[Optional["FileAccessTracker"]] = contextvars.ContextVar(
    "active_tracker", default=None
)


class untracked:
    """Run cash's OWN I/O without it becoming anyone's dependency.

    A nested cached call does its bookkeeping -- resolving configuration,
    reading ``pyproject.toml``, walking up for project markers -- while the
    OUTER call's tracker is live, so those reads were recorded as the outer
    entry's file dependencies: bump the project's version and every cached
    function that calls another one recomputed. The storage-path filters
    (``is_cash_internal``) cannot help, because a config file is not storage.
    A class rather than ``contextlib.contextmanager`` so it costs one
    ContextVar swap, not a generator.
    """

    __slots__ = ("_token", "_observer_token")

    def __enter__(self) -> None:
        self._token = active_tracker.set(None)
        # Nor anyone's observed side effect: cash writing its own bookkeeping
        # file inside a nested call is not the OUTER function writing a file.
        self._observer_token = _active_effect_observer.set(None)

    def __exit__(self, *exc: Any) -> None:
        _active_effect_observer.reset(self._observer_token)
        active_tracker.reset(self._token)
