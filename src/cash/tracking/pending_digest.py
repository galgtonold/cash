"""File digests taken on a thread of their own while a cached body runs.

Kept apart from `file_tracker` so the ``open`` event handler
(`cash.tracking.read_events`) can wait for one without importing the
tracker: it imports nothing of cash's but the path helpers.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from cash._paths import normalize_path

logger = logging.getLogger(__name__)

__all__ = ["PendingDigest", "settle_before_write"]

#: The digests being taken, by resolved path: what a write-mode open waits for.
PENDING: dict[str, PendingDigest] = {}


class PendingDigest:
    """A file's content hash, taken on a thread of its own.

    Started when the body opens the file, so it overlaps the body's own
    read and work; ``result`` waits for it. Until it is done, a write-mode
    open of the same file anywhere in this process first waits for it
    (`settle_before_write`): the digest holds the file as it was before this
    process wrote to it -- an ``np.memmap`` writes without moving a
    timestamp, so the mid-call check could not tell. A write by another
    process that moves no timestamp while the body runs is not seen
    (known-limitations: an edit that keeps size and timestamps).
    """

    __slots__ = ("_digest", "_path", "_thread")

    def __init__(self, path: str, size: int, compute: Callable[[str, int], str | None]) -> None:
        self._path = path
        self._digest: str | None = None
        # The caller's context: the digest is cash's own read, untracked as
        # it would be on the caller's thread.
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(self._run, size, compute), name="cash-file-digest", daemon=True
        )
        PENDING[path] = self
        try:
            self._thread.start()
        except RuntimeError:  # no new threads (interpreter shutting down): here, then
            self._run(size, compute)

    def _run(self, size: int, compute: Callable[[str, int], str | None]) -> None:
        try:
            self._digest = compute(self._path, size)
        except Exception:  # noqa: BLE001 - an unreadable file: no digest, as on the caller's thread
            logger.debug("[TRACKER] Could not hash %r", self._path, exc_info=True)
        finally:
            if PENDING.get(self._path) is self:
                del PENDING[self._path]

    def result(self) -> str | None:
        """The digest, once taken; None when the file could not be read."""
        if self._thread.ident is not None:
            self._thread.join()
        return self._digest


def settle_before_write(path: Any) -> None:
    """Wait for a digest of *path* still being taken (`PendingDigest`) before
    it is opened to write. Called from the ``open`` event; one dict check
    while no digest is pending."""
    if not PENDING:
        return
    try:
        raw = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        resolved = os.path.normcase(normalize_path(os.path.realpath(raw)))
    except (TypeError, ValueError, OSError):
        # Cannot say which file: wait for them all.
        resolved = None
    for pending_path, pending in list(PENDING.items()):
        if resolved is None or os.path.normcase(pending_path) == resolved:
            pending.result()
