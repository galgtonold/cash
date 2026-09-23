"""Noticing that a disk tier was cleared by another process.

`cash clear` on a live service clears the disk and nothing else, so a process
holding results in its RAM tier would go on serving the pre-clear answers. The
disk tier's generation token (`CacheBackend.generation_token`) moves on a
clear; `ClearWatcher` compares it at most once a second -- one ``stat``, not one
per hit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ._base import CacheBackend

logger = logging.getLogger(__name__)

__all__ = ["ClearWatcher"]

_UNSEEN = object()


class ClearWatcher:
    """Tells a tiered backend when its disk tier was cleared under it."""

    #: How often, in seconds, the generation token is compared.
    CHECK_EVERY = 1.0

    def __init__(self) -> None:
        self._generation: Any = _UNSEEN
        self._checked_at = 0.0
        self._stamp_writes_seen = 0

    def cleared(self, disk: CacheBackend) -> bool:
        """Was *disk* cleared since the last look? Throttled to `CHECK_EVERY`."""
        now = time.monotonic()
        if now - self._checked_at < self.CHECK_EVERY:
            return False
        self._checked_at = now
        try:
            token = disk.generation_token()
        except Exception:  # noqa: BLE001 - a check must never break a read
            return False
        # From no stamp to one is a directory being created, not cleared --
        # unless THIS process wrote that stamp since the last look: a process
        # that started cold, wrote the first stamp and then had the directory
        # cleared would otherwise take the change for "still new".
        known = self._generation
        writes = disk.stamp_writes
        # This process stamping AGAIN is evidence on its own: it stamps only a
        # directory it finds unstamped, so a second stamp means the first was
        # removed in between, possibly inside one throttle window.
        restamped = writes > self._stamp_writes_seen and (
            self._stamp_writes_seen >= 1 or writes - self._stamp_writes_seen >= 2
        )
        if known in (_UNSEEN, None) and writes != self._stamp_writes_seen:
            known = disk.written_stamp
        self._stamp_writes_seen = writes
        self._generation = token
        return restamped or (known not in (_UNSEEN, None) and token != known)
