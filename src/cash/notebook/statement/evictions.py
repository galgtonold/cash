"""A statement recomputed because the disk cap had evicted its value.

After a restart, a statement whose entry the cap removed misses like any
other, and its badge said nothing about why. The disk tier notes what its cap
evicts (``cash.backends.eviction_log``), so a miss can ask: the badge then
names the eviction, and when the recompute took long enough to matter,
``CACHE-EVICTED-RECOMPUTE`` warns, once per statement.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ...backends import adaptive_caps, budget_notices
from ...backends.eviction_log import EvictionNote
from ...diagnostics import warn_diagnostic
from ...exceptions import CashCacheIneffectiveWarning

logger = logging.getLogger(__name__)

__all__ = ["EVICTED_MISS_REASON", "EvictedRecomputes"]

#: The badge's miss reason for such a statement.
EVICTED_MISS_REASON = "evicted to make room when the disk cache reached its size cap"


def _first_line(code: str) -> str:
    """The line a reader finds the statement by: its first that is not a
    comment (a loop body's stored code starts with its context marker)."""
    lines = [ln for ln in code.splitlines() if ln.strip()]
    first = next((ln for ln in lines if not ln.lstrip().startswith("#")), lines[0] if lines else "")
    return first.strip()[:80]


class EvictedRecomputes:
    """Attributes misses to the cap's evictions, and warns once per statement."""

    def __init__(self) -> None:
        #: Statements already warned about, by a digest of their code.
        self._warned: set[str] = set()

    def attribute(self, metrics: dict, backend: Any, cache_key: str, code: str, seconds: float) -> bool:
        """Did the cap evict *cache_key*'s entry? Then say so on *metrics*,
        warn when recomputing it took *seconds* worth mentioning, and return
        True. False, touching nothing, otherwise."""
        if not cache_key:
            return False
        try:
            note = backend.eviction_note(cache_key)
        except Exception:  # a miss reason is a diagnostic
            logger.debug("Could not ask the backend about evictions", exc_info=True)
            return False
        if not isinstance(note, EvictionNote):
            return False
        metrics["miss_reason"] = EVICTED_MISS_REASON
        if seconds >= budget_notices.EVICTED_RECOMPUTE_WARN_SECONDS:
            self._warn(backend, code, seconds)
        return True

    def _warn(self, backend: Any, code: str, seconds: float) -> None:
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if digest in self._warned:
            return
        self._warned.add(digest)
        try:
            budget = backend.disk_budget()
        except Exception:  # noqa: BLE001 - the cap only improves the message
            budget = None
        if not isinstance(budget, budget_notices.DiskBudget):
            budget = None
        free = adaptive_caps.free_bytes_on_volume(budget.cache_dir) if budget is not None else 0
        message, fix = budget_notices.evicted_recompute_warning(
            f"the value of `{_first_line(code)}`", seconds, budget, free
        )
        warn_diagnostic(CashCacheIneffectiveWarning, "CACHE-EVICTED-RECOMPUTE", message, fix)
