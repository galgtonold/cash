"""What cash says about the disk cache's size cap: how big, and when it bites.

The disk tier's cap is sized to the machine unless the user sets one, and
eviction at the cap is otherwise silent -- a cache grew to many GiB, or lost
entries, with nothing on screen to say why. So two quiet notices, each at most
once per process per cache folder:

* **the cap**, when caching starts: the folder, how big the cache may grow,
  where that number comes from and how to change it (`describe_budget`);
* **the first eviction**: how much the cap made cash remove (`eviction_text`).

And one warning, ``CACHE-EVICTED-RECOMPUTE``, when an entry the cap evicted is
needed again and recomputing it took long enough to matter
(`evicted_recompute_warning`).

Quiet means not a warning: nothing is wrong. They are INFO records on the
``cash.storage`` logger, which ``verbose=True``, ``debug=True`` or an
application logging at INFO shows, and the notebook prints them into the cell
(``%cash_on`` for the cap, the end of the cell for the eviction).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import NamedTuple

from ..config import format_size, human_bytes
from ..effectiveness import CUMULATIVE_WASTE_SECONDS

__all__ = [
    "EVICTED_RECOMPUTE_WARN_SECONDS",
    "DiskBudget",
    "announce_budget",
    "cap_text",
    "claim_budget_notice",
    "claim_eviction_notice",
    "describe_budget",
    "evicted_recompute_warning",
    "eviction_text",
    "storage_logger",
]

#: Where the notices are logged.
storage_logger = logging.getLogger("cash.storage")


class DiskBudget(NamedTuple):
    """The disk tier's cap as a user should read it."""

    cache_dir: str
    cap: int
    #: Where an adaptive cap comes from (`adaptive_caps.disk_cap_reason`);
    #: None when the user set it.
    why: str | None


def cap_text(cap: int, adaptive: bool) -> str:
    """A cap the user set reads back as they wrote it (``1 MB``, not
    ``976.6 KiB``); one cash sized, in binary units."""
    return human_bytes(cap) if adaptive else format_size(cap)


def describe_budget(budget: DiskBudget) -> str:
    """``caching in <dir>, up to 26.0 GiB (<why>; set max_cache_size to change it)``."""
    size = cap_text(budget.cap, budget.why is not None)
    if budget.why is None:
        return f"caching in {budget.cache_dir}, up to {size} (set by max_cache_size)"
    return f"caching in {budget.cache_dir}, up to {size} ({budget.why}; set max_cache_size to change it)"


def eviction_text(cache_dir: str, cap: str, freed: int, count: int) -> str:
    """The first-eviction notice, *cap* already formatted (`cap_text`). Keep
    it and docs/how-it-works/storage.md saying the same."""
    entries = "1 entry" if count == 1 else f"{count} entries"
    return (
        f"the cache in {cache_dir} reached its {cap} cap, so cash removed "
        f"{entries} ({human_bytes(freed)}), the ones worth least per byte: least compute "
        f"time saved for the space they take. Later removals are not reported; set "
        f"max_cache_size to change the cap."
    )


#: A recompute of an evicted result shorter than this is not worth a warning.
#: The same floor the effectiveness ledger uses for "worth interrupting
#: anyone": the user pays the seconds either way, and two of them lost to a
#: too-small cap is when knowing pays for the noise. The persistence floor
#: (0.1 s) would warn on nearly every entry a busy cache evicts.
EVICTED_RECOMPUTE_WARN_SECONDS = CUMULATIVE_WASTE_SECONDS


def evicted_recompute_warning(what: str, seconds: float, budget: DiskBudget | None, free: int) -> tuple[str, str]:
    """``(what happened, fix)`` for CACHE-EVICTED-RECOMPUTE, both paths.

    *what* names the result (a function's call, a statement's value). Keep it
    and docs/warnings.md#cache-evicted-recompute saying the same thing.
    """
    cap = cap_text(budget.cap, budget.why is not None) if budget is not None else None
    at = f"its {cap} cap" if cap else "its size cap"
    message = (
        f"{what} had been evicted from the disk cache to make room -- the cache is at {at} -- "
        f"so it was computed again, which took {seconds:.1f}s."
    )
    above = f" above {cap}" if cap else ""
    if budget is None or free >= budget.cap:
        fix = (
            f"raise max_cache_size{above} so results like this one stay cached "
            f"(there is {human_bytes(free)} free on that volume), or cache smaller results."
        )
    else:
        fix = (
            f"cache smaller results, or point cache_dir at a roomier volume: only "
            f"{human_bytes(free)} is free on this one, so raising max_cache_size may not help."
        )
    return message, fix


_lock = threading.Lock()
#: Cache folders whose cap this process has shown.
_budget_shown: set[str] = set()
#: Cache folders whose first eviction this process has reported.
_eviction_told: set[str] = set()


def _scope(cache_dir: str) -> str:
    """One name per folder, whatever the spelling (case on Windows, symlinks)."""
    try:
        path = os.path.realpath(cache_dir)
    except OSError:
        path = os.path.abspath(cache_dir)
    return os.path.normcase(path)


def _claim(seen: set[str], cache_dir: str) -> bool:
    scope = _scope(cache_dir)
    with _lock:
        if scope in seen:
            return False
        seen.add(scope)
        return True


def claim_budget_notice(cache_dir: str) -> bool:
    """True the first time this process is about to show *cache_dir*'s cap."""
    return _claim(_budget_shown, cache_dir)


def claim_eviction_notice(cache_dir: str) -> bool:
    """True the first time this process reports an eviction in *cache_dir*."""
    return _claim(_eviction_told, cache_dir)


def announce_budget(cache_dir: str, budget: object) -> None:
    """Log the cap of *cache_dir* once, when a record would reach anyone.

    *budget* is a callable returning the `DiskBudget` (or None), called only
    then: an adaptive cap is measured from the volume, and a process whose log
    shows nothing should not pay for that. Nor is the folder claimed then, so
    a notebook's ``%cash_on`` still shows it.
    """
    if not storage_logger.isEnabledFor(logging.INFO):
        return
    if not claim_budget_notice(cache_dir):
        return
    found = budget() if callable(budget) else budget
    if found is not None:
        storage_logger.info("%s", describe_budget(found))
