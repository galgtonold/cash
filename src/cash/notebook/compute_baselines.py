"""What a computation cost when it was last measured, kept across kernels.

``%cash_stats`` credits a restore as a saving only where it can point at a
measurement of what that computation costs. Until round 30 the only
measurement it would accept was one THIS kernel took, which made the headline
useless in the one reading every tester takes: after a Restart & Run All
nothing has been recomputed, so the net printed as "at least -10.3s, at best
1.3min" -- a range whose floor is exactly minus cash's own overhead (r30s3,
r30s5). "For a team lead the range reads as 'cash may have cost you time',
which the measurement contradicts."

This store keeps those measurements on disk beside the cache, so the next
kernel can still point at one. Two rules keep it honest:

* **The minimum wins.** A computation measured at 9 s and later at 4 s is
  credited 4 s. The stale-high baseline -- a first run with a cold page cache
  or cold imports, credited forever after -- is the bug the verified net
  exists to prevent, and a minimum can only ever ratchet a credit DOWN.
* **It is a measurement, not a promise.** A baseline says what that work cost
  on this machine, not what it would cost today. ``%cash_stats`` labels a net
  credited from it as measured rather than verified.

Best-effort throughout, like the loop-split store: a missing, unreadable,
corrupt or future-versioned file leaves it empty, which means "no baseline",
which is the pre-round-30 behaviour. The failure mode is a less informative
number, never a wrong one.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib

from .versioned_json_store import StoreRegistry, VersionedJsonStore

_STORE_FILENAME = "_compute_baselines.json"
_STORE_VERSION = 1

#: Keep the store small and its write cheap. At the cap the CHEAPEST baselines
#: go: they are the ones whose credit is worth least, and a statement that
#: matters is re-measured the next time it runs anyway.
_MAX_ENTRIES = 4000


def _key(identity: str) -> str:
    """A short digest of the statement source (or call key) being measured.

    Digested rather than stored verbatim so the file stays small and holds no
    copy of the user's source.
    """
    return hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()[:16]


class ComputeBaselineStore(VersionedJsonStore[float]):
    """``identity -> the least seconds it has ever been measured to cost``."""

    FILENAME = _STORE_FILENAME
    VERSION = _STORE_VERSION
    FIELD = "baselines"
    LOG_TAG = "BASELINES"

    def __init__(self, cache_dir: str | None) -> None:
        super().__init__(cache_dir)
        self._dirty = False

    def _load_value(self, value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    def _dump_value(self, value: float) -> float:
        return round(value, 4)

    def get(self, identity: str) -> float | None:
        """The least this computation has been measured to cost, or ``None``."""
        self._ensure_loaded()
        return self._items.get(_key(identity))

    def record(self, identity: str, seconds: float) -> None:
        """Note a measurement. Only a new minimum changes anything.

        Writing is left to :meth:`flush`, which the caller runs once per cell.
        Writing here would put file I/O on the per-statement path; writing
        only at exit would lose the session, since a Restart & Run All kills
        the kernel and no hook runs -- which is exactly the session whose
        measurements the next kernel needs.
        """
        if not seconds or seconds <= 0:
            return
        self._ensure_loaded()
        key = _key(identity)
        known = self._items.get(key)
        if known is not None and known <= seconds:
            return
        self._items[key] = float(seconds)
        self._dirty = True
        if len(self._items) > _MAX_ENTRIES:
            self._evict()

    def _evict(self) -> None:
        keep = sorted(self._items.items(), key=lambda kv: kv[1], reverse=True)
        self._items = dict(keep[:_MAX_ENTRIES])

    def flush(self) -> None:
        """Write the store out, if anything changed."""
        if self._dirty and self._write():
            self._dirty = False

    def clear(self) -> None:
        """Forget every measurement, on disk too.

        ``%cash_stats reset`` says the session's measurements are forgotten;
        leaving them here would credit a later hit against a measurement the
        reset claims not to have.
        """
        self._ensure_loaded()
        self._items.clear()
        self._dirty = False
        self._delete_file()


def _flush_at_exit(store: ComputeBaselineStore) -> None:
    # The first measurement is written straight away and the rest are
    # throttled, so this only catches the last few seconds of a session --
    # and a kernel killed outright runs no hook at all. Cheap insurance,
    # never relied upon.
    with contextlib.suppress(Exception):
        atexit.register(store.flush)


_STORES: StoreRegistry[ComputeBaselineStore] = StoreRegistry(ComputeBaselineStore, _flush_at_exit)


def get_store(cache_dir: str | None) -> ComputeBaselineStore:
    """The shared store for *cache_dir* (one per directory, process-wide)."""
    return _STORES.get(cache_dir)


def store_for_backend(backend) -> ComputeBaselineStore | None:
    """Shared store for *backend*'s cache dir, or ``None`` if unresolvable."""
    return _STORES.for_backend(backend)


def _reset_stores_for_tests() -> None:
    """Drop cached stores. Tests only -- each tmp_path is a fresh session."""
    _STORES.reset()
