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

import contextlib
import hashlib
import json
import logging
import os

from cash.utils import replace_with_retry

logger = logging.getLogger(__name__)

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


class ComputeBaselineStore:
    """``identity -> the least seconds it has ever been measured to cost``."""

    def __init__(self, cache_dir: str | None) -> None:
        self._path = os.path.join(cache_dir, _STORE_FILENAME) if cache_dir else None
        self._baselines: dict[str, float] = {}
        self._loaded = False
        self._dirty = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            logger.debug("[BASELINES] no readable store at %s", self._path)
            return
        if not isinstance(doc, dict) or doc.get("version") != _STORE_VERSION:
            return
        baselines = doc.get("baselines")
        if not isinstance(baselines, dict):
            return
        for key, seconds in baselines.items():
            if isinstance(key, str) and isinstance(seconds, (int, float)) and seconds > 0:
                self._baselines[key] = float(seconds)

    def get(self, identity: str) -> float | None:
        """The least this computation has been measured to cost, or ``None``."""
        self._ensure_loaded()
        return self._baselines.get(_key(identity))

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
        known = self._baselines.get(key)
        if known is not None and known <= seconds:
            return
        self._baselines[key] = float(seconds)
        self._dirty = True
        if len(self._baselines) > _MAX_ENTRIES:
            self._evict()

    def _evict(self) -> None:
        keep = sorted(self._baselines.items(), key=lambda kv: kv[1], reverse=True)
        self._baselines = dict(keep[:_MAX_ENTRIES])

    def flush(self) -> None:
        """Write the store out, if anything changed."""
        if not self._dirty or not self._path:
            return
        doc = {"version": _STORE_VERSION, "baselines": {k: round(v, 4) for k, v in sorted(self._baselines.items())}}
        tmp_path = f"{self._path}.{os.getpid()}.tmp"
        try:
            from cash.backends.file_backend import recreate_cache_dir

            recreate_cache_dir(os.path.dirname(self._path))
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            # Same reason as the loop-split store: on Windows a bare
            # os.replace is DENIED while any handle holds the destination.
            replace_with_retry(tmp_path, self._path)
            self._dirty = False
        except OSError:
            logger.debug("[BASELINES] could not persist to %s", self._path)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    def clear(self) -> None:
        """Forget every measurement, on disk too.

        ``%cash_stats reset`` says the session's measurements are forgotten;
        leaving them here would credit a later hit against a measurement the
        reset claims not to have.
        """
        self._ensure_loaded()
        self._baselines.clear()
        self._dirty = False
        if self._path:
            with contextlib.suppress(OSError):
                os.unlink(self._path)


_STORES: dict[str | None, ComputeBaselineStore] = {}


def get_store(cache_dir: str | None) -> ComputeBaselineStore:
    """The shared store for *cache_dir* (one per directory, process-wide)."""
    store = _STORES.get(cache_dir)
    if store is None:
        store = ComputeBaselineStore(cache_dir)
        _STORES[cache_dir] = store
        # The first measurement is written straight away and the rest are
        # throttled, so this only catches the last few seconds of a session --
        # and a kernel killed outright runs no hook at all. Cheap insurance,
        # never relied upon.
        with contextlib.suppress(Exception):
            import atexit

            atexit.register(store.flush)
    return store


def store_for_backend(backend) -> ComputeBaselineStore | None:
    """Shared store for *backend*'s cache dir, or ``None`` if unresolvable."""
    try:
        from .statement.miss_guard import resolve_cache_dir

        return get_store(resolve_cache_dir(backend))
    except Exception:  # noqa: BLE001 - a baseline is a reporting nicety
        logger.debug("[BASELINES] could not resolve a store", exc_info=True)
        return None


def _reset_stores_for_tests() -> None:
    """Drop cached stores. Tests only -- each tmp_path is a fresh session."""
    _STORES.clear()
