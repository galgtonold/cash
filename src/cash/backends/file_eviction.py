"""Keeping a file cache under its byte cap: size accounting and GDSF eviction.

`FileEvictor` owns everything `FileBackend` needs to stay under
``max_size_bytes``: the running byte total, the adaptive cap, the
GreedyDual-Size-Frequency ranking of what to evict, and the warning when the
cache evicts what it has only just written.
"""

from __future__ import annotations

import heapq
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning
from ._base import gdsf_value
from .adaptive_caps import adaptive_disk_cap_for, free_bytes_on_volume, human_bytes
from .cache_dir import entry_totals
from .entry_format import ENTRY_SUFFIX
from .rank_index import RankIndex

if TYPE_CHECKING:
    from ._writes import PendingWrites
    from .touched_entries import TouchedEntries

logger = logging.getLogger(__name__)

__all__ = ["FileEvictor"]


def _stem(path: str) -> str:
    return os.path.basename(path)[: -len(ENTRY_SUFFIX)]


def _priority(metadata: dict, size: int, base: float) -> float:
    """GDSF: ``H = L + hits * execution_time / size``, L as of *base*."""
    return base + gdsf_value(metadata, size)


class FileEvictor:
    """Size accounting and value-per-byte eviction for one cache directory.

    It reads the backend's per-key bookkeeping through *touched* (see
    `TouchedEntries`) and forgets an entry there when it removes one. Its own
    accounting is guarded by ``touched.lock``, so a removal updates both in one
    step.
    """

    #: Seconds between re-readings of the volume's free space for an adaptive
    #: cap: it sits on the write path, and a network volume answers slowly.
    CAP_REFRESH_INTERVAL = 60.0

    #: An entry evicted within this many writes of being written is the
    #: signature of a cache too small for its working set.
    EVICT_WARN_RECENT_OPS = 3

    #: Below roughly this many entries fitting the cap, the advice changes from
    #: "raise the cap" to "cache something smaller".
    FEW_ENTRIES_FIT = 20

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int | None,
        adaptive: bool,
        *,
        touched: TouchedEntries,
        writes: PendingWrites,
        untracked: Callable[[], Any],
    ) -> None:
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_bytes
        #: The cap came from the machine-scaling policy, not the user, so it
        #: may be re-derived as the volume fills or empties.
        self.adaptive = adaptive
        self.cap_derived_at = 0.0  # 0.0: never
        #: On-disk bytes of the entries. Absolute once `ensure_size_scanned`
        #: has walked the directory, before that only this process's delta.
        self.current_bytes = 0
        self.size_scanned = False
        self.rank_index = RankIndex(cache_dir, untracked)
        self._touched = touched
        self._writes = writes
        self._lock = touched.lock
        self._untracked = untracked
        # Write order, so eviction can tell it is dropping something written
        # only a couple of writes ago.
        self.write_seq = 0
        self.write_seq_by_key: dict[str, int] = {}
        self.warned_thrash = False
        # Candidates, least valuable per byte first, as (path, size, ranked_at),
        # consumed across passes and rebuilt when empty; priorities in rank_h.
        self.queue: deque[tuple[str, int, float]] = deque()
        self.rank_h: dict[str, float] = {}
        # Entries written after the ranking was taken, as a heap of
        # (priority, seq, path, size, ranked_at): under GDSF a cheap large
        # value can be the least valuable thing in the cache the moment it lands.
        self.fresh: list[tuple[float, int, str, int, float]] = []
        self.ranked = False
        # GreedyDual's clock L, and each touched key's L as of its last write
        # (None: read since, re-based at next use). Resumed from the rank index
        # on first need, so a new process does not rank everything it writes
        # below everything already on disk.
        self.clock = 0.0
        self.clock_loaded = False
        self.base: dict[str, float | None] = {}

    @property
    def capped(self) -> bool:
        """Is there a cap to keep to? Only then is anything ranked or evicted."""
        return bool(self.max_size_bytes)

    # -- bookkeeping the backend reports -------------------------------------

    def note_write(self, key: str, old_bytes: int, new_bytes: int) -> None:
        """An entry of *new_bytes* replaced one of *old_bytes* (0 if new)."""
        with self._lock:
            self.current_bytes += new_bytes - old_bytes
            self.write_seq += 1
            self.write_seq_by_key[key] = self.write_seq

    def note_read(self, key: str) -> None:
        """A read: re-based at its next use, so a read never loads the index."""
        with self._lock:
            self.base[key] = None

    def clear(self) -> None:
        """The directory was emptied."""
        self.rank_index.remove()
        with self._lock:
            self.write_seq_by_key.clear()
            self.queue.clear()
            self.rank_h.clear()
            self.fresh = []
            self.ranked = False
            self.base.clear()
            self.clock = 0.0
            self.current_bytes = 0

    # -- size and cap ---------------------------------------------------------

    def scan_size_bytes(self) -> int:
        """Total the entries' bytes with ``scandir`` + ``stat``; no file is opened."""
        with self._untracked():
            totals = entry_totals(self.cache_dir)
        if totals is None:
            logger.debug("Could not scan %s for size", self.cache_dir)
            return 0
        return totals[1]

    def ensure_size_scanned(self) -> None:
        """Establish the on-disk byte total, once, when the first write needs it.

        Only eviction reads the total, so a read-only process never pays the
        walk, and a writer pays it on the write worker. The walk runs outside
        the lock and replaces the running delta: the write that triggered it is
        already on disk, so the directory is the truth.
        """
        if self.size_scanned:
            return
        scanned = self.scan_size_bytes()
        with self._lock:
            if self.size_scanned:
                return
            self.current_bytes = scanned
            self.size_scanned = True
        # The footprint is what the cap was missing; size it now.
        self.refresh_adaptive_cap(force=True)

    def refresh_adaptive_cap(self, force: bool = False) -> None:
        """Re-size an adaptive cap from the volume as it is now.

        A notebook kernel outlives the free-space reading it started with, so
        the cap follows the volume, at most once per `CAP_REFRESH_INTERVAL`.
        Sized from free space plus the cache's own bytes, or the cache would be
        over a cap its own contents caused. An explicit cap is never touched.
        """
        if not self.adaptive:
            return
        now = time.monotonic()
        if not force and now - self.cap_derived_at < self.CAP_REFRESH_INTERVAL:
            return
        self.cap_derived_at = now
        self.max_size_bytes = adaptive_disk_cap_for(self.cache_dir, self.current_bytes)

    # -- ranking --------------------------------------------------------------

    def ensure_clock(self) -> None:
        """Resume the GDSF clock from the rank index, once per process."""
        if self.clock_loaded:
            return
        _ranks, clock, _count = self.rank_index.load()
        with self._lock:
            self.clock = max(self.clock, clock)
            self.clock_loaded = True

    def access_record(self, key: str, meta: dict, path: str) -> tuple[str, float]:
        """The rank-index record for a flushed read of *key*."""
        with self._lock:
            base = self.base.get(key)
            if base is None:
                base = self.base[key] = self.clock
        return _stem(path), _priority(meta, os.path.getsize(path), base)

    def record_rank(self, key: str, path: str, metadata: dict, size: int) -> None:
        """A write: store its priority, and make it a candidate at once.

        Ranked at the recency `rebuild_queue` would give it -- its own mtime,
        or its ``last_access`` if newer -- not at ``time.time()``: mtime is the
        filesystem's clock, and on a coarse ``time.time()`` the write's own
        mtime read as newer than the moment it was ranked, so every fresh
        entry looked read-since and was skipped as a candidate.
        """
        self.ensure_clock()
        # Only once a ranking exists: a ranking taken later walks the directory
        # after this write landed and already holds the entry.
        ranked_at = None
        if self.ranked:
            try:
                ranked_at = os.path.getmtime(path)
            except OSError:
                ranked_at = time.time()
            last_access = metadata.get("last_access")
            if last_access is not None and last_access > ranked_at:
                ranked_at = last_access
        with self._lock:
            clock = self.clock
            self.base[key] = clock
            priority = _priority(metadata, size, clock)
            if self.ranked and ranked_at is not None:
                heapq.heappush(self.fresh, (priority, self.write_seq_by_key.get(key, 0), path, size, ranked_at))
        self.rank_index.append([(_stem(path), priority)])

    def rebuild_queue(self) -> None:
        """Rank every entry for eviction, least valuable per byte first.

        GDSF: the lowest ``H = L + hits * execution_time / size`` goes first,
        and the clock L rises to each victim's H, so an entry that stops being
        read ages out however valuable it was.

        H comes from one ``scandir`` (sizes, mtimes) plus the rank index, never
        from opening entries, which costs orders of magnitude more per entry.
        Entries this process wrote or read are ranked from memory; an entry
        with no record ranks as if its cost were unknown and small.

        Ties go to the least recently used (mtime, or an unflushed in-memory
        ``last_access``), then to the older write by the rank index's line
        order, because mtimes come in coarse steps and a burst of writes shares
        one.

        The ranking is a queue consumed across many passes; entries written
        after it was taken join `fresh`. The index is compacted here once it
        has grown past twice the directory.
        """
        ranks: dict[str, tuple[float, int]] = {}
        try:
            with self._untracked(), os.scandir(self.cache_dir) as entries:
                for entry in entries:
                    if not entry.name.endswith(ENTRY_SUFFIX):
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue  # vanished mid-walk: a slightly late eviction
                    ranks[entry.path] = (st.st_mtime, st.st_size)
        except OSError:
            logger.debug("Could not scan %s to rank evictions", self.cache_dir, exc_info=True)

        indexed, index_clock, index_lines = self.rank_index.load()

        prio: dict[str, float] = {}
        recency: dict[str, float] = {}
        stamps: dict[str, float] = {}
        seqs: dict[str, int] = {}
        with self._lock:
            self.clock = max(self.clock, index_clock)
            self.clock_loaded = True
            clock = self.clock
            for path, (mtime, size) in ranks.items():
                recency[path] = stamps[path] = mtime
                key = self._touched.key_for(path)
                meta = self._touched.metadata(key)
                if meta is not None:
                    last_access = meta.get("last_access")
                    if last_access is not None and last_access > mtime:
                        # What `touched_since` compares against later.
                        stamps[path] = last_access
                        # Only a read not yet on disk orders by it; once
                        # written, mtime is the entry's recency as for every
                        # other entry, and the header's own stamp runs ahead of
                        # the filesystem clock.
                        if self._touched.has_unflushed_read(key):
                            recency[path] = last_access
                    seqs[path] = self.write_seq_by_key.get(key, 0)
                # Only a write or a read in this process re-bases an entry;
                # metadata merely looked at is not a use.
                if meta is not None and key in self.base:
                    base = self.base[key]
                    if base is None:
                        base = self.base[key] = clock
                    prio[path] = _priority(meta, size, base)
                    continue
                recorded = indexed.get(_stem(path))
                if recorded is not None:
                    prio[path] = recorded
                else:
                    prio[path] = _priority(meta or {}, size, 0.0)

        # The index's order is the write order across processes; an entry with
        # no record sorts first. The in-process sequence decides only when the
        # index could not be written.
        recorded_at = {stem: i for i, stem in enumerate(indexed)}
        ordered = sorted(
            ranks,
            key=lambda p: (prio[p], recency[p], recorded_at.get(_stem(p), -1), seqs.get(p, 0)),
        )
        # Each item carries the recency it was ranked at, for `touched_since`.
        self.queue = deque((p, ranks[p][1], stamps[p]) for p in ordered)
        self.rank_h = prio
        self.fresh = []
        self.ranked = True

        if index_lines > 2 * len(ranks) + 64:
            self.rank_index.compact({_stem(p): prio[p] for p in ordered}, clock)

    def _pop_candidate(self) -> tuple[str, int, float, float] | None:
        """The next victim, ``(path, size, ranked_at, priority)``: the lower
        of the ranking's head and the freshest writes' head."""
        head = self.queue[0] if self.queue else None
        fresh = self.fresh[0] if self.fresh else None
        if head is None and fresh is None:
            return None
        if head is not None:
            head_key = (self.rank_h.get(head[0], 0.0), head[2])
        if fresh is None or (head is not None and head_key <= (fresh[0], fresh[4])):
            path, size, ranked_at = self.queue.popleft()
            return path, size, ranked_at, self.rank_h.pop(path, 0.0)
        priority, _seq, path, size, ranked_at = heapq.heappop(self.fresh)
        return path, size, ranked_at, priority

    def touched_since(self, path: str, key: str | None, ranked_at: float) -> bool:
        """Has this entry been read or rewritten since it was ranked?

        A read raises its priority, so the queued value is stale: it is dropped
        as a candidate and ranked properly by the next rebuild. Both signals,
        because a read in this process shows first as ``last_access`` in
        memory, and one in another process only as mtime.
        """
        meta = self._touched.metadata(key) if key else None
        last_access = meta.get("last_access") if meta else None
        if last_access is not None and last_access > ranked_at:
            return True
        try:
            return os.path.getmtime(path) > ranked_at
        except OSError:
            return False  # gone: let the eviction no-op and clean the bookkeeping

    def remove_path(self, path: str, key: str | None = None) -> int:
        """Remove the entry at *path* (*key*'s, looked up when not given), with
        its bookkeeping, and return the bytes freed.

        Measured from the file: the bookkeeping only knows the keys this
        process touched.
        """
        try:
            freed = os.path.getsize(path)
        except OSError:
            freed = 0

        with self._lock:
            if key is None:
                key = self._touched.key_for(path)
            self._touched.forget(key, path)
            if key is not None:
                self.write_seq_by_key.pop(key, None)
                self.base.pop(key, None)
            self.current_bytes -= freed

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Failed to remove cache entry %s: %s", path, exc)
        return freed

    def evict(self) -> None:
        """Evict the least valuable entries per byte while over the cap, to 90% of it."""
        if not self.max_size_bytes:
            return
        self.ensure_size_scanned()
        self.refresh_adaptive_cap()
        if self.current_bytes <= self.max_size_bytes:
            return

        target = self.max_size_bytes * 0.9
        evicted_recent = False
        n_evicted = 0
        rebuilt = False
        own_key = self._writes.current_worker_key()

        while self.current_bytes > target:
            candidate = self._pop_candidate()
            if candidate is None:
                if rebuilt:
                    # Everything ranked this pass was considered; what is left
                    # is in flight or newer. The next write re-examines.
                    break
                self.rebuild_queue()
                rebuilt = True
                if not self.queue and not self.fresh:
                    break
                continue

            path, _size, ranked_at, priority = candidate
            key = self._touched.key_for(path)

            if self.touched_since(path, key, ranked_at):
                continue

            # Never evict a key with ANOTHER write in flight: this runs on the
            # single write worker, and deleting would drain a write queued
            # behind the one running now -- which can never start. The write
            # running now has landed, so its own entry may go.
            if key is not None and key != own_key and self._writes.has_pending(key):
                continue

            written_at = self.write_seq_by_key.get(key) if key else None
            if written_at is not None and self.write_seq - written_at <= self.EVICT_WARN_RECENT_OPS:
                evicted_recent = True

            if self.remove_path(path):
                n_evicted += 1
                with self._lock:
                    self.clock = max(self.clock, priority)

        if n_evicted:
            # The clock lets the next process resume the ranking.
            self.rank_index.append([], clock=self.clock)
        if evicted_recent:
            self.warn_thrash()

    def dominant_entry_size(self) -> int | None:
        """The size at which cumulative bytes cross half the ranked cache.

        "How big are the things filling this cache", which a mean is not: one
        64 MB entry among 3000 small ones is most of the bytes. Computed only
        when the cache is already thrashing, from sizes already in memory; the
        partly drained queues make it a sample, which is all the message needs.
        """
        sizes = [s for _p, s, _m in self.queue]
        sizes.extend(s for _h, _seq, _p, s, _m in self.fresh)
        if not sizes:
            return None
        sizes.sort()
        half = sum(sizes) / 2
        run = 0
        for size in sizes:
            run += size
            if run >= half:
                return size
        return sizes[-1]

    def warn_thrash(self) -> None:
        """Warn once per backend that the cache evicts what it has just written.

        Names the free space rather than prescribing a bigger cap the volume
        may not have, and when only a handful of entries fit, points at caching
        something smaller instead.
        """
        if self.warned_thrash:
            return
        self.warned_thrash = True

        cap = human_bytes(self.max_size_bytes)
        free = free_bytes_on_volume(self.cache_dir)
        if free >= (self.max_size_bytes or 0):
            room = f"raise max_cache_size -- there is {human_bytes(free)} free on that volume, so there is room for it."
        else:
            room = (
                f"cache fewer or smaller results, or point cache_dir at a "
                f"roomier volume: only {human_bytes(free)} is free on this "
                f"one, so raising max_cache_size may not help."
            )

        shape = ""
        dominant = self.dominant_entry_size()
        if dominant and self.max_size_bytes:
            fits = max(1, self.max_size_bytes // dominant)
            if fits < self.FEW_ENTRIES_FIT:
                how_many = "only one fits" if fits == 1 else f"only about {fits} fit"
                shape = (
                    f" Most of it is entries of around {human_bytes(dominant)}, "
                    f"so {how_many} at once; if what you need downstream is a "
                    f"summary of those results rather than the results "
                    f"themselves, caching that instead would fit far more."
                )

        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-THRASH",
            f"the cache is full at its {cap} cap and is evicting entries within "
            f"a couple of writes of storing them, so it is re-writing and "
            f"re-evicting rather than caching durably -- which is slower than "
            f"no cache at all.{shape}",
            room,
        )
