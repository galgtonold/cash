"""In-memory cache backend with LRU eviction."""

from __future__ import annotations

import builtins
import copy
import ctypes
import logging
import pickle
import sys
import time
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from .. import _plain_data
from .._sizing import pandas_nbytes
from ._base import CacheBackend, MetadataDict, gdsf_value
from .serialization import Serializer

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

__all__ = ["InMemoryBackend"]


#: A list or tuple with at most this many items has the frames in it
#: copied as `_copy_frame` copies them, not deep (`_safe_deep_copy`).
_PREMADE_ITEMS_MAX = 64


class InMemoryBackend(CacheBackend):
    """
    In-memory cache backend using a dictionary.
    Supports smart eviction based on memory pressure.
    """

    source_label: str = "RAM"

    def __init__(
        self,
        max_memory_percent: float = 0.9,
        check_interval: int = 10,
        max_entries: int | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """
        Args:
            max_memory_percent: Memory usage percentage (0.0 to 1.0) at which to trigger eviction.
            check_interval: Number of 'set' operations between memory checks.
            max_entries: Maximum number of cache entries. When exceeded, LRU eviction is triggered.
                         None means unlimited entries (eviction only via memory pressure).
            max_size_bytes: Soft byte cap for the RAM tier. When the tracked
                         total exceeds it, least-recently-used entries are evicted down to
                         ~90% of the cap. ``None`` (this CONSTRUCTOR's default) means
                         unbounded — eviction driven only by ``max_entries`` and psutil
                         memory pressure.

                         **``None`` is not what a user gets.** Every backend the
                         factory builds passes a resolved cap
                         (``adaptive_caps.resolve_ram_cap``: a fifth of the memory
                         this process may use, clamped to [512 MiB, 4 GiB]), so the
                         RAM tier is bounded by default and independently of the
                         disk tier. Read alone, the old wording here supported
                         exactly the wrong conclusion, and a ticket was filed on
                         it: "the in-memory tier is unbounded by default".
                         ``cash info`` prints the resolved number.
        """
        self._store: dict[str, tuple[MetadataDict, Any]] = {}  # Stores (metadata, value)
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._current_size_bytes = 0
        #: The footprint the first check of the current memory-pressure episode
        #: left, and the machine's memory percent when that share was taken;
        #: both None outside an episode. See `_evict`.
        self._pressure_floor: int | None = None
        self._pressure_percent: float | None = None
        self._set_count = 0
        #: Keys whose stored value holds only tuples and immutable primitives
        #: below its top: a hit copies the top list alone, with no new check.
        self._immutable_below: builtins.set[str] = set()
        #: Keys whose stored value is a list of dicts of immutable values: a
        #: hit copies each dict with ``map(dict, ...)`` instead of deepcopy.
        self._dict_rows: builtins.set[str] = set()
        #: GreedyDual-Size-Frequency state for the byte cap (see
        #: `_evict_to_byte_cap`): the clock L, and each key's L as of its last
        #: write or read. Kept here, not in the entry's metadata dict, because
        #: that dict is shared with the other tiers and would carry it to disk.
        self._gdsf_clock = 0.0
        self._gdsf_base: dict[str, float] = {}
        #: Access order, for ties. ``last_access`` is wall-clock and collides
        #: within one timer tick, which is how the disk tier's LRU once
        #: degenerated into directory order.
        self._access_seq = 0
        self._seq_by_key: dict[str, int] = {}

    #: How far the machine's memory percent must climb within one pressure
    #: episode before the tier takes a fresh share rather than holding flat.
    #: Two points: well above the jitter of consecutive psutil readings, well
    #: below the nine points between the 90% trigger and the 81% target.
    _PRESSURE_WORSENED_POINTS = 2.0

    #: Types whose instances cannot be mutated, so SHARING one between the
    #: stored entry and the caller is safe. Exact-type membership, never
    #: isinstance: a subclass can carry a mutable ``__dict__``, and an IntEnum
    #: member's type is the user's own class.
    _IMMUTABLE_SCALARS = (int, float, str, bool, bytes, complex, type(None))

    @staticmethod
    def _safe_deep_copy(value: Any, key: str = "<unknown>", *, required: bool = False) -> Any:
        """Copy *value* so the caller cannot reach the stored entry.

        A RAM-tier hit must hand back something independent, or a caller that
        mutates the result poisons every later hit. ``deepcopy`` guarantees
        that but pays a recursive call PER ELEMENT: restoring a 200k-element
        list of ints cost ~13ms, over 4x what rebuilding the list from
        scratch cost, so the cache was slower than no cache at all.

        A container of immutable scalars needs no such walk. Its elements
        cannot be mutated, so copying the container alone already isolates the
        caller -- and a tuple, being immutable itself, needs no copy whatever.
        Measured against ``deepcopy``: 2.1x for a 200k int list, 3.3x for the
        same as a tuple, 1.6x for 50k strings, and no regression on nested
        dicts, where ``all()`` short-circuits on the first element.
        """
        try:
            type_name = type(value).__name__
            if type_name in ("DataFrame", "Series"):
                return InMemoryBackend._copy_frame(value)
            value_type = type(value)
            if value_type is list or value_type is tuple:
                scalars = InMemoryBackend._IMMUTABLE_SCALARS
                if all(type(item) in scalars for item in value):
                    return value if value_type is tuple else list(value)
                # Lists and tuples all the way down, over primitives: copied
                # without a Python call per element (`_plain_data`). deepcopy
                # of two million parsed rows took 1.5 s on every RAM hit.
                done, copied = _plain_data.copy_plain(value)
                if done:
                    return copied
            if value_type is dict:
                # A notebook entry is dicts around the values, and carries the
                # RNG state: `random.getstate()` is a tuple of 625 ints, which
                # deepcopy walks an int at a time -- 2.3M calls for the 2,629
                # entries of one loop over files (round 23, r23s2). Its plain
                # parts go into deepcopy's memo as already copied: a tuple is
                # shared, a list of immutables gets a new list, and two names
                # for one object still come back as one object.
                memo: dict[int, Any] = {}
                InMemoryBackend._premade_copies(value, memo)
                return copy.deepcopy(value, memo)
            if (value_type is list or value_type is tuple) and len(value) <= _PREMADE_ITEMS_MAX:
                # ``frame, summary, n = build()``: a call's result is a tuple,
                # and deepcopy copies a frame in it deep even where a shallow
                # copy is safe (`_copy_frame`).
                memo = {}
                InMemoryBackend._premade_copies(dict(enumerate(value)), memo)
                return copy.deepcopy(value, memo)
            return copy.deepcopy(value)
        except (TypeError, pickle.PicklingError, RecursionError, AttributeError) as exc:
            if required:
                # Storing it would hand every caller the SAME object: a caller
                # mutating a hit changes what later calls get, and two threads
                # get one object to mutate at once (found attacking the
                # decorator before round 26). Isolation is what makes a cached
                # value safe to hand out, so a value that cannot be isolated is
                # not stored -- and it is unpicklable too, so no disk tier
                # could hold it either.
                raise CacheBackendError(
                    f"the result could not be copied ({type(exc).__name__}: {exc}), "
                    f"so caching it would hand every caller the same object"
                ) from exc
            logger.debug("Could not deep-copy value for key %r, returning reference", key)
            return value

    @staticmethod
    def _copy_frame(frame: Any) -> Any:
        """A copy of a pandas frame/series that no later write can reach.

        Under pandas copy-on-write -- always on from pandas 3 -- a SHALLOW copy
        is that already: the first write to either side copies then, and only
        what it writes. A deep copy of every stored frame, and again on every
        RAM hit, was most of cash's own first-run cost on frame-heavy work:
        0.28 of 0.44 s for 20 statements making 1M-row frames, against 0.09 s
        plain (measured after round 28). Without copy-on-write, deep as before.
        """
        return frame.copy(deep=not _pandas_copy_on_write())

    @staticmethod
    def _premade_copies(value: dict, memo: dict[int, Any], depth: int = 0) -> None:
        """Put a copy of each plain container in *value*'s dicts into *memo*."""
        for item in value.values():
            item_type = type(item)
            if item_type.__name__ in ("DataFrame", "Series") and id(item) not in memo:
                memo[id(item)] = InMemoryBackend._copy_frame(item)
            elif item_type is dict:
                if depth < 4:
                    InMemoryBackend._premade_copies(item, memo, depth + 1)
            elif (item_type is tuple or item_type is list) and id(item) not in memo:
                if _plain_data.immutable_below(item):
                    memo[id(item)] = item if item_type is tuple else list(item)

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, without counting an access. See `BaseBackend.peek_metadata`."""
        entry = self._store.get(key)
        return dict(entry[0]) if entry is not None else None

    def peek_entry(self, key: str) -> tuple[MetadataDict, Any] | None:
        """The stored metadata and value themselves: not copied, not counted.

        For writing the entry to another tier as it is
        (``TieredBackend.persist_from_memory``), where a copy of a large frame
        would be pure waste. Both are this tier's own objects: never change
        the value; where the entry is stored (``storage``, ``persist_skipped``)
        is all the caller may update in the metadata, once it is stored elsewhere too.
        """
        return self._store.get(key)

    def get_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, counted as an access the way `get` counts one.

        Without this the base class answered through ``get()``, which
        deep-copies the value only to drop it: the upstream simulation reading
        entries' metadata spent 5.3 s of one 6 s cell copying (round 23, r23s1).
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        metadata = entry[0]
        metadata["last_access"] = time.time()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        metadata.setdefault("source", self.source_label)
        return metadata

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        if key in self._store:
            metadata, value = self._store[key]

            metadata["last_access"] = time.time()
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata.setdefault("source", self.source_label)
            self._touch(key)

            if key in self._immutable_below:
                # Checked when it was stored; the stored value is private.
                return metadata, (list(value) if type(value) is list else value)
            if key in self._dict_rows:
                return metadata, list(map(dict, value))
            return metadata, self._safe_deep_copy(value, key)
        return None, None

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> bool | None:
        """Store *value*; returns False if it was refused (see below)."""
        metadata = self._init_metadata(metadata, key)

        # Plain data is sized, checked and copied from ONE look at it: three
        # separate walks were most of promoting two million parsed rows here.
        plain = _plain_data.profile(value)
        dict_rows_size = None if plain is not None else _plain_data.dict_rows_profile(value)
        if dict_rows_size is not None:
            size = dict_rows_size
        elif plain is None:
            size = self._get_object_size(value)
        else:
            size = plain[0]

        # A value above the eviction target can never stay: the byte cap evicts
        # down to 90% of the cap, and it would be the last one standing. It
        # used to be stored anyway, and the eviction took every older entry and
        # then the value itself -- one oversized write, or one restore of a big
        # disk entry (read-repair promotes into this tier with no size gate),
        # emptied the tier. Refused here, before the copy: copying a frame of
        # gigabytes only to throw it away is its own cost. The previous value
        # for the key goes too, or a later read would serve it as current.
        if self._max_size_bytes is not None and size > self._max_size_bytes * 0.9:
            if key in self._store:
                self._drop(key)
            return False

        if "storage" not in metadata:
            metadata["storage"] = ["RAM"]

        if dict_rows_size is not None:
            # csv.DictReader / JSON records with immutable values: a new dict
            # per row is a complete copy, built in C (round 20: dict rows were
            # 10x slower to cache than the same data as tuples).
            immutable = False
            stored = list(map(dict, value))
        elif plain is None:
            immutable = False
            # Only for a decorator entry, where the stored value IS what the
            # next call hands back. A notebook statement's payload is the
            # variables a cell left behind, and one unisolatable variable among
            # them (an open handle in scope) must not stop the statement being
            # cached -- the notebook re-executes what it cannot restore.
            stored = self._safe_deep_copy(value, key, required=bool((metadata or {}).get("copy_required")))
        else:
            _size, immutable, levels = plain
            stored = _plain_data.copy_plain(value, immutable, levels)[1]
        metadata["size"] = size

        # Byte-cap bookkeeping: on replacement, discount the old entry's size
        # before recording the new one so the running total stays accurate.
        if key in self._store:
            self._current_size_bytes -= self._store[key][0].get("size", 0)
        self._store[key] = (metadata, stored)
        self._touch(key)
        if immutable:
            self._immutable_below.add(key)
        else:
            self._immutable_below.discard(key)
        if dict_rows_size is not None:
            self._dict_rows.add(key)
        else:
            self._dict_rows.discard(key)
        self._current_size_bytes += size

        # Check max_entries limit
        if self.max_entries is not None and len(self._store) > self.max_entries:
            self._evict_lru(len(self._store) - self.max_entries)

        # Enforce the soft byte cap (adaptive RAM cap; None = unbounded).
        if self._max_size_bytes is not None and self._current_size_bytes > self._max_size_bytes:
            self._evict_to_byte_cap()

        # Check memory pressure periodically
        self._set_count += 1
        if self._set_count % self.check_interval == 0:
            self._check_and_evict()

    def _drop(self, key: str) -> None:
        """Remove *key*, keeping the byte-cap running total in sync."""
        self._immutable_below.discard(key)
        self._dict_rows.discard(key)
        self._gdsf_base.pop(key, None)
        self._seq_by_key.pop(key, None)
        entry = self._store.pop(key, None)
        if entry is not None:
            self._current_size_bytes -= entry[0].get("size", 0)

    def delete(self, key: str) -> None:
        self._drop(key)

    def clear(self) -> None:
        self._store.clear()
        self._immutable_below.clear()
        self._dict_rows.clear()
        self._gdsf_base.clear()
        self._seq_by_key.clear()
        self._current_size_bytes = 0
        self._pressure_floor = None
        self._pressure_percent = None
        # Also try to free memory back to OS
        self._try_malloc_trim()

    def list_entries(self) -> list[dict[str, Any]]:
        return [meta for meta, _ in self._store.values()]

    def entry_count(self) -> int:
        return len(self._store)

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        keys_to_delete = []
        for key, (meta, _) in self._store.items():
            if is_expired(meta):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._drop(key)

        if keys_to_delete:
            self._try_malloc_trim()

        return len(keys_to_delete)

    def _get_object_size(self, obj: Any, seen: builtins.set[int] | None = None) -> int:
        """Estimate object size in bytes (recursive)."""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        # Plain data -- lists and tuples over primitives -- is summed a level at
        # a time: the per-element recursion below took 3.5 s to size two
        # million parsed rows being promoted into this tier (round 19). At any
        # depth, not only the top: every notebook entry holds the RNG state, a
        # tuple of 625 ints one dict down (round 23: 1.7M calls in one cell).
        if type(obj) in _plain_data.SEQS:
            plain = _plain_data.size_of(obj)
            if plain is not None:
                return plain

        # int() on every return, without exception. `mem.sum()` below returns a
        # `numpy.int64`, and this value is written into entry metadata as
        # `size` -- so a numpy scalar there makes the metadata file unreadable
        # in any environment without numpy, forever. That is not hypothetical:
        # it escaped from `%cash_on` as a ModuleNotFoundError. The annotation
        # already said `-> int`; this makes it true.
        try:
            # A frame is sized from its column arrays: ``memory_usage`` spent
            # ~0.23 ms building a result Series per store (``cash._sizing``).
            frame_size = pandas_nbytes(obj)
            if frame_size is not None:
                return frame_size
            # Prefer nbytes for numpy/pandas
            if hasattr(obj, "nbytes"):
                return int(obj.nbytes)
            if hasattr(obj, "memory_usage"):
                # pandas DataFrame/Series
                # OPTIMIZATION: Use deep=False for speed (deep=True scans all object columns)
                try:
                    mem = obj.memory_usage(deep=False)
                    if hasattr(mem, "sum"):
                        return int(mem.sum())
                    return int(mem)
                except (TypeError, AttributeError):
                    # Fallback to sys.getsizeof below when memory_usage is unavailable
                    pass

            size = sys.getsizeof(obj)

            if isinstance(obj, dict):
                size += sum(self._get_object_size(v, seen) for v in obj.values())
                # Also keys
                size += sum(self._get_object_size(k, seen) for k in obj)
            elif isinstance(obj, (list, tuple, set)):
                size += sum(self._get_object_size(i, seen) for i in obj)

            return size
        except (TypeError, RecursionError, ValueError):
            logger.debug("Could not estimate size of %s object", type(obj).__name__, exc_info=True)
            return 0

    def _check_and_evict(self) -> None:
        """Check memory usage and evict items if threshold is exceeded."""
        if psutil is None:
            return

        try:
            mem = psutil.virtual_memory()
            if mem.percent / 100.0 > self.max_memory_percent:
                self._evict(mem)
            else:
                # The episode is over: the next one takes a fresh share.
                self._pressure_floor = None
                self._pressure_percent = None
        except (OSError, AttributeError) as exc:
            logger.debug("Memory check failed: %s", exc)

    def _evict(self, mem: Any = None) -> None:
        """Give back this tier's SHARE of the machine's memory pressure.

        Same order as the byte cap (`_evict_to_byte_cap`): least value per
        byte first. It used to score ``execution_time * access_count / size``,
        which put every entry not yet read at zero -- a 30-second result went
        before a 1 ms one that had been read once.

        **How much.** This used to drop entries until the WHOLE MACHINE fell
        under the target. When the pressure is someone else's, that never
        happens, so one check emptied the tier -- measured: 200 entries to 0
        in a single call -- and every later check emptied it again. Round 27
        ran five testers on one box, each with a cash kernel and an uncached
        oracle kernel over its full dataset: r27s3's parameter sweep went from
        13.4 s to ~100 s, stayed there through reruns and even through
        reverting the edit it blamed, while the uncached kernel beside it
        barely moved. Its `cached=` count fell from 601 to ~170 and never
        recovered. A clean replay of the same seven steps on a quiet machine
        stays at 9-12 s throughout. Emptying a cache the machine's pressure
        does not come from costs its user everything and the machine nothing.

        So it sheds its share: ``overshoot * own / in_use`` bytes, where
        ``own`` is this tier's footprint. A tier that IS most of the memory in
        use sheds nearly the whole overshoot, as before; one that holds a
        sliver of it sheds a sliver.

        **Once per episode.** Under pressure that never relents, taking the
        share again on every check (one per ``check_interval`` writes) is the
        same drain, geometrically -- ~15% per check across the hundreds of
        writes one sweep makes. So the first check of an episode takes the
        share, records the footprint it left, and later checks in the same
        episode only hold the tier at that level: new entries displace the
        least valuable old ones rather than the tier shrinking again. The
        episode ends at the first check that finds no pressure.

        Falls back to the old drain-to-target when the reading carries no
        ``total`` -- psutil always provides it, so that is only ever a test
        double -- because the share cannot be computed without it.
        """
        target_percent = self.max_memory_percent * 0.9
        share = self._pressure_share(mem, target_percent)
        if share is not None:
            self._shed(share)
            return

        items = [
            (self._gdsf_priority(key, meta), self._seq_by_key.get(key, 0), key)
            for key, (meta, _val) in self._store.items()
        ]
        items.sort()

        evicted_count = 0

        for priority, _seq, key in items:
            if key in self._store:
                self._drop(key)
                self._gdsf_clock = max(self._gdsf_clock, priority)
                evicted_count += 1

                if psutil is None:
                    break
                mem = psutil.virtual_memory()
                if mem.percent / 100.0 <= target_percent:
                    break

        if evicted_count > 0:
            self._try_malloc_trim()

    def _pressure_share(self, mem: Any, target_percent: float) -> float | None:
        """Bytes this tier should give back now, or None when it cannot tell.

        The first check of a pressure episode: its proportional share of the
        overshoot. Every later one: whatever it has grown past the level the
        first one left.
        """
        total = getattr(mem, "total", None)
        percent = getattr(mem, "percent", None)
        if not isinstance(total, (int, float)) or not isinstance(percent, (int, float)) or total <= 0:
            return None
        own = self._current_size_bytes
        worsening = (
            self._pressure_percent is not None and percent > self._pressure_percent + self._PRESSURE_WORSENED_POINTS
        )
        if self._pressure_floor is not None and not worsening:
            return max(0.0, own - self._pressure_floor)
        # A fresh share: the episode's first check, or pressure that has got
        # WORSE since the last one. Holding flat is right while the pressure is
        # steady; if it keeps climbing, something -- possibly this tier, if the
        # memory it freed has not gone back to the OS yet -- is still growing,
        # and refusing to shed again would let the machine swap. The old loop
        # erred towards emptying the cache; this must not err the other way.
        self._pressure_percent = percent
        in_use = total * percent / 100.0
        overshoot = in_use - total * target_percent
        if overshoot <= 0 or in_use <= 0:
            return 0.0
        return min(float(own), overshoot * min(1.0, own / in_use))

    def _shed(self, nbytes: float) -> None:
        """Drop the least valuable entries until *nbytes* are freed."""
        freed = 0
        if nbytes > 0:
            items = sorted(
                (self._gdsf_priority(key, meta), self._seq_by_key.get(key, 0), key)
                for key, (meta, _val) in self._store.items()
            )
            for priority, _seq, key in items:
                if freed >= nbytes:
                    break
                entry = self._store.get(key)
                if entry is None:
                    continue
                freed += entry[0].get("size", 0) or 0
                self._drop(key)
                self._gdsf_clock = max(self._gdsf_clock, priority)
            if freed:
                self._try_malloc_trim()
        self._pressure_floor = self._current_size_bytes

    def _touch(self, key: str) -> None:
        """Record a write or read: it re-bases the entry's GDSF priority."""
        self._gdsf_base[key] = self._gdsf_clock
        self._access_seq += 1
        self._seq_by_key[key] = self._access_seq

    def _gdsf_priority(self, key: str, meta: MetadataDict) -> float:
        """``H = L + hits * execution_time / size``, L as of the last access."""
        return self._gdsf_base.get(key, 0.0) + gdsf_value(meta, meta.get("size", 1))

    def _evict_to_byte_cap(self) -> None:
        """Evict by value per byte until under ~90% of the byte cap.

        GreedyDual-Size-Frequency: the lowest ``H = L + hits * cost / size``
        goes first, and the clock ``L`` rises to each victim's H, so an entry
        that stops being read ages below newer ones however valuable it was.
        Recency alone treated a 30 s result like a 50 ms one of the same size,
        and 4 MB like 1 MB; in the eviction simulation (benchmarks/eviction_sim)
        that difference was most of the loss. In a notebook: a folder loop's
        per-file frames (20 ms each) were all dropped for a later cell's
        full-table copies (0.1-3 s per 570 MB), so the loop's next run read
        every file again (round 23, r23s2). Ties go to the least recently
        touched.

        Evicts down to 90% of the cap, giving headroom so the next few writes
        don't immediately re-trigger eviction. No-op when the cap is unset or
        already satisfied.
        """
        if not self._max_size_bytes or self._current_size_bytes <= self._max_size_bytes:
            return
        target = self._max_size_bytes * 0.9
        items = [
            (self._gdsf_priority(key, meta), self._seq_by_key.get(key, 0), key)
            for key, (meta, _) in self._store.items()
        ]
        items.sort()  # least valuable per byte first
        evicted = 0
        for priority, _seq, key in items:
            if self._current_size_bytes <= target:
                break
            self._drop(key)
            self._gdsf_clock = max(self._gdsf_clock, priority)
            evicted += 1
        if evicted:
            self._try_malloc_trim()

    def _evict_lru(self, count: int):
        """Evict the N least-recently-used entries."""
        if count <= 0:
            return
        items = []
        for key, (meta, _) in self._store.items():
            last_access = meta.get("last_access", 0)
            items.append((last_access, key))
        items.sort()  # oldest first
        for _, key in items[:count]:
            self._drop(key)

    def _try_malloc_trim(self) -> None:
        """Try to clean up memory on Linux."""
        if sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except (OSError, AttributeError):
                # Best-effort memory cleanup; safe to ignore on non-glibc systems
                pass


_COW: list[bool] = []


def _pandas_copy_on_write() -> bool:
    """Whether pandas copy-on-write is in force (always, from pandas 3)."""
    if _COW:
        return _COW[0]
    import sys

    pd = sys.modules.get("pandas")
    if pd is None:
        return False  # not decided yet: nothing to copy without pandas
    try:
        on = int(str(pd.__version__).split(".")[0]) >= 3
        if not on:
            on = bool(pd.get_option("mode.copy_on_write") is True)
    except Exception:  # noqa: BLE001 - unknown: the safe answer is "deep copy"
        on = False
    _COW.append(on)
    return on
