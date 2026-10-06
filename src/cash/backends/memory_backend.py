"""In-memory cache backend with LRU eviction."""

from __future__ import annotations

import builtins
import copy
import ctypes
import io
import logging
import pickle
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from .. import _plain_data, kept_state
from .._lazy_module import LazyModule
from ..sizing import memory_footprint
from ..value_types import IMMUTABLE_PRIMS
from ._base import CacheBackend, MetadataDict, gdsf_value
from .serialization import Serializer

psutil = LazyModule("psutil")  # imported on first use: ~11 ms off `import cash`

logger = logging.getLogger(__name__)

__all__ = ["InMemoryBackend"]


#: A list or tuple with at most this many items has the frames in it
#: copied as `_copy_frame` copies them, not deep (`_safe_deep_copy`).
_PREMADE_ITEMS_MAX = 64


class InMemoryBackend(CacheBackend):
    """Entries held in this process's memory, gone when the process ends.

    A hit returns a copy, so changing it does not change the entry. Entries
    are evicted when the byte cap or the entry cap is reached, or when the
    machine runs short of memory.
    """

    source_label: str = "RAM"
    cost_kind: str = "ram"

    def __init__(
        self,
        max_memory_percent: float = 0.9,
        check_interval: int = 10,
        max_entries: int | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """
        Args:
            max_memory_percent: Share of the machine's memory in use (0.0 to
                1.0) above which entries are evicted.
            check_interval: Writes between two memory checks.
            max_entries: Most entries held. ``None``: no limit.
            max_size_bytes: Byte cap. Past it, the entries worth least per
                byte are evicted down to about 90% of the cap. ``None``: no
                cap. A RAM tier built from configuration always has one (a
                fifth of the memory this process may use, between 512 MiB and
                4 GiB); ``cash info`` shows it.
        """
        self._store: dict[str, tuple[MetadataDict, Any]] = {}  # Stores (metadata, value)
        #: Guards the store and its bookkeeping: every method reads or changes
        #: several of these together, and eviction walks the whole store, so
        #: two threads using one tier would otherwise see it half-updated. A
        #: hit's copy and a write's copy run outside it: they touch only the
        #: value, which a stored entry never changes.
        self._lock = threading.RLock()
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
        #: Per key, `_copy_frame`'s verdict on each pandas frame in the stored
        #: value, by the frame's ``id``: whether its cells need a deep copy.
        #: Decided once when the value is stored; the stored frames are private
        #: and alive as long as the entry, so the ids stay theirs.
        self._frame_cells: dict[str, dict[int, bool]] = {}
        #: GreedyDual-Size-Frequency state for the byte cap (see
        #: `_evict_to_byte_cap`): the clock L, and each key's L as of its last
        #: write or read. Kept here, not in the entry's metadata dict, because
        #: that dict is shared with the other tiers and would carry it to disk.
        self._gdsf_clock = 0.0
        self._gdsf_base: dict[str, float] = {}
        #: Access order, for ties. ``last_access`` is wall-clock and collides
        #: within one timer tick, so it cannot order entries written together.
        self._access_seq = 0
        self._seq_by_key: dict[str, int] = {}

    #: How far the machine's memory percent must climb within one pressure
    #: episode before the tier takes a fresh share rather than holding flat.
    #: Two points: well above the jitter of consecutive psutil readings, well
    #: below the nine points between the 90% trigger and the 81% target.
    _PRESSURE_WORSENED_POINTS = 2.0

    #: What memory pressure never takes the tier below, and what it may grow to
    #: while the pressure lasts. A share is at most a fifth of the tier (the
    #: overshoot over the memory in use, with the target at 81%), so below
    #: this it would free at most ~3 MiB, which relieves no machine, while
    #: every entry it drops is recomputed. And a session that starts on a full
    #: machine must still be able to hold a working set, not only the few
    #: entries it had at the first check.
    _PRESSURE_KEEPS_BYTES = 16 * 1024**2

    @staticmethod
    def _safe_deep_copy(
        value: Any,
        key: str = "<unknown>",
        *,
        required: bool = False,
        known_cells: dict[int, bool] | None = None,
        record_cells: dict[int, bool] | None = None,
    ) -> Any:
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

        *known_cells* and *record_cells* carry `_copy_frame`'s verdict on the
        pandas frames in *value* (see there): a hit looks the stored frames up
        in *known_cells*, a store records its copies in *record_cells*.

        A polars frame is not a pandas one: it has no ``copy()``, and
        ``deepcopy`` of it is a ``clone()``, which shares its immutable
        buffers and so costs about a millisecond whatever its size.
        """
        try:
            value_type = type(value)
            if _is_pandas_frame(value_type):
                return InMemoryBackend._copy_frame(value, known_cells, record_cells)
            if value_type.__name__ == "ndarray" and value_type is getattr(sys.modules.get("numpy"), "ndarray", None):
                if not value.dtype.hasobject:
                    return _copy_array(value)
            if value_type is list or value_type is tuple:
                if all(type(item) in IMMUTABLE_PRIMS for item in value):
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
                # deepcopy would walk an int at a time. Its plain
                # parts go into deepcopy's memo as already copied: a tuple is
                # shared, a list of immutables gets a new list, and two names
                # for one object still come back as one object.
                memo: dict[int, Any] = {}
                InMemoryBackend._premade_copies(value, memo, known_cells, record_cells)
                return InMemoryBackend._deep_copy(value, memo, known_cells, record_cells)
            if (value_type is list or value_type is tuple) and len(value) <= _PREMADE_ITEMS_MAX:
                # ``frame, summary, n = build()``: a call's result is a tuple;
                # its frames are copied as `_copy_frame` copies them.
                memo = {}
                InMemoryBackend._premade_copies(dict(enumerate(value)), memo, known_cells, record_cells)
                return InMemoryBackend._deep_copy(value, memo, known_cells, record_cells)
            return InMemoryBackend._deep_copy(value, {}, known_cells, record_cells)
        except (TypeError, pickle.PicklingError, RecursionError, AttributeError) as exc:
            if required:
                # Storing it would hand every caller the SAME object: a caller
                # mutating a hit changes what later calls get, and two threads
                # get one object to mutate at once. Isolation is what makes a cached
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
    def _copy_frame(
        frame: Any,
        known_cells: dict[int, bool] | None = None,
        record_cells: dict[int, bool] | None = None,
        memo: dict[int, Any] | None = None,
    ) -> Any:
        """A copy of a pandas frame/series that no later write can reach.

        Deep, on every store and every hit. A shallow copy is not enough even
        under pandas copy-on-write: copy-on-write covers writes made through
        pandas, but ``s.array`` of any column and ``s.values`` of a nullable
        or categorical column are writable handles to the block itself, so
        ``df["score"].values[0] = 100`` on a returned frame would land in the
        stored entry and in every later hit.

        A deep pandas copy does not copy the Python objects in an object
        column: a list, dict or array in a cell stayed one object shared by
        the entry, the caller and every later hit, so ``df["tags"].iloc[0].append(...)`` changed
        what the next call got. A frame holding such cells is copied through
        pickle, which copies them too.

        Finding out scans every object column (``infer_dtype``, O(rows)):
        22 ms per hit at a million rows. A stored frame is private to its
        entry and never written, so the answer for it cannot change: the store
        records it for the copy it keeps (*record_cells*, by ``id``), and a
        hit reads it back (*known_cells*) instead of scanning again.

        *memo*, when given, receives ``id(cell) -> copy`` for each cell
        copied: a cell list returned beside its frame stays the frame's.
        """
        mutable = known_cells.get(id(frame)) if known_cells is not None else None
        if mutable is None:
            mutable = _holds_mutable_cells(frame)
        copied = None
        if mutable:
            copied = InMemoryBackend._copy_cells(frame, memo)
            if copied is None:
                try:
                    copied = pickle.loads(kept_state.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))
                except Exception:  # noqa: BLE001 - cells that cannot be copied are shared
                    logger.debug("could not copy the cells of a %s", type(frame).__name__)
        if copied is None:
            copied = frame.copy(deep=True)
        if record_cells is not None:
            record_cells[id(copied)] = mutable
        return copied

    @staticmethod
    def _deep_copy(
        value: Any,
        memo: dict[int, Any],
        known_cells: dict[int, bool] | None,
        record_cells: dict[int, bool] | None,
    ) -> Any:
        """A copy of *value* as a disk hit hands it back: a pickle round trip.

        ``copy.deepcopy`` differed from the disk tier in ways a caller sees:
        it trusts a class's ``__deepcopy__``, so one that returns ``self``
        ("immutable, no need to copy") handed every hit the entry's own
        object; it drops the attributes of a subclass of a C type
        (`kept_state` keeps them); it makes a read-only numpy array
        writable; and it recurses a few Python frames per level, so a chain
        of a few hundred linked objects could not be copied at all. The round
        trip is the disk tier's own copy, run in C: about 5x faster than
        ``deepcopy`` on 10,000 small dataclasses. Array data goes out of band
        and is copied once, read-only staying read-only.

        What *memo* already holds (``id(original) -> copy``, `_premade_copies`)
        and every frame met on the way are not pickled but put into the copy
        as they are: a pandas frame copied by `_copy_frame` (its cells a
        container at a time, and a cell list returned beside the frame still
        the frame's), a polars one cloned.

        A value pickle refuses (a lambda, a lock: no disk tier can hold it
        either) is copied by ``deepcopy`` (`_deepcopy_with_frames`).
        """
        frames = _frame_types()
        ndarray = getattr(sys.modules.get("numpy"), "ndarray", None)

        def persistent_id(obj: Any) -> int | None:
            obj_type = type(obj)
            if obj_type in _ATOMS:
                return None
            key = id(obj)
            if key in memo:
                return key
            if obj_type is ndarray and not obj.dtype.hasobject:
                memo[key] = _copy_array(obj)  # its data alone: no pickling of its dtype and shape
                return key
            if frames and isinstance(obj, frames):
                if _is_pandas_frame(type(obj)):
                    memo[key] = InMemoryBackend._copy_frame(obj, known_cells, record_cells, memo)
                else:
                    memo[key] = _copy_polars(obj)
                return key
            return None

        try:
            buffers: list[pickle.PickleBuffer] = []
            stream = kept_state.dumps(
                value,
                pickle.HIGHEST_PROTOCOL,
                buffer_callback=buffers.append,
                persistent_id=persistent_id if memo or frames or ndarray else None,
            )
            unpickler = pickle.Unpickler(
                io.BytesIO(stream),
                buffers=[bytes(b) if memoryview(b).readonly else bytearray(b) for b in buffers],
            )
            unpickler.persistent_load = memo.__getitem__
            return unpickler.load()
        except Exception:  # noqa: BLE001 - whatever pickle refuses, deepcopy may copy
            logger.debug("could not copy a %s by pickle; deepcopy instead", type(value).__name__, exc_info=True)
        return InMemoryBackend._deepcopy_with_frames(value, memo, known_cells, record_cells)

    @staticmethod
    def _deepcopy_with_frames(
        value: Any,
        memo: dict[int, Any],
        known_cells: dict[int, bool] | None,
        record_cells: dict[int, bool] | None,
    ) -> Any:
        """``copy.deepcopy(value, memo)``, with every pandas frame in it copied
        as `_copy_frame` copies it, wherever it sits.

        deepcopy copies a frame with ``DataFrame.__deepcopy__``, which leaves
        the lists and dicts in its object columns shared. `_premade_copies`
        reaches the frames of the common shapes before the copy; one held by
        an object, a nested tuple or a deep dict is found in deepcopy's own
        memo afterwards (it keeps every original it copied alive there). If
        one of those holds mutable cells, the value is copied again with each
        such frame copied by `_copy_frame` up front.
        """
        before = dict(memo)
        copied = copy.deepcopy(value, memo)
        if "pandas" not in sys.modules:
            return copied
        frames = [item for item in memo.get(id(memo), ()) if _is_pandas_frame(type(item))]
        if not frames:
            return copied
        cells = {}
        for frame in frames:
            mutable = known_cells.get(id(frame)) if known_cells is not None else None
            cells[id(frame)] = _holds_mutable_cells(frame) if mutable is None else mutable
        if not any(cells.values()):
            if record_cells is not None:
                for frame in frames:
                    record_cells[id(memo[id(frame)])] = False
            return copied
        for frame in frames:
            before[id(frame)] = InMemoryBackend._copy_frame(frame, cells, record_cells)
        return copy.deepcopy(value, before)

    @staticmethod
    def _copy_cells(frame: Any, memo: dict[int, Any] | None = None) -> Any:
        """A copy of *frame* whose list cells are new lists, without pickling it.

        A column of ``(action, datetime)`` lists is plain data, and a pickle
        round trip of it pays per leaf: 2.5 s a store and again a hit at
        800,000 rows. Its cells are copied one container at a time instead
        (`_plain_data.spine_copy`); a column of immutables (tuples of
        numbers, strings) is shared as it is. None when a cell is not plain
        data, or one list sits in two cells: the pickle round trip then
        copies it, with the sharing kept.
        """
        try:
            if _mutable_labels(frame):
                return None  # only the pickle round trip copies those
            is_series = getattr(frame, "ndim", 2) == 1
            positions = [0] if is_series else [i for i, dtype in enumerate(frame.dtypes) if str(dtype) == "object"]
            columns = [frame] if is_series else [frame.iloc[:, i] for i in positions]
            cells = [column.tolist() for column in columns]
            writable = [n for n, column_cells in enumerate(cells) if not _plain_data.immutable_below(column_cells)]
            flat = [cell for n in writable for cell in cells[n]]
            copied_flat = _plain_data.spine_copy(flat, memo)
            if memo is not None:
                memo.pop(id(flat), None)  # this list is ours, and gone on return
            if copied_flat is None:
                return None
            copied = frame.copy(deep=True)
            import pandas as pd

            offset = 0
            for n in writable:
                values = pd.Series(copied_flat[offset : offset + len(cells[n])], dtype=object).array
                offset += len(cells[n])
                if is_series:
                    # In place: a new Series would drop its attrs, its flags
                    # and a subclass, which a disk hit keeps.
                    copied.iloc[:] = values
                    return copied
                copied.isetitem(positions[n], values)
            return copied
        except Exception:
            logger.debug("could not copy the cells of a %s cell by cell", type(frame).__name__, exc_info=True)
            return None

    @staticmethod
    def _premade_copies(
        value: dict,
        memo: dict[int, Any],
        known_cells: dict[int, bool] | None = None,
        record_cells: dict[int, bool] | None = None,
        depth: int = 0,
    ) -> None:
        """Put a copy of each plain container in *value*'s dicts into *memo*.

        The frames first: a list that is also one of their cells is then
        found in the memo as the frame's copy of it.
        """
        for item in value.values():
            if _is_pandas_frame(type(item)) and id(item) not in memo:
                memo[id(item)] = InMemoryBackend._copy_frame(item, known_cells, record_cells, memo)
        for item in value.values():
            item_type = type(item)
            if _is_pandas_frame(item_type):
                pass
            elif item_type is dict:
                if depth < 4:
                    InMemoryBackend._premade_copies(item, memo, known_cells, record_cells, depth + 1)
            elif (item_type is tuple or item_type is list) and id(item) not in memo:
                if _plain_data.immutable_below(item):
                    memo[id(item)] = item if item_type is tuple else list(item)
                else:
                    # Nested plain data (a parsed log: rows holding lists of
                    # tuples). Every list it holds goes into the memo too, so
                    # a name bound to one of them still shares it with the copy.
                    _plain_data.spine_copy(item, memo)

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, without counting an access. See `BaseBackend.peek_metadata`."""
        with self._lock:
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
        with self._lock:
            return self._store.get(key)

    def get_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, counted as an access the way `get` counts one.

        Not through ``get()``, which deep-copies the value only to drop it --
        the upstream simulation reads many entries' metadata.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            metadata = entry[0]
            metadata["last_access"] = time.time()
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata.setdefault("source", self.source_label)
            return metadata

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None, None
            metadata, value = entry

            metadata["last_access"] = time.time()
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata.setdefault("source", self.source_label)
            self._touch(key)
            immutable_below = key in self._immutable_below
            dict_rows = key in self._dict_rows
            known_cells = self._frame_cells.get(key)

        if immutable_below:
            # Checked when it was stored; the stored value is private.
            return metadata, (list(value) if type(value) is list else value)
        if dict_rows:
            return metadata, list(map(dict, value))
        return metadata, self._safe_deep_copy(value, key, known_cells=known_cells)

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
            size = memory_footprint(value)
        else:
            size = plain[0]

        # A value above the eviction target can never stay: the byte cap evicts
        # down to 90% of the cap, so storing it would evict every older entry
        # and then the value itself -- one oversized write, or one restore of a
        # big disk entry (read-repair promotes into this tier with no size
        # gate), would empty the tier. Refused here, before the copy: copying a frame of
        # gigabytes only to throw it away is its own cost. The previous value
        # for the key goes too, or a later read would serve it as current.
        if self._max_size_bytes is not None and size > self._max_size_bytes * 0.9:
            with self._lock:
                self._drop(key)
            return False

        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        frame_cells: dict[int, bool] = {}
        if dict_rows_size is not None:
            # csv.DictReader / JSON records with immutable values: a new dict
            # per row is a complete copy, built in C, instead of a deepcopy.
            immutable = False
            stored = list(map(dict, value))
        elif plain is None:
            immutable = False
            # Only for a decorator entry, where the stored value IS what the
            # next call hands back. A notebook statement's payload is the
            # variables a cell left behind, and one unisolatable variable among
            # them (an open handle in scope) must not stop the statement being
            # cached -- the notebook re-executes what it cannot restore.
            stored = self._safe_deep_copy(
                value, key, required=bool((metadata or {}).get("copy_required")), record_cells=frame_cells
            )
        else:
            _size, immutable, levels = plain
            stored = _plain_data.copy_plain(value, immutable, levels)[1]
        metadata["size"] = size

        with self._lock:
            # Byte-cap bookkeeping: on replacement, discount the old entry's size
            # before recording the new one so the running total stays accurate.
            previous = self._store.get(key)
            if previous is not None:
                self._current_size_bytes -= previous[0].get("size", 0)
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
            if frame_cells:
                self._frame_cells[key] = frame_cells
            else:
                self._frame_cells.pop(key, None)
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
        """Remove *key*, keeping the byte-cap running total in sync. Called
        with the lock held."""
        self._immutable_below.discard(key)
        self._dict_rows.discard(key)
        self._frame_cells.pop(key, None)
        self._gdsf_base.pop(key, None)
        self._seq_by_key.pop(key, None)
        entry = self._store.pop(key, None)
        if entry is not None:
            self._current_size_bytes -= entry[0].get("size", 0)

    def delete(self, key: str) -> None:
        with self._lock:
            self._drop(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._immutable_below.clear()
            self._dict_rows.clear()
            self._frame_cells.clear()
            self._gdsf_base.clear()
            self._seq_by_key.clear()
            self._current_size_bytes = 0
            self._pressure_floor = None
            self._pressure_percent = None
        # Also try to free memory back to OS
        self._try_malloc_trim()

    def list_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [meta for meta, _ in self._store.values()]

    def entry_count(self) -> int:
        return len(self._store)

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        with self._lock:
            keys_to_delete = [key for key, (meta, _) in self._store.items() if is_expired(meta)]
            for key in keys_to_delete:
                self._drop(key)

        if keys_to_delete:
            self._try_malloc_trim()

        return len(keys_to_delete)

    def _check_and_evict(self) -> None:
        """Check memory usage and evict items if threshold is exceeded."""
        mem = _memory_reading()
        if mem is None:
            return
        if mem.percent / 100.0 > self.max_memory_percent:
            self._evict(mem)
        else:
            # The episode is over: the next one takes a fresh share.
            self._pressure_floor = None
            self._pressure_percent = None

    def _evict(self, mem: Any) -> None:
        """Give back this tier's SHARE of the machine's memory pressure.

        Same order as the byte cap (`_evict_to_byte_cap`): least value per
        byte first. Scoring by ``execution_time * access_count / size`` would
        put every entry not yet read at zero, ahead of a 1 ms one read once.

        **How much.** Not until the whole machine falls under the target:
        when the pressure is someone else's that never happens, and every
        check would empty the tier -- costing its user everything and the
        machine nothing. It sheds its share: ``overshoot * own / in_use`` bytes, where
        ``own`` is this tier's footprint. A tier that IS most of the memory in
        use sheds nearly the whole overshoot; one that holds a sliver of it
        sheds a sliver.

        **Once per episode.** Under pressure that never relents, taking the
        share again on every check (one per ``check_interval`` writes) is the
        same drain, geometrically -- ~15% per check across the hundreds of
        writes one sweep makes. So the first check of an episode takes the
        share, records the footprint it left, and later checks in the same
        episode only hold the tier at that level: new entries displace the
        least valuable old ones rather than the tier shrinking again. The
        episode ends at the first check that finds no pressure.
        """
        self._shed(self._pressure_share(mem, self.max_memory_percent * 0.9))

    def _pressure_share(self, mem: Any, target_percent: float) -> float:
        """Bytes this tier should give back now.

        The first check of a pressure episode: its proportional share of the
        overshoot. Every later one: whatever it has grown past the level the
        first one left. Neither goes below `_PRESSURE_KEEPS_BYTES`.
        """
        total = mem.total
        percent = mem.percent
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
        # and refusing to shed again would let the machine swap.
        self._pressure_percent = percent
        in_use = total * percent / 100.0
        overshoot = in_use - total * target_percent
        if overshoot <= 0 or in_use <= 0:
            return 0.0
        return min(float(max(0, own - self._PRESSURE_KEEPS_BYTES)), overshoot * min(1.0, own / in_use))

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
        self._pressure_floor = max(self._current_size_bytes, self._PRESSURE_KEEPS_BYTES)

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
        Recency alone would treat a 30 s result like a 50 ms one of the same
        size, and 4 MB like 1 MB (see benchmarks/eviction_sim). Ties go to the
        least recently touched.

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


#: Whether a failed memory reading has been logged in this process.
_READING_FAILURE_LOGGED: list[bool] = []


def _memory_reading() -> Any | None:
    """psutil's reading of the machine's memory, or None when there is none.

    The pressure check is advice about memory, never part of a store: the
    value is already in the tier when it runs, on every ``check_interval``-th
    write. So a reading that fails -- whatever it raises -- skips the check
    rather than failing that write. A reading without a numeric percentage
    and total counts as none. Logged once per process: a check that cannot
    run stays that way, and a line per ten writes would say nothing new.
    """
    try:
        mem = psutil.virtual_memory()
        percent = mem.percent
    except Exception as exc:  # noqa: BLE001 - see docstring: a probe never fails a write
        if not _READING_FAILURE_LOGGED:
            _READING_FAILURE_LOGGED.append(True)
            logger.warning(
                "cash: could not read the machine's memory (%s: %s); the RAM tier skips its memory-pressure check",
                type(exc).__name__,
                exc,
            )
        return None
    total = getattr(mem, "total", None)
    for number in (percent, total):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return None
    return mem if total > 0 else None


#: What ``pandas.api.types.infer_dtype`` calls an object column whose cells
#: are all immutable (str, bytes, numbers, dates, Decimal...), so a copy of
#: the column may share them. Anything else ("mixed", "unknown-array", ...)
#: may hold a list, dict or array.
_IMMUTABLE_CELLS = frozenset(
    {
        "empty",
        "string",
        "bytes",
        "integer",
        "floating",
        "mixed-integer-float",
        "decimal",
        "complex",
        "boolean",
        "datetime64",
        "datetime",
        "date",
        "timedelta64",
        "timedelta",
        "time",
        "period",
    }
)


#: Types `InMemoryBackend._deep_copy`'s pickler copies itself, without a look.
_ATOMS = frozenset({str, int, float, bool, type(None), bytes, complex})


def _copy_array(array: Any) -> Any:
    """A copy of a numpy array of numbers, read-only if it was, as a
    pickle round trip makes it."""
    copied = array.copy(order="K")
    if not array.flags.writeable:
        copied.flags.writeable = False
    return copied


def _copy_polars(frame: Any) -> Any:
    """A copy of a polars frame or series: a clone, which shares its immutable
    buffers, with the Python objects of an ``Object`` column copied too.

    A clone shares those objects with the entry and every hit, and pickle
    refuses an ``Object`` column, so no disk tier holds one: the RAM entry
    is the only copy there is.
    """
    import polars as pl

    copied = copy.deepcopy(frame)
    columns = [copied] if isinstance(copied, pl.Series) else copied.get_columns()
    fresh = [
        pl.Series(column.name, InMemoryBackend._deep_copy(column.to_list(), {}, None, None), dtype=pl.Object)
        for column in columns
        if column.dtype == pl.Object
    ]
    if not fresh:
        return copied
    return fresh[0] if isinstance(copied, pl.Series) else copied.with_columns(fresh)


def _frame_types() -> tuple[type, ...]:
    """The pandas and polars frame and series classes, of those imported.

    Never imports either: a value cannot hold a frame of a library that
    is not imported.
    """
    modules = (sys.modules.get("pandas"), sys.modules.get("polars"))
    if _FRAME_TYPES and _FRAME_TYPES[0] == modules:
        return _FRAME_TYPES[1]
    found: list[type] = []
    complete = True
    for module in modules:
        frame, series = getattr(module, "DataFrame", None), getattr(module, "Series", None)
        if isinstance(frame, type) and isinstance(series, type):
            found += [frame, series]
        elif module is not None:
            complete = False  # still importing: asked again next time
    if complete:
        _FRAME_TYPES[:] = [modules, tuple(found)]
    return tuple(found)


#: `_frame_types`' last answer: ``[modules, types]``.
_FRAME_TYPES: list = []


def _is_pandas_frame(value_type: type) -> bool:
    """A pandas DataFrame or Series, or any subclass of one.

    By class, not by name: polars, cudf and others call their frames
    ``DataFrame`` too, and have no ``copy(deep=...)``. A subclass defined
    outside pandas (a user's own, geopandas') is a frame too: its deep copy
    shares the lists in its cells just the same. Never imports pandas: a
    value cannot hold a frame before pandas is imported.
    """
    pd = sys.modules.get("pandas")
    try:
        return issubclass(value_type, (pd.DataFrame, pd.Series))  # type: ignore[union-attr]
    except (AttributeError, TypeError):  # no pandas, or one still importing
        return False


def _holds_mutable_cells(frame: Any) -> bool:
    """Does a DataFrame or Series hold a Python object that can be changed in place?

    Object columns can, and so can the labels (`_mutable_labels`); each is
    classified by pandas' C-level ``infer_dtype``, not a Python loop over
    its cells.
    """
    try:
        from pandas.api.types import infer_dtype

        if getattr(frame, "ndim", 2) == 1:
            columns = [frame] if str(frame.dtype) == "object" else []
        else:
            columns = [frame.iloc[:, i] for i, dtype in enumerate(frame.dtypes) if str(dtype) == "object"]
        return any(infer_dtype(column, skipna=True) not in _IMMUTABLE_CELLS for column in columns) or (
            _mutable_labels(frame)
        )
    except Exception:  # noqa: BLE001 - cannot tell: the plain deep copy
        return False


def _mutable_labels(frame: Any) -> bool:
    """Does a frame's index, column index or a categorical's categories hold
    a Python object that can be changed in place?

    A deep pandas copy copies these arrays but not the objects in them: a
    hashable object with mutable state (a sensor keyed by name that carries
    its calibration) stayed one object shared by the entry and every hit.
    """
    from pandas.api.types import infer_dtype

    indexes = [frame.index] if getattr(frame, "ndim", 2) == 1 else [frame.index, frame.columns]
    dtypes = [frame.dtype] if getattr(frame, "ndim", 2) == 1 else list(frame.dtypes)
    arrays = []
    for index in indexes:
        arrays.extend(getattr(index, "levels", None) or [index])
    dtypes += [array.dtype for array in arrays]
    arrays += [dtype.categories for dtype in dtypes if str(dtype) == "category"]
    return any(
        str(array.dtype) == "object" and infer_dtype(array, skipna=True) not in _IMMUTABLE_CELLS for array in arrays
    )
