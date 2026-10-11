"""In-memory cache backend with LRU eviction."""

from __future__ import annotations

import builtins
import copy
import ctypes
import datetime
import decimal
import functools
import gc
import io
import logging
import pickle
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from .. import _plain_data, kept_state
from . import frame_sharing
from .._lazy_module import LazyModule
from ..sizing import memory_footprint
from ..value_types import IMMUTABLE_PRIMS
from ._base import CacheBackend, MetadataDict, gdsf_value
from .serialization import Serializer

psutil = LazyModule("psutil")  # imported on first use: ~11 ms off `import cash`

logger = logging.getLogger(__name__)

__all__ = ["NO_PRIVATE_COPY", "InMemoryBackend"]

#: Metadata key: this tier's copy of the value is not a stand-in for the
#: value the caller stored (a closed file, stored by name, copies back into
#: a closed file, which does not pickle), so `InMemoryBackend.private_copy`
#: never offers it to another tier.
NO_PRIVATE_COPY = "no_private_copy"


#: A list or tuple with at most this many items has the frames in it
#: copied as `_copy_frame` copies them, not deep (`_safe_deep_copy`).
_PREMADE_ITEMS_MAX = 64


#: `InMemoryBackend._frame_cells` verdict for a pandas table stored frozen
#: (`frame_sharing`): a hit hands out a shallow copy of it.
_SHARED = "shared"


#: `InMemoryBackend._frame_cells` verdict for a table stored as
#: `_TableWithBytes`: only a copy reads it out.
_BYTES = "bytes"


#: `InMemoryBackend._copy_plans` value for an entry the first hit plans.
_PLAN_ON_FIRST_HIT = object()

#: `InMemoryBackend._copy_holding_bytes` for an entry it cannot hand out.
_UNSERVABLE = object()

#: JSON-like data with at least this many containers is kept as `marshal`
#: bytes (`_Marshalled`). Below it the copies a container at a time cost
#: little, and a hit hands back what `_plain_data.copy_plan` copies.
_MARSHAL_NODES_MIN = 4096


class _Marshalled:
    """A stored value kept as `marshal` bytes (`_plain_data.marshal_dumps`):
    a hit reads a new copy out of them, at C speed.

    JSON-like data -- parsed records, an index of lists -- was copied a
    container at a time on the store and again on every hit: 1.7 s and
    2.3 s for a million nested records, more than building them took.
    Kept as bytes, the store writes it once (0.2 s) and a hit reads it
    (0.4 s), and the tier holds 50 MB where the copy took 700 MB.
    """

    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __deepcopy__(self, memo: dict) -> Any:
        # A copy that falls back to deepcopy (`_deepcopy_with_frames`) reads
        # a part kept as bytes out, as the pickle round trip does.
        return _plain_data.marshal_loads(self.data)

    def __reduce__(self) -> tuple:
        # The pickle round trip of `_deep_copy` reads it out too when it
        # runs without its persistent_id hook.
        return _plain_data.marshal_loads, (self.data,)


def _marshals(facts: Any) -> bool:
    """Is JSON-like data with these `_plain_data.TreeFacts` kept as marshal
    bytes: big enough, and nothing in it marshal would give back otherwise?"""
    return facts.nodes >= _MARSHAL_NODES_MIN and facts.types <= _plain_data.MARSHAL_TYPES


#: `_marshal_parts` looks this far into dicts, and into dicts this small:
#: a notebook entry's payload, its ``variables``.
_PARTS_DEPTH = 2
_PARTS_KEYS = 32


def _without(value: dict, parts: dict[int, Any], depth: int = 0) -> dict:
    """*value* with each of *parts* (`_marshal_parts`) in its dicts as None:
    what is left to size once the parts are sized by their own walk. A
    parent walked to size it would walk the part again."""
    rest = {}
    for name, item in value.items():
        if id(item) in parts:
            item = None
        elif type(item) is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
            item = _without(item, parts, depth + 1)
        rest[name] = item
    return rest


def _read_parts(value: dict, memo: dict[int, Any], depth: int = 0) -> None:
    """``memo[id(part)] = what it holds`` for each `_Marshalled` where
    `_marshal_parts` puts them: in *value*'s dicts, as far as it looks."""
    for item in value.values():
        kind = type(item)
        if kind is _Marshalled:
            if id(item) not in memo:
                memo[id(item)] = _plain_data.marshal_loads(item.data)
        elif kind is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
            _read_parts(item, memo, depth + 1)


def _marshal_parts(value: dict, depth: int = 0, found: dict | None = None) -> dict[int, tuple[Any, Any]]:
    """``{id(part): (part, facts)}`` for the values of *value*, and of the
    small dicts in it, that are kept as marshal bytes (`_marshals`).

    Only a part nothing outside it reaches into (``held_once``): a list in
    it that another variable also names must stay that variable's list. The
    part itself may be held twice -- two names for one list -- since one
    `_Marshalled` stands for it wherever the copy meets it.
    """
    found = {} if found is None else found
    for item in value.values():
        kind = type(item)
        if kind is not dict and kind is not list and kind is not tuple or id(item) in found:
            continue
        facts = _plain_data.tree_facts(item)
        if facts is not None and facts.held_once and _marshals(facts):
            found[id(item)] = (item, facts)
        elif kind is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
            # A namespace of variables, one of which another name holds.
            _marshal_parts(item, depth + 1, found)
    return found


class _TableWithBytes:
    """A pandas table stored with its object columns of lists and dicts kept
    as `marshal` bytes (`_plain_data.marshal_dumps`), the way big JSON-like
    values are (`_Marshalled`).

    Such cells were copied one container at a time on the store and on every
    hit: 1.6 s and 1.4 s for a million rows of two-item lists, more than
    building the table. Written once as bytes, a hit reads new cells out of
    them at C speed. *table* is the rest of it, with those columns holding
    None, kept as any table is (shared when it can be frozen).
    """

    __slots__ = ("columns", "table")

    def __init__(self, table: Any, columns: tuple[int, ...], data: bytes) -> None:
        self.table = table
        self.columns = (columns, data)

    @staticmethod
    def make(frame: Any, record_cells: dict) -> _TableWithBytes | None:
        """*frame* kept this way, or None when one of its columns of objects
        holds anything but `_plain_data.MARSHAL_TYPES`, or a list or dict that
        something outside its cell also holds (that sharing must survive the
        copy: the cell copy keeps it), or its labels hold such objects."""
        try:
            import numpy as np

            if _mutable_labels(frame):
                return None
            is_series = getattr(frame, "ndim", 2) == 1
            positions = (0,) if is_series else tuple(i for i, dtype in enumerate(frame.dtypes) if str(dtype) == "object")
            lists = [frame.to_numpy().tolist()] if is_series else [frame.iloc[:, i].to_numpy().tolist() for i in positions]
            writable = tuple(n for n, cells in enumerate(lists) if not _plain_data.immutable_below(cells))
            if not writable:
                return None
            columns = [lists[n] for n in writable]
            del lists
            # Each cell is held by the table's array and its column's list
            # (and, for several columns, by the list of them all).
            if len(columns) == 1:
                facts = _plain_data.held_items_facts(columns[0], 1)
            else:
                facts = _plain_data.held_items_facts([cell for cells in columns for cell in cells], 2)
            if facts is None or not facts.held_once:
                return None
            data = _plain_data.marshal_dumps(columns, facts)
            if data is None:
                return None
            del columns
            empty = np.full(len(frame), None, dtype=object)
            if is_series:
                rest = frame._constructor(empty, index=frame.index, name=frame.name).__finalize__(frame)
            else:
                rest = frame.copy(deep=False)
                for n in writable:
                    rest.isetitem(positions[n], empty.copy())
            # Its other object columns passed `immutable_below`: no scan again.
            rest = InMemoryBackend._copy_frame(rest, {id(rest): False}, record_cells)
            return _TableWithBytes(rest, tuple(positions[n] for n in writable), data)
        except Exception:  # noqa: BLE001 - copied a cell at a time instead
            logger.debug("could not keep a %s's cells as bytes", type(frame).__name__, exc_info=True)
            return None

    def read_out(self, known_cells: dict | None) -> Any:
        """A new table: *table* copied as stored, the cells read out of the bytes."""
        copied = InMemoryBackend._copy_frame(self.table, known_cells)
        positions, data = self.columns
        columns = _plain_data.marshal_loads(data)
        if getattr(copied, "ndim", 2) == 1:
            # In place: a new Series would drop its attrs, its flags and a
            # subclass, which a disk hit keeps.
            copied.iloc[:] = _object_array(columns[0])
            return copied
        for position, cells in zip(positions, columns):
            copied.isetitem(position, _object_array(cells))
        return copied

    def __deepcopy__(self, memo: dict) -> Any:
        return self.read_out(None)

    def __reduce__(self) -> tuple:
        return _read_out_table, (self.table, self.columns)


def _read_out_table(table: Any, columns: tuple) -> Any:
    kept = _TableWithBytes(table, *columns)
    return kept.read_out(None)


def _object_array(cells: list) -> Any:
    """A numpy object array of *cells*, each a cell: a list stays a list,
    where ``np.array`` would make a dimension of it."""
    import numpy as np

    try:
        return np.fromiter(cells, dtype=object, count=len(cells))
    except (TypeError, ValueError):  # numpy before 1.23
        array = np.empty(len(cells), dtype=object)
        for i, cell in enumerate(cells):
            array[i] = cell
        return array


class _Unservable(Exception):
    """A part kept as bytes sits inside an object no copy can be made of:
    it cannot be read out, so the entry cannot be handed out at all."""


def _reaches_marshalled(value: Any) -> bool:
    """Does *value* hold a `_Marshalled` anywhere below it?

    For an object the hit could not copy (`InMemoryBackend._copy_holding_bytes`),
    whose parts no copy read out. Walks what the collector sees it refer to,
    never into classes, modules or code, which a stored part never sits in.
    """
    stop = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.CodeType)
    seen: builtins.set[int] = set()
    todo = [value]
    while todo:
        obj = todo.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if type(obj) is _Marshalled or type(obj) is _TableWithBytes:
            return True
        if type(obj) in _ATOMS or isinstance(obj, stop):
            continue
        todo.extend(gc.get_referents(obj))
    return False


class InMemoryBackend(CacheBackend):
    """Entries held in this process's memory, gone when the process ends.

    A hit returns a copy, so changing it does not change the entry (a pandas
    table: a shallow copy of frozen data, `frame_sharing`). Entries
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
        self._frame_cells: dict[str, dict[int, bool | None]] = {}
        #: Per key, how a hit copies a dict stored through `spine_copy` --
        #: a notebook entry, a JSON-like result -- part by part
        #: (`_plain_data.copy_plan`). Decided once, by the first hit: the
        #: stored value is private and never written.
        self._copy_plans: dict[str, Any] = {}
        #: Keys whose stored value holds parts kept as marshal bytes
        #: (`_marshal_parts`): only a copy reads them out.
        self._holds_bytes: builtins.set[str] = set()
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
        by_reference: list[bool] | None = None,
        by_spine: list[bool] | None = None,
        walk: list | None = None,
        premade: dict[int, Any] | None = None,
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

        *walk* is `_plain_data.tree_walk` of *value*, when the caller has it.
        *premade* is what to put in the copy for parts of a dict *value*,
        ``id(part) -> what``: the parts kept as marshal bytes (`_marshal_parts`).
        """
        try:
            value_type = type(value)
            if _is_pandas_frame(value_type):
                return InMemoryBackend._copy_frame(value, known_cells, record_cells)
            if value_type is _TableWithBytes:
                return value.read_out(known_cells)
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
            if premade is None and (value_type is dict or value_type is list or value_type is tuple):
                # JSON-like data -- an index, records, a dict of lists: one
                # step per container, where deepcopy took one per leaf.
                copied = _plain_data.spine_copy(value, walk=walk)
                if copied is not None:
                    if by_spine is not None:
                        by_spine.append(True)
                    return copied
            if value_type is dict:
                # A notebook entry is dicts around the values, and carries the
                # RNG state: `random.getstate()` is a tuple of 625 ints, which
                # deepcopy would walk an int at a time. Its plain
                # parts go into deepcopy's memo as already copied: a tuple is
                # shared, a list of immutables gets a new list, and two names
                # for one object still come back as one object.
                memo: dict[int, Any] = dict(premade) if premade else {}
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
            if by_reference is not None:
                by_reference.append(True)
            return value

    @staticmethod
    def _copy_frame(
        frame: Any,
        known_cells: dict[int, bool] | None = None,
        record_cells: dict[int, bool] | None = None,
        memo: dict[int, Any] | None = None,
    ) -> Any:
        """A copy of a pandas frame/series that no later write can reach.

        Under pandas copy-on-write, a table of numbers, dates and text is
        copied once, when stored, and that copy -- cash's own, which no caller
        holds -- is frozen and shared: every hit hands out a shallow copy of
        it (`frame_sharing`). A deep copy on every hit cost 0.2 s per 250 MB,
        most of an unchanged Run All. Copy-on-write alone is not enough --
        ``s.array`` of any column and ``s.values`` of a nullable or
        categorical column are writable handles to the block itself -- so
        the frozen data is read-only and marked shared for good. The
        caller's own table is never frozen: cash does not change what an
        object the caller holds allows. Any other table (nullable or
        categorical columns, mutable labels, a subclass) is copied deep, on
        the store and on every hit.

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
        if mutable is _SHARED:
            # Frozen when stored (`frame_sharing`): a shallow copy, which
            # pandas copies before any write, is as independent as a deep one.
            return frame_sharing.hand_out(frame)
        if record_cells is not None and frame_sharing.enabled() and frame_sharing.frozen_already(frame):
            # Stored: frozen by cash already (a hit, or a table just read
            # from disk, `promote`), so no holder can write its data: kept
            # by a shallow copy, nothing copied and nothing changed.
            stored = frame_sharing.hand_out(frame)
            record_cells[id(stored)] = _SHARED
            return stored
        if mutable is None:
            mutable = _holds_mutable_cells(frame)
        copied = None
        if mutable and record_cells is not None:
            # Stored: its columns of lists and dicts kept as bytes, the rest
            # as any other table.
            kept = _TableWithBytes.make(frame, record_cells)
            if kept is not None:
                record_cells[id(kept)] = _BYTES
                return kept
        if mutable:
            copied = InMemoryBackend._copy_cells(frame, memo)
            if copied is None:
                try:
                    copied = _round_trip(frame)
                except Exception:  # noqa: BLE001 - cells that cannot be copied are shared
                    logger.debug("could not copy the cells of a %s", type(frame).__name__)
                    # Recorded as None: a store that must isolate its value
                    # refuses it (`set`), as it does a bare uncopiable object.
                    mutable = None
        if copied is None:
            copied = frame.copy(deep=True)
        if record_cells is not None:
            if mutable is False and frame_sharing.freeze(copied, cells_known=True):
                mutable = _SHARED  # private, now frozen: hits share it
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
        #: Every object copied here, kept alive until the copy is done: *memo*
        #: is keyed by ``id``, and an array or frame that a ``__getstate__``
        #: builds for the pickle is freed once written, so the next one could
        #: get its address and be handed the first one's copy.
        alive: list[Any] = []

        def persistent_id(obj: Any) -> int | None:
            obj_type = type(obj)
            if obj_type in _ATOMS:
                return None
            key = id(obj)
            if key in memo:
                return key
            if obj_type is ndarray and not obj.dtype.hasobject:
                memo[key] = _copy_array(obj)  # its data alone: no pickling of its dtype and shape
                alive.append(obj)
                return key
            if obj_type is _Marshalled:
                memo[key] = _plain_data.marshal_loads(obj.data)  # a part kept as bytes, read out
                alive.append(obj)
                return key
            if obj_type is _TableWithBytes:
                memo[key] = obj.read_out(known_cells)
                alive.append(obj)
                return key
            if record_cells is not None and getattr(obj_type, "_cash_stored_as_is", False):
                # Stored as it is: an immutable stand-in whose copy is
                # something else (a closed file's `ClosedStream` copies into
                # the closed file, which no disk tier can write). A hit
                # copies it as usual.
                memo[key] = obj
                return key
            if frames and isinstance(obj, frames):
                if _is_pandas_frame(type(obj)):
                    memo[key] = InMemoryBackend._copy_frame(obj, known_cells, record_cells, memo)
                else:
                    memo[key] = _copy_polars(obj)
                alive.append(obj)
                return key
            return None

        current_names: set[int] = set()

        def renamed_id(obj: Any) -> int | None:
            if isinstance(obj, _BY_NAME) and id(obj) not in current_names and id(obj) not in memo:
                current = _named_now(obj)
                if current is None:
                    current_names.add(id(obj))  # pickled by its name, as it is
                else:
                    memo[id(obj)] = current
            return persistent_id(obj)

        # A class pickles by name, and only while its module still names
        # that very class: once a notebook re-runs the cell defining it, a
        # stored instance's class is the old one. A disk hit looks the class
        # up by name and gets the new one; so does the second attempt here.
        stores = record_cells is not None  # `_cash_stored_as_is` is looked for
        for hook in (persistent_id if memo or frames or ndarray or stores else None, renamed_id):
            try:
                buffers: list[pickle.PickleBuffer] = []
                stream = kept_state.dumps(
                    value,
                    pickle.HIGHEST_PROTOCOL,
                    buffer_callback=buffers.append,
                    persistent_id=hook,
                )
                unpickler = pickle.Unpickler(
                    io.BytesIO(stream),
                    buffers=[bytes(b) if memoryview(b).readonly else bytearray(b) for b in buffers],
                )
                unpickler.persistent_load = memo.__getitem__
                return unpickler.load()
            except pickle.PicklingError:
                logger.debug("could not copy a %s by pickle", type(value).__name__, exc_info=True)
            except Exception:  # noqa: BLE001 - whatever pickle refuses, deepcopy may copy
                logger.debug("could not copy a %s by pickle; deepcopy instead", type(value).__name__, exc_info=True)
                break
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
            elif type(item) is _TableWithBytes and id(item) not in memo:
                memo[id(item)] = item.read_out(known_cells)
        for item in value.values():
            item_type = type(item)
            if _is_pandas_frame(item_type):
                pass
            elif item_type is dict:
                # JSON-like data (a variable holding an index, a namespace of
                # such variables) is copied whole; any other dict is looked
                # into, and so is one holding what *memo* has already (a
                # part kept as bytes): copying it would walk that part.
                if id(item) in memo:
                    continue
                holds_premade = len(item) <= _PREMADE_ITEMS_MAX and not memo.keys().isdisjoint(map(id, item.values()))
                if (holds_premade or _plain_data.spine_copy(item, memo) is None) and depth < 4:
                    InMemoryBackend._premade_copies(item, memo, known_cells, record_cells, depth + 1)
            elif (item_type is tuple or item_type is list) and id(item) not in memo:
                if _plain_data.immutable_below(item):
                    memo[id(item)] = item if item_type is tuple else list(item)
                else:
                    # Nested plain or JSON-like data (a parsed log: rows
                    # holding lists of tuples; records). Every list and dict
                    # it holds goes into the memo too, so a name bound to one
                    # of them still shares it with the copy.
                    _plain_data.spine_copy(item, memo)

    def promote(self, key: str, value: Any, metadata: MetadataDict) -> bool:
        """Keep *value*, just read from a slower tier, when that costs no copy;
        False, with nothing kept, when it would.

        The value is the caller's alone: nothing else has seen it. Its pandas
        tables are frozen where they are (`frame_sharing`) and kept as they
        are, so a restore of a 400 MB table costs no second copy. A value the
        store would have to copy -- an array, records, an object -- is not
        kept on its first read (`TieredBackend.get` keeps it on the second):
        a restored value is mostly read once per process, and copying it
        took longer than reading it from disk.
        """
        if not _cheap_to_keep(value):
            return False
        for frame in _frames_in(value):
            if not frame_sharing.freeze(frame):
                return False
        return self.set(key, value, metadata) is not False

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, without counting an access. See `BaseBackend.peek_metadata`."""
        with self._lock:
            entry = self._store.get(key)
            return dict(entry[0]) if entry is not None else None

    def peek_entry(self, key: str, *, value: bool = True) -> tuple[MetadataDict, Any] | None:
        """The stored metadata and value themselves: not copied, not counted.

        For writing the entry to another tier as it is
        (``TieredBackend.persist_from_memory``), where a copy of a large frame
        would be pure waste. Both are this tier's own objects: never change
        the value; where the entry is stored (``storage``, ``persist_skipped``)
        is all the caller may update in the metadata, once it is stored elsewhere too.
        A value kept as marshal bytes, or holding parts kept so
        (`_Marshalled`), is read out of them, into a copy
        (`_copy_holding_bytes`): the bytes never leave the tier. None when a
        part sits in what no copy can be made of. Unless *value* is False:
        then the value is None.
        """
        with self._lock:
            entry = self._store.get(key)
        if entry is not None and not value:
            return entry[0], None
        if entry is not None and type(entry[1]) is _Marshalled:
            return entry[0], _plain_data.marshal_loads(entry[1].data)
        if entry is not None and key in self._holds_bytes:
            copied = self._copy_holding_bytes(entry[1], key, self._frame_cells.get(key))
            return None if copied is _UNSERVABLE else (entry[0], copied)
        return entry

    def private_copy(self, key: str, default: Any = None, metadata: MetadataDict | None = None) -> Any:
        """The stored value for *key* to write to another tier, when it is a
        copy only this tier holds, which nothing changes; else *default*.
        With *metadata*, only the value stored with that very dict (the
        write that just stored it, not an earlier or a later one).

        Every stored value is that -- a hit copies it, or shares a frozen
        table (`frame_sharing`) -- except one kept by reference (it could not
        be copied: the caller holds it), one holding parts kept as bytes
        (reading those out is a full copy), and one whose metadata says
        `NO_PRIVATE_COPY` (its copy is not what the caller's value stores
        as). A value kept whole as bytes is read out into a new object,
        which no one else holds.
        """
        with self._lock:
            entry = self._store.get(key)
            holds_bytes = key in self._holds_bytes
        if entry is None or holds_bytes or entry[0].get("by_reference") or entry[0].get(NO_PRIVATE_COPY):
            return default
        if metadata is not None and entry[0] is not metadata:
            return default
        if type(entry[1]) is _Marshalled:
            return _plain_data.marshal_loads(entry[1].data)
        return entry[1]

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
            plan = self._copy_plans.get(key)
            holds_bytes = key in self._holds_bytes

        if type(value) is _Marshalled:
            return metadata, _plain_data.marshal_loads(value.data)
        if holds_bytes:
            # Parts kept as bytes: only a copy reads them out, never a plan
            # or the stored value itself.
            copied = self._copy_holding_bytes(value, key, known_cells)
            if copied is _UNSERVABLE:
                with self._lock:
                    entry = self._store.get(key)
                    if entry is not None and entry[1] is value:
                        self._drop(key)
                return None, None
            return metadata, copied
        if immutable_below:
            # Checked when it was stored; the stored value is private.
            return metadata, (list(value) if type(value) is list else value)
        if dict_rows:
            return metadata, list(map(dict, value))
        if plan is _PLAN_ON_FIRST_HIT:
            plan = _plain_data.copy_plan(value)
            with self._lock:
                entry = self._store.get(key)
                if entry is not None and entry[1] is value:
                    self._copy_plans[key] = plan
        if plan is not None:
            copied = _plain_data.copy_by_plan(value, plan)
            if copied is not None:
                return metadata, copied
        return metadata, self._safe_deep_copy(value, key, known_cells=known_cells)

    def _copy_holding_bytes(self, value: Any, key: str, known_cells: dict[int, bool] | None) -> Any:
        """A copy of a stored *value* holding parts kept as marshal bytes
        (`_marshal_parts`) with every part read out, or `_UNSERVABLE`.

        The copy of the whole reads them out wherever they sit. When it
        cannot be made -- the store's copy turned something into what no
        copy takes again: a closed file a ``with`` left bound, stored as a
        closed file -- the stored value itself must not go out, with the
        bytes in it. Its dicts are rebuilt instead, the parts in them read
        out once each, and every other value in them copied on its own,
        with those reads in the memo: a tuple holding the records still
        holds the records the name does. A value that cannot be copied is
        handed out as it is, as the whole would have been -- unless a part
        sits inside it, which nothing can read out: then `_UNSERVABLE`.
        """
        # Each part read out once, for the copy of the whole and, should
        # that fail, for the copy a value at a time.
        memo: dict[int, Any] = {}
        if type(value) is dict:
            _read_parts(value, memo)
        fell_back: list[bool] = []
        copied = self._safe_deep_copy(value, key, known_cells=known_cells, by_reference=fell_back, premade=memo)
        if not fell_back:
            return copied
        if type(value) is not dict:
            return _UNSERVABLE  # a table kept with bytes that could not be read out
        try:
            return self._copy_spine(value, memo, known_cells)
        except _Unservable:
            logger.debug("a part kept as bytes sits in what cannot be copied: %r is not served", key)
            return _UNSERVABLE

    @staticmethod
    def _copy_spine(value: dict, memo: dict[int, Any], known_cells: dict[int, bool] | None, depth: int = 0) -> dict:
        """*value*'s dicts (as far as `_marshal_parts` looks) rebuilt, and
        each other value in them copied on its own (`_copy_holding_bytes`)."""
        copied = {}
        for name, item in value.items():
            if id(item) in memo:
                copied[name] = memo[id(item)]
            elif type(item) is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
                copied[name] = memo[id(item)] = InMemoryBackend._copy_spine(item, memo, known_cells, depth + 1)
            elif type(item) in _ATOMS:
                copied[name] = item
            else:
                # Its own memo: a copy that fails half-way leaves half-made
                # copies in it, which another value must not be given.
                trial = dict(memo)
                try:
                    copied[name] = InMemoryBackend._deep_copy(item, trial, known_cells, None)
                except Exception:  # noqa: BLE001 - what cannot be copied is shared, as the whole was
                    if _reaches_marshalled(item):
                        raise _Unservable from None
                    copied[name] = item
                else:
                    memo.update(trial)
                memo[id(item)] = copied[name]  # two names for it still share it
        return copied

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> bool | None:
        """Store *value*; returns False if it was refused (see below)."""
        metadata = self._init_metadata(metadata, key)

        # Plain data is sized, checked and copied from ONE look at it: three
        # separate walks were most of promoting two million parsed rows here.
        plain = _plain_data.profile(value)
        dict_rows_size = None if plain is not None else _plain_data.dict_rows_profile(value)
        # JSON-like data -- a notebook entry's payload -- is walked once
        # (`tree_facts`, which a notebook's checks may have made already),
        # for its size and how to copy it: big and of marshal's types alone,
        # it is kept as marshal bytes; else copied a container at a time,
        # over that walk (inside a `one_look`, which keeps no walk, over a
        # new one). A dict that is not JSON-like as a whole -- a payload
        # with numpy's RNG state beside the variables -- has such parts kept
        # as bytes (`_marshal_parts`), and the rest copied as before.
        facts = walk = parts = None
        if plain is None and dict_rows_size is None:
            if _plain_data.in_a_look():
                facts = _plain_data.tree_facts(value)
            else:
                facts, walk = _plain_data.tree_facts_and_walk(value)
            if facts is None and type(value) is dict:
                parts = _marshal_parts(value)
        whole = facts is not None and _marshals(facts)
        if whole:
            walk = None  # not copied a container at a time
        if dict_rows_size is not None:
            size = dict_rows_size
        elif facts is not None:
            size = facts.size
        elif parts:
            # Each part sized by its walk, the rest as before.
            size = memory_footprint(_without(value, parts)) + sum(part_facts.size for _part, part_facts in parts.values())
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

        frame_cells: dict[int, bool | None] = {}
        plan = None
        holds_bytes = False
        if dict_rows_size is not None:
            # csv.DictReader / JSON records with immutable values: a new dict
            # per row is a complete copy, built in C, instead of a deepcopy.
            immutable = False
            stored = list(map(dict, value))
        elif whole and (marshalled := _plain_data.marshal_dumps(value, facts)) is not None:
            immutable = False
            stored = _Marshalled(marshalled)
        elif plain is None:
            immutable = False
            premade = None
            if parts:
                premade = {}
                for part_id, (part, part_facts) in parts.items():
                    data = _plain_data.marshal_dumps(part, part_facts)
                    if data is not None:
                        premade[part_id] = _Marshalled(data)
            elif facts is not None and walk is None:
                walk = _plain_data.tree_walk(value)  # for the copy, a container at a time
            # Only for a decorator entry, where the stored value IS what the
            # next call hands back. A notebook statement's payload is the
            # variables a cell left behind, and one unisolatable variable among
            # them (an open handle in scope) must not stop the statement being
            # cached -- the notebook re-executes what it cannot restore.
            required = bool((metadata or {}).get("copy_required"))
            fell_back: list[bool] = []
            by_spine: list[bool] = []
            stored = self._safe_deep_copy(
                value,
                key,
                required=required,
                record_cells=frame_cells,
                by_reference=fell_back,
                by_spine=by_spine,
                walk=walk,
                premade=premade,
            )
            holds_bytes = (bool(premade) or _BYTES in frame_cells.values()) and not fell_back
            if by_spine and type(stored) is dict:
                # Nothing in it is reached twice: its parts can be copied
                # each on its own, each the fastest way it allows. Planned
                # by the first hit: the plan walks every leaf again (3 ms of
                # a 200,000-int list's 15 ms store), and most entries are
                # never read.
                plan = _PLAN_ON_FIRST_HIT
            if required and None in frame_cells.values():
                # A frame whose object cells pickle cannot copy (a worker
                # holding a lock) would share those cells with every hit.
                raise CacheBackendError(
                    "the result holds a pandas frame whose object cells could not be copied, "
                    "so caching it would hand every caller the same objects"
                )
            if fell_back or None in frame_cells.values():
                # Said on the entry, so a reader that must not hand out the
                # stored object itself (a notebook call or statement hit) can
                # refuse it: every hit returns this very object, or a frame
                # sharing its object cells with it.
                metadata["by_reference"] = True
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
            if plan is not None:
                self._copy_plans[key] = plan
            else:
                self._copy_plans.pop(key, None)
            if holds_bytes:
                self._holds_bytes.add(key)
            else:
                self._holds_bytes.discard(key)
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
        self._copy_plans.pop(key, None)
        self._holds_bytes.discard(key)
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
            self._copy_plans.clear()
            self._holds_bytes.clear()
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
IMMUTABLE_CELLS = frozenset(
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


#: Exact types of a cell no one can change in place, for an object array
#: `infer_dtype` calls "mixed": ints beside strings in an index a row was
#: added to by label (``df.loc['action_time'] = ...``), floats (NaN) beside
#: dates. Exact, since a subclass of str can carry a ``__dict__``.
_IMMUTABLE_CELL_TYPES: frozenset[type] = frozenset(
    {
        int,
        float,
        complex,
        bool,
        str,
        bytes,
        type(None),
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        decimal.Decimal,
    }
)


def immutable_cells(array: Any) -> bool:
    """Is every cell of the object *array* a value no one can change in place?

    ``infer_dtype`` answers at C speed for one kind of cell; for a mix it
    says only "mixed", and the cells' exact types are read instead, also at
    C speed. A 1.7-million-row frame with such an index was copied through a
    pickle round trip for the RAM tier: 6 s of a 4.4 s statement.
    """
    from pandas.api.types import infer_dtype

    kind = infer_dtype(array, skipna=True)
    if kind in IMMUTABLE_CELLS:
        return True
    if kind not in ("mixed", "mixed-integer", "mixed-integer-float"):
        return False
    return _immutable_cell_types().issuperset(map(type, array))


@functools.cache
def _immutable_cell_types() -> frozenset[type]:
    """`_IMMUTABLE_CELL_TYPES` and pandas' and numpy's scalars."""
    import numpy as np
    import pandas as pd

    return (
        _IMMUTABLE_CELL_TYPES
        | {pd.Timestamp, pd.Timedelta, type(pd.NaT), np.datetime64, np.timedelta64}
        | {t for t in np.sctypeDict.values() if issubclass(t, np.number) or t is np.bool_}
    )


#: `_cheap_to_keep` looks this far into a value, at containers this small.
_CHEAP_DEPTH = 3
_CHEAP_ITEMS = 64
#: An array at most this big is copied for the RAM tier on its first read.
_CHEAP_ARRAY_BYTES = 1 << 20


def _cheap_to_keep(value: Any, depth: int = 0) -> bool:
    """Can the RAM tier keep *value* without copying much of it: atoms,
    small arrays and pandas tables it can freeze (`frame_sharing`), in a few
    small dicts, lists and tuples (a notebook entry's payload)?"""
    kind = type(value)
    if kind in _ATOMS:
        return True
    if _is_pandas_frame(kind):
        return frame_sharing.enabled() and frame_sharing.freezable(value)
    if kind.__name__ == "ndarray" and kind is getattr(sys.modules.get("numpy"), "ndarray", None):
        return not value.dtype.hasobject and value.nbytes <= _CHEAP_ARRAY_BYTES
    if kind in (dict, list, tuple) and depth < _CHEAP_DEPTH and len(value) <= _CHEAP_ITEMS:
        items = value.values() if kind is dict else value
        if kind is dict and not all(type(name) in _ATOMS for name in value):
            return False
        return all(_cheap_to_keep(item, depth + 1) for item in items)
    return False


def _frames_in(value: Any, depth: int = 0) -> list:
    """The pandas tables `_cheap_to_keep` found in *value*."""
    kind = type(value)
    if _is_pandas_frame(kind):
        return [value]
    if kind in (dict, list, tuple) and depth < _CHEAP_DEPTH:
        items = value.values() if kind is dict else value
        return [frame for item in items for frame in _frames_in(item, depth + 1)]
    return []


#: Types `InMemoryBackend._deep_copy`'s pickler copies itself, without a look.
_ATOMS = frozenset({str, int, float, bool, type(None), bytes, complex})


#: What pickle stores as a reference by name, checked against what that name holds.
_BY_NAME = (type, types.FunctionType)


def _named_now(obj: Any) -> Any:
    """The class or function *obj*'s module now names where *obj* was
    defined, when that is another object of the same kind; else None.

    What a pickle round trip from disk resolves the name to. A notebook cell
    re-run defines a new class under the old name, and pickle refuses an
    instance of the old one ("not the same object as __main__.Fit").
    """
    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    qualname = getattr(obj, "__qualname__", None)
    if module is None or not isinstance(qualname, str) or "<locals>" in qualname:
        return None
    current: Any = module
    for part in qualname.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    if current is obj:
        return None
    if isinstance(obj, type) and isinstance(current, type):
        return current
    if isinstance(obj, types.FunctionType) and isinstance(current, types.FunctionType):
        return current
    return None


def _round_trip(value: Any) -> Any:
    """*value* through pickle and back, a class its module has since
    redefined replaced by the one it names now (`_named_now`)."""
    try:
        return pickle.loads(kept_state.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except pickle.PicklingError:
        renamed: dict[int, Any] = {}
        current_names: set[int] = set()

        def renamed_id(obj: Any) -> int | None:
            key = id(obj)
            if key in renamed:
                return key
            if isinstance(obj, _BY_NAME) and key not in current_names:
                current = _named_now(obj)
                if current is not None:
                    renamed[key] = current
                    return key
                current_names.add(key)
            return None

        stream = kept_state.dumps(value, pickle.HIGHEST_PROTOCOL, persistent_id=renamed_id)
        unpickler = pickle.Unpickler(io.BytesIO(stream))
        unpickler.persistent_load = renamed.__getitem__
        return unpickler.load()


def _copy_array(array: Any) -> Any:
    """A copy of a numpy array of numbers, read-only if it was, as a
    pickle round trip makes it. A big one times the copy (`copy_seconds`)."""
    if array.nbytes < _TIMED_COPY_BYTES:
        copied = array.copy(order="K")
    else:
        started = _copy_clock()
        copied = array.copy(order="K")
        _note_copy_speed(array.nbytes, _copy_clock() - started)
    if not array.flags.writeable:
        copied.flags.writeable = False
    return copied


#: An array copy at least this big is timed (`_note_copy_speed`).
_TIMED_COPY_BYTES = 1 << 20
#: Bytes per second this process copied its last big arrays at, newest last
#: (`_copy_clock`): the copy speed of THIS machine and its memory, which a
#: fitted model cannot know (a hit of an 80 MB array measured 0.2-0.8 s on
#: one machine, against 0.02 s fitted). Empty until measured.
_COPY_SPEED: list[float] = []
def _fine_thread_clock() -> Any:
    """The clock a copy is timed with: this thread's CPU time, where it
    resolves a microsecond; else the wall clock (Windows: 15.6 ms ticks).

    CPU time, because a copy that waited for a core says nothing about the
    copy: the wall time of the cell it would replace waited too.
    """
    try:
        if time.get_clock_info("thread_time").resolution <= 1e-6 and _ticks_finely(time.thread_time):
            return time.thread_time
    except (AttributeError, ValueError, OSError):
        pass
    return time.perf_counter


def _ticks_finely(clock: Any) -> bool:
    """Whether *clock* really moves within half a millisecond of busy work.

    The reported resolution is not enough: Windows reports 1e-07 for
    `thread_time`, but GetThreadTimes moves in 15.6 ms ticks, so a copy timed
    with it reads 0 s.
    """
    deadline = time.perf_counter() + 5e-4
    first = clock()
    while time.perf_counter() < deadline:
        if clock() != first:
            return True
    return False


_copy_clock = _fine_thread_clock()

#: How many readings `copy_seconds` takes the fastest of. The fastest, not
#: the mean: one copy caught behind other work says little about the next,
#: and a cell's caching should not flip with every burst of load.
_COPY_READINGS = 8


def _note_copy_speed(nbytes: int, seconds: float) -> None:
    if seconds <= 0:
        return
    _COPY_SPEED.append(nbytes / seconds)
    del _COPY_SPEED[:-_COPY_READINGS]


def copy_seconds(nbytes: int) -> float:
    """Seconds this process takes to copy *nbytes* of array data, measured
    (a probe copy the first time nothing has been measured yet)."""
    if not _COPY_SPEED:
        import numpy as np

        probe = np.ones(_PROBE_BYTES // 8)
        for _ in range(2):  # the first touches fresh pages
            _copy_array(probe)
        if not _COPY_SPEED:  # the clock did not move during the probe
            started = time.perf_counter()
            probe.copy(order="K")
            _note_copy_speed(probe.nbytes, max(time.perf_counter() - started, 1e-9))
    return nbytes / max(_COPY_SPEED)


_PROBE_BYTES = 8 << 20

#: What a hit of a table the tier shares costs (`frame_sharing`): a shallow
#: copy, whatever its size.
SHARED_HIT_SECONDS = 1e-4


def hit_seconds(value: Any, size_bytes: int) -> float | None:
    """What handing out a RAM hit of *value* (about *size_bytes*) costs in
    this process, or None when the tier cannot say: a table it shares costs
    a shallow copy; an array or a table it copies, a copy at the measured
    speed (`copy_seconds`)."""
    kind = type(value)
    try:
        if _is_pandas_frame(kind):
            dtypes = [value.dtype] if value.ndim == 1 else list(value.dtypes)
            if any(str(dtype) == "object" for dtype in dtypes):
                return None  # its cells decide (`_holds_mutable_cells`): not scanned here
            if frame_sharing.enabled() and frame_sharing.freezable(value):
                return SHARED_HIT_SECONDS
            return copy_seconds(size_bytes)
        if kind.__name__ == "ndarray" and kind is getattr(sys.modules.get("numpy"), "ndarray", None):
            if not value.dtype.hasobject:
                return copy_seconds(size_bytes)
    except Exception:  # noqa: BLE001 - an estimate it cannot make: the fitted one
        logger.debug("no measured hit cost for a %s", kind.__name__, exc_info=True)
    return None


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
        if getattr(frame, "ndim", 2) == 1:
            columns = [frame] if str(frame.dtype) == "object" else []
        else:
            columns = [frame.iloc[:, i] for i, dtype in enumerate(frame.dtypes) if str(dtype) == "object"]
        return any(not immutable_cells(column) for column in columns) or (
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
    indexes = [frame.index] if getattr(frame, "ndim", 2) == 1 else [frame.index, frame.columns]
    dtypes = [frame.dtype] if getattr(frame, "ndim", 2) == 1 else list(frame.dtypes)
    arrays = []
    for index in indexes:
        arrays.extend(getattr(index, "levels", None) or [index])
    dtypes += [array.dtype for array in arrays]
    arrays += [dtype.categories for dtype in dtypes if str(dtype) == "category"]
    return any(
        str(array.dtype) == "object" and not immutable_cells(array) for array in arrays
    )
