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
import itertools
import logging
import pickle
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

from cash.exceptions import CacheBackendError

from .. import _native, _plain_data, cost_model, held_reads, kept_state, stored_pickle
from .._lazy_module import LazyModule
from ..cost_model import GENERIC_FAMILY
from ..sizing import memory_footprint, pickled_size_estimate
from ..source_reading import read_code_file
from ..value_types import IMMUTABLE_PRIMS
from . import frame_sharing, heap_blocks, iterative_copy, set_order, shared_blocks
from ._base import MEASURED_ROUND_TRIP, CacheBackend, MetadataDict, gdsf_value
from .entry_format import SplitPayload
from .serialization import Serializer

psutil = LazyModule("psutil")  # imported on first use: ~11 ms off `import cash`

logger = logging.getLogger(__name__)

__all__ = ["NO_PRIVATE_COPY", "ROUND_TRIP_CLASS", "InMemoryBackend"]

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


class _Pickled:
    """A stored value kept as the pickle stream and out-of-band buffers a
    disk entry holds (``entry_format.MAGIC_SPLIT``): a hit unpickles a new
    copy out of them, as a disk hit does, without reading a file.

    The tier's copy of an object of a class of its own was a pickle round
    trip on the store, and the unpickling was most of it: a statsmodels
    result rebuilds its design matrices (7 s of a 10 s fit), so a value
    that was never read again cost a second fit. Kept as the bytes, the
    store pays the pickling alone, and the disk tier writes these very
    bytes (`stored_payload`) instead of pickling the value again.

    The bytes are cash's own, made when the value is stored: the stream is
    a new ``bytes`` and each buffer is copied out of the value's memory
    (``tobytes``), so nothing the caller does to the value afterwards --
    in Python or in C -- reaches them. A hit hands out a value over new
    buffers (a read-only one, which numpy will not make writable, shared).
    """

    __slots__ = ("buffers", "readonly", "served", "stream", "top_class")

    def __init__(self, stream: bytes, buffers: list[bytes], readonly: list[bool], top_class: str | None = None) -> None:
        self.stream = stream
        self.buffers = buffers
        self.readonly = readonly
        #: The class (`class_key`) a reading is credited to, for a value that
        #: is an object of a class of its own; None for a notebook entry.
        self.top_class = top_class
        #: Whether a hit has read it out yet (the first one is timed).
        self.served = False

    def load(self) -> Any:
        """A new copy of the value."""
        buffers = [data if ro else bytearray(data) for data, ro in zip(self.buffers, self.readonly)]
        return pickle.loads(self.stream, buffers=buffers)

    def payload(self) -> Any:
        """The bytes as the file tier writes them."""
        return SplitPayload(self.stream, list(self.buffers)) if self.buffers else self.stream

    @property
    def nbytes(self) -> int:
        return len(self.stream) + sum(len(data) for data in self.buffers)

    def __deepcopy__(self, memo: dict) -> Any:
        return self.load()

    def __reduce__(self) -> tuple:
        return _load_pickled, (self.stream, self.buffers, self.readonly)


def _load_pickled(stream: bytes, buffers: list[bytes], readonly: list[bool]) -> Any:
    return _Pickled(stream, buffers, readonly).load()


def _snapshot(value: Any) -> _Pickled | None:
    """*value* pickled as the file tier pickles it (`PickleSerializer.serialize_split`),
    each large buffer copied out of band; None when it does not pickle.

    The pickle made of this very value a moment ago, when there is one, is
    taken, not made again: the one `premade_snapshot` made for a call's
    store, or the one a digest of it made for the store that follows
    (`stored_pickle.kept`). A statement's entry holding a value so digested
    (`stored_pickle.nested`) holds that value's pickle as it is, nested in
    its own (`_snapshot_nesting`); one whose digest ran out of its budget
    has it made now, and raises `stored_pickle.HashBudgetExceeded` past
    what the store may spend on it (`room_past_budget`)."""
    premade = getattr(_PREMADE, "entry", None)
    if premade is not None and premade[0] is value:
        _PREMADE.entry = None
        return premade[1]
    part = stored_pickle.kept(value)
    if part is not None:
        return _pickled(value, part)
    inner = stored_pickle.nested(value) if type(value) is dict else None
    if inner is not None:
        made = _snapshot_nesting(value, *inner)
        if made is not None:
            return made
    try:
        part = stored_pickle.pickle_value(value, keep=True, digest=False)
    except Exception:  # copied the usual way, which says why it cannot be
        logger.debug("could not keep a %s as pickle bytes", type(value).__name__, exc_info=True)
        return None
    return _pickled(value, part)


def _pickled(value: Any, part: stored_pickle.Part) -> _Pickled:
    kind = type(value)
    top_class = None if kind is dict or _copied_as_data(kind) else class_key(kind)
    return _Pickled(part.stream, part.buffers, part.readonly, top_class)


def _snapshot_nesting(value: dict, inner: Any, part: stored_pickle.Part) -> _Pickled | None:
    """*value* pickled with *inner*, a value in it, as *part*, the pickle a
    digest of *inner* made: nested as it is (`stored_pickle.load_part`),
    its buffers handed out of band as they are, with no second pickle of
    *inner*. The caller checked that nothing else in *value* is part of
    *inner* (`stored_pickle.nest`), so the two pickles share no object."""
    ours = {id(data): data for data in part.buffers}

    def override(obj: Any) -> Any:
        if obj is inner:
            return stored_pickle.load_part, (part.stream, *map(pickle.PickleBuffer, part.buffers))
        return NotImplemented

    try:
        made = stored_pickle.pickle_value(
            value,
            keep=True,
            digest=False,
            reducer_override=override,
            kept_buffers=lambda data: ours.get(id(data)) is data,
        )
    except Exception:  # pickled the usual way
        logger.debug("could not nest a kept pickle in a %s", type(value).__name__, exc_info=True)
        return None
    return _pickled(value, made)


#: ``(value, _Pickled)``: the snapshot `premade_snapshot` made for the
#: store about to run on this thread.
_PREMADE = threading.local()


#: The most a stored output's pickle is made for its store when its change
#: check ran out of budget (`room_past_budget`): what one store may copy
#: anyway.
PAST_BUDGET_MAX_BYTES = 2 << 30


def room_past_budget(backend: Any) -> int | None:
    """How many bytes the pickle of a statement's output whose digest ran
    out of its budget may take when the store makes it
    (`stored_pickle.keeping`): what one entry of *backend*'s RAM tier may
    hold (see `InMemoryBackend.set`), at most `PAST_BUDGET_MAX_BYTES`; None
    without a RAM tier first, which is the only tier that takes those
    bytes."""
    tiers = getattr(backend, "backends", None) or [backend]
    first = tiers[0] if tiers else None
    if not isinstance(first, InMemoryBackend):
        return None
    cap = first._max_size_bytes
    return PAST_BUDGET_MAX_BYTES if cap is None else min(int(cap * 0.9), PAST_BUDGET_MAX_BYTES)


def kept_as_pickle(value: Any) -> bool:
    """Would the tier keep *value*, a call's result, as its pickle bytes
    (`_Pickled`), were it big enough: an object of a class of its own?"""
    kind = type(value)
    return kind is not dict and not _copied_as_data(kind)


def premade_snapshot(value: Any) -> stored_pickle.Part | None:
    """`_snapshot` of *value*, made now for the store that follows on this
    thread, which takes it instead of pickling *value* again, as the
    `stored_pickle.Part` it was made as, its digest taken
    (`stored_pickle.object_digest`). None when it does not pickle.

    Only for a store that runs before anything can change *value*: the
    bytes are what it was when they were made. `release_premade` drops one
    the store did not take."""
    try:
        part = stored_pickle.pickle_value(value, keep=True)
    except Exception:  # copied the usual way, which says why it cannot be
        logger.debug("could not keep a %s as pickle bytes", type(value).__name__, exc_info=True)
        _PREMADE.entry = None
        return None
    _PREMADE.entry = (value, _pickled(value, part))
    return part


def release_premade() -> None:
    """Drop the snapshot `premade_snapshot` made, if no store took it."""
    _PREMADE.entry = None


def _at_least(value: Any, nbytes: int, nodes: int = 2048, items: int = 64) -> bool:
    """Do the first objects in *value* -- what its dicts, lists, tuples and
    instance dicts hold, level by level, at most *nodes* of them and
    *items* from each container -- take at least *nbytes* themselves
    (``sys.getsizeof``)? Cheap whatever *value*'s size: a graph's adjacency
    dict alone is a megabyte at 40,000 nodes. True means big; False only
    that these few objects are not."""
    total = 0
    seen: set[int] = set()
    level = [value]
    while level and len(seen) < nodes:
        found: list = []
        for obj in level:
            if id(obj) in seen or len(seen) >= nodes:
                continue
            seen.add(id(obj))
            kind = type(obj)
            if kind.__sizeof__ is not object.__sizeof__ and kind not in _SIZED_CHEAPLY:
                continue  # its own __sizeof__ may read it all (pandas: every string)
            try:
                total += sys.getsizeof(obj)
            except TypeError:
                continue
            if total >= nbytes:
                return True
            if kind is dict:
                found.extend(itertools.islice(obj.values(), items))
            elif kind is list or kind is tuple:
                found.extend(obj[:items])
            elif kind not in _ATOMS and not isinstance(obj, type):
                attrs = getattr(obj, "__dict__", None)
                if type(attrs) is dict:
                    found.extend(itertools.islice(attrs.values(), items))
        level = found
    return False


#: Types whose ``__sizeof__`` `_at_least` calls: builtin containers and atoms.
_SIZED_CHEAPLY = frozenset({dict, list, tuple, set, frozenset, str, bytes, bytearray, int, float})


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


def _marshal_parts(
    value: dict, depth: int = 0, found: dict | None = None, skip: Any = frozenset()
) -> dict[int, tuple[Any, Any]]:
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
        if kind is not dict and kind is not list and kind is not tuple or id(item) in found or id(item) in skip:
            continue
        facts = _plain_data.tree_facts(item)
        if facts is not None and facts.held_once and _marshals(facts):
            found[id(item)] = (item, facts)
        elif kind is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
            # A namespace of variables, one of which another name holds.
            _marshal_parts(item, depth + 1, found, skip)
    return found


#: `_shared_parts` takes a list or tuple of at least this many rows: below
#: it marshal's bytes cost a hit little.
_SHARED_ROWS_MIN = _MARSHAL_NODES_MIN

#: The C walk kernels (`cash._native.kernels`), or None.
_K = _native.kernels


def _shared_parts(value: dict, depth: int = 0, found: dict | None = None) -> dict[int, list | tuple]:
    """``{id(part): part}`` for the big lists and tuples of rows -- tuples
    of strings, numbers and tuples -- in *value* and the small dicts in it
    (`_marshal_parts`' reach): kept as new lists over the same rows, and
    copied so by every hit.

    Read back out of marshal bytes, 300,000 parsed ``(str, str, str)`` rows
    cost a hit 1.3 s, twice what parsing them took; a list over the same
    rows, which nothing can write into, costs milliseconds."""
    if found is None and depth == 0 and _K is not None and type(value) is dict:
        # The same walk in C: a Python step per value of 200,000 record
        # dicts was a third of storing them.
        return _K.shared_parts(value, _SHARED_ROWS_MIN, _PARTS_DEPTH, _PARTS_KEYS, _plain_data.immutable_below)
    found = {} if found is None else found
    for item in value.values():
        kind = type(item)
        if kind is list or kind is tuple:
            if (
                id(item) not in found
                and len(item) >= _SHARED_ROWS_MIN
                and type(item[0]) is tuple
                and _plain_data.immutable_below(item)
            ):
                found[id(item)] = item
        elif kind is dict and depth < _PARTS_DEPTH and len(item) <= _PARTS_KEYS:
            _shared_parts(item, depth + 1, found)
    return found


def _shared_copies(shared: dict[int, list | tuple]) -> dict[int, list | tuple]:
    """A hit's copy of each part `_shared_parts` found, by the stored part's id."""
    return {part_id: list(part) if type(part) is list else part for part_id, part in shared.items()}


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
            positions = (
                (0,) if is_series else tuple(i for i, dtype in enumerate(frame.dtypes) if str(dtype) == "object")
            )
            lists = (
                [frame.to_numpy().tolist()] if is_series else [frame.iloc[:, i].to_numpy().tolist() for i in positions]
            )
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
        except Exception:
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
        #: Per key, the big lists and tuples of immutables in the stored value
        #: (`_shared_parts`), by id: a hit gives each a new list over the same
        #: items.
        self._shared: dict[str, dict[int, list | tuple]] = {}
        #: Keys whose stored value is cash's own by construction: nothing
        #: in it is an object the caller's value also holds (`private_copy`).
        self._own: builtins.set[str] = set()
        #: Keys whose stored value holds pandas tables kept uncopied with
        #: their digest (`frame_sharing.share_checked`), ``key -> [(table,
        #: digest)]``: a hit, and a disk write of it, first checks they still
        #: hash so.
        self._checked: dict[str, list[tuple[Any, str]]] = {}
        #: Keys STORE-RESULT-WRITTEN was said for (`_written_after_store`).
        self._written: builtins.set[str] = set()
        #: Held while tables kept uncopied are taken over (`_adopt_tables`).
        self._adopt_lock = threading.Lock()
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
        #: ``(key, metadata, value)`` of the entry the last `set` stored and
        #: then evicted at once (the byte cap took the new entry itself): its
        #: copy is still the tier's own, and `private_copy` hands it to the
        #: disk write that follows instead of having the caller's value
        #: serialized before the store returns. Released by `release_handoff`.
        self._handoff: tuple[str, MetadataDict, Any] | None = None

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
    def _time_copy(
        value: Any, key: str, *, required: bool, walk: list | None, premade: dict[int, Any] | None
    ) -> float | None:
        """Seconds unpickling another copy of *value* takes, the copy thrown
        away; None when that copy was not a pickle round trip throughout."""
        fell_back: list[bool] = []
        by_deepcopy: list[bool] = []
        _UNPICKLED.seconds = 0.0
        try:
            InMemoryBackend._safe_deep_copy(
                value,
                key,
                required=required,
                record_cells={},
                by_reference=fell_back,
                by_spine=[],
                walk=walk,
                premade=premade,
                by_deepcopy=by_deepcopy,
            )
        except Exception:  # noqa: BLE001 - the first copy stands; this one only times
            return None
        return None if (fell_back or by_deepcopy) else _UNPICKLED.seconds

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
        by_deepcopy: list[bool] | None = None,
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
        ``deepcopy`` of it is a ``clone()``, which shares its buffers --
        possibly the caller's numpy memory. `_copy_polars` copies a
        caller's, and clones the tier's own, about a millisecond whatever
        its size.

        *walk* is `_plain_data.tree_walk` of *value*, when the caller has it.
        *premade* is what to put in the copy for parts of a dict *value*,
        ``id(part) -> what``: the parts kept as marshal bytes (`_marshal_parts`).
        *by_deepcopy* gets True when the copy fell back to ``deepcopy``
        (`_deep_copy`), which trusts what a class's ``__deepcopy__`` returns.
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
                copied = _copy_object_array(value)
                if copied is not None:
                    return copied
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
                return InMemoryBackend._deep_copy(value, memo, known_cells, record_cells, by_deepcopy)
            if (value_type is list or value_type is tuple) and len(value) <= _PREMADE_ITEMS_MAX:
                # ``frame, summary, n = build()``: a call's result is a tuple;
                # its frames are copied as `_copy_frame` copies them.
                memo = {}
                InMemoryBackend._premade_copies(dict(enumerate(value)), memo, known_cells, record_cells)
                return InMemoryBackend._deep_copy(value, memo, known_cells, record_cells, by_deepcopy)
            return InMemoryBackend._deep_copy(value, {}, known_cells, record_cells, by_deepcopy)
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
        reuse = None
        if copied is None:
            arrow = not _stored_frame(frame, known_cells, record_cells)
            checked = getattr(_CHECKED, "frames", None) if record_cells is not None else None
            if (
                checked is not None
                and mutable is False
                and arrow
                and frame.ndim == 2
                and not any(heap_blocks.in_heap(array) for array in frame_sharing.block_arrays(frame))
            ):
                # Stored by `set`: a big table of numbers and dates is kept
                # uncopied, with its digest, checked on every hit
                # (`_still_as_stored`), and adopted once its caller lets go
                # of it (`InMemoryBackend._adopt_tables`).
                kept = frame_sharing.share_checked(frame)
                if kept is not None and _own_metadata(frame, kept[0]):
                    checked.append(kept)
                    record_cells[id(kept[0])] = False
                    return kept[0]
            if record_cells is not None and mutable is False and arrow:
                # Stored: columns the tier already keeps are not copied again.
                reuse = shared_blocks.BlockReuse.for_frame(frame)
            timed = shared_blocks.bytes_to_copy(frame) if record_cells is not None and frame.ndim == 2 else 0
            started = time.perf_counter()
            copied = frame_sharing.copy_keeping_blocks(frame, arrow=arrow, reuse=reuse)
            if timed >= _STORE_TIMED_BYTES:
                _note_store_copy_speed(timed, time.perf_counter() - started)
        if not _own_metadata(frame, copied):
            # An attribute deepcopy refuses: the pickle round trip copies it,
            # as a disk hit would.
            try:
                copied = _round_trip(frame)
            except Exception:  # noqa: BLE001 - shared: recorded so, as uncopiable cells are
                logger.debug("could not copy the attributes of a %s", type(frame).__name__)
                mutable = None
        if record_cells is not None:
            if mutable is False and frame_sharing.freeze(copied, cells_known=True):
                mutable = _SHARED  # private, now frozen: hits share it
            record_cells[id(copied)] = mutable
            if reuse is not None and mutable in (False, _SHARED):
                reuse.keep(copied)
        return copied

    @staticmethod
    def _deep_copy(
        value: Any,
        memo: dict[int, Any],
        known_cells: dict[int, bool] | None,
        record_cells: dict[int, bool] | None,
        by_deepcopy: list[bool] | None = None,
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
        either) is copied by ``deepcopy`` (`_deepcopy_with_frames`), and
        *by_deepcopy*, when given, gets True.
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
            if obj_type is ndarray and (copied := _copy_object_array(obj)) is not None:
                memo[key] = copied
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
                elif record_cells is not None:
                    memo[key] = _store_polars(obj)  # stored: hits clone it
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
                    on_retry=buffers.clear,
                )
                unpickler = pickle.Unpickler(
                    io.BytesIO(stream),
                    buffers=[bytes(b) if memoryview(b).readonly else bytearray(b) for b in buffers],
                )
                unpickler.persistent_load = memo.__getitem__
                started = time.perf_counter()
                try:
                    return unpickler.load()
                finally:
                    _UNPICKLED.seconds = getattr(_UNPICKLED, "seconds", 0.0) + time.perf_counter() - started
            except pickle.PicklingError:
                logger.debug("could not copy a %s by pickle", type(value).__name__, exc_info=True)
            except Exception as exc:
                if kept_state.too_deep(exc):
                    # Deeper than the pickler may recurse: deepcopy recurses
                    # deeper still, so the copy keeps its own stack.
                    return iterative_copy.deep_copy(value, memo, persistent_id)
                logger.debug("could not copy a %s by pickle; deepcopy instead", type(value).__name__, exc_info=True)
                break
        if by_deepcopy is not None:
            by_deepcopy.append(True)
        before = dict(memo)
        if not kept_state.FLAT_FIRST:  # see there: deepcopy too runs out of C stack first
            try:
                return InMemoryBackend._deepcopy_with_frames(value, memo, known_cells, record_cells)
            except (RecursionError, MemoryError) as exc:
                if not kept_state.too_deep(exc):
                    raise
        # What deepcopy left in *memo* is half built: start again from what
        # was there before it.
        memo.clear()
        memo.update(before)
        return iterative_copy.deep_copy(value, memo, persistent_id)

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
        # A subclass's attributes (`_own_metadata`): deepcopy's frame copy
        # leaves them shared too.
        if not any(cells.values()) and not any(_metadata_held(frame) for frame in frames):
            for frame in frames:
                # deepcopy's frame copy keeps the labels and the Arrow data.
                frame_sharing.own_parts(memo[id(frame)], arrow=not _stored_frame(frame, known_cells, record_cells))
                if record_cells is not None:
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
            copied = frame_sharing.copy_keeping_blocks(frame)
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
                if _plain_data.immutable_below(item) or _of_immutable_cells(item):
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
        are, so a restore of a 400 MB table costs no second copy; a polars
        frame is kept as a clone, which shares its memory, polars' own. A value the
        store would have to copy -- an array, records, an object -- is not
        kept on its first read (`TieredBackend.get` keeps it on the second):
        a restored value is mostly read once per process, and copying it
        took longer than reading it from disk.
        """
        if not _cheap_to_keep(value):
            return False
        for frame in _frames_in(value):
            if _is_polars_frame(type(frame)):
                _remember_own_polars(frame)  # read from disk: its store is a clone
            elif not frame_sharing.freeze(frame):
                return False
        return self.set(key, value, metadata) is not False

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, without counting an access. See `BaseBackend.peek_metadata`."""
        held_reads.flush()
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
        if entry is not None and type(entry[1]) is _Pickled:
            try:
                return entry[0], entry[1].load()
            except Exception:  # noqa: BLE001 - as a hit: what cannot be rebuilt is not served
                return None
        if entry is not None and key in self._holds_bytes:
            copied = self._copy_holding_bytes(entry[1], key, self._frame_cells.get(key))
            return None if copied is _UNSERVABLE else (entry[0], copied)
        return entry

    def holds_own_copy(self, key: str) -> bool:
        """Is the value stored for *key* cash's own by construction, so that
        nothing the caller does to its value can change it?

        Only when every part of it was copied in a way that keeps nothing of
        the caller's: plain data, data kept as bytes, a pickle round trip,
        arrays, and frozen tables (`frame_sharing`). Not a table copied deep
        by pandas (a subclass, a nullable, categorical or object column: its
        Python objects stay the caller's), nor what ``deepcopy`` copied, nor
        a value kept by reference, nor tables kept uncopied
        (`checked_copy`). A background write of a value that is not would
        serialize what the caller did to it after the call.
        """
        with self._lock:
            return key in self._own and key not in self._checked

    def may_share_with_caller(self, key: str) -> bool:
        """May the value stored for *key* share a part with the caller's?

        Any value not cash's own (`holds_own_copy`): one kept by reference is
        the caller's, one ``deepcopy`` copied shares what it does not copy
        (a function's closure: a local helper and the generator it draws
        from; whatever a class's ``__deepcopy__`` returns as itself), and a
        table pandas copied deep keeps the Python objects of its object
        cells. A copy of it made later holds what the caller has done to
        those parts since, not what was stored. Tables kept uncopied
        (`checked_copy`) are not counted: their check finds a change, and
        the write is dropped.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            return bool(entry[0].get("by_reference")) or (key not in self._own and key not in self._checked)

    def private_copy(self, key: str, default: Any = None, metadata: MetadataDict | None = None) -> Any:
        """The stored value for *key* to write to another tier, when it is a
        copy only this tier holds, which nothing changes; else *default*.
        With *metadata*, only the value stored with that very dict (the
        write that just stored it, not an earlier or a later one).

        Only a value `holds_own_copy` vouches for, and not one holding parts
        kept as bytes (reading those out is a full copy), nor one whose
        metadata says `NO_PRIVATE_COPY` (its copy is not what the caller's
        value stores as). A value kept whole as bytes is read out into a
        new object, which no one else holds. *default* tells the caller to
        serialize its own value before it returns.
        """
        with self._lock:
            entry = self._store.get(key)
            holds_bytes = key in self._holds_bytes
            own = key in self._own and key not in self._checked
            if entry is None and self._handoff is not None and self._handoff[0] == key:
                # Stored and evicted at once: its copy is still the tier's own.
                entry, holds_bytes, own = self._handoff[1:], False, True
        if entry is None or not own or holds_bytes or entry[0].get("by_reference") or entry[0].get(NO_PRIVATE_COPY):
            return default
        if metadata is not None and entry[0] is not metadata:
            return default
        if type(entry[1]) is _Marshalled:
            return _plain_data.marshal_loads(entry[1].data)
        if type(entry[1]) is _Pickled:
            # Its bytes go to disk as they are (`stored_payload`); reading a
            # copy out of them is the cost they were kept to save.
            return default
        return entry[1]

    def write_copy(self, key: str, value: Any, metadata: MetadataDict, default: Any = None) -> Any:
        """A copy of *value*, cash's own, for a background write to another
        tier when this tier keeps none it can vouch for (`private_copy`):
        it refused the value (over its eviction target, or slower to
        restore than to compute), dropped it, or copied it in a way that
        may keep something of the caller's (a table with nullable or
        ``str`` columns of Python strings: pandas' deep copy shares its
        labels). Else *default*, and the caller serializes its own value
        before it returns.

        Made here, on the caller's thread, for a value whose copy costs
        about a copy of its memory and keeps nothing of the caller's
        (`_write_copy`): tables of numbers, dates, text and nullable
        columns (labels copied too), arrays, atoms, in a few dicts, lists
        and tuples. *key* and *metadata* name the entry, for the log.
        """
        if metadata.get(NO_PRIVATE_COPY):
            return default  # its copy does not store as the value does
        try:
            if not _write_copyable(value):
                return default
            return _write_copy(value, {})
        except Exception:  # serialized by the caller instead, as before
            logger.debug("no copy of %r to write", key, exc_info=True)
            return default

    def checked_copy(self, key: str, metadata: MetadataDict) -> tuple[Any, Any] | None:
        """``(stored value, check)`` for a background write of *key*'s
        entry, stored with that very *metadata*, when its value is the
        tier's own but for pandas tables kept uncopied
        (`frame_sharing.share_checked`); else None. The writer calls
        ``check()`` once the bytes are written: False when the tables no
        longer hash as they did when stored, and what was written may hold
        what the caller wrote since, said by STORE-RESULT-WRITTEN."""
        with self._lock:
            entry = self._store.get(key)
            checked = self._checked.get(key)
            own = key in self._own and key not in self._holds_bytes
        if entry is None or checked is None or not own or entry[0] is not metadata:
            return None
        if entry[0].get("by_reference") or entry[0].get(NO_PRIVATE_COPY):
            return None

        def check() -> bool:
            if _still_as_stored(checked):
                return True
            self._written_after_store(key, metadata)
            return False

        return entry[1], check

    def get_metadata(self, key: str) -> MetadataDict | None:
        """The metadata, counted as an access the way `get` counts one.

        Not through ``get()``, which deep-copies the value only to drop it --
        the upstream simulation reads many entries' metadata.
        """
        held_reads.flush()
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
        metadata, value = self._get_copy(key)
        # The copy's sets iterate as the stored ones do (`set_order`).
        set_order.restore(value, metadata)
        return metadata, value

    def _get_copy(self, key: str) -> tuple[MetadataDict | None, Any | None]:
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
            shared = self._shared.get(key)
            checked = self._checked.get(key)

        if checked is not None and not _still_as_stored(checked):
            # A table kept uncopied shares its memory with the caller's, and
            # the caller wrote it past pandas since (``df["x"].array[0] =
            # ...``): a miss, said once.
            self._written_after_store(key, metadata)
            with self._lock:
                entry = self._store.get(key)
                if entry is not None and entry[1] is value:
                    self._drop(key)
            return None, None
        if checked is not None:
            known_cells = self._adopt_tables(key, value, checked, known_cells)
        if type(value) is _Marshalled:
            return metadata, _plain_data.marshal_loads(value.data)
        if type(value) is _Pickled:
            return self._read_pickled(key, metadata, value)
        if holds_bytes or shared:
            # Parts kept as bytes: only a copy reads them out, never a plan
            # or the stored value itself. Parts kept as lists over shared
            # immutables go into the copy as new lists.
            copied = self._copy_holding_bytes(value, key, known_cells, shared)
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

    def _adopt_tables(
        self, key: str, value: Any, checked: list[tuple[Any, Any]], known_cells: dict[int, Any] | None
    ) -> dict[int, Any] | None:
        """Take over the pandas tables of *key*'s entry kept uncopied
        (`frame_sharing.share_checked`) whose caller let go of their data,
        their digests just checked: frozen, they are the tier's own, and
        this hit and every later one shares them (`frame_sharing.adopt`).
        One the caller still holds stays as it is, and a hit copies it.
        Returns the entry's table verdicts, updated.

        The entry's list of checked values is replaced, never changed: a
        background write of the entry checks the list it was handed, which
        still names the tables it wrote."""
        if known_cells is None or not any(_is_pandas_frame(type(kept)) for kept, _digest in checked):
            return known_cells
        with self._adopt_lock:
            adopted = [kept for kept, _digest in checked if _is_pandas_frame(type(kept)) and frame_sharing.adopt(kept)]
        if not adopted:
            return known_cells
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry[1] is not value or self._checked.get(key) is not checked:
                return known_cells
            cells = dict(known_cells)
            for kept in adopted:
                cells[id(kept)] = _SHARED
            self._frame_cells[key] = cells
            left = [(kept, digest) for kept, digest in checked if not any(kept is done for done in adopted)]
            if left:
                self._checked[key] = left
            else:
                self._checked.pop(key, None)
            return cells

    def _read_pickled(self, key: str, metadata: MetadataDict, kept: _Pickled) -> tuple[MetadataDict | None, Any]:
        """A hit of a value kept as pickle bytes: a new copy read out of them.

        The first one is timed, as the store used to time its round trip:
        the reading goes on the entry (``measured_round_trip_seconds``) and
        to its class (`round_trip_seconds`), so the next store of one is
        judged on it; and a hit no faster than the compute keeps the value
        out of this tier from now on (it is handed out this once). Bytes
        that no longer load (a class gone) are a miss, as on disk."""
        started = time.perf_counter()
        try:
            value = kept.load()
        except Exception:  # a value that cannot be rebuilt here is recomputed
            logger.debug("could not read %r out of its pickle bytes", key, exc_info=True)
            with self._lock:
                entry = self._store.get(key)
                if entry is not None and entry[1] is kept:
                    self._drop(key)
            return None, None
        if not kept.served:
            kept.served = True
            if MEASURED_ROUND_TRIP not in metadata:
                seconds = time.perf_counter() - started
                if _judged_by_compute(metadata) and seconds >= metadata["execution_time"]:
                    # About to drop it: one stalled reading (a full garbage
                    # collection, the writer holding the interpreter lock) is
                    # not enough, so it is read again and the faster kept.
                    seconds = min(seconds, _read_seconds(kept))
                self._learn_reading(key, metadata, kept, seconds)
        return metadata, value

    def _learn_reading(self, key: str, metadata: MetadataDict, kept: _Pickled, seconds: float) -> None:
        """Put what reading *kept* out took on its entry and its class, and
        drop the entry when that is no faster than computing it."""
        size = int(metadata.get("size") or 0)
        metadata[MEASURED_ROUND_TRIP] = seconds
        class_name = kept.top_class
        if class_name is None and not metadata.get("call_refs") and metadata.get("cost_model_family") is not None:
            class_name = metadata.get(ROUND_TRIP_CLASS)
        if class_name is not None:
            note_round_trip(class_name, size, seconds)
        if _judged_by_compute(metadata) and seconds >= metadata["execution_time"]:
            with self._lock:
                entry = self._store.get(key)
                if entry is not None and entry[1] is kept:
                    self._drop(key)

    def release_handoff(self) -> None:
        """Let go of the copy of an entry its own store evicted, which no
        disk write took (`private_copy`)."""
        with self._lock:
            self._handoff = None

    def stored_payload(self, key: str, metadata: MetadataDict | None = None) -> Any:
        """The bytes a value kept as pickle bytes (`_Pickled`) is held in,
        as the file tier writes them; None for any other entry. With
        *metadata*, only the entry stored with that very dict (or one its
        own store evicted at once, `private_copy`)."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None and self._handoff is not None and self._handoff[0] == key:
                entry = self._handoff[1:]
        if entry is None or type(entry[1]) is not _Pickled:
            return None
        if metadata is not None and entry[0] is not metadata:
            return None
        return entry[1].payload()

    def _copy_holding_bytes(
        self, value: Any, key: str, known_cells: dict[int, bool] | None, shared: dict[int, Any] | None = None
    ) -> Any:
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
        memo: dict[int, Any] = _shared_copies(shared) if shared else {}
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
        """Store *value*; returns False if it was refused (see `_set`). Its
        big arrays are kept out of glibc's heap (`heap_blocks`)."""
        with heap_blocks.per_store():
            return self._set(key, value, metadata, serializer)

    def _set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> bool | None:
        """Store *value*; returns False if it was refused (see below)."""
        # Reads the fast hit path held back count in what eviction weighs.
        held_reads.flush()
        metadata = self._init_metadata(metadata, key)
        if set_order.FIELD not in metadata:
            set_order.record(value, metadata)

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
        # Big lists of rows of strings and numbers (`_shared_parts`) are not
        # kept as bytes: a hit gets a new list over the same rows. In an
        # entry JSON-like throughout, the copy a container at a time shares
        # them on the store and the first hit's plan gives each a new list
        # (`copy_plan`); in any other, they are kept beside the rest.
        shared: dict[int, list | tuple] = {}
        if plain is None and dict_rows_size is None:
            if type(value) is dict:
                shared = _shared_parts(value)
            if _plain_data.in_a_look():
                facts = _plain_data.tree_facts(value)
            else:
                facts, walk = _plain_data.tree_facts_and_walk(value)
            if facts is None and type(value) is dict:
                parts = _marshal_parts(value, skip=shared)
        whole = not shared and facts is not None and _marshals(facts)
        if facts is not None:
            shared = {}
        if whole:
            walk = None  # not copied a container at a time
        # An object of a class of its own is kept as its pickle bytes
        # (`_Pickled`), which are made first: what the tier holds is those
        # bytes, so they are its size. Walking the object for its footprint
        # took four times its pickling (1.35 s against 0.35 s for a graph of
        # 350,000 edges), on every store, for a number the bytes give.
        pickled = None
        # An entry holding a value whose pickle its digest made a moment ago
        # (`stored_pickle.nested`) is kept as bytes too: that pickle is made,
        # or, when the digest ran out of its budget, is made for this store.
        nested = type(value) is dict and stored_pickle.nesting(value)
        if (
            plain is None
            and dict_rows_size is None
            and facts is None
            and not parts
            and not shared
            and (nested or _copied_by_round_trip(value, metadata))
        ):
            # Refused before it is pickled when its arrays alone are over the
            # eviction target, or would be slower to read back than to
            # compute (both checked again below, on the bytes).
            estimate = pickled_size_estimate(value)
            over = self._max_size_bytes is not None and estimate > self._max_size_bytes * 0.9
            if over or _slower_than_its_compute(value, estimate, metadata):
                with self._lock:
                    self._drop(key)
                return False
            if nested or estimate >= _ROUND_TRIP_MIN_BYTES or _at_least(value, _ROUND_TRIP_MIN_BYTES):
                # Not for a small value, which is copied as any other: its
                # pickle would be made only to be dropped.
                try:
                    pickled = _snapshot(value)
                except stored_pickle.HashBudgetExceeded:
                    # Its pickle, left to this store, took longer than its
                    # statement is worth (or more bytes than one entry may
                    # hold): not kept, and not copied another way either.
                    with self._lock:
                        self._drop(key)
                    return False
        if pickled is not None and pickled.nbytes >= _ROUND_TRIP_MIN_BYTES:
            size = pickled.nbytes
        elif dict_rows_size is not None:
            size = dict_rows_size
        elif facts is not None:
            size = facts.size
        elif parts:
            # Each part sized by its walk, the rest as before.
            size = memory_footprint(_without(value, parts)) + sum(
                part_facts.size for _part, part_facts in parts.values()
            )
        elif plain is None:
            size = memory_footprint(value)
        else:
            size = plain[0]
        if pickled is not None and size < _ROUND_TRIP_MIN_BYTES:
            pickled = None  # small: copied as any other value (`_safe_deep_copy`)

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

        if _slower_than_its_compute(value, size, metadata):
            # Its class's copies were timed here: a hit would take longer
            # than computing it again. Not copied; the disk tier is asked
            # with that time too (`PersistencePolicy.decide`), and says no.
            with self._lock:
                self._drop(key)
            return False

        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        frame_cells: dict[int, bool | None] = {}
        plan = None
        holds_bytes = False
        checked: list[tuple[Any, str]] = []
        #: Is the stored value cash's own by construction (`private_copy`)?
        own = True
        if dict_rows_size is not None:
            # csv.DictReader / JSON records with immutable values: a new dict
            # per row is a complete copy, built in C, instead of a deepcopy.
            immutable = False
            stored = list(map(dict, value))
        elif whole and (marshalled := _plain_data.marshal_dumps(value, facts)) is not None:
            immutable = False
            stored = _Marshalled(marshalled)
        elif pickled is not None:
            # An object of a class of its own: kept as the bytes its pickle
            # round trip would have read back at once (`_Pickled`). A hit
            # reads them, and the disk tier writes them.
            immutable = False
            stored = pickled
            learned = _learned_round_trip(value, size, metadata)
            if learned is not None:
                metadata[MEASURED_ROUND_TRIP] = learned
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
            marshalled_parts = bool(premade)
            if shared:
                # Stored as new lists over the same rows, by the copy below.
                premade = {**(premade or {}), **_shared_copies(shared)}
            # Only for a decorator entry, where the stored value IS what the
            # next call hands back. A notebook statement's payload is the
            # variables a cell left behind, and one unisolatable variable among
            # them (an open handle in scope) must not stop the statement being
            # cached -- the notebook re-executes what it cannot restore.
            required = bool((metadata or {}).get("copy_required"))
            fell_back: list[bool] = []
            by_spine: list[bool] = []
            by_deepcopy: list[bool] = []
            _UNPICKLED.seconds = 0.0
            _CHECKED.frames = checked = []
            try:
                stored = self._safe_deep_copy(
                    value,
                    key,
                    required=required,
                    record_cells=frame_cells,
                    by_reference=fell_back,
                    by_spine=by_spine,
                    walk=walk,
                    premade=premade,
                    by_deepcopy=by_deepcopy,
                )
            finally:
                _CHECKED.frames = None
            if not (fell_back or by_deepcopy) and size >= _ROUND_TRIP_MIN_BYTES:
                # The unpickling alone: what a disk hit does again, after
                # reading the bytes (the pickling and the array copies here
                # are this tier's own).
                unpickle_seconds = _UNPICKLED.seconds
                if _may_refuse(unpickle_seconds, metadata):
                    # A reading that may keep the value off disk, or out of
                    # this tier, is taken again and the faster one kept: one
                    # unpickling stalled by something else (a full garbage
                    # collection, the writer thread holding the interpreter
                    # lock) read 3.3 s for a graph that unpickles in 0.25 s,
                    # and its 1.4 s build was kept nowhere, so a restart
                    # rebuilt it and everything keyed on it. A class that is
                    # slow to load is slow both times.
                    again = self._time_copy(value, key, required=required, walk=walk, premade=premade)
                    if again is not None:
                        unpickle_seconds = min(unpickle_seconds, again)
                if _copied_by_round_trip(value, metadata):
                    # On the entry, for the disk tier's decision just after
                    # this (`PersistencePolicy.decide`): the copy was a pickle
                    # round trip, and a disk hit unpickles it again.
                    metadata[MEASURED_ROUND_TRIP] = unpickle_seconds
                _note_round_trip(value, size, unpickle_seconds)
                if _hit_no_faster(value, unpickle_seconds, metadata):
                    # A hit is this copy again: no faster than computing it.
                    with self._lock:
                        self._drop(key)
                    return False
            holds_bytes = (marshalled_parts or _BYTES in frame_cells.values()) and not fell_back
            if shared and not fell_back:
                kept = {id(premade[part_id]): premade[part_id] for part_id in shared}
                shared = kept
            # Its own only when every part was copied in a way that leaves
            # nothing of the caller's in it: a pickle round trip, a spine
            # copy, arrays, parts read out of bytes, frozen tables, and
            # tables copied in full (`_frames_own`). A deep pandas copy keeps
            # the Python objects of its object cells (a str subclass with
            # attributes), and ``deepcopy`` trusts a class's
            # ``__deepcopy__``: such a value may still share a part with the
            # caller's.
            # Tables kept uncopied (`frame_sharing.share_checked`) aside:
            # those share memory the caller can still write (`_checked`).
            own = not fell_back and not by_deepcopy and _frames_own(frame_cells, stored)
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
                    "the result holds a pandas frame whose object cells or attributes could not be copied, "
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
            if shared and own:
                self._shared[key] = shared
            else:
                self._shared.pop(key, None)
            if own:
                self._own.add(key)
            else:
                self._own.discard(key)
            if checked:
                self._checked[key] = checked
            else:
                self._checked.pop(key, None)
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

            # Evicted by its own store: kept for the disk write that follows.
            evicted_at_once = key not in self._store and own and not checked and not metadata.get("by_reference")
            self._handoff = (key, metadata, stored) if evicted_at_once and not holds_bytes else None

    def _written_after_store(self, key: str, metadata: MetadataDict) -> None:
        """Say that a table *key*'s entry kept uncopied was written since its
        store, once per entry: the entry is dropped, and the call runs again.
        It was dropped with only a debug log, so a caller writing its result
        past pandas never had it cached and never heard why."""
        with self._lock:
            if key in self._written:
                return
            self._written.add(key)
        name = next((metadata.get(f) for f in ("func_name", "function") if metadata.get(f)), None)
        what = f"the result of {name}" if name else f"the cached value {key!r}"
        from ..diagnostics import warn_diagnostic
        from ..exceptions import CashCacheStoreFailedWarning

        warn_diagnostic(
            CashCacheStoreFailedWarning,
            "STORE-RESULT-WRITTEN",
            f"{what} holds a pandas table cash kept without copying it, and "
            f"its data was written after it was stored, past pandas "
            f"(``.array[i] = ...``, ``.to_numpy()`` written, or a numpy array "
            f"the table was made over). The entry was dropped, so the next "
            f"call computes it again.",
            "write a copy of the result rather than the result itself, or "
            "write it through pandas (``df.loc[i, 'x'] = ...``), which "
            "copies before it writes.",
            location=("<cash>", 1),
        )

    def _drop(self, key: str) -> None:
        """Remove *key*, keeping the byte-cap running total in sync. Called
        with the lock held."""
        self._immutable_below.discard(key)
        self._dict_rows.discard(key)
        self._frame_cells.pop(key, None)
        self._copy_plans.pop(key, None)
        self._holds_bytes.discard(key)
        self._shared.pop(key, None)
        self._own.discard(key)
        self._checked.pop(key, None)
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
            self._shared.clear()
            self._own.clear()
            self._checked.clear()
            self._gdsf_base.clear()
            self._seq_by_key.clear()
            self._current_size_bytes = 0
            self._pressure_floor = None
            self._pressure_percent = None
        # Also try to free memory back to OS
        self._try_malloc_trim()

    def list_entries(self) -> list[dict[str, Any]]:
        held_reads.flush()
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
        held_reads.flush()
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


class _MeminfoReading(NamedTuple):
    """The two numbers of psutil's ``virtual_memory()`` the RAM tier reads."""

    total: int
    percent: float


def _meminfo_reading() -> _MeminfoReading | None:
    """``virtual_memory()``'s total and percent, read from ``/proc/meminfo``
    as psutil reads them on Linux, or None where that file says otherwise.

    psutil takes ``MemTotal`` and ``MemAvailable`` (kB) and gives the percent
    in use as ``(total - available) / total``, rounded to one decimal. Read
    directly, a process need not import psutil (16-27 ms) at its first store.
    A kernel without ``MemAvailable`` (before 3.14), one that reports none
    available or more than the total (psutil estimates those from other
    lines), or a reading that does not parse, leaves the answer to psutil.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        # Untracked (`read_code_file`): cash's own read, never a statement's input.
        fields: dict[bytes, int] = {}
        for line in read_code_file("/proc/meminfo").splitlines():
            name, _, rest = line.partition(b":")
            if name in (b"MemTotal", b"MemAvailable"):
                number, unit = rest.split()
                if unit != b"kB":
                    return None
                fields[name] = int(number) * 1024
        total, available = fields[b"MemTotal"], fields[b"MemAvailable"]
    except Exception:  # noqa: BLE001 - no /proc, or a format it does not have: psutil answers
        return None
    if not 0 < available <= total:
        return None
    return _MeminfoReading(total, round((total - available) / total * 100, 1))


def _virtual_memory() -> Any:
    """The machine's memory, with psutil's ``total`` and ``percent``: from
    ``/proc/meminfo`` on Linux (`_meminfo_reading`), from psutil elsewhere."""
    reading = _meminfo_reading()
    return reading if reading is not None else psutil.virtual_memory()


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
        mem = _virtual_memory()
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


def _of_immutable_cells(items: list | tuple) -> bool:
    """Is *items* flat, of values no one can change in place, pandas' and
    numpy's scalars among them (`_immutable_cell_types`)? Then a new list is
    a complete copy. ``list(df['action_time'])``, 1.7 million Timestamps, was
    copied through a pickle round trip for the RAM tier: 16 s after a 2.9 s
    statement. Asked of the exact types at C speed, and only once pandas is
    imported: before that, no such scalar exists."""
    if "pandas" not in sys.modules:
        return False
    return _immutable_cell_types().issuperset(map(type, items))


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
    small arrays, pandas tables it can freeze (`frame_sharing`) and polars
    frames, in a few small dicts, lists and tuples (a notebook entry's
    payload)?"""
    kind = type(value)
    if kind in _ATOMS:
        return True
    if _is_pandas_frame(kind):
        return frame_sharing.enabled() and frame_sharing.freezable(value)
    if _is_polars_frame(kind):
        # Its store is a clone (`_own_polars`), which shares the immutable
        # buffers: no copy. Not with an Object column, whose Python objects
        # are copied.
        return not _has_object_column(value)
    if kind.__name__ == "ndarray" and kind is getattr(sys.modules.get("numpy"), "ndarray", None):
        return not value.dtype.hasobject and value.nbytes <= _CHEAP_ARRAY_BYTES
    if kind in (dict, list, tuple) and depth < _CHEAP_DEPTH and len(value) <= _CHEAP_ITEMS:
        items = value.values() if kind is dict else value
        if kind is dict and not all(type(name) in _ATOMS for name in value):
            return False
        return all(_cheap_to_keep(item, depth + 1) for item in items)
    return False


def _write_copyable(value: Any, depth: int = 0) -> bool:
    """Is *value* made only of atoms, arrays of plain values and plain
    pandas tables `_write_copy` copies in full (`_copies_in_full`), in a
    few dicts, lists and tuples?"""
    kind = type(value)
    if kind in _ATOMS:
        return True
    if _is_pandas_frame(kind):
        return _copies_in_full(value)
    if kind.__name__ == "ndarray" and kind is getattr(sys.modules.get("numpy"), "ndarray", None):
        return not value.dtype.hasobject
    if kind in (dict, list, tuple):
        if _of_atoms(value):
            return True  # any size: copied at C speed
        if depth >= _CHEAP_DEPTH or len(value) > _CHEAP_ITEMS:
            return False
        if kind is dict and not all(type(name) in _ATOMS for name in value):
            return False
        items = value.values() if kind is dict else value
        return all(_write_copyable(item, depth + 1) for item in items)
    return False


def _of_atoms(value: dict | list | tuple) -> bool:
    """Are a dict's keys and values, or a list's or tuple's items, all `_ATOMS`?"""
    atom = _ATOMS.__contains__
    if type(value) is dict:
        return all(map(atom, map(type, value))) and all(map(atom, map(type, value.values())))
    return all(map(atom, map(type, value)))


def _copies_in_full(frame: Any) -> bool:
    """Does a copy of each array *frame* keeps its data and labels in
    (`_full_copy_values`) leave nothing the caller can change: a plain
    DataFrame or Series without ``attrs``, of numbers, dates and
    durations, Arrow data, nullable columns and Python strings, under
    labels of those kinds (or a range)?"""
    import pandas as pd

    if type(frame) not in (pd.DataFrame, pd.Series) or frame.attrs:
        return False
    if not all(_full_copy_values(blk.values) for blk in frame._mgr.blocks):
        return False
    axes = (frame.index,) if frame.ndim == 1 else (frame.index, frame.columns)
    for axis in axes:
        if isinstance(axis, pd.RangeIndex):
            continue
        if isinstance(axis, pd.MultiIndex) or not _full_copy_values(axis._data):
            return False
    return True


def _full_copy_values(values: Any) -> bool:
    """Is ``values.copy()`` a copy of all *values* hold that the caller can
    reach: their memory, and Python objects only where none can change
    (strings, a column of immutable cells)? Arrow data counts: the copy
    that keeps a table (`frame_sharing.copy_keeping_blocks`) copies all of
    it that is not memory only Arrow holds."""
    import numpy as np
    import pandas as pd
    from pandas.core.arrays.arrow import ArrowExtensionArray
    from pandas.core.arrays.masked import BaseMaskedArray

    if type(values) is np.ndarray:
        return not values.dtype.hasobject or immutable_cells(values.ravel())
    if type(values) in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray, pd.arrays.StringArray):
        return True  # one array; a StringArray holds only str and missing values
    return isinstance(values, (BaseMaskedArray, ArrowExtensionArray)) and type(values).__module__.startswith("pandas.")


def _write_copy(value: Any, memo: dict[int, Any], depth: int = 0) -> Any:
    """A copy of a `_write_copyable` *value*: each table and array copied
    in full, containers rebuilt over the copies, atoms as they are. One
    object met twice is copied once (*memo*), as pickle keeps it one."""
    kind = type(value)
    if kind in _ATOMS:
        return value
    found = memo.get(id(value))
    if found is not None:
        return found
    if _is_pandas_frame(kind):
        copied = _full_frame_copy(value)
    elif _of_atoms(value):
        copied = value if kind is tuple else kind(value)
    elif kind is dict:
        copied = {name: _write_copy(item, memo, depth + 1) for name, item in value.items()}
    elif kind is list or kind is tuple:
        copied = kind(_write_copy(item, memo, depth + 1) for item in value)
    else:  # an array of plain values
        copied = _copy_array(value)
    memo[id(value)] = copied
    return copied


def _frames_own(frame_cells: dict[int, Any], stored: Any) -> bool:
    """Is each table the RAM tier copied into *stored* (*frame_cells*: its
    verdict by ``id``) a copy holding nothing of the caller's? A frozen one
    or one kept with its cells as bytes is; so is one copied block by block
    whose columns and labels are all of kinds that copy leaves nothing
    shared of (`_copies_in_full`): a hit copies it again, so it is never
    handed out either."""
    if all(cells in (_SHARED, _BYTES) for cells in frame_cells.values()):
        return True
    found = {id(frame): frame for frame in _frames_in(stored)}
    return all(
        cells in (_SHARED, _BYTES) or (cells is False and key in found and _copies_in_full(found[key]))
        for key, cells in frame_cells.items()
    )


def _full_frame_copy(frame: Any) -> Any:
    """*frame* copied block by block, its labels and Arrow data with it
    (`frame_sharing.copy_keeping_blocks`)."""
    return frame_sharing.copy_keeping_blocks(frame)


def _frames_in(value: Any, depth: int = 0) -> list:
    """The pandas tables and polars frames `_cheap_to_keep` found in *value*."""
    kind = type(value)
    if _is_pandas_frame(kind) or _is_polars_frame(kind):
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
    """What a copy holds in place of *obj*, a class or function its module
    has since named another of (`kept_state.named_now`), or None.

    A class is the one named now, which a disk hit also resolves the name
    to. A function is itself: a copy holds what the value held, as plain
    Python does, and the function named now may do something else."""
    current = kept_state.named_now(obj)
    if current is None or isinstance(obj, type):
        return current
    return obj


def _metadata_held(frame: Any) -> list[str]:
    """The ``_metadata`` attributes *frame* holds an object in that can be
    changed in place (a subclass's ``info`` dict, geopandas' ``crs``).

    pandas copies these by reference into every copy it makes, deep or not
    (``__finalize__``): a stored table, every hit of it and the caller's
    result held one dict between them.
    """
    names = getattr(type(frame), "_metadata", None)
    if not names:
        return []
    own = getattr(frame, "__dict__", {})
    return [name for name in names if name in own and type(own[name]) not in _ATOMS]


def _own_metadata(frame: Any, copied: Any) -> bool:
    """Give *copied* (a copy of *frame*) its own copy of each attribute
    `_metadata_held` names that it still shares with *frame*, all from one
    ``deepcopy`` memo (two names for one object stay one). False, with
    *copied* unchanged, when one of them cannot be deep-copied."""
    names = [name for name in _metadata_held(frame) if copied.__dict__.get(name) is frame.__dict__[name]]
    if not names:
        return True
    memo: dict[int, Any] = {}
    try:
        fresh = {name: copy.deepcopy(frame.__dict__[name], memo) for name in names}
    except Exception:
        logger.debug("could not deep-copy the attributes of a %s", type(frame).__name__, exc_info=True)
        return False
    for name, value in fresh.items():
        object.__setattr__(copied, name, value)
    return True


def _stored_frame(frame: Any, known_cells: dict[int, Any] | None, record_cells: dict[int, Any] | None) -> bool:
    """Is *frame*, about to be copied, a table this tier stored (a hit), not
    one a caller handed in? Its Arrow data is then the tier's own copy."""
    return record_cells is None and known_cells is not None and id(frame) in known_cells


def _round_trip(value: Any) -> Any:
    """*value* through pickle and back, a class or function its module has
    since named another of replaced as `_named_now` says."""
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
    pickle round trip makes it. A big one times the copy (`copy_seconds`).
    An array no one can write (`_unwritable`) is not copied."""
    if _unwritable(array):
        return array
    if array.nbytes < _TIMED_COPY_BYTES:
        copied = _copy_values(array)
    else:
        started = _copy_clock()
        copied = _copy_values(array)
        _note_copy_speed(array.nbytes, _copy_clock() - started)
    if not array.flags.writeable:
        copied.flags.writeable = False
    return copied


def _unwritable(array: Any) -> bool:
    """Can no one write *array*'s data, ever?

    Read-only is not enough: an array that owns its data and was made
    read-only is made writeable again by ``setflags(write=True)``, and a
    read-only view changes with a base that is writeable, or that is made
    so. Only an array whose chain of bases ends in ``bytes`` (``np.frombuffer``
    of bytes), directly or through a memoryview of them, is held by memory
    Python never writes."""
    base = array
    while type(base) is type(array):
        if base.flags.writeable:
            return False
        base = base.base
    if type(base) is memoryview:
        if not base.readonly:
            return False
        base = base.obj
    return type(base) is bytes


def _copy_values(array: Any) -> Any:
    """``array.copy(order="K")``, a record array's padding zeroed.

    numpy copies a record dtype field by field into fresh memory, so the
    gaps between fields (seven bytes after a scikit-learn tree node's
    ``uint8`` field) held whatever that memory held: the same values, but
    a restored model's pickle differed from process to process."""
    fresh = heap_blocks.fresh_copy(array)
    if fresh is not None:
        return fresh
    dtype = array.dtype
    if dtype.fields is None and dtype.subdtype is None:
        return array.copy(order="K")
    import numpy as np

    # ``np.zeros``, not ``zeros_like``: that one fills field by field too.
    fortran = array.flags.f_contiguous and not array.flags.c_contiguous
    copied = np.zeros(array.shape, dtype, order="F" if fortran else "C")
    np.copyto(copied, array)
    return copied


def _copy_object_array(array: Any) -> Any | None:
    """A copy of a numpy array of objects no one can change in place
    (`immutable_cells`), the cells shared, read-only if it was; None for any
    other array. ``df['action_time'].unique()`` over 1.9 million Timestamps
    was copied through a pickle round trip for the RAM tier: 17 s after a
    4 s statement. Not timed: sharing cells is no measure of copy speed."""
    if array.dtype.kind != "O" or "pandas" not in sys.modules:
        return None
    if not immutable_cells(array.ravel(order="K")):
        return None
    copied = array.copy(order="K")
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
    with it reads 0 s. Nor is seeing it move: one of those ticks falls in
    the half millisecond about one time in thirty, and the process then
    timed every copy with it. A fine clock moves by less than
    `_FINE_STEP_S` from one reading to the next.
    """
    deadline = time.perf_counter() + 5e-4
    last = clock()
    while time.perf_counter() < deadline:
        now = clock()
        if now != last:
            if now - last <= _FINE_STEP_S:
                return True
            last = now  # a tick, or this thread was descheduled between readings
    return False


#: The most a clock `_ticks_finely` takes to be fine moves between two
#: readings of it in a busy loop.
_FINE_STEP_S = 1e-4


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

#: Bytes per second the stores of tables copied at, the fastest of the last
#: `_COPY_READINGS`. Apart from `_COPY_SPEED`: a stored table is copied into
#: fresh pages, which on some machines took five times as long as the
#: copy of an array whose pages were reused, and a hit's cost is priced by
#: the latter.
_STORE_COPY_SPEED: list[float] = []

#: Smaller table stores are not timed: a copy that fits in pages the
#: process has just freed runs ten times faster than a big one. Timed by
#: the wall clock: in a virtual machine the faults on fresh pages are
#: mostly spent outside the thread's CPU time, and they are the cost.
_STORE_TIMED_BYTES = 16 << 20


def _note_store_copy_speed(nbytes: int, seconds: float) -> None:
    if seconds <= 0:
        return
    _STORE_COPY_SPEED.append(nbytes / seconds)
    del _STORE_COPY_SPEED[:-_COPY_READINGS]


def _probe_store_copy_speed() -> None:
    """Time one copy into fresh memory, as a store of a table copies, when no
    store has been timed yet (`_STORE_COPY_SPEED`). A big table the tier
    keeps uncopied (`frame_sharing.share_checked`) is never timed, and the
    memory it holds is what the caller's next tables then take fresh:
    priced by `copy_seconds` alone, from pages a probe had just reused, a
    cleaning cell's every version looked free to keep, and the tier filled
    with them, evicting what a re-run would have restored."""
    import numpy as np

    source = np.ones(_STORE_TIMED_BYTES // 8)
    started = time.perf_counter()
    fresh = source.copy()
    _note_store_copy_speed(fresh.nbytes, max(time.perf_counter() - started, 1e-9))


def store_copy_seconds(value: Any) -> float | None:
    """What storing *value* in this tier costs in copying, at the speed
    stores of tables were measured at (`_STORE_COPY_SPEED`, else
    `copy_seconds`), or None when the tier cannot say: a numpy array
    of plain data, or a pandas table it shares (its blocks the tier copied
    before and Arrow data left out, `shared_blocks.bytes_to_copy`). Any
    other value -- Python objects, a table copied deep -- is None."""
    kind = type(value)
    try:
        if _is_pandas_frame(kind):
            dtypes = [value.dtype] if value.ndim == 1 else list(value.dtypes)
            if any(str(dtype) == "object" for dtype in dtypes):
                return None
            if not (frame_sharing.enabled() and frame_sharing.freezable(value)):
                return None
            if value.ndim == 1:
                nbytes = int(value.array.nbytes) if getattr(value.array, "_pa_array", None) is None else 0
            else:
                nbytes = shared_blocks.bytes_to_copy(value)
            if not _STORE_COPY_SPEED:
                _probe_store_copy_speed()
            if _STORE_COPY_SPEED:
                return nbytes / max(_STORE_COPY_SPEED)
            return copy_seconds(nbytes)
        if kind.__name__ == "ndarray" and kind is getattr(sys.modules.get("numpy"), "ndarray", None):
            if not value.dtype.hasobject:
                return copy_seconds(value.nbytes)
    except Exception:
        logger.debug("no store cost for a %s", kind.__name__, exc_info=True)
    return None


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
    except Exception:
        logger.debug("no measured hit cost for a %s", kind.__name__, exc_info=True)
    return round_trip_seconds(value, size_bytes)


#: Seconds the RAM tier's copies on this thread spent unpickling, since the
#: store in progress began (`InMemoryBackend.set`).
_UNPICKLED = threading.local()

#: Seconds per byte unpickling the tier's copy of an object of a class took
#: -- ``__setstate__`` and all -- by class, the fastest of the last
#: `_COPY_READINGS`. A statsmodels results object rebuilds its
#: design matrix when it is unpickled: 0.74 s for 141 MiB, five times what
#: the cost model's generic rate says.
_ROUND_TRIP_RATES: dict[str, list[float]] = {}

#: Smaller copies are not timed: their fixed costs would read as a rate.
_ROUND_TRIP_MIN_BYTES = 1 << 20

#: Classes remembered at most.
_ROUND_TRIP_CLASSES = 256


def class_key(kind: type) -> str:
    """The name `round_trip_seconds` remembers a class's readings under."""
    return f"{getattr(kind, '__module__', '')}.{getattr(kind, '__qualname__', kind.__name__)}"


def _note_round_trip(value: Any, nbytes: int, seconds: float) -> None:
    """Remember what unpickling the copy of *value*, of about *nbytes*, took:
    only for an object of a class of its own (a frame, an array or a builtin
    container is copied otherwise, and has its own measure). A notebook
    entry is a dict of the statement's variables: its store credits the
    class of its biggest one (`note_round_trip`)."""
    kind = type(value)
    if kind is dict or _copied_as_data(kind):
        return
    note_round_trip(class_key(kind), nbytes, seconds)


def _hit_no_faster(value: Any, unpickle_seconds: float, metadata: MetadataDict) -> bool:
    """Did unpickling the copy of *value*, an object of a class of its own,
    take at least the compute its entry records? A RAM hit copies it the
    same way, so it would be no faster than computing it again."""
    if type(value) is dict or _copied_as_data(type(value)) or not _judged_by_compute(metadata):
        return False
    return unpickle_seconds >= metadata["execution_time"]


def _may_refuse(unpickle_seconds: float, metadata: MetadataDict) -> bool:
    """Could *unpickle_seconds*, read for the entry *metadata* describes,
    keep it out of this tier (`_hit_no_faster`) or off disk? The disk tier
    refuses a restore that saves too little of the compute
    (``PersistencePolicy.pays_to_restore``, 20% by default); a reading of
    half the compute or more is close enough to either line to be taken
    again."""
    return _judged_by_compute(metadata) and 2 * unpickle_seconds >= metadata["execution_time"]


def _judged_by_compute(metadata: MetadataDict) -> bool:
    """May the tier refuse the entry *metadata* describes for costing more
    than it saves? Not one the caller decided to keep (a decorator's,
    ``@cash:persist``, one that must be isolated), nor one recording no compute."""
    if metadata.get("decorator_entry") or metadata.get("force_persist") or metadata.get("copy_required"):
        return False
    return bool(metadata.get("execution_time"))


def _slower_than_its_compute(value: Any, size: int, metadata: MetadataDict) -> bool:
    """Would a hit of *value* (about *size* bytes) take at least as long as
    the compute its entry records, by what unpickling objects of its class
    took here (`round_trip_seconds`)? Then that time goes on *metadata*.
    Before one of its class was read back here, by the fitted cost model's
    price of reading one of its size into RAM: the first graph or model of
    a session is judged before it is pickled, as the next ones are.
    Never for an entry the caller decided to keep (`_judged_by_compute`)."""
    kind = type(value)
    if kind is dict or _copied_as_data(kind) or not _judged_by_compute(metadata):
        return False
    predicted = round_trip_seconds(value, size)
    if predicted is None:
        # No object of its class read back here yet: the fitted cost model's
        # price of reading one of its size into RAM. Not put on the entry as
        # a reading: nothing was read.
        return cost_model.estimated_restore_time(kind.__name__, size, "ram") >= metadata["execution_time"]
    if predicted < metadata["execution_time"]:
        return False
    metadata[MEASURED_ROUND_TRIP] = predicted
    return True


#: Metadata key: the class (`class_key`) of the biggest value in a
#: notebook entry, which a hit of the entry kept as pickle bytes credits
#: its reading to (`InMemoryBackend._read_pickled`).
ROUND_TRIP_CLASS = "round_trip_class"


def _read_seconds(kept: _Pickled) -> float:
    """Seconds reading a copy out of *kept* takes, the copy thrown away;
    infinite when it cannot be read."""
    started = time.perf_counter()
    try:
        kept.load()
    except Exception:  # noqa: BLE001 - what cannot be read out is not worth keeping
        return float("inf")
    return time.perf_counter() - started


def restore_too_slow(value: Any, metadata: MetadataDict, seconds: float) -> bool:
    """Would `restored_no_faster` say *seconds* were no faster than the
    compute, without crediting the reading to anything? For a second
    reading first."""
    if not _credited_class(value, metadata):
        return False
    return _judged_by_compute(metadata) and seconds >= metadata["execution_time"]


def _credited_class(value: Any, metadata: MetadataDict) -> str | None:
    """The class (`class_key`) a reading of *value* is credited to: its
    own, for an object of a class of its own; for a notebook entry, that of
    its biggest variable (`ROUND_TRIP_CLASS`) -- unless the entry holds
    references to call entries (``call_refs``) in its place, whose reading
    says nothing of the object's."""
    kind = type(value)
    if kind is dict:
        if metadata.get("cost_model_family") != GENERIC_FAMILY or metadata.get("call_refs"):
            return None
        return metadata.get(ROUND_TRIP_CLASS)
    return None if _copied_as_data(kind) else class_key(kind)


def restored_no_faster(value: Any, metadata: MetadataDict, seconds: float) -> bool:
    """Credit what restoring *value* from a slower tier took, *seconds*, to
    its class (of its biggest value, for a notebook entry): the next store
    of one is judged on it (`round_trip_seconds`). True when the restore
    was no faster than the compute its entry records, for an entry the
    policy may refuse (`_judged_by_compute`): then it is not worth keeping
    there.

    The first object of a class is written to disk judged by the cost model
    alone, without being read back first; its first restore is the reading.
    Only an object of a class of its own: a frame or an array is restored
    as data, at the rate the cost model fits."""
    name = _credited_class(value, metadata)
    if not name:
        return False
    note_round_trip(name, int(metadata.get("cost_model_size_bytes") or metadata.get("size") or 0), seconds)
    return _judged_by_compute(metadata) and seconds >= metadata["execution_time"]


def _learned_round_trip(value: Any, size: int, metadata: MetadataDict) -> float | None:
    """What unpickling *value* is predicted to take from earlier readings
    of its class (of its biggest value, for a notebook entry), or None."""
    if type(value) is not dict:
        return round_trip_seconds(value, size)
    readings = _ROUND_TRIP_RATES.get(metadata.get(ROUND_TRIP_CLASS) or "")
    return size * min(readings) if readings else None


def _copied_by_round_trip(value: Any, metadata: MetadataDict) -> bool:
    """Was the copy of *value* a pickle round trip of an object of a class
    of its own? A notebook entry is a dict of a statement's variables: only
    when its biggest one is such an object (its ``cost_model_family``)."""
    if type(value) is dict:
        return metadata.get("cost_model_family") == GENERIC_FAMILY
    return not _copied_as_data(type(value))


def _copied_as_data(kind: type) -> bool:
    """Is a value of type *kind* copied as data -- a frame, an array, a
    builtin container or scalar -- rather than through a pickle round trip
    of an object of a class of its own?"""
    if kind in _COPIED_AS_DATA:
        return True
    return _is_pandas_frame(kind) or _is_polars_frame(kind) or kind.__name__ == "ndarray"


def note_round_trip(key: str, nbytes: int, seconds: float) -> None:
    """Remember that unpickling an object of the class named *key*
    (`class_key`), of about *nbytes*, took *seconds*."""
    if nbytes < _ROUND_TRIP_MIN_BYTES or seconds <= 0:
        return
    readings = _ROUND_TRIP_RATES.pop(key, [])
    readings.append(seconds / nbytes)
    del readings[:-_COPY_READINGS]
    _ROUND_TRIP_RATES[key] = readings  # the most recently seen class last
    while len(_ROUND_TRIP_RATES) > _ROUND_TRIP_CLASSES:
        del _ROUND_TRIP_RATES[next(iter(_ROUND_TRIP_RATES))]


_COPIED_AS_DATA = frozenset({list, tuple, set, frozenset, str, bytes, bytearray})


def round_trip_seconds(value: Any, nbytes: int) -> float | None:
    """What unpickling *value*, about *nbytes*, is measured to cost in this
    process, from the tier's earlier copies of objects of its class; None
    when none was timed. A RAM hit copies it that way, and a disk hit
    unpickles it: either takes at least this."""
    readings = _ROUND_TRIP_RATES.get(class_key(type(value)))
    if not readings:
        return None
    return nbytes * min(readings)


def _copy_polars(frame: Any) -> Any:
    """A copy of a polars frame, series or lazy frame holding nothing of
    *frame*'s.

    Polars data is Arrow data, and may be the caller's numpy memory itself:
    ``pl.Series(a)``, ``pl.DataFrame({"x": a})`` and ``pl.from_numpy`` copy
    nothing, so a clone, which shares its buffers, changed with ``a``. The
    columns are copied through their Arrow data (`_owned_polars`), all of
    it: pyarrow sees every buffer polars exports as foreign memory, polars'
    own and numpy's alike, so none passes as Arrow's own
    (`frame_sharing.arrow_owned`). A value the RAM tier holds as its own
    (`_own_polars`, a hit) is cloned instead: it shares nothing the caller
    had. The
    Python objects of an ``Object`` column are copied too: pickle refuses
    such a column, so no disk tier holds one, and the RAM entry is the only
    copy there is.
    """
    import polars as pl

    share = _own_polars(frame)  # polars writes no buffer a clone shares
    if isinstance(frame, pl.LazyFrame):
        if share:
            return frame.clone()
        # A plan holds the data it was made from: pickled, it is written out
        # with it. A plan that does not pickle (a Python function in it) is
        # shared, as a clone shares it.
        try:
            return pickle.loads(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            logger.debug("could not copy a LazyFrame's data", exc_info=True)
            return frame.clone()
    if isinstance(frame, pl.Series):
        if frame.dtype == pl.Object:
            return pl.Series(frame.name, InMemoryBackend._deep_copy(frame.to_list(), {}, None, None), dtype=pl.Object)
        return frame.clone() if share else _owned_polars(frame)
    objects = [column for column in frame.get_columns() if column.dtype == pl.Object]
    if not objects:
        return frame.clone() if share else _owned_polars(frame)
    names = {column.name for column in objects}
    rest = frame.select([name for name in frame.columns if name not in names])
    rest = rest.clone() if share else _owned_polars(rest)
    by_name = {column.name: column for column in rest.get_columns()}
    by_name.update((column.name, _copy_polars(column)) for column in objects)
    return pl.DataFrame([by_name[name] for name in frame.columns])


#: ``id -> weak reference`` for each polars value whose data is the RAM
#: tier's own (`_remember_own_polars`): a copy it stored, or a value just
#: read from disk. Copied by a clone, which shares that data.
_OWN_POLARS: dict[int, Any] = {}
_OWN_POLARS_LOCK = threading.Lock()


def _remember_own_polars(frame: Any) -> None:
    """Note that *frame*, held by the RAM tier alone and never handed out,
    holds data cash owns: its copies (hits) may share it."""
    key = id(frame)
    try:
        ref = weakref.ref(frame, lambda _ref, key=key: _OWN_POLARS.pop(key, None))
    except TypeError:
        return
    with _OWN_POLARS_LOCK:
        _OWN_POLARS[key] = ref


def _own_polars(frame: Any) -> bool:
    ref = _OWN_POLARS.get(id(frame))
    return ref is not None and ref() is frame


def _owned_polars(frame: Any) -> Any:
    """A polars DataFrame or Series with no ``Object`` column, copied: its
    Arrow data copied (`frame_sharing.owned_arrow_data`) and rebuilt as
    the disk tier rebuilds it, or else polars' own pickle round trip."""
    from .serialization import rebuilt_polars

    copied = rebuilt_polars(frame, frame_sharing.owned_arrow_data)
    if copied is not None:
        return copied
    return pickle.loads(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))


#: The pandas tables the store in progress on this thread kept uncopied
#: (`frame_sharing.share_checked`), ``[(table, digest)]``; None outside
#: `InMemoryBackend.set`.
_CHECKED = threading.local()


def _store_polars(frame: Any) -> Any:
    """The RAM tier's stored copy of a polars frame, series or lazy frame:
    a copy (`_copy_polars`), or a clone of one the tier holds already
    (`_own_polars`).

    It was kept as a clone with its digest, and a hit checked the digest:
    a frame made over a numpy array (``pl.Series(a)``) holds the array's
    memory, and a caller writing ``a`` right after the call made every hit
    a miss, so the function ran on every call. Polars does not say which
    memory it allocated itself, so a frame is copied unless the tier made
    it. Measured on 0.5-2 GB frames: the copy took 0.14 s per GB once
    memory is warm (Windows; 0.2 s on Linux), against 0.1-0.2 s for the
    clone and its digest, plus a 0.06 s per GB digest at each hit and disk
    write. On memory the process never touched before, the first copy took
    3-5 s per GB on a Linux machine slow to hand out pages, where a plain
    numpy copy took 8 s. The copy doubles the RAM the frame takes."""
    copied = _copy_polars(frame)
    _remember_own_polars(copied)
    return copied


def _still_as_stored(checked: list[tuple[Any, Any]]) -> bool:
    """Do the pandas tables kept uncopied (`frame_sharing.share_checked`)
    still hash to their digests?"""
    return all(frame_sharing.checked_digest(kept) == digest for kept, digest in checked)


def _is_polars_frame(value_type: type) -> bool:
    """Exactly a polars DataFrame, Series or LazyFrame: a subclass may hold
    state of its own that a clone shares. Never imports polars."""
    pl = sys.modules.get("polars")
    return pl is not None and value_type in (
        getattr(pl, "DataFrame", None),
        getattr(pl, "Series", None),
        getattr(pl, "LazyFrame", None),
    )


def _has_object_column(frame: Any) -> bool:
    """Does a polars DataFrame or Series hold an ``Object`` column? Polars
    nests none inside a List or Struct. A LazyFrame read from a disk tier
    holds none: pickle refuses one."""
    import polars as pl

    if isinstance(frame, pl.LazyFrame):
        return False
    dtypes = [frame.dtype] if isinstance(frame, pl.Series) else frame.dtypes
    return any(dtype == pl.Object for dtype in dtypes)


def _frame_types() -> tuple[type, ...]:
    """The pandas and polars frame and series classes, and polars'
    LazyFrame, of those imported.

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
            if module is modules[1]:
                # Pickled, a plan and the data it holds took 2 s a hit at 80 MB.
                lazy = getattr(module, "LazyFrame", None)
                if isinstance(lazy, type):
                    found.append(lazy)
                else:
                    complete = False
        elif module is not None:
            complete = False  # still importing: asked again next time
    if complete:
        _FRAME_TYPES[:] = [modules, tuple(found)]
    return tuple(found)


#: `_frame_types`' last answer: ``[modules, types]``.
_FRAME_TYPES: list = []


#: A value whose store would copy this many bytes or more goes to the disk
#: tier alone, when the disk tier takes it (`TieredBackend.set`).
DISK_ALONE_MIN_BYTES = 64 * 1024**2


def bytes_a_store_copies(value: Any, _depth: int = 0) -> int:
    """About how many bytes `InMemoryBackend.set` would copy to keep *value*:
    its size (`pickled_size_estimate`), less the tables and polars frames it
    keeps uncopied (`frame_sharing.keeps_uncopied`).
    Looks into the dicts, lists and tuples around them (a notebook entry is
    a dict of the statement's variables). Copies nothing."""
    value_type = type(value)
    if _depth < 4:
        if value_type is dict:
            return sum(bytes_a_store_copies(item, _depth + 1) for item in value.values())
        if (value_type is list or value_type is tuple) and len(value) <= _PREMADE_ITEMS_MAX:
            return sum(bytes_a_store_copies(item, _depth + 1) for item in value)
    if _is_pandas_frame(value_type) and frame_sharing.keeps_uncopied(value):
        return 0
    return pickled_size_estimate(value)


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
        return any(not immutable_cells(column) for column in columns) or (_mutable_labels(frame))
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
    return any(str(array.dtype) == "object" and not immutable_cells(array) for array in arrays)
