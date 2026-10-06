"""How a call's arguments become the args segment of its key."""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import pickle
import sys
import threading
import types
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .. import _plain_data
from .._clock import perf_counter as _perf_counter
from .._memo import ARGUMENTS, FRAMES, LruMemo
from ..canonical_form import (
    NOT_HOOKED,
    ContentHashing,
    TooDeepValueError,
    canonical_bytes,
    canonical_call_bytes,
    canonical_marker_bytes,
    stable_key_repr,
)
from ..content_hashers import BUILTIN_CONTENT, builtin_family_of, builtin_hash, is_native_panic
from ..exceptions import CashCacheIneffectiveWarning
from ..lineage_tag import own_tag
from ..value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, IMMUTABLE_PRIMS, writable_types
from .cash_key import cash_key_method, cash_key_method_of_type, type_name

if TYPE_CHECKING:
    from .cached_function import CachedFunction
    from .cash_key import KeyCheck
    from .frozen import FrozenResults
    from .reporting import Notices

logger = logging.getLogger(__name__)

#: A census taken while one cache key is built, shared by the code fold and the
#: argument hash so a big argument is looked at once (`plain_census`). None
#: outside a key build: after the body has run, an argument may have changed.
PLAIN_CENSUS = threading.local()


def plain_census(value: Any) -> tuple[str, Any, dict] | None:
    """What kind of plain data *value* is, memoized for the key build in progress.

    ``("plain", value, shared)`` for lists and tuples of primitives
    (`_plain_data.is_plain`), ``("plain_aliased", (value, repeats), shared)``
    for such data holding one list more than once (`_plain_data.aliases`),
    ``("plain_numpy", (value, repeats), shared)`` for plain data with numpy
    scalars among its leaves, ``("dict_rows", (keys, rows[, repeats]), shared)``
    for a list of dicts sharing their keys (`_plain_data.dict_rows`),
    ``("tree", (value, repeats), shared)`` for other JSON-like data
    (`_plain_data.tree_levels`), None for anything else. *shared* locates
    the containers inside that another argument could also hold
    (`_plain_data.sharing`).
    """
    memo = getattr(PLAIN_CENSUS, "memo", None)
    if memo is not None:
        hit = memo.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1]
    found: tuple[str, Any, dict] | None = None
    shape = _plain_data.sharing(value)
    if shape is not None:
        repeats, shared, numpy = shape
        if numpy:
            found = ("plain_numpy", (value, repeats), shared)
        else:
            found = ("plain_aliased", (value, repeats), shared) if repeats else ("plain", value, shared)
    else:
        rows = _plain_data.dict_rows_unchecked(value)
        if rows is not None:
            # A row's values are held by its dict and by its tuple in *rows*.
            shape = _plain_data.sharing(rows[1], held_twice_at=1)
            if shape is not None and not shape[2]:
                repeats, shared, _numpy = shape
                shared.update(_plain_data.shared_rows(value))
                found = ("dict_rows", (*rows, repeats) if repeats else rows, shared)
            # Its tuples reference every value: held here, they would make
            # each one look shared to the next census.
            del rows
        if found is None:
            shape = _plain_data.sharing(value, tree=True)
            if shape is not None:
                found = ("tree", (value, shape[0]), shape[1])
    if memo is not None:
        memo[id(value)] = (value, found)
    return found


def plain_key_part(value: Any) -> Any:
    """*value*, or -- for plain data -- a marker holding the digest of its content.

    Each plain argument is keyed by its content on its own, pickled without the
    memo (`_plain_data.pickle_unshared`), so one small dict beside two million
    rows does not send the rows down the general path.
    """
    if type(value) not in _plain_data.TREE_NODES:
        return value
    census = plain_census(value)
    if census is None:
        return value
    kind, data, _shared = census
    if kind == "plain_numpy":
        h = hashlib.sha256(_plain_data.level_key_bytes(data[0]))
        h.update(pickle.dumps(data[1], protocol=4))
        return (f"__cash_{kind}__", h.hexdigest())
    return (f"__cash_{kind}__", hashlib.sha256(_plain_data.pickle_unshared(data)).hexdigest())


def shared_across(values: list, keyed: list, walked: dict) -> tuple:
    """Where one writable container is reachable from two arguments.

    ``((argument, level, position), where it was first met)`` per repeat:
    a container held by two arguments changes in both when written, where
    two equal copies change in one, so the two calls key apart. Plain and
    JSON-like arguments are keyed by digests of their own content, so their
    shared containers come from the census (`plain_census`), and are
    compared with each other, with every argument itself, and with what
    the canonical walk of the other arguments met (*walked*, its ``_seen``).
    *keyed* is what each of *values* is keyed as (`hash_payload`'s digests).
    """
    first: dict[int, tuple] = {}
    found: list = []
    writable = writable_types()
    for i, (value, digest) in enumerate(zip(values, keyed)):
        census = plain_census(value) if digest is value and type(value) in _plain_data.TREE_NODES else None
        spots = [(id(value), (i, -2, 0))] if isinstance(value, writable) else []
        if census is not None:
            spots.extend((oid, (i, *where)) for oid, where in census[2].items())
        for oid, where in spots:
            earlier = first.get(oid)
            if earlier is not None:
                found.append((where, earlier))
                continue
            first[oid] = where
            met = walked.get(oid) if census is not None else None
            if met is not None:
                found.append((where, ("walked", met[0])))
    return tuple(found)


#: Values whose identity is code plus what it captures. A hasher registered for
#: one of these types covers every such value in the process, and the obvious
#: one -- by name -- gives every closure one factory makes the same identity.
#: `ArgHasher.first_unhashable_arg` found only built-in-typed arguments.
NO_SUSPECT = object()


CODE_VALUE_TYPES = (types.FunctionType, types.MethodType, functools.partial)


#: The fix for an unhashable code value. It must NOT suggest
#: `register_hasher(function, ...)`: following that advice is how a second
#: closure got the first one's result.
CODE_ARG_FIX = (
    "pass a module-level function in its place, and give the values it "
    "captures to the cached function as plain arguments, where they reach the "
    "key. Do not register a hasher for function: every closure one factory "
    "makes shares a name, so a hasher keyed on it hands one closure's result "
    "to another. See known-limitations.md, 'A closure or lambda passed as an "
    "argument'."
)


def unhashable_arg_fix(value: Any, type_name: str) -> str:
    """The fix line for an argument of *type_name* that could not be hashed."""
    if isinstance(value, CODE_VALUE_TYPES):
        return CODE_ARG_FIX
    if cash_key_method(value) is not None:
        return (
            f"fix {type_name}.__cash_key__(): it must return a value cash can hash, such as a "
            f"string, a number or a tuple of them, and must not raise."
        )
    return (
        f"if {type_name} is your own class, give it a __cash_key__(self) method that returns what "
        f"identifies an instance; otherwise register a hasher with cash.register_hasher({type_name}, ...), "
        f"or pass the argument by a hashable value."
    )


#: What `ArgHasher._nested_hasher` does with a value of a type: nothing, or
#: key it by its class's ``__cash_key__``. Decided once per type: it is asked
#: of every non-primitive value a key walk meets.
_PASS = "pass"
_KEYED = "keyed"
_HOOK_KINDS: dict[type, str] = {}


def _hook_kind(t: type) -> str:
    try:
        return _HOOK_KINDS[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _PASS
    if t in BUILTIN_CONTAINERS or issubclass(t, (type, types.ModuleType, *CODE_VALUE_TYPES)):
        kind = _PASS
    else:
        kind = _KEYED if cash_key_method_of_type(t) is not None else _PASS
    if len(_HOOK_KINDS) >= 4096:
        _HOOK_KINDS.clear()
    _HOOK_KINDS[t] = kind
    return kind


#: How deep ``__cash_key__`` may nest (a key naming an object with its own
#: key); past it, the keys are taken to loop.
_CASH_KEY_MAX_DEPTH = 32
_CASH_KEY_DEPTH = threading.local()


#: Who wrote a value's ``_cash_lineage_hash``, in ``_cash_lineage_src``. Only the
#: notebook's statement layer keeps the tag current as the value changes, so
#: only its tag stands in for the value's content (see `ArgHasher.hash_payload`).
LINEAGE_SRC_STATEMENT = "statement"
LINEAGE_SRC_DECORATOR = "decorator"
#: Written for a function decorated ``frozen=True``: the user's promise that the
#: result is not modified afterwards, trusted like the statement layer's tag and
#: audited now and then (`FrozenResults.audit`).
LINEAGE_SRC_FROZEN = "frozen"


_COW_PANDAS: bool | None = None


#: The costliest argument of the key most recently hashed on this thread:
#: ``(label, seconds, type name, producer, pandas without copy-on-write)``.
#: A description, never the value: a reference here would keep a large
#: argument alive after its caller dropped it.
ARG_COST = threading.local()


def is_cow_pandas(value: Any) -> bool:
    """Is *value* a pandas DataFrame/Series under copy-on-write?

    Copy-on-write is the only mode in pandas 3 and opt-in before. Checked
    without importing pandas: a pandas object means it is already loaded.
    """
    global _COW_PANDAS
    t = type(value)
    if t.__name__ not in ("DataFrame", "Series") or not (t.__module__ or "").startswith("pandas"):
        return False
    if _COW_PANDAS is None:
        try:
            import pandas as pd

            major = int(pd.__version__.split(".", 1)[0])
            _COW_PANDAS = major >= 3 or pd.options.mode.copy_on_write is True
        except Exception:  # noqa: BLE001 - unknown pandas: no memo, hash every time
            _COW_PANDAS = False
    return _COW_PANDAS


def frame_memoable(value: Any) -> bool:
    """Can the copy-on-write memo check *value* instead of reading it
    (`ArgHasher._memo_content_digest`)? A pandas frame under copy-on-write
    whose blocks pandas alone can write (`frame_borrows_its_data`).

    A handle recorded and dropped since does not count: it costs the memo a
    fresh hash, and how a walk keys an object holding the frame must not
    change because ``.values`` was read once."""
    if not is_cow_pandas(value):
        return False
    try:
        return not frame_borrows_its_data(value, since=sys.maxsize)
    except Exception:  # noqa: BLE001 - a pandas internals change: no memo
        return False


def frame_signature(obj: Any) -> tuple:
    """What must stay the same for a pandas object's content hash to hold.

    Under copy-on-write, a frame whose data another frame also references
    cannot be written in place: every write path (``loc``/``iloc``/``at``,
    column assignment, ``inplace=True`` methods, ``update``, ``insert``,
    ``pop``) first gives the written frame NEW block arrays, and writes
    through ``.values`` / ``to_numpy()`` raise (the arrays are read-only).
    So the identities of the block arrays, the manager and the axes are an
    exact change signal. The
    axis NAMES are compared by value, because ``df.index.name = ...``
    renames the same Index object and the content hash includes them; so
    are the index ``freq`` and ``attrs``, which the hash also holds.

    Two ways around copy-on-write are closed separately. An Arrow-backed
    array (pandas 3's strings, in a column or an axis) is written by
    swapping the Arrow array it holds, so that array's identity is in the
    signature. Every other array is written in place through a handle
    pandas hands out, which is recorded (`watch_array_handles`), or is not
    memoised at all (`frame_borrows_its_data`).
    """
    mgr = obj._mgr
    blocks = tuple((id(block.values), id(getattr(block.values, "_pa_array", None))) for block in mgr.blocks)
    attrs = pickle.dumps(stable_key_repr(obj.attrs, BUILTIN_CONTENT), protocol=4) if obj.attrs else None
    index = (id(obj.index), _axis_arrays_signature(obj.index), tuple(obj.index.names))
    index += (repr(getattr(obj.index, "freq", None)), attrs)
    if hasattr(obj, "columns"):
        columns = (id(obj.columns), _axis_arrays_signature(obj.columns), tuple(obj.columns.names))
        return (id(mgr), blocks, *columns, *index)
    return (id(mgr), blocks, *index, obj.name)


def _axis_arrays_signature(axis: Any) -> tuple:
    return tuple((id(values), id(getattr(values, "_pa_array", None))) for values in _axis_arrays(axis))


def _axis_arrays(axis: Any) -> list:
    """The arrays an axis keeps its labels in; a ``RangeIndex`` keeps none
    (and its ``_data`` would build one)."""
    kind = type(axis).__name__
    if kind == "RangeIndex":
        return []
    if kind == "MultiIndex":
        return [level._data for level in axis._levels] + list(axis._codes)
    return [axis._data]


def frame_borrows_its_data(obj: Any, held: Any = None, since: int = 0) -> bool:
    """Whether *obj*'s data sits in memory something outside pandas may write.

    The memo trusts copy-on-write: while the memo's shallow copy *held*
    shares the blocks, every write through pandas gives *obj* new arrays,
    which `frame_signature` sees. Copy-on-write only governs pandas' own
    objects. An ndarray the frame was built over with ``copy=False`` (or as
    ``index=arr``, which pandas does not copy), or a view of the ndarray a
    block is a view of, is written straight past it: same blocks, changed
    data. The memo answered 10.0 where the frame really summed to 109.0.

    So every array *obj*'s blocks and axes keep their data in, and every
    array those are views of (up to the ndarray that owns the memory), must
    be referenced only by pandas: by a block or an axis of *obj*, of *held*,
    or of another pandas object copy-on-write tracks as sharing them
    (``df.assign(...)``, ``df[cols]``, ``df["c"]``, ``df.reset_index()``),
    or by one of those arrays' views. ``sys.getrefcount`` says how many
    references an array has; any it has beyond those is someone else's
    handle and the frame is hashed afresh. A reference counted as pandas'
    that is not would let a handle through, so each one counted is a
    reference known to exist: an attribute, a list slot, a view's ``base``.

    Memory an ndarray does not own at the end of the chain -- an Arrow
    buffer (``read_parquet``), a memory map, a Python buffer -- can be
    shared in ways no reference count shows, and so can an extension array
    other than dates, durations and Arrow strings (a categorical or nullable
    array is handed out writable by ``.values``). Such frames are hashed on
    every call. The same goes for a frame some of whose memory a handle has
    reached (`watch_array_handles`) since its content was hashed (*since*,
    an `exposure_mark`), and for a check that cannot run (a pandas
    internals change).

    Called when the memo stores a frame, as well as when it is looked up:
    a caller's array written and then dropped between two calls leaves no
    reference to find at the second.
    """
    if -1 in _EXPOSED:
        return True
    try:
        found = _frame_memory(obj, held)
        if found is None:
            return True
        arrays, pandas_refs = found
        if _EXPOSED and any(_EXPOSED.get(id(array), _NOT_EXPOSED)[1] > since for array in arrays):
            return True
        return any(extra != _refcount_baseline() for extra in _refs_beyond(arrays, pandas_refs))
    except Exception:  # noqa: BLE001 - a pandas internals change: re-hash, the safe answer
        return True


def _frame_memory(obj: Any, held: Any) -> tuple[list, dict[int, int]] | None:
    """``(arrays, {id(array): references pandas holds})`` for everything
    *obj* keeps its data in, or ``None`` when some of it cannot be vouched
    for (see `frame_borrows_its_data`).

    A function of its own so that none of its locals is left pointing at an
    array when the references are counted.
    """
    import numpy as np
    import pandas as pd
    from pandas.core.indexes.frozen import FrozenList

    date_arrays = (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray)
    holders: dict[int, Any] = {}
    todo: list = []

    def hold(holder: Any) -> None:
        if id(holder) not in holders:
            holders[id(holder)] = holder
            todo.append(holder)

    for frame in (obj, held) if held is not None else (obj,):
        for block in frame._mgr.blocks:
            hold(block)
        hold(frame.index)
        if hasattr(frame, "columns"):
            hold(frame.columns)

    arrays: list = []
    pandas_refs: dict[int, int] = {}

    def reference(array: Any) -> bool:
        """Count one reference pandas holds to *array*, then the ones it
        holds itself; False when some of that memory cannot be vouched for."""
        while True:
            key = id(array)
            if key in pandas_refs:
                pandas_refs[key] += 1
                return True
            if getattr(array, "_pa_array", None) is not None:
                # Immutable; a write swaps the Arrow array, which the
                # signature sees. Only strings: a numeric Arrow array can
                # be a zero-copy view of an ndarray someone else writes.
                return _is_arrow_string(array._pa_array.type)
            pandas_refs[key] = 1
            arrays.append(array)
            if isinstance(array, date_arrays):
                array = array._ndarray
            elif type(array) is not np.ndarray:
                return False
            elif array.base is None:
                return bool(array.flags.owndata)
            elif type(array.base) is np.ndarray:
                array = array.base
            else:
                return False

    block_type = pd.core.internals.blocks.Block
    while todo:
        holder = todo.pop()
        if isinstance(holder, block_type):
            values = holder.values
            if type(values) is np.ndarray and values.dtype == object:
                return None  # Python objects change in place: `s[0].append(...)`
            if not reference(values):
                return None
            shared = holder.refs.referenced_blocks
        elif isinstance(holder, pd.MultiIndex):
            for level in holder._levels:
                hold(level)
            hold(holder._codes)  # a list indexes made by ``_view`` share
            shared = holder._references.referenced_blocks if holder._references is not None else []
            _hold_cached_indexes(holder, hold, pd)
        elif isinstance(holder, pd.RangeIndex):
            continue
        elif type(holder) is FrozenList:
            if not all(reference(codes) for codes in holder):
                return None
            continue
        elif type(holder).__module__ == "pandas._libs.index":
            if not reference(holder.values):
                return None
            continue
        elif isinstance(holder, pd.Index):
            values = holder._data
            if not reference(values):
                return None
            engine = holder._cache.get("_engine")
            if engine is not None:
                # pandas' lookup table for the labels; indexes made from this
                # one by ``_view`` share it, so it is counted once, as a holder.
                hold(engine)
            shared = holder._references.referenced_blocks if holder._references is not None else []
            _hold_cached_indexes(holder, hold, pd)
        else:
            return None
        for ref in shared:
            other = ref()
            if other is not None:
                hold(other)
    return arrays, pandas_refs


def _hold_cached_indexes(index: Any, hold: Callable, pd: Any) -> None:
    """Indexes pandas keeps in *index*'s cache (a MultiIndex's ``levels``)
    hold its arrays too."""
    for cached in index._cache.values():
        for item in cached if isinstance(cached, (list, tuple)) else (cached,):
            if isinstance(item, pd.Index):
                hold(item)


def _is_arrow_string(arrow_type: Any) -> bool:
    import pyarrow as pa

    kinds = (pa.types.is_string, pa.types.is_large_string, getattr(pa.types, "is_string_view", None))
    return any(kind is not None and kind(arrow_type) for kind in kinds)


def _refs_beyond(arrays: list, pandas_refs: dict[int, int]) -> list[int]:
    """``sys.getrefcount`` of each of *arrays* less the references pandas
    holds; `_refcount_baseline` for an array nothing else references."""
    return [sys.getrefcount(array) - pandas_refs[id(array)] for array in arrays]


def _refcount_baseline() -> int:
    """What `_refs_beyond` reads for an array only its list holds.

    Measured rather than written down: what ``sys.getrefcount`` counts
    besides the holders varies across Python versions (3.14 counts one
    fewer), and a baseline one too high lets a caller's array through
    as the frame's own -- the stale answer this check exists to stop.
    """
    global _REFCOUNT_BASELINE
    baseline = _REFCOUNT_BASELINE
    if baseline is None:
        import numpy as np

        probe = [np.empty(1)]
        baseline = _REFCOUNT_BASELINE = _refs_beyond(probe, {id(probe[0]): 0})[0]
    return baseline


#: See ``_refcount_baseline``.
_REFCOUNT_BASELINE: int | None = None


#: ``id -> (weak reference, exposure number)`` for every array a handle
#: has reached since the frame memo started (`watch_array_handles`), with
#: the number of the latest handle that reached it.
_EXPOSED: dict[int, tuple[Any, int]] = {}
_NOT_EXPOSED = (None, 0)
#: How many handles have been recorded (`exposure_mark`).
_EXPOSURES = [0]


def exposure_mark() -> int:
    """The number of the latest handle recorded: a frame hashed after it is
    unaffected by every handle up to it, if no handle is still alive."""
    return _EXPOSURES[0]


def watch_array_handles() -> None:
    """Record each handle pandas gives out to the memory it keeps, through
    which that memory can be written. Done once, when the frame memo first
    stores a frame: until then nothing relies on a frame staying unwritten.

    ``.array`` (of a Series or an Index), ``pd.array(s, copy=False)`` and a
    date index's ``asi8`` are the public handles to a numpy-backed array
    pandas keeps: ``s.array[0] = 100.0`` writes into the block while its
    identity stays, so the memo's content hash would no longer describe the
    frame. So does ``pd.Index(df["a"]).array[0] = 100.0``, since the index
    shares the column's memory. The handle is usually gone by the next
    call, so the only trace is the one left here. pandas does not call
    ``.array`` or ``pd.array`` itself, so an ordinary workload records
    nothing; it reads ``asi8`` (``resample``, ``rolling``), so that one is
    recorded only when code outside pandas asks for it.

    The read-only views ``.values``, ``to_numpy()`` and ``np.asarray`` give
    of a numpy-backed column or index are handles too: numpy lets
    ``arr.flags.writeable = True`` make one writable again (the block under
    it is), which is the usual answer to pandas' "assignment destination is
    read-only". pandas reads these itself all the time, so they are recorded
    only when code outside pandas and cash asks, and only a read-only view.

    A recorded handle that is dropped leaves its write, if any, in the
    frame; it costs the frame one fresh hash (*since* in
    `frame_borrows_its_data`), not the memo for good. One still alive keeps
    a reference that `frame_borrows_its_data` counts.
    """
    global _WATCHING
    if _WATCHING:
        return
    import pandas as pd
    from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin

    from pandas.core.base import IndexOpsMixin
    from pandas.core.generic import NDFrame
    from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin

    for cls, name, outside_only in (
        (pd.Series, "array", False),
        (pd.Index, "array", False),
        (DatetimeIndexOpsMixin, "asi8", True),
        (pd.Series, "values", _READ_ONLY_VIEW),
        (pd.DataFrame, "values", _READ_ONLY_VIEW),
        (pd.Index, "values", _READ_ONLY_VIEW),
        (DatetimeTimedeltaMixin, "values", _READ_ONLY_VIEW),
        (IndexOpsMixin, "to_numpy", _READ_ONLY_VIEW),
        (pd.DataFrame, "to_numpy", _READ_ONLY_VIEW),
        (pd.Series, "__array__", _READ_ONLY_VIEW),
        (NDFrame, "__array__", _READ_ONLY_VIEW),
        (pd.Index, "__array__", _READ_ONLY_VIEW),
    ):
        original = cls.__dict__.get(name)
        if isinstance(original, types.FunctionType):  # a method: wrapped as it is
            setattr(cls, name, _noting(original, outside_only))
            continue
        if isinstance(original, property) and original.fget is not None:
            getter = original.fget
        elif hasattr(original, "__get__"):  # Index.array: a cached property
            getter = functools.partial(_get_through, original)
        else:
            continue
        setattr(cls, name, property(_noting(getter, outside_only), doc=original.__doc__))
    original_array = pd.array
    noting_array = _noting(original_array, False)

    @functools.wraps(original_array)
    def array(*args: Any, **kwargs: Any) -> Any:
        copy = kwargs.get("copy", args[2] if len(args) > 2 else True)
        return (original_array if copy else noting_array)(*args, **kwargs)

    pd.array = array
    _WATCHING = True


_WATCHING = False


def _get_through(descriptor: Any, instance: Any) -> Any:
    return descriptor.__get__(instance, type(instance))


#: `_noting`'s *outside_only* for a read-only view (`watch_array_handles`).
_READ_ONLY_VIEW = "read-only view"


def _noting(accessor: Callable, outside_only: bool | str) -> Callable:
    @functools.wraps(accessor)
    def noting(*args: Any, **kwargs: Any) -> Any:
        handle = accessor(*args, **kwargs)
        if outside_only:
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller.startswith("pandas."):
                return handle
            if outside_only is _READ_ONLY_VIEW:
                flags = getattr(handle, "flags", None)
                if flags is None or flags.writeable or caller.startswith("cash."):
                    return handle  # a copy, an extension array, or cash's own read
        try:
            _note_exposed(handle)
        except Exception:  # noqa: BLE001 - recording must never break the user's read
            _EXPOSED[-1] = (None, sys.maxsize)  # unknown: treat every memoised frame as written
        return handle

    return noting


def _note_exposed(handle: Any) -> None:
    """Record the arrays a writable *handle* reaches (`_memory_of`)."""
    _EXPOSURES[0] += 1
    number = _EXPOSURES[0]
    for part in _memory_of(handle):
        key = id(part)
        known = _EXPOSED.get(key)
        if known is not None:
            _EXPOSED[key] = (known[0], number)
            continue
        try:
            _EXPOSED[key] = (weakref.ref(part, lambda _ref, key=key: _EXPOSED.pop(key, None)), number)
        except TypeError:
            _EXPOSED[key] = (part, number)


def _memory_of(value: Any) -> list:
    """*value*, the array inside an extension array, and every array those
    are views of: whatever a write through *value* can land in."""
    found = [value]
    inner = getattr(value, "_ndarray", None)
    if inner is not None:
        found.append(inner)
    views = []
    for part in found:
        base = getattr(part, "base", None)
        while base is not None and len(views) < 64:
            views.append(base)
            base = getattr(base, "base", None)
    return found + views


#: Types whose code must not participate in any cache key. Process-wide,
#: not per-instance: a marker is a property of the type, and a user who
#: marks it once should not have to repeat it per Cash instance.
#:
#: Holds STRONG references deliberately, so a registered class can never
#: be garbage collected. Considered and rejected a WeakSet: opaque types
#: are registered by hand, at import time, in the tens at most for any
#: real user -- not generated in volume -- so the leak this trades away
#: has no realistic scale to bite at. A WeakSet would also silently
#: un-register a type the moment nothing else references it, which is
#: the opposite of "mark it once and forget about it."
OPAQUE_TYPES: set = set()


def mark_opaque(*types_: type) -> None:
    """Exclude *types_* from code-surface hashing: what ``cash.opaque`` records."""
    OPAQUE_TYPES.update(types_)


def is_opaque(obj: Any) -> bool:
    """True when *obj* -- a class, or an instance of one -- must not have
    its code hashed into a cache key.

    The type itself must be in ``OPAQUE_TYPES`` (``cash.opaque``); a
    subclass of an opaque class is not covered. It may carry its own
    freshly-written methods the user actively edits, and inheriting the
    mark would silently exempt that code from ever invalidating the cache.
    A subclass that wants the same treatment is marked itself (pinned by
    ``test_a_subclass_of_an_opaque_class_does_not_inherit_opacity``).

    Never raises: a metaclass that defines
    ``__eq__`` without ``__hash__`` makes the CLASS ITSELF unhashable
    (Python's data-model default, not just its instances), so
    ``target in OPAQUE_TYPES`` can raise ``TypeError`` on a real,
    if unusual, class shape. An opacity check must not be the thing
    that breaks an otherwise-cacheable call.
    """
    try:
        if isinstance(obj, functools.partial):
            # A partial is the function it wraps plus arguments, both of
            # which are keyed. Declaring `functools.partial` opaque would
            # silence EVERY partial in the process, including ones over code
            # the user then edits.
            return False
        target = obj if isinstance(obj, type) else type(obj)
        return target in OPAQUE_TYPES
    except Exception as e:  # noqa: BLE001 - opacity check must never break a call
        logger.debug("[CORE] opacity check failed for %r: %s", obj, e)
        return False


def _rough_size(labelled: tuple[str, Any]) -> int:
    """How big an argument looks, to pick the one a key's time is charged
    to. Never raises: a scipy sparse matrix defines ``__len__`` only to
    raise, and that TypeError, out of a description, made every sparse
    argument unhashable."""
    value = labelled[1]
    try:
        return len(value)
    except Exception:  # noqa: BLE001 - no length: its size will do
        try:
            return sys.getsizeof(value)
        except Exception:  # noqa: BLE001 - a size is only a guess; none will do
            return 0


def _raise_panic_as_unhashable(exc: BaseException) -> None:
    """Raise a native library's panic as the TypeError of an unhashable value
    (see `ArgHasher.hash_payload`); return for anything else."""
    if is_native_panic(exc):
        raise TypeError(f"hashing an argument panicked inside a native library: {exc}") from exc


#: `ArgHasher.normalize_call_args` without a signature: the name's own.
_BY_NAME: Any = object()

class _CostliestArg:
    """The argument of one payload that took longest to hash: its label,
    seconds, type name, the cached function that produced it, and whether
    it is a pandas frame without copy-on-write (`ARG_COST`)."""

    def __init__(self, frozen: FrozenResults) -> None:
        self._frozen = frozen
        self.costliest: tuple | None = None

    def timed(self, label: str, value: Any, hash_one: Callable[[Any], Any]) -> Any:
        """``hash_one(value)``, timed and kept when it is the costliest yet."""
        t0 = _perf_counter()
        digest = hash_one(value)
        seconds = _perf_counter() - t0
        if self.costliest is None or seconds > self.costliest[1]:
            producer = own_tag(value, "_cash_lineage_producer")
            if producer is None and self._frozen.arrays and id(value) in self._frozen.arrays:
                producer = self._frozen.arrays[id(value)][1]
            if producer is None and self._frozen.containers and id(value) in self._frozen.containers:
                producer = self._frozen.containers[id(value)][1]
            old_pandas = (
                type(value).__name__ in ("DataFrame", "Series")
                and (type(value).__module__ or "").startswith("pandas")
                and not is_cow_pandas(value)
            )
            self.costliest = (label, seconds, type(value).__name__, producer, old_pandas)
        return digest

    def charge_payload(self, raw: list[tuple[str, Any]], seconds: float) -> None:
        """Charge the payload walk's *seconds* to the largest of the *raw*
        arguments (those that went into the payload as they are)."""
        if self.costliest is None or seconds > self.costliest[1]:
            label, value = max(raw, key=_rough_size)
            producer = own_tag(value, "_cash_lineage_producer")
            if producer is None and self._frozen.containers and id(value) in self._frozen.containers:
                producer = self._frozen.containers[id(value)][1]
            self.costliest = (label, seconds, type(value).__name__, producer, False)


class ArgHasher:
    """Canonical arguments and their content hashes, with the memos that keep
    re-hashing an unchanged argument cheap."""

    def __init__(
        self,
        cached: dict[str, CachedFunction],
        frozen: FrozenResults,
        notices: Notices,
        key_check: KeyCheck | None = None,
    ) -> None:
        self._cached = cached
        #: Checks each ``__cash_key__`` against the content it names, in the
        #: background (`KeyCheck`); None checks nothing.
        self.key_check = key_check
        self._frozen = frozen
        self._notices = notices
        # id(arg) -> (weakref, lineage_hash, content_hash). Lets a repeated call
        # with the SAME unmutated argument skip re-hashing a possibly-huge
        # input; `hash_payload` validates each read (weakref identity +
        # lineage).
        self._memo: LruMemo[int, tuple] = LruMemo(ARGUMENTS)
        # id(frame) -> (weakref, shallow copy, signature, content hash): the
        # pandas copy-on-write memo, see `_frame_memo_store`.
        self._frame_memo: LruMemo[int, tuple] = LruMemo(FRAMES)
        #: type -> (callable(value) -> str, source hash), from
        #: ``cash.register_hasher``. The source hash is part of the argument
        #: hash, so editing a hasher's body invalidates what it keyed.
        self.type_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}
        #: The same, for ``register_hasher(..., override=True)``: consulted
        #: BEFORE cash's own content hashers. Separate so the hot path skips
        #: the question with one empty check.
        self.override_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}
        #: How a key walk reads a frame, array or table inside an argument:
        #: through the copy-on-write memo (`_memo_content_digest`).
        self._content = ContentHashing(builtin_family_of, self._memo_content_digest, frame_memoable)

    def register_hasher(self, type_: type, hasher_fn: Callable[[Any], str], src_hash: str, *, override: bool) -> None:
        """Make *hasher_fn* the identity of *type_* values, replacing any earlier one."""
        # One type, one registration: re-registering must not leave the
        # previous entry behind in the other registry, still winning.
        self.type_hashers.pop(type_, None)
        self.override_hashers.pop(type_, None)
        if override:
            self.override_hashers[type_] = (hasher_fn, src_hash)
        else:
            self.type_hashers[type_] = (hasher_fn, src_hash)
        # A memoized hash was produced by whichever hasher was in effect
        # before; drop them so the new registration is not shadowed for
        # objects already seen.
        self._memo.clear()
        self._frame_memo.clear()

    def keys_by_registration(self, value: Any) -> bool:
        """Is *value* keyed by a hasher the user registered, rather than by
        what it holds?

        Not for code (a function, a method, a partial, a class): its code is
        what it is, and a hasher registered for ``types.FunctionType`` must
        not take every function's body out of the key.
        """
        if isinstance(value, (type, *CODE_VALUE_TYPES)):
            return False
        if cash_key_method(value) is not None:
            return True
        if not (self.override_hashers or self.type_hashers):
            return False
        try:
            return any(
                isinstance(value, type_)
                for registry in (self.override_hashers, self.type_hashers)
                for type_ in registry
            )
        except Exception:  # noqa: BLE001 - a lookup must never break a call
            return False

    def keys_by_registration_only(self, value: Any) -> bool:
        """Is *value* keyed by a hasher registered for its type (not by
        ``__cash_key__``)?"""
        try:
            return any(
                isinstance(value, type_)
                for registry in (self.override_hashers, self.type_hashers)
                for type_ in registry
            )
        except Exception:  # noqa: BLE001 - a lookup must never break a call
            return False

    def warn_unhashable_args(
        self, func_name: str, args: tuple, kwargs: dict, cause: BaseException | None = None
    ) -> None:
        """KEY-UNHASHABLE-ARG, naming the argument when one can be singled out.

        *cause* is what hashing the arguments raised, when known: a value
        nested deeper than pickle follows is said to be that, not unhashable
        for want of a hasher.
        """
        if isinstance(cause, TooDeepValueError):
            self._notices.warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "<too deep>",
                f"@cash.cache on {func_name}: an argument is nested too deeply to key (deeper than pickle "
                "follows, such as a long linked list), so this call and every call like it does not cache.",
                code="KEY-UNHASHABLE-ARG",
                fix=(
                    "pass a flatter value, or give its class a __cash_key__(self) method returning what "
                    "identifies it (a version, an id, a digest of its content)."
                ),
            )
            return
        arg_type_name = self.first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            which = (
                "an argument could not be hashed, and cash cannot say which -- the value is nested inside a container"
            )
            suggestion = (
                "find the nested value, then register a hasher for its "
                "type with cash.register_hasher(SomeType, ...) or pass "
                "something hashable in its place."
            )
        else:
            which = f"an argument of type {arg_type_name} could not be hashed"
            suggestion = unhashable_arg_fix(self.first_unhashable_arg(args, kwargs), arg_type_name)
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: {which}, so this call and every call like it does not cache.",
            code="KEY-UNHASHABLE-ARG",
            fix=suggestion,
        )

    def warn_key_build_failed(self, func_name: str, args: tuple, kwargs: dict, e: Exception) -> None:
        """KEY-BUILD-FAILED: a step of the key build raised where it did not expect to."""
        arg_type_name = self.first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            hint = (
                "check the function's arguments -- cash could not identify "
                "the offending type; if the exception does not belong to "
                "your code, report it as a bug with the traceback."
            )
        elif isinstance(self.first_unhashable_arg(args, kwargs), CODE_VALUE_TYPES):
            hint = CODE_ARG_FIX
        else:
            hint = (
                f"register a hasher with "
                f"cash.register_hasher({arg_type_name}, ...) if "
                f"{arg_type_name} is the unhashable argument."
            )
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: cache-key generation raised "
            f"{type(e).__name__} ({e}) somewhere it did not anticipate, so "
            f"this call does not cache.",
            code="KEY-BUILD-FAILED",
            fix=hint,
        )

    def first_unhashable_arg_type(self, args: tuple, kwargs: dict) -> str:
        """Return the qualname of the argument that could not be hashed, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. See
        `ArgHasher.first_unhashable_arg` for how the argument is found.
        """
        suspect = self.first_unhashable_arg(args, kwargs)
        return "<unknown>" if suspect is NO_SUSPECT else type(suspect).__qualname__

    def first_unhashable_arg(self, args: tuple, kwargs: dict) -> Any:
        """The argument that could not be hashed, or ``NO_SUSPECT``.

        Each candidate is hashed ALONE and the first that fails is named, so
        ``score(df, lambda d: d * 2)`` blames the lambda, not the DataFrame
        (whose hasher cash rejects, and which with override=True would re-key
        every DataFrame function). This runs only on the failure path. Strings,
        numbers, None and built-in containers are skipped: a scalar always
        hashes, and a container holding the culprit is reported as "nested",
        which says more than naming the list. When no single candidate fails
        on its own, the first non-built-in is the best remaining guess.
        """
        candidates = [a for a in (*args, *kwargs.values()) if not isinstance(a, IMMUTABLE_PRIMS + BUILTIN_CONTAINERS)]
        for candidate in candidates:
            try:
                self.hash_payload((candidate,), {})
            except Exception:  # noqa: BLE001 - exactly what we are looking for
                return candidate
        return candidates[0] if candidates else NO_SUSPECT

    def normalize_call_args(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        signature: Any = _BY_NAME,
    ) -> tuple[tuple, dict]:
        """Bind ``(args, kwargs)`` to the function signature and apply defaults.

        Collapses logically-identical calls written in different forms - ``f(1)``
        vs ``f(1, y=10)`` (the default) vs ``f(x=1, y=10)``, and kwargs in any
        order - to one canonical argument shape so they share a cache key
        instead of producing wasteful misses.

        Best-effort: any introspection or bind failure (builtins with no
        signature, ``*args`` calls that don't match, deliberately mismatched
        calls) returns the inputs unchanged, so behavior never regresses.
        """
        # Read once per decoration: a notebook cell re-run with an edited
        # default gets a new `CachedFunction`, and with it the new signature.
        # The key build passes its own wrapper's *signature*: the name's slot
        # holds the function decorated last under it.
        if signature is _BY_NAME:
            cf = self._cached.get(func_name)
            signature = cf.signature if cf is not None else None
        sig = signature
        if sig is None:
            return args, kwargs
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            # The call doesn't match the signature (the function itself would
            # raise when invoked). Leave the raw form untouched.
            return args, kwargs
        # ``bound.arguments`` is ordered by parameter definition, so the result
        # is canonical regardless of how the caller wrote the call. Re-express
        # named params as kwargs; keep *args positional. ``**kwargs`` keeps the
        # caller's order: it is a dict the body can read the order of
        # (``pd.DataFrame(kwargs)``, ``dict(**kwargs)`` forwarded), like a dict
        # argument. (We only build a payload to hash, so routing named params
        # through kwargs is purely for determinism.)
        canon_args: list[Any] = []
        canon_kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                continue
            val = bound.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                canon_args.extend(val)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                # Under its own name: a `**kwargs` entry may be named like a
                # parameter (`def request(url, /, **params)` called as
                # `request("/a", url="x")`), and both are inputs.
                for k in val:
                    canon_kwargs[f"{name}:{k}"] = val[k]
            else:
                canon_kwargs[name] = val
        return tuple(canon_args), canon_kwargs

    def _memo_arg_hash(self, arg: Any, lineage: str, content_hash: str) -> None:
        """Record ``id(arg) -> (weakref, lineage, content_hash)`` for the session.
        Values that cannot be weak-referenced are skipped (they simply keep
        full-hashing).
        """
        try:
            wref = weakref.ref(arg)
        except TypeError:
            return
        self._memo[id(arg)] = (wref, lineage, content_hash)

    def _frame_memo_lookup(self, obj: Any) -> str | None:
        """The content hash recorded for *obj*, if *obj* has not changed since."""
        entry = self._frame_memo.get(id(obj))
        if entry is None:
            return None
        wref, held, signature, content_hash, since = entry
        if frame_borrows_its_data(obj, held, since):
            self._frame_memo.pop(id(obj), None)
            return None
        try:
            if wref() is obj and frame_signature(obj) == signature:
                return content_hash
        except Exception:  # noqa: BLE001 - a pandas internals change: just re-hash
            pass
        self._frame_memo.pop(id(obj), None)
        return None

    def _frame_memo_store(self, obj: Any, content_hash: str, since: int) -> None:
        """Remember *obj*'s content hash, and hold a shallow copy of it.

        *since* is the `exposure_mark` taken before the content was hashed:
        a handle recorded up to it no longer matters once it is gone.

        The shallow copy shares the data and is what makes the signature
        exact: while cash references the blocks, pandas must copy before any
        write. Cost: the first in-place write to each block afterwards copies
        that block, once. The entry, copy included, goes when *obj* is
        collected, or when the memo drops it to make room.

        Nothing is stored for a frame whose data something outside pandas
        can write (`frame_borrows_its_data`): a caller could write through
        its array and drop it before the next call, leaving nothing to see.
        """
        try:
            watch_array_handles()
            held = obj.copy(deep=False)
            if frame_borrows_its_data(obj, held, since):
                return
            signature = frame_signature(obj)
            memo = self._frame_memo
            key = id(obj)
            wref = weakref.ref(obj, lambda _ref, key=key, memo=memo: memo.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup; hash every time
            return
        self._frame_memo[key] = (wref, held, signature, content_hash, since)

    def cash_key_hash(self, value: Any, method: Callable) -> str:
        """*value*'s identity from its class's ``__cash_key__``.

        The returned value is hashed like an argument of its own, so it may be
        a string, a tuple, a frame or another object with a key. The class's
        name goes in, so two classes returning the same id key apart, and so
        does the method's code, so editing it invalidates what it keyed.
        """
        from .function_identity import hash_callable_source

        name = type_name(value)
        depth = getattr(_CASH_KEY_DEPTH, "n", 0)
        if depth >= _CASH_KEY_MAX_DEPTH:
            raise TypeError(f"{name}.__cash_key__() keys on objects whose keys lead back to it")
        _CASH_KEY_DEPTH.n = depth + 1
        try:
            try:
                identity = method(value)
            except Exception as exc:  # reported as an unhashable argument
                raise TypeError(f"{name}.__cash_key__() raised {type(exc).__name__}: {exc}") from exc
            if identity is value:
                raise TypeError(f"{name}.__cash_key__() returned the object itself")
            digest = self.hash_payload((identity,), {})
        finally:
            _CASH_KEY_DEPTH.n = depth
        src = hash_callable_source(method)
        key_id = hashlib.sha256(f"{name}:{src}:{digest}".encode("utf-8")).hexdigest()
        if self.key_check is not None:
            self.key_check.note(value, key_id, method)
        return f"__cash_key__:{key_id}"

    def _nested_hasher(self, value: Any) -> Any:
        """A value's stand-in inside an argument, or `NOT_HOOKED`.

        A registered hasher's identity first (the top level's order; a
        ``Store`` inside a list was pickled instead, which failed on the lock
        it holds), then the class's ``__cash_key__``. Asked of every
        non-primitive value a key walk meets, so what a type is gets decided
        once per type (`_hook_kind`).
        """
        if self.override_hashers or self.type_hashers:
            for registry in (self.override_hashers, self.type_hashers):
                for type_, (hasher_fn, src_hash) in registry.items():
                    if isinstance(value, type_):
                        return ("__cash_hashed__", f"{src_hash}:{hasher_fn(value)}")
        if _hook_kind(type(value)) is _KEYED:
            return ("__cash_hashed__", self.cash_key_hash(value, cash_key_method(value)))
        return NOT_HOOKED

    def _memo_content_digest(self, value: Any) -> str | None:
        """`builtin_hash` for a frame, array or table inside an argument,
        through the copy-on-write memo when it is a pandas frame: an
        unchanged frame inside an object or a list is checked, not read
        again (``ContentHashing.digest`` of the key walk)."""
        cow = is_cow_pandas(value)
        if cow:
            digest = self._frame_memo_lookup(value)
            if digest is not None:
                return digest
        since = exposure_mark()
        digest = builtin_hash(value)
        if cow and digest is not None:
            self._frame_memo_store(value, digest, since)
        return digest

    def plain_value_digest(self, value: Any) -> str | None:
        """``hash_payload((value,), {})`` for plain or JSON-like data
        (`plain_key_part`), without the general walk around it; None for
        any other value.

        The same digest: such a value is keyed by the digest of its content
        alone, holds no hook's value, and alone shares nothing with another
        argument.
        """
        if self.override_hashers or self.type_hashers or self._frozen.arrays or self._frozen.containers:
            return None
        if type(value) not in _plain_data.TREE_NODES:
            return None
        marker = plain_key_part(value)
        if marker is value:
            return None
        return hashlib.sha256(canonical_marker_bytes(marker)).hexdigest()

    def hash_payload(self, args: tuple, kwargs: dict) -> str:
        """Hash one concrete ``(args, kwargs)`` form. May raise on unpicklable
        values; the caller decides whether to retry with a different form.

        A panic out of a Rust extension while hashing (polars raises one on
        values it cannot hash) is a ``BaseException``: it passed every
        handler on the way out and ended the user's call. It is raised as the
        ``TypeError`` of an unhashable argument instead, so the call runs
        uncached with KEY-UNHASHABLE-ARG.
        """
        if not (self.override_hashers or self.type_hashers or self._frozen.arrays or self._frozen.containers):
            # Every argument a number, a string, None: the canonical bytes
            # built directly, the same ones the walk below makes (a
            # tenth of its cost, which is most of a small-argument hit).
            fast = canonical_call_bytes(args, kwargs)
            if fast is not None:
                ARG_COST.last = None
                return hashlib.sha256(fast).hexdigest()
        # Timed per argument -- two clock reads each -- so that a
        # CACHE-NET-LOSS verdict can name the argument that costs the time.
        cost = _CostliestArg(self._frozen)
        try:
            hashed_args = tuple(cost.timed(f"#{i}", a, self.arg_hash) for i, a in enumerate(args))
            hashed_kwargs = {k: cost.timed(k, v, self.arg_hash) for k, v in kwargs.items()}
        except BaseException as exc:
            _raise_panic_as_unhashable(exc)
            raise
        # An argument with no hasher of its own goes into the payload AS IS,
        # and its cost is the walk and the pickle below, not the lookup timed
        # above -- so CACHE-NET-LOSS named a 2M-row list as taking "about 0ms
        # to hash". The payload's time is charged to the largest
        # such argument.
        raw = [
            (label, value)
            for (label, value), digest in zip(
                [(f"#{i}", a) for i, a in enumerate(args)] + list(kwargs.items()),
                list(hashed_args) + list(hashed_kwargs.values()),
            )
            if digest is value and type(value) not in CODELESS_PRIMS
        ]
        payload_t0 = _perf_counter()
        args_bytes = self._payload_bytes(args, kwargs, hashed_args, hashed_kwargs)
        if raw:
            cost.charge_payload(raw, _perf_counter() - payload_t0)
        ARG_COST.last = cost.costliest
        return hashlib.sha256(args_bytes).hexdigest()

    def _payload_bytes(self, args: tuple, kwargs: dict, hashed_args: tuple, hashed_kwargs: dict) -> bytes:
        """The canonical bytes of the hashed arguments (`canonical_bytes`).

        One canonical form: sets in a stable order, every container tagged
        with its type, a container met twice marked. Plain and JSON-like data
        is keyed by a digest of each argument on its own, so a container two
        arguments share is marked here (`shared_across`).
        """
        try:
            # A list, not ``map``: a StopIteration raised inside ``map`` ends
            # it early, and the arguments after it would leave the key.
            form: tuple = (
                tuple([plain_key_part(a) for a in hashed_args]),
                {k: plain_key_part(v) for k, v in hashed_kwargs.items()},
            )
            walked: dict = {}
            args_bytes = canonical_bytes(form, self._content, hook=self._nested_hasher, seen=walked)
            shared = shared_across([*args, *kwargs.values()], [*hashed_args, *hashed_kwargs.values()], walked)
            if shared:
                args_bytes += pickle.dumps(("__cash_shared__", shared), protocol=4)
        except BaseException as exc:
            _raise_panic_as_unhashable(exc)
            raise
        return args_bytes

    def arg_hash(self, arg: Any) -> Any:
        """One argument's key part. The first step that answers wins:

        1. The class's ``__cash_key__``, unless a registered hasher
           covers the type: the user said what identifies the value,
           and that holds across restarts.
        2. A frozen array or container (``frozen=True`` results): its
           audited digest, without reading the content again.
        3. The content digest memoised for this very object, while its
           statement or frozen lineage tag is unchanged (`_memo`): a
           within-session speedup that returns the content digest, never
           the tag. The decorator's own tags are not trusted here, since
           nothing moves them when the object is mutated.
        4. A pandas copy-on-write frame's memoised digest, checked
           exactly (`_frame_memo_lookup`).
        5. A hasher registered with ``override=True``: the user's
           identity beats reading the content.
        6. A builtin content hasher (pandas, numpy, polars, pyarrow,
           ...): byte-stable across processes, so a persisted entry
           survives a restart where a session tag would not.
        7. The statement or frozen lineage tag, for a value with no
           content hasher: cheap and current within the session.
        8. A hasher registered for the type.
        9. The value itself, which the payload walk pickles.
        """
        method = cash_key_method(arg)
        if method is not None and not (
            (self.override_hashers or self.type_hashers) and self.keys_by_registration_only(arg)
        ):
            return self.cash_key_hash(arg, method)
        lineage = self._trusted_lineage(arg)
        frozen_hash = self._frozen_digest(arg)
        if frozen_hash is not None:
            return frozen_hash
        if lineage is not None:
            entry = self._memo.get(id(arg))
            if entry is not None:
                wref, memo_lineage, content_hash = entry
                if memo_lineage == lineage and wref() is arg:
                    return content_hash
        # pandas >= 3 copy-on-write: an exact "has this frame changed?"
        # check instead of a trusted tag. See `_frame_memo_lookup`.
        frame_memo = lineage is None and is_cow_pandas(arg)
        if frame_memo:
            content_hash = self._frame_memo_lookup(arg)
            if content_hash is not None:
                return content_hash

        # Overriding hashers, ahead of everything cash would do itself.
        # The user has said their identity for this type beats content
        # hashing, which is the only way to stop re-reading a 800MB array
        # on every call. Guarded by the emptiness check so the ordinary
        # case pays one dict truth test, not a loop.
        if self.override_hashers:
            for type_, (hasher_fn, src_hash) in self.override_hashers.items():
                if isinstance(arg, type_):
                    return f"{src_hash}:{hasher_fn(arg)}"

        since = exposure_mark()
        content_digest = builtin_hash(arg)
        if content_digest is not None:
            if lineage is not None:
                self._memo_arg_hash(arg, lineage, content_digest)
            elif frame_memo:
                self._frame_memo_store(arg, content_digest, since)
            return content_digest
        # Notebook lineage hash: the authoritative, cheap identity for
        # values that carry NO content hasher (custom objects). Kept ahead
        # of registered hashers so a lineage-carrying object short-circuits
        # its (possibly expensive) registered hasher within a session
        # (test_hasher_priority_cash_hash_first).
        if lineage is not None:
            return lineage
        for type_, (hasher_fn, src_hash) in self.type_hashers.items():
            if isinstance(arg, type_):
                # Embed the hasher source hash so that changing the
                # hasher's body invalidates dependent cache entries
                # even when the hasher's output coincidentally matches.
                return f"{src_hash}:{hasher_fn(arg)}"
        return arg

    def _trusted_lineage(self, arg: Any) -> str | None:
        """*arg*'s own lineage tag, when it is one the key may trust: a
        statement's, or a frozen result's that still passes its audit.

        The instance's OWN tag: one inherited from a tagged class made
        every instance key alike (see cash.lineage_tag).
        """
        lineage = own_tag(arg)
        if lineage is not None:
            src = own_tag(arg, "_cash_lineage_src")
            if src == LINEAGE_SRC_FROZEN:
                if not self._frozen.audit(arg):
                    lineage = None
            elif src != LINEAGE_SRC_STATEMENT:
                lineage = None
        return lineage

    def _frozen_digest(self, arg: Any) -> str | None:
        """The audited digest of *arg* when it is a frozen array or container."""
        if self._frozen.arrays and id(arg) in self._frozen.arrays:
            frozen_hash = self._frozen.array_hash(arg)
            if frozen_hash is not None:
                return frozen_hash
        if self._frozen.containers and id(arg) in self._frozen.containers:
            frozen_hash = self._frozen.container_hash(arg)
            if frozen_hash is not None:
                return frozen_hash
        return None

    def serialize_args(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        normalized: tuple[tuple, dict] | None = None,
        failure: list | None = None,
    ) -> str | None:
        """Hash the arguments, canonicalised.

        *normalized* lets a caller that has ALREADY canonicalised pass the
        result in rather than have it recomputed. That is not an optimisation:
        the code channel (`CodeArgs.fold_code_args`) and this value channel must key
        off the SAME bound arguments, or `f()` and `f(<the default>)` -- the
        same logical call -- disagree in one channel and split into two cache
        entries. One canonicalisation, shared, is the only way that invariant
        holds by construction rather than by two call sites staying in step.
        *failure*, when given, gets what hashing raised when this returns None.
        """
        if normalized is None:
            normalized = self.normalize_call_args(func_name, args, kwargs)
        try:
            return self.hash_payload(*normalized)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            # Normalization can fold a default value into the payload (so
            # f(1) keys identically to f(1, y=<default>)). If that default is
            # unpicklable it must not make a call that hashed fine before stop
            # caching - retry with the raw, un-normalized form first.
            if normalized[0] is not args or normalized[1] is not kwargs:
                try:
                    return self.hash_payload(args, kwargs)
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    pass
            # Pickle failure here is surfaced via CashCacheIneffectiveWarning in
            # KeyBuilder.resolve (which sees the None return). Keep this log at
            # debug level so it's available when explicitly enabled but doesn't
            # double-warn.
            logger.debug("Could not serialize arguments for %s: %s", func_name, e)
            if failure is not None:
                failure.append(e)
            return None

    def note_arg_cost(self, func_name: str, unkeyed: bool = False) -> None:
        """Keep the costliest argument to hash seen for *func_name*.

        Only its description is kept -- parameter, type, seconds, the cached
        function that produced it, and *unkeyed*: ``key=`` replaced it in the
        key, so it was hashed only for the in-place-change check -- never the
        value, which may be large.
        """
        cost = getattr(ARG_COST, "last", None)
        ARG_COST.last = None
        if cost is None:
            return
        label, seconds, type_name, producer, old_pandas = cost
        cf = self._cached.get(func_name)
        if cf is None or (cf.arg_cost is not None and cf.arg_cost[2] >= seconds):
            return
        cf.arg_cost = (label, type_name, seconds, producer, old_pandas, unkeyed)
