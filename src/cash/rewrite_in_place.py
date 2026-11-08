"""Putting a stored state back into the live object a computation changed.

``W, H = fit_nmf(X, k)`` where ``fit_nmf`` writes into ``X``: the result alone
is not what the computation did, so it ran every time. It is stored with the
state ``X`` was left in, and a hit hands the result back and writes that
state into the object ``X`` is bound to, which keeps its identity: another
name, a container or a view holding the same object sees the change, as in
a plain run.

Only objects whose whole state a write in place reaches are taken
(`refusal`): an exact NumPy array of a non-object dtype, an exact pandas
``DataFrame`` or ``Series`` whose cells are plain values, and an exact
``list``, ``dict`` or ``set`` of immutable leaves. A class instance, a model,
a memory map, an array of objects, a list of lists (a restore would hand
back copies of the inner lists that something else may hold) run every
time, as before.

A hit writes only into an object of the same shape as the stored state
(`mismatch`): an array of the same shape and dtype that can be written; a
frame or Series with the same index, columns, dtypes and names (a column the
computation added cannot be put back into the frame in place). Everything
is checked before anything is written, so a refused hit leaves the object as
it was and the computation runs on it.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

from cash._user_attrs import attribute_or
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = [
    "WRITTEN_BACK",
    "ArgumentsLeft",
    "FrameState",
    "HeldByCash",
    "KeyedBefore",
    "Shape",
    "before_call",
    "Columns",
    "release_frame",
    "mismatch",
    "refusal",
    "rewritable_kind",
    "rewrite",
    "shape_change",
    "shape_of",
    "snapshot",
]

#: The kind of effect a call's observer records for the arguments whose
#: state after the call its entry stores, and a hit writes back.
WRITTEN_BACK = "argument written back"

#: Containers rewritten by their contents.
_CONTAINERS = (list, dict, set)
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


def _numpy_array(obj: Any) -> bool:
    kind = type(obj)
    return kind.__name__ == "ndarray" and kind.__module__ == "numpy"


def _pandas_kind(obj: Any) -> str | None:
    """``"frame"`` or ``"series"`` for an exact pandas DataFrame or Series."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return None
    kind = type(obj)
    if kind is pd.DataFrame:
        return "frame"
    if kind is pd.Series:
        return "series"
    return None


def rewritable_kind(obj: Any) -> bool:
    """Whether *obj* is of a kind `rewrite` writes into (the checks of
    `refusal` aside)."""
    return type(obj) in _CONTAINERS or _numpy_array(obj) or _pandas_kind(obj) is not None


def _leaves_only(items: Any) -> bool:
    return set(map(type, items)) <= _LEAVES


def _plain_numpy_dtype(dtype: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(dtype, np.dtype) and not dtype.hasobject


def _column_refusal(values: Any, dtype: Any, frame: bool) -> str | None:
    """Why a column (or a Series) of *dtype* cannot be rewritten, or None."""
    if _plain_numpy_dtype(dtype):
        return None
    np = sys.modules.get("numpy")
    if np is not None and isinstance(dtype, np.dtype):
        # An object column: its cells are copies after a restore, which is
        # the same only for values nothing changes in place.
        return None if _leaves_only(values) else "a column holds objects that can change in place"
    if frame:
        # An extension column is put back whole, with its dtype.
        return None
    return f"a Series of dtype {dtype} is not put back in place"


def refusal(obj: Any) -> str | None:
    """Why *obj* cannot have a stored state written back into it, or None."""
    if type(obj) in _CONTAINERS:
        items = [*obj.keys(), *obj.values()] if type(obj) is dict else obj
        if not _leaves_only(items):
            return f"a {type(obj).__name__} holding objects that can change in place is not put back in place"
        return None
    if _numpy_array(obj):
        if obj.dtype.hasobject:
            return "an array of objects is not put back in place"
        if not obj.flags.writeable:
            return "a read-only array is not put back in place"
        return None
    kind = _pandas_kind(obj)
    if kind == "frame":
        try:
            for _, column in obj.items():
                why = _column_refusal(column.array, column.dtype, frame=True)
                if why is not None:
                    return why
        except Exception:  # noqa: BLE001 - a frame that cannot be read is not written
            return "the frame could not be read"
        return None
    if kind == "series":
        try:
            return _column_refusal(obj.array, obj.dtype, frame=False)
        except Exception:  # noqa: BLE001 - a Series that cannot be read is not written
            return "the Series could not be read"
    return f"a {type(obj).__name__} is not put back in place, and a new one would not be that object"


def _same_axis(a: Any, b: Any) -> bool:
    """Whether two pandas indexes are the same labels, dtype, names and type."""
    return (
        type(a) is type(b)
        and a.dtype == b.dtype
        and list(a.names) == list(b.names)
        and attribute_or(a, "freq", None) == attribute_or(b, "freq", None)
        and a.equals(b)
    )


class Shape:
    """What of an object a write in place cannot change: an array's shape
    and dtype, a frame's or Series' axes, dtypes, name, attrs and flags
    (`shape_of`). A list, dict or set has none."""

    __slots__ = ("arrays", "kind", "parts")

    def __init__(self, kind: type, parts: tuple) -> None:
        self.kind = kind
        self.parts = parts
        #: A frame's `Columns`, when taken (`before_call`).
        self.arrays: Columns | None = None


def before_call(obj: Any, release: bool = True) -> Shape:
    """*obj*'s `Shape` before a call that may change it, with a frame's
    `Columns`, which `snapshot` compares after it.

    With *release*, a frame's entry in the frame-hash memo is dropped
    first: the memo's shallow copy holds the frame's blocks, and while it
    does, pandas copies a block before any write to it. The call then runs
    on the frame as the caller holds it, and which columns it writes into
    and which it gives a new array is what plain pandas does with that
    frame. Without, a frame the memo holds is noted as such
    (`Columns.cash_held`), and a call that gives one of its columns a new
    array is not stored (`snapshot`): pandas may have copied it only
    because of the memo."""
    if release:
        release_frame(obj)
    shape = shape_of(obj)
    if _pandas_kind(obj) == "frame":
        shape.arrays = Columns.of(obj)
    return shape


def release_frame(obj: Any) -> None:
    """Drop what cash holds of *obj*'s blocks (`FrameDigests.forget`)."""
    if _pandas_kind(obj) != "frame":
        return
    hashing = sys.modules.get("cash.decorator.arg_hashing")
    if hashing is not None:
        hashing.FRAME_DIGESTS.forget(obj)


def _held_by_cash(obj: Any) -> bool:
    hashing = sys.modules.get("cash.decorator.arg_hashing")
    return hashing is not None and id(obj) in hashing.FRAME_DIGESTS


def shape_of(obj: Any) -> Shape:
    """*obj*'s `Shape`. Holds its index objects, which pandas never changes
    in place, and a copy of its attrs."""
    if type(obj) in _CONTAINERS:
        return Shape(type(obj), ())
    if _numpy_array(obj):
        return Shape(type(obj), (obj.shape, obj.dtype))
    flags = obj.flags.allows_duplicate_labels
    if _pandas_kind(obj) == "frame":
        return Shape(type(obj), (obj.columns, obj.index, tuple(obj.dtypes), dict(obj.attrs), flags))
    return Shape(type(obj), (obj.index, obj.dtype, obj.name, dict(obj.attrs), flags))


def _weak(obj: Any) -> Callable[[], Any]:
    """A weak reference to *obj*, or, for an object that takes none, a
    callable that returns it."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _root(array: Any) -> Any:
    """What owns an ndarray's memory: the end of its ``base`` chain."""
    np = sys.modules["numpy"]
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


class Columns(NamedTuple):
    """Which array holds each column of a frame (*tokens*: a NumPy column's
    memory, address and strides, and its row in its block; an extension
    column's array object and the address ranges of its memory), whether
    another pandas object shares each column's block (*shared*, None where
    pandas does not say), and, per column, weak references to the array
    object and to what owns its memory (*held*): memory that is gone was
    replaced, whatever array took its address since. Strong ones would
    count as names for the arrays, which the frame-hash memo reads as the
    frame's data being borrowed."""

    tokens: tuple
    shared: tuple
    held: tuple
    #: The frame-hash memo held the frame's blocks (`before_call`).
    cash_held: bool = False

    @classmethod
    def of(cls, frame: Any) -> Columns:
        np = sys.modules["numpy"]
        mgr = frame._mgr
        blocks, blknos, blklocs = mgr.blocks, mgr.blknos, mgr.blklocs
        tokens, shared, held = [], [], []
        for position in range(frame.shape[1]):
            block = blocks[blknos[position]]
            values = block.values
            if isinstance(values, np.ndarray):
                tokens.append((values.__array_interface__["data"][0], values.strides, int(blklocs[position])))
                held.append((None, (_weak(_root(values)),)))
            else:
                owners: list = []
                tokens.append((id(values), _memory_of(values, owners)))
                held.append((_weak(values), tuple(_weak(_root(owner)) for owner in owners)))
            refs = attribute_or(block, "refs", None)
            shared.append(bool(refs.has_reference()) if refs is not None else None)
        return cls(tuple(tokens), tuple(shared), tuple(held), _held_by_cash(frame))


def _memory_of(values: Any, owners: list) -> frozenset:
    """The address ranges an extension array's own arrays and buffers
    span: two array objects over the same memory (``fillna(inplace=True)``
    on an ``Int64`` column wraps the very arrays it filled in a new one)
    are one column written into."""
    from cash.argument_sharing import memory_of

    return frozenset(memory_of(values, owners))


def _column_replaced(before: Any, held: tuple, now: Any, values: Any) -> bool | None:
    """Whether a column whose array was *before* (a `Columns` token, with
    its *held* references) was given a new one, now *now*, held in
    *values*: False when it was written into, None when that cannot be
    told (a new array over part of the old memory)."""
    array, owners = held
    if array is not None and array() is values:
        return False  # the same array object, whatever it holds now
    if any(owner() is None for owner in owners):
        return True  # its memory is gone
    if array is None:
        return before != now  # a NumPy column: its memory and place in its block
    if before[1] and before[1] == now[1]:
        return False  # a new array over the very same memory
    if any(lo < hi2 and lo2 < hi for lo, hi in before[1] for lo2, hi2 in now[1]):
        return None
    return True


class HeldByCash(ValueError):
    """A call gave a column of a frame the frame-hash memo held a new array:
    pandas copies a block cash holds before writing it, so whether plain
    pandas would have written into it cannot be told (`snapshot`)."""


class FrameState(NamedTuple):
    """A frame's state after a call, and the *replaced* positions of the
    columns the call gave a new array (``df["a"] = df["a"] + 1``); it wrote
    into the others (``df.loc[0, "a"] = 5``). A hit does the same, so a
    name for a column's array (``h = df["a"].array``) sees the write as it
    does in a plain run, or keeps the old array as it does there.

    *shared* is whether another pandas object shared each column's block
    before the call (`Columns`): pandas copies a shared block before it
    writes it, so which columns a call gives a new array depends on it, and
    a hit is served only on a frame whose columns are shared the same way."""

    frame: Any
    replaced: tuple[int, ...]
    shared: tuple | None = None


def _same(a: Any, b: Any) -> bool:
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(a, pd.Index):
        return isinstance(b, pd.Index) and _same_axis(a, b)
    if type(a) is tuple:
        return type(b) is tuple and len(a) == len(b) and all(map(_same, a, b))
    try:
        return bool(type(a) is type(b) and a == b)
    except Exception:  # noqa: BLE001 - not comparable: not the same
        return False


def shape_change(before: Shape, after: Any) -> str | None:
    """Why *after* cannot be written into an object of the shape *before*,
    or None: the computation added a column, reshaped an array, changed a
    dtype or the index."""
    if before.kind is not type(after):
        return f"it is bound to a {before.kind.__name__}, not a {type(after).__name__}"
    try:
        now = shape_of(after)
        if not _same(before.parts, now.parts):
            return "its shape, dtypes, index or columns differ from the stored ones"
    except Exception:  # noqa: BLE001 - not comparable: not written
        return "it could not be compared with the stored state"
    return None


def mismatch(live: Any, stored: Any) -> str | None:
    """Why *stored* cannot be written into *live*, or None when `rewrite`
    can write it without failing halfway, and as the call wrote it."""
    if isinstance(stored, FrameState):
        if _pandas_kind(live) == "frame":
            release_frame(live)
            try:
                shared = Columns.of(live).shared
            except Exception:  # noqa: BLE001 - its columns cannot be read: not written
                return "its columns could not be read"
            if stored.shared is None or shared != stored.shared:
                return (
                    "another pandas object shares its columns' memory differently than in the call it was "
                    "stored from, so pandas would copy different columns before writing them"
                )
        stored = stored.frame
    if type(live) is not type(stored):
        return f"it is bound to a {type(live).__name__}, not a {type(stored).__name__}"
    why = refusal(live) or refusal(stored)
    if why is not None:
        return why
    try:
        before = shape_of(live)
    except Exception:  # noqa: BLE001 - not comparable: not written
        return "it could not be compared with the stored state"
    return shape_change(before, stored)


def rewrite(live: Any, stored: Any) -> None:
    """Write the state of *stored* into *live* (`mismatch` said it fits)."""
    kind = type(live)
    if kind is list:
        live[:] = stored
    elif kind is dict or kind is set:
        live.clear()
        live.update(stored)
    elif _numpy_array(live):
        sys.modules["numpy"].copyto(live, stored, casting="no")
    elif _pandas_kind(live) == "frame":
        release_frame(live)
        replaced = range(stored.shape[1]) if not isinstance(stored, FrameState) else stored.replaced
        frame = stored.frame if isinstance(stored, FrameState) else stored
        np = sys.modules["numpy"]
        for position in range(frame.shape[1]):
            # The stored column's own array, so its dtype comes with it and
            # nothing is aligned on labels: as a new array where the call
            # gave the column one, else written into the very array that
            # holds the column, as the call wrote into it.
            values = frame.iloc[:, position].array.copy()
            if position in replaced:
                live.isetitem(position, values)
                continue
            mgr = live._mgr
            held = mgr.blocks[mgr.blknos[position]].values
            if isinstance(held, np.ndarray):
                held[mgr.blklocs[position]] = np.asarray(values)
            else:
                held[:] = values
    else:
        live.iloc[:] = stored.to_numpy(copy=True)


def snapshot(obj: Any, arrays: Columns | None = None) -> Any:
    """A copy of *obj*'s state that nothing else holds, for an entry to
    keep. A frame's comes as a `FrameState`, whose replaced columns are
    those whose array differs from *arrays*, its `Columns` before the call
    (all of them when that was not taken)."""
    if type(obj) in _CONTAINERS:
        return type(obj)(obj)
    if _pandas_kind(obj) == "frame":
        now = Columns.of(obj).tokens
        before = arrays.tokens if arrays is not None else ()
        mgr = obj._mgr
        verdicts = [
            True
            if position >= len(before)
            or (type(before[position][0]) is int) != (type(token[0]) is int)
            or len(before[position]) != len(token)
            else _column_replaced(
                before[position], arrays.held[position], token, mgr.blocks[mgr.blknos[position]].values
            )
            for position, token in enumerate(now)
        ]
        if None in verdicts:
            raise ValueError("a column was given a new array over part of its old memory")
        if arrays is not None and arrays.cash_held and any(verdicts):
            raise HeldByCash("pandas may have copied a column only because cash's frame-hash memo held it")
        replaced = tuple(position for position, verdict in enumerate(verdicts) if verdict)
        return FrameState(obj.copy(), replaced, arrays.shared if arrays is not None else None)
    return obj.copy()


class KeyedBefore(NamedTuple):
    """The state an argument was in before the call, as the key held it:
    by all it holds, under the key's argument hash *args_hash*. A hit whose
    key holds the argument the same way is on an argument in that state."""

    args_hash: str


class ArgumentsLeft:
    """A decorated call's result, stored with the state it left the
    arguments it changed in place in: ``{parameter: (before, state
    after)}``, *before* a `KeyedBefore` or the argument's fingerprint. A
    hit writes each state into the argument the caller handed in, while
    that argument is in the state *before* names."""

    __slots__ = ("reads", "result", "states")

    def __init__(self, result: Any, states: dict[str, tuple[str, Any]], reads: tuple | None = None) -> None:
        self.result = result
        self.states = states
        #: What the code the call ran reads outside its arguments
        #: (`argument_sharing.reads_of`): a hit checks the arguments it
        #: writes share nothing with it. None: not known, so not served.
        self.reads = reads

    def __reduce__(self) -> tuple:
        return (ArgumentsLeft, (self.result, self.states, self.reads))
