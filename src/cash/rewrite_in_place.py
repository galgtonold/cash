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
from typing import Any, NamedTuple

from cash._user_attrs import attribute_or
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = [
    "WRITTEN_BACK",
    "ArgumentsLeft",
    "KeyedBefore",
    "Shape",
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

    __slots__ = ("kind", "parts")

    def __init__(self, kind: type, parts: tuple) -> None:
        self.kind = kind
        self.parts = parts


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
    can write it without failing halfway."""
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
        for position in range(stored.shape[1]):
            # The stored column's own array, so its dtype comes with it and
            # nothing is aligned on labels.
            live.isetitem(position, stored.iloc[:, position].array.copy())
    else:
        live.iloc[:] = stored.to_numpy(copy=True)


def snapshot(obj: Any) -> Any:
    """A copy of *obj*'s state that nothing else holds, for an entry to keep."""
    if type(obj) in _CONTAINERS:
        return type(obj)(obj)
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

    __slots__ = ("result", "states")

    def __init__(self, result: Any, states: dict[str, tuple[str, Any]]) -> None:
        self.result = result
        self.states = states

    def __reduce__(self) -> tuple:
        return (ArgumentsLeft, (self.result, self.states))
