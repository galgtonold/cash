"""The in-memory size of a pandas frame or series, without ``memory_usage``.

``DataFrame.memory_usage()`` builds a result Series -- one per column, then
concatenated with the index's -- and that is nearly all it costs: ~0.23 ms for
a 200-row, five-column frame, deep or not, against ~0.05 ms to sum the column
arrays' ``nbytes`` to the same number. Cash sized every stored frame twice, once
for the RAM tier's cap and once for the restore-cost estimate; in a loop over a
thousand small files that was 18% of the cell (round 23).

``deep=True`` is also unbounded where it differs at all: a column of Python
objects is walked value by value, seconds for millions of strings. Those are
sampled here instead -- the callers want the order of magnitude, not the byte.
Arrow-backed strings (pandas 3's default) report their real size as ``nbytes``.
"""
from __future__ import annotations

import random
import sys
from typing import Any

#: Values looked at per column of Python objects.
_OBJECT_SAMPLE = 64


def _holds_python_objects(dtype: Any) -> bool:
    """Object dtype, or strings stored as Python objects rather than in Arrow.

    By name: categorical and string dtypes report ``kind == 'O'`` too, and
    their arrays' ``nbytes`` is already the real size."""
    return str(dtype) == "object" or getattr(dtype, "storage", None) == "python"


def _sampled_bytes(values: Any) -> int:
    """Pointers plus the sampled mean size of what they point at.

    A random sample, seeded by the length so one value always gets one size.
    Not every k-th value: data with a period -- rows cycling through a few
    shapes -- lands a fixed stride on one of them (measured: 22% low on a
    column repeating every 5 rows, stride 3,125).
    """
    n = len(values)
    if not n:
        return 0
    if n <= _OBJECT_SAMPLE:
        sample = values
    else:
        sample = values[sorted(random.Random(n).sample(range(n), _OBJECT_SAMPLE))]
    return 8 * n + int(sum(map(sys.getsizeof, sample)) / len(sample) * n)


def _column_bytes(col: Any) -> int:
    """One column's data, as ``memory_usage(deep=True)`` counts it."""
    if str(col.dtype) == "category":
        # Codes, plus the categories counted deep: they are strings held as
        # Python objects on pandas < 3.
        categorical = col.array
        return int(categorical.codes.nbytes) + _index_bytes(categorical.categories)
    if _holds_python_objects(col.dtype):
        return _sampled_bytes(col.to_numpy(dtype=object))
    return int(col.array.nbytes)


def _index_bytes(index: Any) -> int:
    """The index's own ``nbytes``: a RangeIndex is a few numbers, not n of them."""
    if type(index).__name__ != "MultiIndex" and _holds_python_objects(index.dtype):
        return _sampled_bytes(index.to_numpy(dtype=object))
    return int(index.nbytes)


def _arrays_bytes(obj: Any) -> int | None:
    """A frame's column arrays summed straight from its block manager, or None
    when a column needs a closer look (objects, categories) or the manager's
    shape is not the one known. ``items()`` boxes every column as a Series:
    ~1.5 ms a call for a 20-column group frame, sized once per element of a
    comprehension over 360 of them (round 25, r25s5)."""
    try:
        arrays = obj._mgr.arrays
    except Exception:  # noqa: BLE001 - a private attribute: any surprise means "no"
        return None
    total = 0
    for arr in arrays:
        dtype = getattr(arr, "dtype", None)
        if dtype is None or _holds_python_objects(dtype) or str(dtype) == "category":
            return None
        nbytes = getattr(arr, "nbytes", None)
        if not isinstance(nbytes, int):
            return None
        total += nbytes
    return total


def pandas_nbytes(obj: Any) -> int | None:
    """The size of a DataFrame or Series, index included; None for anything else."""
    kind = type(obj).__name__
    try:
        if kind == "DataFrame":
            fast = _arrays_bytes(obj)
            if fast is not None:
                return _index_bytes(obj.index) + fast
            return _index_bytes(obj.index) + sum(_column_bytes(col) for _, col in obj.items())
        if kind == "Series":
            return _index_bytes(obj.index) + _column_bytes(obj)
    except (AttributeError, TypeError, ValueError):
        return None
    return None


#: How deep into tuples, lists and dicts `pickled_lower_bound` looks.
_BOUND_DEPTH = 2


def _array_bound(arr: Any, seen: set[int]) -> int:
    """At least what one array adds to a pickle: a numpy buffer's bytes, one
    byte per element of anything else (Python objects, Arrow, extension
    arrays). Once per array object -- pickle writes a shared one once, and a
    pandas 3 frame shares its string columns with the series taken from it."""
    if id(arr) in seen:
        return 0
    seen.add(id(arr))
    if str(getattr(arr, "dtype", "")) == "category":
        return _array_bound(arr.codes, seen)
    if type(arr).__name__ == "ndarray" and arr.dtype.kind != "O":
        return int(arr.nbytes)
    try:
        return len(arr)
    except TypeError:
        return 0


def pickled_lower_bound(value: Any, _depth: int = 0, _seen: set[int] | None = None) -> int:
    """A size ``pickle.dumps(value)`` is sure to reach, read off the arrays.

    For a frame, a series, a numpy array and tuples, lists and dicts of them;
    0 for anything else. Enough to see that a value is too big to be worth
    storing without pickling it to find out: r28s5's 1.7 GiB result took
    2.7 s to pickle, after 2.8 s of compute.
    """
    seen = set() if _seen is None else _seen
    kind = type(value).__name__
    try:
        if kind == "DataFrame":
            return sum(_array_bound(arr, seen) for arr in value._mgr.arrays)
        if kind == "Series":
            # ``_values``: the ndarray itself for a numpy dtype (``.array``
            # wraps it), the extension array otherwise.
            return _array_bound(value._values, seen)
        if kind == "ndarray":
            return _array_bound(value, seen)
    except Exception:  # noqa: BLE001 - a bound is optional: no bound is 0
        return 0
    if _depth >= _BOUND_DEPTH:
        return 0
    if type(value) in (tuple, list):
        return sum(pickled_lower_bound(v, _depth + 1, seen) for v in value)
    if type(value) is dict:
        return sum(pickled_lower_bound(v, _depth + 1, seen) for v in value.values())
    return 0
