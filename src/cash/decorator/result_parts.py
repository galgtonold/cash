"""Parts of a result that share memory a hit hands back as separate copies.

``a = np.arange(n); return a, a[2:5]``, or ``return a, pl.Series(a)``: the
view, or the series polars made over the array's memory, changes with
``a[2] = -1`` on the computing run. A hit copies each part on its own, the
RAM tier as a disk hit does, and from then on the two are separate: the
same caller code sees the write in both parts on the first run and in one
on later ones. The same holds for a pandas frame made over an array without
a copy (pandas 2 without copy-on-write: ``pd.Series(a)``).

Such a result is not stored (STORE-PARTS-SHARED): the call runs every time,
which is what the code means. Looked for in the containers around the parts
(a tuple, list or dict, a few levels down, and the attributes of an object):
two numpy arrays whose data overlap (``X, y = data[:, :-1], data[:, -1]``
share a base but no element, and are stored), and a numpy array beside a
frame over it. One object reached twice is not a split: a copy keeps it one
object. Two frames sharing memory need no check, as neither polars nor
pandas under copy-on-write writes memory a frame shares.
"""

from __future__ import annotations

import sys
import types
from typing import Any

__all__ = ["split_by_a_hit"]

#: How deep, and over how many items in all, the containers around a
#: result's arrays and frames are looked into.
_DEPTH = 4
_ITEMS = 1024
#: Work `numpy.shares_memory` may spend on one pair before it is taken as
#: shared: an exact answer for arrays of awkward strides can take long.
_MAX_WORK = 100_000


def split_by_a_hit(result: Any) -> tuple[str, str] | None:
    """``(array, other)``, the paths of two parts of *result*
    (``"result[0]"``), when *other*, an array or a frame, holds memory of
    the array *array*; else None."""
    np = sys.modules.get("numpy")
    if np is None:
        return None
    arrays: list[tuple[str, Any]] = []
    frames: list[tuple[str, Any]] = []
    _collect(result, "result", 0, [_ITEMS], np.ndarray, arrays, frames)
    if not arrays:
        return None
    by_owner: dict[int, list[tuple[str, Any]]] = {}
    for path, array in arrays:
        by_owner.setdefault(id(_owner(array, np.ndarray)), []).append((path, array))
    for group in by_owner.values():
        # Views of one owner, each object once: only those can overlap.
        group = list({id(array): (path, array) for path, array in reversed(group)}.values())[::-1]
        for index, (path, array) in enumerate(group):
            for other_path, other in group[index + 1 :]:
                if _shares(np, array, other):
                    return path, other_path
    for frame_path, frame in frames:
        for column in _column_arrays(frame, np.ndarray):
            for array_path, array in arrays:
                if _shares(np, array, column):
                    return array_path, frame_path
    return None


def _owner(array: Any, ndarray: type) -> Any:
    """What holds *array*'s memory: the end of its chain of numpy bases."""
    while type(array.base) is ndarray:
        array = array.base
    return array.base if array.base is not None else array


def _collect(value: Any, path: str, depth: int, budget: list[int], ndarray: type, arrays: list, frames: list) -> None:
    budget[0] -= 1
    if budget[0] < 0:
        return
    kind = type(value)
    if kind is ndarray:
        if not value.dtype.hasobject:
            arrays.append((path, value))
        return
    if _is_frame(kind):
        frames.append((path, value))
        return
    if depth >= _DEPTH:
        return
    if kind is tuple or kind is list:
        for index, item in enumerate(value):
            _collect(item, f"{path}[{index}]", depth + 1, budget, ndarray, arrays, frames)
    elif kind is dict:
        for name, item in list(value.items()):
            _collect(item, f"{path}[{name!r}]", depth + 1, budget, ndarray, arrays, frames)
    elif not isinstance(value, (type, types.ModuleType, types.FunctionType)):
        # An object holding both: ``Result(raw=a, series=pl.Series(a))``.
        try:
            own = vars(value)
        except TypeError:
            return
        if type(own) is dict:
            for name, item in list(own.items()):
                _collect(item, f"{path}.{name}", depth + 1, budget, ndarray, arrays, frames)


def _is_frame(kind: type) -> bool:
    pl = sys.modules.get("polars")
    if pl is not None and kind in (getattr(pl, "DataFrame", None), getattr(pl, "Series", None)):
        return True
    pd = sys.modules.get("pandas")
    try:
        return pd is not None and issubclass(kind, (pd.DataFrame, pd.Series))
    except (AttributeError, TypeError):  # pandas still importing
        return False


def _column_arrays(frame: Any, ndarray: type) -> list:
    """The numpy views of *frame*'s data that copy nothing: polars columns
    of numbers and dates without nulls, and pandas' numpy blocks. What has
    none (text, nulls, Arrow columns) cannot be numpy memory anyway."""
    found = []
    pl = sys.modules.get("polars")
    if pl is not None and type(frame) in (pl.DataFrame, pl.Series):
        columns = frame.get_columns() if type(frame) is pl.DataFrame else [frame]
        for column in columns:
            try:
                found.append(column.to_numpy(allow_copy=False))
            except Exception:  # noqa: BLE001 - not without a copy: not an array's memory
                continue
        return found
    try:
        blocks = vars(frame)["_mgr"].arrays
    except (KeyError, AttributeError, TypeError):
        return found
    return [block for block in blocks if type(block) is ndarray]


def _shares(np: Any, array: Any, column: Any) -> bool:
    if not np.may_share_memory(array, column):
        return False
    try:
        return bool(np.shares_memory(array, column, max_work=_MAX_WORK))
    except Exception:  # noqa: BLE001 - too much work to tell: taken as shared
        return True
