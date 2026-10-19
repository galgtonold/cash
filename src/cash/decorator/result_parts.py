"""Parts of a result that share memory a hit hands back as separate copies.

``a = np.arange(n); return a, a[2:5]``, or ``return a, pl.Series(a)``: the
view, or the series polars made over the array's memory, changes with
``a[2] = -1`` on the computing run. A hit copies each part on its own, the
RAM tier as a disk hit does, and from then on the two are separate: the
same caller code sees the write in both parts on the first run and in one
on later ones. The same holds for a pandas frame made over an array without
a copy (pandas 2 without copy-on-write: ``pd.Series(a)``), and, there, for a
table and a column of it (``return df, df["x"]``).

Such a result is not stored (STORE-PARTS-SHARED): the call runs every time,
which is what the code means. Looked for in the containers around the parts
(a tuple, list or dict, a few levels down, and the attributes of an object,
slots included): two numpy arrays whose data overlap (``X, y = data[:, :-1],
data[:, -1]`` share a base but no element, and are stored), a numpy array
and a frame over it, and two pandas frames over the same memory when pandas
does not copy on write. One object reached twice is not a split: a copy
keeps it one object.

A pair no one can write through is stored: two read-only arrays, and a
read-only array over a frame that never writes the memory it shares, polars
or pandas under copy-on-write (``return df, df["y"].to_numpy()``).
"""

from __future__ import annotations

import sys
import types
from typing import Any

__all__ = ["split_by_a_hit"]

#: How deep, and over how many containers in all, the containers around a
#: result's arrays and frames are looked into. Plain values (numbers, text)
#: and the arrays and frames themselves are not counted, and the containers
#: are opened level by level, so a long list beside two parts that share
#: memory does not hide them.
_DEPTH = 4
_CONTAINERS = 1024
#: Work `numpy.shares_memory` may spend on one pair before it is taken as
#: shared: an exact answer for arrays of awkward strides can take long.
_MAX_WORK = 100_000

_PLAIN = frozenset({int, float, complex, bool, str, bytes, type(None)})
_ARRAY, _POLARS, _PANDAS = 0, 1, 2


def split_by_a_hit(result: Any) -> tuple[str, str] | None:
    """``(part, other)``, the paths of two parts of *result*
    (``"result[0]"``) whose memory overlaps, where a write through one would
    show in the other; else None."""
    np = sys.modules.get("numpy")
    if np is None:
        return None
    parts = _collect(result, np)
    has_array = any(type(part) is np.ndarray for _path, part in parts)
    if not has_array and len(parts) < 2:
        return None
    pandas_copies = _pandas_copies_on_write()
    if not has_array and pandas_copies:
        return None
    # (start, end, order, kind, path, memory, frame id): every array, and
    # the numpy memory of every frame column, by the bytes it spans.
    spans: list[tuple] = []
    for order, (path, part) in enumerate(parts):
        if type(part) is np.ndarray:
            _add_span(np, spans, order, _ARRAY, path, part, None)
            continue
        kind = _POLARS if _is_polars(type(part)) else _PANDAS
        for column in _column_arrays(part, np.ndarray):
            _add_span(np, spans, order, kind, path, column, id(part))
    spans.sort(key=lambda span: span[0])
    open_spans: list[tuple] = []
    for span in spans:
        open_spans = [other for other in open_spans if other[1] > span[0]]
        for other in open_spans:
            if _matters(span, other, pandas_copies) and _shares(np, span[5], other[5]):
                first, second = sorted((span, other), key=lambda s: s[2])
                return _render(first[4]), _render(second[4])
        open_spans.append(span)
    return None


def _matters(one: tuple, other: tuple, pandas_copies: bool) -> bool:
    """Could a write through one of these two parts show in the other?"""
    if one[3] != _ARRAY and other[3] != _ARRAY:
        # Two frames: only pandas without copy-on-write writes the memory a
        # frame shares, and a frame's own columns are one part.
        return one[6] != other[6] and one[3] == other[3] == _PANDAS and not pandas_copies
    if one[3] == _ARRAY and other[3] == _ARRAY:
        return one[5].flags.writeable or other[5].flags.writeable
    array, frame = (one, other) if one[3] == _ARRAY else (other, one)
    if array[5].flags.writeable:
        return True
    return frame[3] == _PANDAS and not pandas_copies


def _add_span(np: Any, spans: list, order: int, kind: int, path: Any, array: Any, frame: int | None) -> None:
    if array.size == 0:
        return
    start, end = _byte_bounds(np)(array)
    spans.append((start, end, order, kind, path, array, frame))


def _byte_bounds(np: Any) -> Any:
    utils = getattr(getattr(np, "lib", None), "array_utils", None)
    return getattr(utils, "byte_bounds", None) or np.byte_bounds


def _collect(result: Any, np: Any) -> list[tuple[Any, Any]]:
    """The arrays and frames in *result*, each object once, in the order
    met, level by level, with their paths (`_render`)."""
    ndarray, generic = np.ndarray, np.generic
    parts: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    frame_kinds: dict[type, bool] = {}
    budget = _CONTAINERS
    level: list[tuple[Any, Any]] = [("result", result)]
    for depth in range(_DEPTH + 1):
        deeper: list[tuple[Any, Any]] = []
        for path, value in level:
            kind = type(value)
            if kind in _PLAIN or isinstance(value, generic):
                continue
            if kind is ndarray:
                if not value.dtype.hasobject and id(value) not in seen:
                    seen.add(id(value))
                    parts.append((path, value))
                continue
            frame = frame_kinds.get(kind)
            if frame is None:
                frame = frame_kinds[kind] = _is_frame(kind)
            if frame:
                if id(value) not in seen:
                    seen.add(id(value))
                    parts.append((path, value))
                continue
            if depth >= _DEPTH or budget <= 0:
                continue
            budget -= 1
            deeper.extend(_items(value, path, generic))
        if not deeper:
            break
        level = deeper
    return parts


def _items(value: Any, path: Any, generic: type) -> list[tuple[Any, Any]]:
    """The parts *value* holds that may hold memory, with their paths: a
    path is ``(parent path, index or name, how it is written)``."""
    if isinstance(value, (tuple, list)):
        kinds = set(map(type, value))
        if all(kind in _PLAIN or issubclass(kind, generic) for kind in kinds):
            return []  # numbers and text: nothing to look at, however many
        fields = getattr(type(value), "_fields", None)  # a namedtuple
        if isinstance(fields, tuple) and len(fields) == len(value):
            return [((path, name, "."), item) for name, item in zip(fields, value) if type(item) not in _PLAIN]
        return [((path, index, "[]"), item) for index, item in enumerate(value) if type(item) not in _PLAIN]
    if isinstance(value, dict):
        return [((path, name, "[]"), item) for name, item in list(dict.items(value)) if type(item) not in _PLAIN]
    if isinstance(value, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        return []
    # An object holding both: ``Result(raw=a, series=pl.Series(a))``.
    found = []
    own = getattr(value, "__dict__", None)
    if type(own) is dict:
        found = [((path, name, "."), item) for name, item in list(own.items())]
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            stored = f"_{cls.__name__.lstrip('_')}{name}" if name.startswith("__") and not name.endswith("__") else name
            try:
                found.append(((path, name, "."), object.__getattribute__(value, stored)))
            except (AttributeError, TypeError):
                continue
    return found


def _render(path: Any) -> str:
    """``"result[0].raw"`` for ``(("result", 0, "[]"), "raw", ".")``."""
    steps = []
    while type(path) is tuple:
        path, step, how = path
        steps.append(f".{step}" if how == "." else f"[{step!r}]")
    return path + "".join(reversed(steps))


def _is_polars(kind: type) -> bool:
    pl = sys.modules.get("polars")
    return pl is not None and kind in (getattr(pl, "DataFrame", None), getattr(pl, "Series", None))


def _is_frame(kind: type) -> bool:
    if _is_polars(kind):
        return True
    pd = sys.modules.get("pandas")
    try:
        return pd is not None and issubclass(kind, (pd.DataFrame, pd.Series))
    except (AttributeError, TypeError):  # pandas still importing
        return False


def _pandas_copies_on_write() -> bool:
    """Does pandas copy before it writes memory two frames share?"""
    pd = sys.modules.get("pandas")
    if pd is None:
        return True
    try:
        return int(pd.__version__.split(".", 1)[0]) >= 3 or pd.options.mode.copy_on_write is True
    except Exception:  # noqa: BLE001 - unknown: taken as writing in place
        return False


def _column_arrays(frame: Any, ndarray: type) -> list:
    """The numpy views of *frame*'s data that copy nothing: polars columns
    of numbers and dates without nulls, and pandas' numpy blocks. What has
    none (text, nulls, Arrow columns) cannot be numpy memory anyway."""
    found = []
    if _is_polars(type(frame)):
        columns = frame.get_columns() if hasattr(frame, "get_columns") else [frame]
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


def _shares(np: Any, array: Any, other: Any) -> bool:
    try:
        return bool(np.shares_memory(array, other, max_work=_MAX_WORK))
    except Exception:  # noqa: BLE001 - too much work to tell: taken as shared
        return True
