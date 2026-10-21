"""A call's result frame holding its argument frame's own cells, kept so on
a hit.

``flt(df)`` returning ``df[mask]`` hands back a frame whose object column
holds the very lists ``df``'s does: ``w['tokens'].iloc[0] is
df['tokens'].iloc[2]``, and a later ``df['tokens'].iloc[2].append(...)``
shows through ``w``. A hit's result is a copy, which holds copies of them.

The call entry records where each such cell sits in the result and in the
argument (`cell_links`), and a hit puts the argument's own cell back in
each place (`relink_cells`). The key says the argument holds what it held
then, row for row, so the cell found there now is the one the call would
have handed back.

Only frames and series the result is, or holds at its top level, and
frames and series passed as arguments themselves; a cell anything else
holds too stays where the share check sees it, and the call is not stored.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

__all__ = [
    "RELINK_FAILED",
    "addresses",
    "cell_links",
    "frames_in",
    "is_frame",
    "is_object_dtype",
    "linked_cell_ids",
    "object_columns_on_both_sides",
    "relink_cells",
]

#: More cells than this to put back, and the call is not stored: the
#: places are metadata, read for every entry.
_MAX_LINKED = 4_000_000

#: `relink_cells` could not put a cell back: the hit is not served.
RELINK_FAILED = object()


def _pandas() -> Any:
    return sys.modules.get("pandas")


def is_object_dtype(dtype: Any) -> bool:
    """Whether *dtype* is numpy's object dtype: a column of Python objects,
    each a cell. Not a categorical, period or interval column, whose array
    is built anew on each read and handed back as a plain object array, nor
    pandas' string dtype, whose strings are made for the read."""
    return isinstance(dtype, sys.modules["numpy"].dtype) and dtype.kind == "O"


def _object_columns(value: Any, pd: Any) -> list[tuple[int, Any]]:
    """``(position, array)`` of *value*'s object columns that hold more than
    values pandas tells apart (`infer_dtype`); position -1 for a series."""
    from cash.notebook.shared_objects import _VALUE_COLUMNS

    if isinstance(value, pd.Series):
        columns = [(-1, value)]
    else:
        columns = [(i, value.iloc[:, i]) for i, dtype in enumerate(value.dtypes) if is_object_dtype(dtype)]
    found = []
    for position, column in columns:
        if not is_object_dtype(column.dtype):
            continue
        array = column.to_numpy(copy=False)
        if pd.api.types.infer_dtype(array, skipna=True) not in _VALUE_COLUMNS:
            found.append((position, array))
    return found


def _frames_in(value: Any, pd: Any) -> list[tuple[list, Any]]:
    """``(path, frame)`` for *value* when it is a frame or series, or for
    each one in it at its top level (a tuple's, a list's or a dict's)."""
    kinds = (pd.Series, pd.DataFrame)
    if isinstance(value, kinds):
        return [([], value)]
    if type(value) in (tuple, list):
        return [([i], item) for i, item in enumerate(value) if isinstance(item, kinds)]
    if type(value) is dict:
        return [([key], item) for key, item in value.items() if type(key) in (str, int) and isinstance(item, kinds)]
    return []


def frames_in(value: Any) -> list[tuple[list, Any]]:
    """`_frames_in`, or none without pandas."""
    pd = _pandas()
    return _frames_in(value, pd) if pd is not None else []


def is_frame(value: Any) -> bool:
    """Whether *value* is a pandas frame or series."""
    pd = _pandas()
    return pd is not None and isinstance(value, (pd.Series, pd.DataFrame))


def addresses(array: Any) -> Any:
    """The addresses of the objects in the slots of the object *array*, as
    an int64 array: read from its memory, where the slots hold them."""
    np = sys.modules["numpy"]
    n = len(array)
    if not n:
        return np.empty(0, dtype=np.int64)
    data = array.__array_interface__["data"][0]
    size = ctypes.sizeof(ctypes.c_void_p)
    step = array.strides[0] // size
    if step == 0:
        return np.full(n, id(array[0]), dtype=np.int64)
    first = data if step > 0 else data + (n - 1) * step * size
    span = (n - 1) * abs(step) + 1
    slots = np.frombuffer((ctypes.c_void_p * span).from_address(first), dtype=np.uintp)
    return slots[::step].astype(np.int64)


def object_columns_on_both_sides(result: Any, args: tuple, kwargs: dict) -> bool:
    """Whether a frame *result* is or holds at its top level and a frame
    among the arguments both have object columns holding more than values."""
    pd = _pandas()
    if pd is None:
        return False
    if not any(_object_columns(frame, pd) for _path, frame in _frames_in(result, pd)):
        return False
    return any(
        isinstance(value, (pd.Series, pd.DataFrame)) and _object_columns(value, pd)
        for value in (*args, *kwargs.values())
    )


def cell_links(result: Any, args: tuple, kwargs: dict) -> tuple[list[dict], dict[int, Any]] | None:
    """Where the cells of *result*'s frames that an argument frame holds
    sit in each, and those cells' addresses by result frame, sorted; None
    when there are none, or too many to record. Plain builtins, as
    metadata must be."""
    pd = _pandas()
    if pd is None:
        return None
    np = sys.modules["numpy"]
    frames = _frames_in(result, pd)
    if not frames:
        return None
    sources = [(["a", i], value) for i, value in enumerate(args)]
    sources += [(["k", name], value) for name, value in kwargs.items()]
    labels, ids = [], []
    for place, value in sources:
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        for position, array in _object_columns(value, pd):
            labels.append((place, position, len(array)))
            ids.append(addresses(array))
    if not ids:
        return None
    every = np.concatenate(ids)
    order = np.argsort(every, kind="stable")
    ordered = every[order]
    starts = np.cumsum([0] + [len(part) for part in ids])
    links: list[dict] = []
    linked: dict[int, list[Any]] = {}
    total = 0
    for path, frame in frames:
        for position, array in _object_columns(frame, pd):
            own = addresses(array)
            at = np.minimum(np.searchsorted(ordered, own), len(ordered) - 1)
            rows = np.flatnonzero(ordered[at] == own)
            if not len(rows):
                continue
            # A cell an argument holds in two places -- one list in two
            # rows, or in two frames -- cannot say which place it was taken
            # from: an argument holding equal lists in those rows has the
            # same key, and its other one would be put back.
            if (np.searchsorted(ordered, own[rows], side="right") - at[rows] > 1).any():
                return None
            total += len(rows)
            if total > _MAX_LINKED:
                return None
            found = order[at[rows]]
            source = np.searchsorted(starts, found, side="right") - 1
            for label in np.unique(source):
                mine = source == label
                place, column, _length = labels[label]
                links.append(
                    {
                        "result": path,
                        "column": position,
                        "rows": _rows(rows[mine], len(array), np),
                        "source": place,
                        "source_column": column,
                        "source_rows": _rows(found[mine] - starts[label], -1, np),
                    }
                )
            linked.setdefault(id(frame), []).append(own[rows])
    if not links:
        return None
    return links, {key: np.unique(np.concatenate(parts)) for key, parts in linked.items()}


def _rows(rows: Any, length: int, np: Any) -> bytes | None:
    """*rows* as bytes, or None when they are every row of a column of
    *length* in order."""
    if len(rows) == length and (length == 0 or (rows[0] == 0 and rows[-1] == length - 1)):
        return None
    return np.asarray(rows, dtype=np.int64).tobytes()


def _read_rows(raw: bytes | None, length: int, np: Any) -> Any:
    if raw is None:
        return np.arange(length, dtype=np.int64)
    return np.frombuffer(raw, dtype=np.int64)


def linked_cell_ids(links: tuple[list[dict], dict[int, Any]] | None) -> dict[int, Any]:
    """The addresses of the cells *links* puts back, sorted, by the result
    frame holding them (`frames_together`)."""
    return links[1] if links else {}


def _column(value: Any, position: int) -> Any:
    """The array behind *value*'s column at *position* (-1: the series),
    or None when it is not an object column."""
    column = value if position == -1 else value.iloc[:, position]
    if not is_object_dtype(column.dtype):
        return None
    return column.to_numpy(copy=False)


def _walk(value: Any, path: list) -> Any:
    for step in path:
        value = value[step]
    return value


def relink_cells(value: Any, args: tuple, kwargs: dict, links: list[dict] | None) -> Any:
    """A hit's *value* with each cell *links* names put back as the
    argument's own; `RELINK_FAILED` when one cannot be."""
    if not links:
        return value
    pd = _pandas()
    if pd is None:
        return RELINK_FAILED
    np = sys.modules["numpy"]
    try:
        for link in links:
            frame = _walk(value, link["result"])
            kind, where = link["source"]
            source = args[where] if kind == "a" else kwargs[where]
            if not isinstance(frame, (pd.Series, pd.DataFrame)) or not isinstance(source, (pd.Series, pd.DataFrame)):
                return RELINK_FAILED
            target = _column(frame, link["column"])
            cells = _column(source, link["source_column"])
            if target is None or cells is None:
                return RELINK_FAILED
            rows = _read_rows(link["rows"], len(target), np)
            source_rows = _read_rows(link["source_rows"], len(cells), np)
            if len(rows) != len(source_rows) or (
                len(rows) and (rows.max() >= len(target) or source_rows.max() >= len(cells))
            ):
                return RELINK_FAILED
            # Through pandas, never into the block: a hit's table may share
            # its blocks with the stored one, and pandas copies them first.
            column = target.copy()
            column[rows] = cells[source_rows]
            if link["column"] == -1:
                frame.iloc[:] = column
            else:
                frame.isetitem(link["column"], column)
            if len(rows) and _column(frame, link["column"])[rows[0]] is not cells[source_rows[0]]:
                return RELINK_FAILED
    except (KeyError, IndexError, TypeError, ValueError):
        return RELINK_FAILED
    return value
