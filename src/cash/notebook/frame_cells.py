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

import contextlib
import ctypes
import pickle
import sys
import weakref
from collections.abc import Callable, Iterator
from typing import Any

from cash._user_attrs import instance_of
from cash.kept_state import user_state_runs

__all__ = [
    "RELINK_FAILED",
    "addresses",
    "cell_links",
    "frames_in",
    "in_sorted",
    "is_frame",
    "is_object_dtype",
    "linked_cell_ids",
    "object_columns_on_both_sides",
    "relink_cells",
    "sorted_cells",
    "sorted_unique",
]

#: More cells than this to put back, and the call is not stored: the
#: places are metadata, read for every entry.
_MAX_LINKED = 4_000_000

#: `relink_cells` could not put a cell back: the hit is not served.
RELINK_FAILED = object()


def loaded_pandas() -> Any:
    return sys.modules.get("pandas")


def is_object_dtype(dtype: Any) -> bool:
    """Whether *dtype* is numpy's object dtype: a column of Python objects,
    each a cell. Not a categorical, period or interval column, whose array
    is built anew on each read and handed back as a plain object array, nor
    pandas' string dtype, whose strings are made for the read."""
    return isinstance(dtype, sys.modules["numpy"].dtype) and dtype.kind == "O"


def object_columns(value: Any, pd: Any) -> list[tuple[int, Any]]:
    """``(position, array)`` of *value*'s object columns that hold more than
    values pandas tells apart (`column_values_only`); position -1 for a series."""
    from cash.notebook.shared_objects import column_values_only

    if instance_of(value, pd.Series):
        columns = [(-1, value)]
    else:
        columns = [(i, value.iloc[:, i]) for i, dtype in enumerate(value.dtypes) if is_object_dtype(dtype)]
    found = []
    for position, column in columns:
        if not is_object_dtype(column.dtype):
            continue
        array = column.to_numpy(copy=False)
        if not column_values_only(array, pd):
            found.append((position, array))
    return found


def frames_with(value: Any, pd: Any) -> list[tuple[list, Any]]:
    """``(path, frame)`` for *value* when it is a frame or series, or for
    each one in it at its top level (a tuple's, a list's or a dict's)."""
    kinds = (pd.Series, pd.DataFrame)
    if instance_of(value, kinds):
        return [([], value)]
    if type(value) in (tuple, list):
        return [([i], item) for i, item in enumerate(value) if instance_of(item, kinds)]
    if type(value) is dict:
        return [([key], item) for key, item in value.items() if type(key) in (str, int) and instance_of(item, kinds)]
    return []


def frames_in(value: Any) -> list[tuple[list, Any]]:
    """`frames_with`, or none without pandas."""
    pd = loaded_pandas()
    return frames_with(value, pd) if pd is not None else []


def is_frame(value: Any) -> bool:
    """Whether *value* is a pandas frame or series."""
    pd = loaded_pandas()
    return pd is not None and instance_of(value, (pd.Series, pd.DataFrame))


#: Within `reading_once`: `addresses` and `slots_digest` by the id of the
#: array read, each with the array (kept alive, so no other takes its id).
_READ_ONCE: dict[tuple, tuple[Any, Any]] | None = None
#: `kept_state.user_state_runs` when `_READ_ONCE` was last emptied.
_READ_SINCE = 0
#: Called when `_READ_ONCE` is emptied for code of the user's that ran.
_ON_REREAD: list[Callable[[], None]] = []


@contextlib.contextmanager
def reading_once() -> Iterator[None]:
    """Within it, each array's slots are read once (`addresses`,
    `slots_digest`): a statement over a hundred frames sharing one frame's
    cells read each of their slots five times over. Unless a pickle of
    cash's runs code of the user's in between (a ``__reduce__`` that may
    write into a frame's slots, `kept_state.user_state_runs`): what was read
    is then read again, once."""
    global _READ_ONCE, _READ_SINCE
    if _READ_ONCE is not None:
        yield
        return
    _READ_ONCE = {}
    _READ_SINCE = user_state_runs()
    try:
        yield
    finally:
        _READ_ONCE = None


def when_read_again(forget: Callable[[], None]) -> None:
    """Have *forget* called each time `read_once` empties what was read:
    for reads kept elsewhere within `reading_once`."""
    _ON_REREAD.append(forget)


def read_once() -> dict[tuple, tuple[Any, Any]] | None:
    """`_READ_ONCE`, emptied first (with whatever `_ON_REREAD` empties)
    when code of the user's ran since. Called before any read kept within
    `reading_once` is used."""
    global _READ_SINCE
    memo = _READ_ONCE
    if memo is not None:
        runs = user_state_runs()
        if runs != _READ_SINCE:
            _READ_SINCE = runs
            memo.clear()
            for reread in _ON_REREAD:
                reread()
    return memo


def addresses(array: Any) -> Any:
    """The addresses of the objects in the slots of the object *array*, as
    an int64 array: read from its memory, where the slots hold them. Within
    `reading_once`, read once for each memory of slots (`_slots_of`)."""
    memo = read_once()
    if memo is None:
        return _addresses(array)
    key = ("a", *_slots_of(array))
    kept = memo.get(key)
    if kept is None:
        kept = memo[key] = (array, _addresses(array))
    return kept[1]


def sampled_addresses(array: Any, count: int) -> Any:
    """`addresses` of every so many of *array*'s slots, about *count* of
    them spread over it (``addresses(array)[::step]``), reading those
    slots only."""
    return _addresses(array[:: max(1, len(array) // count)])


def _slots_of(array: Any) -> tuple[int, int, int]:
    """The memory of *array*'s slots, as `read_once` keeps what was read of
    them: two views of one column (``df`` and ``df[:]``, or the array a
    frame read and the run of it a share check reads) are read once. The
    array is kept beside what was read, so its memory is not freed and
    taken by another while the read is kept."""
    return (array.__array_interface__["data"][0], len(array), array.strides[0])


def _addresses(array: Any) -> Any:
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


def sorted_unique(values: Any) -> Any:
    """`numpy.unique` of the int64 *values*, by one sort: numpy 2.5's own
    takes some twenty times as long on a hundred thousand addresses."""
    np = sys.modules["numpy"]
    ordered = np.sort(values)
    if len(ordered) < 2:
        return ordered
    return ordered[np.concatenate(([True], ordered[1:] != ordered[:-1]))]


#: `sorted_cells` by the memory a column's slots take: what they held, by
#: digest, their addresses sorted, and a weak reference to the array that
#: owns the memory. Kept within `_SORTED_CELLS_MAX` addresses.
_SORTED_CELLS: dict[tuple[int, int, int], tuple[bytes, Any, Any]] = {}
_SORTED_CELLS_MAX = 8_000_000


def sorted_cells(array: Any) -> Any:
    """The addresses of the objects in the slots of the object *array*,
    sorted and each once (`sorted_unique` of `addresses`).

    Read again for each statement, as the share check is: fifty filters of
    one frame each checked against all the frames before them sorted the
    same hundred thousand addresses 1,250 times. The sort is kept by the
    memory the slots take, and served while the slots hold what they held
    then, by a digest of them: a cell put in another's place, or another
    column at the same memory, is read and sorted anew.

    Past the bound, the sorts of memory no longer held are let go, and a
    new one is not kept while the bound is still spent: the frames read on
    each statement are read in the same order every time, and letting the
    oldest go to keep the newest kept none of them once there were more
    than the bound holds (a hundred frames of a hundred thousand rows each
    sorted anew, on every statement)."""
    if len(array) < 1024:
        return sorted_unique(addresses(array))
    digest = slots_digest(array)
    key = (array.__array_interface__["data"][0], len(array), array.strides[0])
    kept = _SORTED_CELLS.get(key)
    if kept is not None and kept[0] == digest:
        return kept[1]
    ordered = sorted_unique(addresses(array))
    owner = memory_owner(array)
    held = sum(len(entry[1]) for entry in _SORTED_CELLS.values()) - (len(kept[1]) if kept is not None else 0)
    if held + len(ordered) > _SORTED_CELLS_MAX:
        for gone in [k for k, entry in _SORTED_CELLS.items() if entry[2]() is None]:
            held -= len(_SORTED_CELLS.pop(gone)[1])
    if held + len(ordered) <= _SORTED_CELLS_MAX:
        try:
            ref = weakref.ref(owner)
        except TypeError:
            return ordered
        _SORTED_CELLS[key] = (digest, ordered, ref)
    elif kept is not None:
        del _SORTED_CELLS[key]
    return ordered


def holds_any_of(array: Any, ordered: Any) -> bool:
    """Do the slots of the object *array* hold any of the objects whose
    addresses are the sorted int64 *ordered*? By its kept sort when it has
    one (`sorted_cells`); else each slot is looked up in *ordered*, a few
    thousand ids, rather than a hundred thousand addresses sorted for it:
    past the bound on kept sorts, a hundred frames sharing one frame's
    cells were each sorted again on every statement."""
    if not len(ordered) or not len(array):
        return False
    # A frame holding them mostly holds them all along: its first slots
    # answer, and the rest are not read.
    if len(array) <= _FIRST_SLOTS:
        return bool(in_sorted(addresses(array), ordered).any())
    if in_sorted(_addresses(array[:_FIRST_SLOTS]), ordered).any():
        return True
    raw = addresses(array)
    if len(raw) >= 1024:
        kept = _SORTED_CELLS.get((array.__array_interface__["data"][0], len(array), array.strides[0]))
        if kept is not None and kept[0] == slots_digest(array):
            return bool(in_sorted(ordered, kept[1]).any())
    return bool(in_sorted(raw[_FIRST_SLOTS:], ordered).any())


#: The slots `holds_any_of` reads first.
_FIRST_SLOTS = 2048


def slots_digest(array: Any) -> bytes:
    """A digest of the addresses the slots of the object *array* hold.
    Within `reading_once`, taken once for each memory of slots
    (`_slots_of`)."""
    memo = read_once()
    if memo is None:
        return _slots_digest(_slots_bytes(array))
    key = ("d", *_slots_of(array))
    kept = memo.get(key)
    if kept is None:
        kept = memo[key] = (array, _slots_digest(_slots_bytes(array)))
    return kept[1]


def _slots_bytes(array: Any) -> Any:
    """The addresses *array*'s slots hold, as `addresses` reads them, or
    the memory of the slots itself where they are one run in order: the
    same bytes, without the copy (the most of a digest's time)."""
    if len(array) and array.strides[0] == ctypes.sizeof(ctypes.c_void_p):
        np = sys.modules["numpy"]
        data = array.__array_interface__["data"][0]
        return np.frombuffer((ctypes.c_void_p * len(array)).from_address(data), dtype=np.int64)
    return addresses(array)


def memory_owner(array: Any) -> Any:
    """The array at the end of *array*'s chain of bases: the one holding
    its memory, alive while any view of it is."""
    np = sys.modules["numpy"]
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def _slots_digest(raw: Any) -> bytes:
    """A digest of the addresses *raw*: BLAKE3 when it is there."""
    try:
        from blake3 import blake3
    except ImportError:
        import hashlib

        return hashlib.blake2b(raw.view("uint8"), digest_size=16).digest()
    return blake3(raw.view("uint8")).digest()


def in_sorted(values: Any, ordered: Any) -> Any:
    """Which of the int64 *values* are among the sorted int64 *ordered*,
    as a mask: one binary search each, where `numpy.isin` sorts both
    arrays again on every call (a hundred thousand addresses against a
    hundred thousand ids, for each of fifty frames)."""
    np = sys.modules["numpy"]
    if not len(ordered):
        return np.zeros(len(values), dtype=bool)
    at = np.searchsorted(ordered, values)
    at[at == len(ordered)] = 0
    return ordered[at] == values


def object_columns_on_both_sides(result: Any, args: tuple, kwargs: dict) -> bool:
    """Whether a frame *result* is or holds at its top level and a frame
    among the arguments both have object columns holding more than values."""
    pd = loaded_pandas()
    if pd is None:
        return False
    if not any(object_columns(frame, pd) for _path, frame in frames_with(result, pd)):
        return False
    return any(
        instance_of(value, (pd.Series, pd.DataFrame)) and object_columns(value, pd)
        for value in (*args, *kwargs.values())
    )


def cell_links(result: Any, args: tuple, kwargs: dict) -> tuple[list[dict], dict[int, Any]] | None:
    """Where the cells of *result*'s frames that an argument frame holds
    sit in each, and those cells' addresses by result frame, sorted; None
    when there are none, or too many to record. Plain builtins, as
    metadata must be."""
    pd = loaded_pandas()
    if pd is None:
        return None
    np = sys.modules["numpy"]
    frames = frames_with(result, pd)
    if not frames:
        return None
    sources = [(["a", i], value) for i, value in enumerate(args)]
    sources += [(["k", name], value) for name, value in kwargs.items()]
    labels, ids, columns = [], [], []
    for place, value in sources:
        if not instance_of(value, (pd.Series, pd.DataFrame)):
            continue
        for position, array in object_columns(value, pd):
            labels.append((place, position, len(array)))
            ids.append(addresses(array))
            columns.append(array)
    every = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
    order = np.argsort(every, kind="stable")
    ordered = every[order]
    starts = np.cumsum([0] + [len(part) for part in ids])
    # With no cell held twice by the arguments, a result column in an
    # argument column's own memory (``df.iloc[3:]``) holds its cells where
    # that column does, row for row, unsearched.
    once = len(ordered) < 2 or not bool((ordered[1:] == ordered[:-1]).any())
    spans = [_slots_span(array) for array in columns] if once else []
    links: list[dict] = []
    linked: dict[int, list[Any]] = {}
    total = 0
    for path, frame in frames:
        for position, array in object_columns(frame, pd) if ids else ():
            own = addresses(array)
            inside = _inside(array, spans)
            if inside is not None and len(own):
                label, first = inside
                total += len(own)
                if total > _MAX_LINKED:
                    return None
                place, column, _length = labels[label]
                links.append(
                    {
                        "result": path,
                        "column": position,
                        "rows": None,
                        "source": place,
                        "source_column": column,
                        "source_rows": _rows(first + np.arange(len(own), dtype=np.int64), -1, np),
                    }
                )
                linked.setdefault(id(frame), []).append(own)
                continue
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
            for label in sorted_unique(source).tolist():
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
        for position, categories in categorical_columns(frame, pd):
            # A categorical's objects are its categories, which a filter
            # keeps as they are: the very dtype the argument's column has.
            source = _same_dtype(frame.dtypes.iloc[position], sources, pd)
            if source is None:
                continue
            place, column = source
            links.append(
                {"result": path, "column": position, "dtype_of": True, "source": place, "source_column": column}
            )
            linked.setdefault(id(frame), []).append(addresses(categories))
    if not links:
        return None
    return links, {key: sorted_unique(np.concatenate(parts)) for key, parts in linked.items()}


def _slots_span(array: Any) -> tuple[int, int, int] | None:
    """``(first, end, step)``: the bytes of memory the slots of the object
    *array* take, and the step between two; None when it is not one
    dimension, or steps backwards or not at all."""
    step = array.strides[0] if array.ndim == 1 else 0
    if step <= 0 or not len(array):
        return None
    first = array.__array_interface__["data"][0]
    return first, first + len(array) * step, step


def _inside(array: Any, spans: list[tuple[int, int, int] | None]) -> tuple[int, int] | None:
    """``(i, row)`` when the slots of *array* are rows ``row`` on of the
    array whose memory ``spans[i]`` is, one for one: its very slots."""
    own = _slots_span(array)
    if own is None:
        return None
    first, end, step = own
    for i, span in enumerate(spans):
        if span is None or span[2] != step:
            continue
        if span[0] <= first and end <= span[1] and (first - span[0]) % step == 0:
            return i, (first - span[0]) // step
    return None


def categorical_columns(value: Any, pd: Any) -> list[tuple[int, Any]]:
    """``(position, categories)`` of a frame *value*'s categorical columns
    whose categories, an object array, hold more than values pandas tells
    apart; none for a series."""
    from cash.notebook.shared_objects import VALUE_COLUMNS

    if not instance_of(value, pd.DataFrame):
        return []
    found = []
    for position, dtype in enumerate(value.dtypes):
        if not isinstance(dtype, pd.CategoricalDtype):
            continue
        categories = dtype.categories.to_numpy(copy=False)
        if is_object_dtype(categories.dtype) and pd.api.types.infer_dtype(categories, skipna=True) not in VALUE_COLUMNS:
            found.append((position, categories))
    return found


def _same_dtype(dtype: Any, sources: list[tuple[list, Any]], pd: Any) -> tuple[list, int] | None:
    """``(place, column)`` of the one column among the argument frames of
    *sources* whose dtype is *dtype* itself, by identity; None when none
    is, or more than one."""
    found = []
    for place, value in sources:
        if instance_of(value, pd.Series):
            if value.dtype is dtype:
                found.append((place, -1))
        elif instance_of(value, pd.DataFrame):
            found.extend((place, i) for i, other in enumerate(value.dtypes) if other is dtype)
    return found[0] if len(found) == 1 else None


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


def put_back_dtypes(value: Any, links: list[dict]) -> set[int]:
    """The ids of the categorical dtypes of *value*'s frames that *links*
    take from an argument on a hit (``dtype_of``): their objects are the
    argument's own again."""
    framed = {tuple(path): frame for path, frame in frames_in(value)}
    found = {
        id(dtype_at(framed[tuple(link["result"])], link["column"]))
        for link in links
        if link.get("dtype_of") and tuple(link["result"]) in framed
    }
    del framed  # a holder of each frame, to the counts the caller takes
    return found


def dtype_at(frame: Any, position: int) -> Any:
    return frame.dtype if position == -1 else frame.dtypes.iloc[position]


def linked_cell_ids(links: tuple[list[dict], dict[int, Any]] | None) -> dict[int, Any]:
    """The addresses of the cells *links* puts back, sorted, by the result
    frame holding them (`frames_together`)."""
    return links[1] if links else {}


def column_at(value: Any, position: int) -> Any:
    """The array behind *value*'s column at *position* (-1: the series),
    or None when it is not an object column."""
    column = value if position == -1 else value.iloc[:, position]
    if not is_object_dtype(column.dtype):
        return None
    return column.to_numpy(copy=False)


def _relink_dtype(frame: Any, position: int, source: Any, source_column: int, pd: Any) -> bool:
    """Make the categorical column of *frame* at *position* take the dtype
    of *source*'s column, its categories the argument's own objects; False
    when the two do not line up."""
    if not instance_of(frame, pd.DataFrame):
        return False
    column = frame.iloc[:, position]
    dtype = (source if source_column == -1 else source.iloc[:, source_column]).dtype
    if not isinstance(column.dtype, pd.CategoricalDtype) or not isinstance(dtype, pd.CategoricalDtype):
        return False
    if column.dtype.ordered != dtype.ordered or not _same_categories(column.dtype.categories, dtype.categories):
        return False
    codes = column.cat.codes.to_numpy()
    frame.isetitem(position, pd.Categorical.from_codes(codes, dtype=dtype))
    # pandas 2 makes an equal dtype of its own around the same categories.
    return frame.dtypes.iloc[position].categories is dtype.categories


def _same_categories(served: Any, own: Any) -> bool:
    """Are *served*, the stored result's categories, *own*, the argument's,
    position by position? Pickled the same, as the key reads them: ``==``
    is the user's, and a class without ``__eq__`` calls its copies unequal.
    A code points at a position, so the argument's categories in another
    order would give other cells."""
    if len(served) != len(own):
        return False
    try:
        return pickle.dumps(served.tolist(), 5) == pickle.dumps(own.tolist(), 5)
    except Exception:  # noqa: BLE001 - a category that does not pickle: not shown the same
        return False


def _walk(value: Any, path: list) -> Any:
    for step in path:
        value = value[step]
    return value


def relink_cells(value: Any, args: tuple, kwargs: dict, links: list[dict] | None) -> Any:
    """A hit's *value* with each cell *links* names put back as the
    argument's own; `RELINK_FAILED` when one cannot be."""
    if not links:
        return value
    pd = loaded_pandas()
    if pd is None:
        return RELINK_FAILED
    np = sys.modules["numpy"]
    try:
        for link in links:
            frame = _walk(value, link["result"])
            kind, where = link["source"]
            source = args[where] if kind == "a" else kwargs[where]
            if not instance_of(frame, (pd.Series, pd.DataFrame)) or not instance_of(source, (pd.Series, pd.DataFrame)):
                return RELINK_FAILED
            if link.get("dtype_of"):
                if not _relink_dtype(frame, link["column"], source, link["source_column"], pd):
                    return RELINK_FAILED
                continue
            target = column_at(frame, link["column"])
            cells = column_at(source, link["source_column"])
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
            if len(rows) and column_at(frame, link["column"])[rows[0]] is not cells[source_rows[0]]:
                return RELINK_FAILED
    except (KeyError, IndexError, TypeError, ValueError):
        return RELINK_FAILED
    return value
