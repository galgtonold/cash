"""Which variables keep their data in the same memory as another, across
libraries.

A numpy view sits on its base (`shared_objects.view_bases`), and pandas'
copy-on-write keeps two pandas objects apart. Neither covers memory one
library borrows from another without a copy:

* ``s = pl.Series(arr)``, ``pl.DataFrame({'a': arr})``, ``pa.array(arr)``,
  ``pd.Series(arr, copy=False)``: the frame reads ``arr``'s buffer, so
  ``arr[:10] = 0`` changes it, and through ``pd.Series(arr, copy=False)``,
  ``s.iloc[0] = 1`` changes ``arr`` too;
* ``s = pl.from_pandas(df)``: a write to ``df`` in place changes ``s``;
* ``x = df['a'].to_numpy()``: a read-only view of the block, which the next
  ``df.loc[0, 'a'] = 1`` writes past.

No reference count shows these (polars and Arrow hold the owner from native
code, or not at all), so they are found by address: two objects alive at the
same time whose buffers overlap share memory. Only pairs that one side can
write in place are linked: numpy with pandas, polars or Arrow, and pandas
with polars or Arrow. polars and Arrow never write into a buffer they share,
and pandas copies before it writes into a block another pandas object
shares.
"""

from __future__ import annotations

import bisect
import logging
import sys
from collections.abc import Callable, Container, Iterable, Mapping
from itertools import chain
from typing import Any

from cash import _plain_data

from .shared_objects import (
    EXACT_CONTAINER_TYPES,
    EXACT_VALUE_TYPES,
    VALUE_TYPES,
    check_walk_budget,
    children_of,
    is_value,
    library_value_types,
)

logger = logging.getLogger(__name__)

__all__ = ["borrowed_memory_names", "borrows_from", "holds_a_borrower", "lazy_frame_sources", "memory_spans"]

_NUMPY, _PANDAS, _POLARS, _ARROW = "numpy", "pandas", "polars", "arrow"

#: For an object of each family, the families of the variables whose memory
#: it may share with a write in place on one side: a numpy view of a pandas
#: block (``df['a'].to_numpy()``) and the frames built over an array or a
#: pandas frame without a copy. numpy with numpy is `view_bases`'s.
_PARTNERS = {
    _NUMPY: (_PANDAS,),
    _PANDAS: (_NUMPY,),
    _POLARS: (_NUMPY, _PANDAS),
    _ARROW: (_NUMPY, _PANDAS),
}

_Span = tuple[int, int]


def _family(value: Any) -> str | None:
    """The library *value* keeps its data with, or None (`_family_of_type`)."""
    return _family_of_type(type(value))


#: `_family_of_type` by type: a library's type only exists once it is
#: imported, so the answer never changes.
_FAMILIES: dict[type, str | None] = {}


def _family_of_type(kind: type) -> str | None:
    """`_family` by type alone, for the types numpy's, pandas', polars' and
    Arrow's modules define (None for the rest)."""
    try:
        return _FAMILIES[kind]
    except KeyError:
        pass
    except TypeError:  # an unhashable metaclass
        return None
    module = str(getattr(kind, "__module__", "")).partition(".")[0]
    family = _library_family(kind) if module in ("numpy", "pandas", "polars", "pyarrow") else None
    _FAMILIES[kind] = family
    return family


def _library_family(kind: type) -> str | None:
    np, pd = sys.modules.get("numpy"), sys.modules.get("pandas")
    pl, pa = sys.modules.get("polars"), sys.modules.get("pyarrow")
    if np is not None and issubclass(kind, np.ndarray):
        return _NUMPY
    if pd is not None and issubclass(kind, (pd.Series, pd.DataFrame, pd.Index)):
        return _PANDAS
    if pl is not None and issubclass(kind, (pl.Series, pl.DataFrame)):
        return _POLARS
    if pa is not None and issubclass(kind, (pa.Array, pa.ChunkedArray, pa.Table, pa.RecordBatch)):
        return _ARROW
    return None


#: The span of an object whose buffers a library would not show: all of
#: memory, so it overlaps every other object's. Taken as sharing memory with
#: each candidate, it is re-derived rather than restored as a copy that
#: might miss a write; a library change costs speed, never a stale value.
_EVERYWHERE: _Span = (0, 1 << 64)

#: The families whose buffers could not be read, each logged once.
_UNREADABLE: set[str] = set()


def memory_spans(value: Any) -> list[_Span]:
    """The ``[start, end)`` address ranges *value*'s data sits in: a numpy
    array's, a pandas object's blocks and axes, a polars frame's fixed-width
    columns (every chunk), an Arrow array's buffers. Empty for anything else.
    When the library will not say (its internals changed), `_EVERYWHERE`:
    an object whose memory is unknown may share any other's."""
    family = _family(value)
    try:
        if family == _NUMPY:
            return _ndarray_spans(value)
        if family == _PANDAS:
            return _pandas_spans(value)
        if family == _POLARS:
            return _polars_spans(value)
        if family == _ARROW:
            return _arrow_spans(value)
    except Exception:
        if family not in _UNREADABLE:
            _UNREADABLE.add(family)
            logger.warning(
                "cash: cannot read where a %s object keeps its data; it is taken as sharing memory "
                "with the arrays and frames beside it (re-derived, not restored).",
                family,
                exc_info=True,
            )
        return [_EVERYWHERE]
    return []


def _ndarray_spans(array: Any) -> list[_Span]:
    if not array.size:
        return []
    np = sys.modules["numpy"]
    byte_bounds = getattr(getattr(np.lib, "array_utils", None), "byte_bounds", None) or np.byte_bounds
    low, high = byte_bounds(array)
    return [(low, high)]


def _pandas_spans(value: Any) -> list[_Span]:
    """Through the blocks and the axes' arrays, not ``.values`` or
    ``.array``, which the frame memo counts as handles given out."""
    pd = sys.modules["pandas"]
    arrays: list[Any] = []
    if isinstance(value, pd.Index):
        arrays.append(_index_data(value, pd))
    else:
        arrays.extend(block.values for block in value._mgr.blocks)
        arrays.extend(_index_data(axis, pd) for axis in value.axes)
    spans: list[_Span] = []
    for array in arrays:
        spans.extend(_pandas_array_spans(array))
    return spans


def _index_data(index: Any, pd: Any) -> Any:
    """The array an index keeps its labels in; None for a range (which keeps
    none) and a MultiIndex (whose ``_data`` would build every tuple)."""
    if isinstance(index, (pd.RangeIndex, pd.MultiIndex)):
        return None
    return index._data


def _pandas_array_spans(array: Any) -> list[_Span]:
    np = sys.modules["numpy"]
    if array is None:
        return []
    if isinstance(array, np.ndarray):
        return _ndarray_spans(array)
    spans: list[_Span] = []
    # The numpy arrays an extension array is made of (dates, masked numbers,
    # categoricals, Python strings), or the Arrow array it wraps.
    for name in ("_ndarray", "_data", "_mask", "_codes"):
        part = getattr(array, name, None)
        if isinstance(part, np.ndarray):
            spans.extend(_ndarray_spans(part))
    pa_array = getattr(array, "_pa_array", None)
    if pa_array is not None:
        spans.extend(_arrow_spans(pa_array))
    return spans


def _polars_spans(value: Any) -> list[_Span]:
    """Every chunk of every column whose data a numpy array can lend: the
    fixed-width ones, and the flat values under an ``Array`` column (what
    ``pl.Series(a2d)`` keeps without a copy)."""
    pl = sys.modules["polars"]
    columns = value.get_columns() if isinstance(value, pl.DataFrame) else [value]
    spans: list[_Span] = []
    for column in columns:
        spans.extend(_polars_series_spans(column, pl))
    return spans


def _polars_series_spans(series: Any, pl: Any) -> list[_Span]:
    dtype = series.dtype
    if isinstance(dtype, pl.Array):
        return _polars_series_spans(series.reshape((-1,)), pl)
    if isinstance(dtype, pl.Struct):
        spans: list[_Span] = []
        for field in series.struct.unnest().get_columns():
            spans.extend(_polars_series_spans(field, pl))
        return spans
    width = _polars_width(dtype, pl)
    if width is None:
        return []
    spans = []
    chunks = series.get_chunks() if series.n_chunks() > 1 else [series]
    for chunk in chunks:
        span = _polars_chunk_span(chunk, width)
        if span is not None:
            spans.append(span)
    return spans


def _polars_chunk_span(chunk: Any, width: int) -> _Span | None:
    """Where the values of one polars chunk sit; None when it has none.

    polars 1 says so itself (``_get_buffer_info``); polars 2 dropped that, so
    the values are read as a numpy view (``to_numpy(allow_copy=False)``, of
    the physical type: a date's integers), and a chunk with nulls, which has
    no such view, through the Arrow array it exports without a copy. Raises
    when none of these works (`memory_spans`)."""
    buffer_info = getattr(chunk, "_get_buffer_info", None)
    if buffer_info is not None:
        address, offset, length = buffer_info()
        return (address, address + (offset + length) * width) if length else None
    if not len(chunk):
        return None
    if not chunk.null_count():
        view = chunk.to_physical().to_numpy(allow_copy=False)
        spans = _ndarray_spans(view)
        return spans[0] if spans else None
    import pyarrow as pa

    (array,) = pa.chunked_array(chunk.to_physical()).chunks
    values = array.buffers()[1]
    start = values.address + array.offset * width
    return (start, start + len(array) * width)


def _polars_width(dtype: Any, pl: Any) -> int | None:
    """Bytes per value of a fixed-width polars type (temporal ones by their
    physical integer), or None for any other."""
    physical = dtype.to_physical() if hasattr(dtype, "to_physical") else dtype
    for types, width in (
        ((pl.Int8, pl.UInt8), 1),
        ((pl.Int16, pl.UInt16), 2),
        ((pl.Int32, pl.UInt32, pl.Float32), 4),
        ((pl.Int64, pl.UInt64, pl.Float64), 8),
    ):
        if any(physical == kind for kind in types):
            return width
    for name, width in (("Int128", 16), ("UInt128", 16), ("Float16", 2)):
        kind = getattr(pl, name, None)
        if kind is not None and physical == kind:
            return width
    return None


def _arrow_spans(value: Any) -> list[_Span]:
    pa = sys.modules["pyarrow"]
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        arrays: Iterable[Any] = value.columns
    else:
        arrays = [value]
    spans: list[_Span] = []
    for array in arrays:
        chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
        for chunk in chunks:
            for buffer in chunk.buffers():
                if buffer is not None and buffer.size:
                    spans.append((buffer.address, buffer.address + buffer.size))
    return spans


class _Spans:
    """Address ranges, asked whether any of them overlaps a given range."""

    def __init__(self, spans: Iterable[_Span]) -> None:
        ordered = sorted(spans)
        self._starts = [start for start, _end in ordered]
        self._reach: list[int] = []
        reach = 0
        for _start, end in ordered:
            reach = max(reach, end)
            self._reach.append(reach)

    def __bool__(self) -> bool:
        return bool(self._starts)

    def overlaps(self, span: _Span) -> bool:
        start, end = span
        # The ranges starting before *end*; one of them reaches past *start*.
        below = bisect.bisect_left(self._starts, end)
        return below > 0 and self._reach[below - 1] > start


#: Fewest items of a container below a value that `_borrowers_in` asks
#: about as one tree of JSON-like data rather than walking it.
_TREE_MIN = 32


def _borrowers_in(values: Iterable[Any]) -> list[tuple[Any, str]]:
    """The objects among *values*, or held in them, that may share memory
    with another library's (`_PARTNERS`), with their family. A numpy array
    only when it is a view: one that owns its memory lends it, it borrows
    none."""
    value_types = VALUE_TYPES + library_value_types()
    exact, containers = EXACT_VALUE_TYPES, EXACT_CONTAINER_TYPES
    found: list[tuple[Any, str]] = []
    seen: set[int] = set()
    stack = [value for value in values if not _plain_data.is_tree(value)]
    pops = 0
    while stack:
        pops += 1
        if not pops & 1023:
            check_walk_budget()
        obj = stack.pop()
        if id(obj) in seen or is_value(obj, value_types):
            continue
        seen.add(id(obj))
        family = _family(obj)
        if family is not None:
            if family != _NUMPY or obj.base is not None:
                found.append((obj, family))
            continue
        if type(obj) in (list, tuple) and _rows_of_values(obj, exact):
            continue
        children = children_of(obj)
        if not children or exact.issuperset(map(type, children)):
            continue
        for child in children:
            kind = type(child)
            if kind in exact:
                continue
            if kind in containers:
                # A record of plain values (`{'a': 1, 'b': 'x'}`) holds no
                # frame: skipped at C speed, not pushed, for a list of millions.
                items = child.values() if kind is dict else child
                if exact.issuperset(map(type, items)) and (kind is not dict or exact.issuperset(map(type, child))):
                    continue
                if len(child) >= _TREE_MIN and kind is not tuple and _plain_data.is_tree(child):
                    continue  # JSON-like data below: no frame, read at C speed
            stack.append(child)
    return found


def _rows_of_values(items: list | tuple, exact: frozenset[type]) -> bool:
    """Whether *items* are dicts of plain values (parsed records), asked at C
    speed: a million of them hold no frame."""
    kinds = set(map(type, items))
    if kinds != {dict}:
        return False
    return exact.issuperset(map(type, chain.from_iterable(map(dict.values, items))))


def _candidates(
    user_ns: Mapping[str, Any],
    families: set[str],
    skip_name: Callable[[str], bool],
    skip_ids: set[int],
    containers: Container[str] | None,
) -> list[tuple[str, Any, str]]:
    """``(name, object, family)`` for each variable bound to an object of
    *families*, or to a builtin container directly holding one (``data =
    {'x': arr}``), when the container's name is in *containers* (every
    container when None)."""
    found: list[tuple[str, Any, str]] = []
    for name, value in list(user_ns.items()):
        if name.startswith("_") or skip_name(name) or id(value) in skip_ids:
            continue
        family = _family(value)
        if family is not None:
            if family in families:
                found.append((name, value, family))
            continue
        if containers is not None and name not in containers:
            continue
        if type(value) is dict:
            items: Iterable[Any] = value.values()
        elif type(value) in (list, tuple):
            items = value
        else:
            continue
        check_walk_budget()
        # The item types at C speed: a list of a million numbers holds no
        # array, and is not asked item by item.
        if not any(_family_of_type(kind) in families for kind in set(map(type, items))):
            continue
        for item in items:
            family = _family(item)
            if family in families and id(item) not in skip_ids:
                found.append((name, item, family))
    return found


def borrowed_memory_names(
    values: Iterable[Any],
    user_ns: Mapping[str, Any],
    skip_name: Callable[[str], bool] = lambda name: False,
    containers: Container[str] | None = None,
) -> set[str]:
    """The variables of *user_ns* whose data shares memory with an object
    among *values*, or held in them, where one side can write it in place
    (`_PARTNERS`): ``arr`` for ``s = pl.Series(arr)``, ``df`` for ``x =
    df['a'].to_numpy()``.

    A variable is a candidate when it is bound to a numpy array or a pandas
    object, or to a dict, list or tuple directly holding one; with
    *containers*, only the containers so named are looked into (a
    statement's inputs: what it built the output from, where a list of
    millions it never read costs nothing). The
    candidates' memory is looked up first, so a statement whose outputs
    could borrow from no variable costs a scan of the names, not the
    outputs' buffers. Walks the outputs under the walk budget in progress
    (`check_walk_budget`).
    """
    if "numpy" not in sys.modules:
        return set()
    borrowers = _borrowers_in(values)
    if not borrowers:
        return set()
    families = {partner for _obj, family in borrowers for partner in _PARTNERS[family]}
    own = {id(obj) for obj, _family in borrowers}
    candidates = _candidates(user_ns, families, skip_name, own, containers)
    if not candidates:
        return set()
    lent: dict[str, list[_Span]] = {}
    for obj, family in borrowers:
        check_walk_budget()
        spans = memory_spans(obj)
        for partner in _PARTNERS[family]:
            lent.setdefault(partner, []).extend(spans)
    index = {family: _Spans(spans) for family, spans in lent.items()}
    names: set[str] = set()
    for name, obj, family in candidates:
        if name in names:
            continue
        spans_of = index.get(family)
        if not spans_of:
            continue
        check_walk_budget()
        if any(spans_of.overlaps(span) for span in memory_spans(obj)):
            names.add(name)
    return names


def lazy_frame_sources(values: Iterable[Any], inputs: Iterable[str], user_ns: Mapping[str, Any]) -> set[str]:
    """The inputs a polars LazyFrame among *values* may keep its data in.

    A lazy frame shows no buffers, so ``pl.LazyFrame({'a': arr})`` or
    ``df.lazy()`` cannot be matched by address (`borrowed_memory_names`).
    Its plan holds what the statement built it from, so each input bound to
    a numpy array, a pandas, polars or Arrow object, or another lazy frame,
    is taken as a source: a write to ``arr`` reaches ``lf`` directly, one
    to the array under ``df`` through ``df``'s own link."""
    pl = sys.modules.get("polars")
    if pl is None or not any(isinstance(value, pl.LazyFrame) for value in values):
        return set()
    return {
        name
        for name in inputs
        if name in user_ns and (_family(user_ns[name]) is not None or isinstance(user_ns[name], pl.LazyFrame))
    }


def borrows_from(value: Any, sources: Iterable[Any]) -> bool:
    """Whether *value*, or an object it holds, shares memory with one of
    *sources* (or an array or frame a dict, list or tuple among them holds)
    across libraries, as `borrowed_memory_names` asks of variables: a call's
    result ``pl.Series(a)`` with its argument ``a``."""
    return bool(borrowed_memory_names([value], {f"source{index}": source for index, source in enumerate(sources)}))


def holds_a_borrower(value: Any) -> bool:
    """Whether *value* is, or holds, an object that may keep its data in
    another library's memory (a pandas, polars or Arrow object, a numpy
    view)."""
    return bool(_borrowers_in([value]))
