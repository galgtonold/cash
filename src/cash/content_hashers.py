"""Content hashers, one per library cash knows: pandas, numpy, polars,
PyArrow, modin, dask and scipy.sparse.

`builtin_hash` reads every byte of a value together with its schema --
column names, dtypes, an array's memory layout -- so the same values under
two dtypes or two layouts hash apart. The decorator keys arguments on these
hashes (directly, and inside containers through `BUILTIN_CONTENT` or its own
memo around it), and the notebook's value hash (`cash.value_hash`) uses
them for every frame, array and table it meets.

The bytes themselves -- array buffers, columns -- go through
`cash.bulk_digest`, which hashes a big buffer on several threads. Python
objects held inside one (an object column, ``attrs``) are hashed as a typed
buffer when a column holds only floats, only bools or only ints, else in
their canonical key form (`cash.canonical_form.canonical_bytes`).
`builtin_family_of` remembers per type which hasher claims it.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import weakref
from collections.abc import Callable
from typing import Any

from . import _plain_data
from ._memo import ARROW_VALUES, LruMemo
from .bulk_digest import DIRECT_MAX_BYTES, BulkHasher, fold_buffer
from .canonical_form import ContentHashing, canonical_bytes
from .kept_state import packed_records
from .sizing import SPARSE_PARTS, held_values
from .value_types import LEAF_TYPES

logger = logging.getLogger(__name__)


def builtin_hash_family(type_: type) -> str | None:
    """Name the built-in content hasher that claims *type_*, or ``None``.

    Asked of a bare TYPE as well as of a value: ``register_hasher`` needs it
    to tell a user that the hasher they are registering would never run.

    Matched on module PREFIX, so a user's own subclass defined in their own
    module is deliberately not claimed -- registering a hasher for it has
    always worked and still does.
    """
    type_name = getattr(type_, "__name__", "")
    module = getattr(type_, "__module__", "") or ""
    if module.startswith("pandas") and type_name in ("DataFrame", "Series"):
        return "pandas"
    if type_name == "ndarray" and module.startswith("numpy"):
        return "numpy"
    if module.startswith("polars"):
        return "polars"
    if module.startswith("pyarrow"):
        return "pyarrow"
    if module.startswith("modin"):
        return "modin"
    if module.startswith("dask"):
        return "dask"
    if module.startswith("scipy.sparse") and hasattr(type_, "tocsr") and hasattr(type_, "format"):
        return "scipy.sparse"
    return None


def builtin_family_of(type_: type) -> str | None:
    """`builtin_hash_family`, remembered per type: asked of every value a
    key walks."""
    try:
        return _FAMILIES[type_]
    except KeyError:
        family = _FAMILIES[type_] = builtin_hash_family(type_)
        return family
    except TypeError:  # a class whose metaclass makes it unhashable
        return builtin_hash_family(type_)


_FAMILIES: dict[type, str | None] = {}


def builtin_hash(value: Any) -> str | None:
    """Every byte of *value*, with its schema, for the library types cash knows.

    A hex digest for pandas, numpy, polars, PyArrow, modin and dask values, or
    ``None`` when no built-in hasher claims *value*'s type or hashing it failed
    -- a normal answer: the caller falls through to its own next step.

    Every hasher folds in what the values alone do not say: column and index
    names, dtypes, the memory layout of an array. The same values under two
    dtypes are two different objects to the code reading them.
    """
    family = builtin_hash_family(type(value))
    if family == "pandas":
        return hash_pandas(value)
    if family == "numpy":
        return hash_numpy(value)
    if family == "polars":
        return hash_polars(value)
    if family == "pyarrow":
        return ARROW_DIGESTS.digest(value)
    if family == "modin":
        return hash_modin(value)
    if family == "dask":
        return hash_dask(value)
    if family == "scipy.sparse":
        return hash_sparse(value)
    return None


#: How a key walk reads a frame, array or table with no memo in front:
#: every byte, through `builtin_hash`.
BUILTIN_CONTENT = ContentHashing(builtin_family_of, builtin_hash)


def hash_pandas(value: Any) -> str | None:
    """Hash a pandas DataFrame or Series over values AND schema.

    ``hash_pandas_object`` covers row values + index values but NOT the
    schema labels: column names, ``Series.name``, and index name(s) are
    invisible to it, so ``df.rename(columns=...)`` (or an empty frame of any
    shape) collided with the original and returned its cached result. Fold
    the labels in as a digest prefix.

    The dtypes go in for the same reason, and it is the sharper one: the same
    values under two dtypes are two different objects to the body. A tz-naive
    and a tz-aware series collided, and the tz-aware call was served the naive
    one's ``TypeError: Cannot convert tz-naive timestamps``; so did
    ``int64``/``Int64`` (pd.NA semantics), ``int64``/``int32`` and a
    categorical against an object column.

    The values are hashed column by column and index level by index level
    (`_fold_pandas_values`), each in the form that keeps it exact.
    """
    try:
        import pandas as pd

        index = value.index
        index_dtypes = [_pandas_dtype_key(dt) for dt in getattr(index, "dtypes", [index.dtype])]
        # What the labels and dtypes leave out, and code reads: the column
        # axis's own name (``melt``, ``stack`` and ``reset_index`` name
        # columns after it), an index's ``freq`` (``shift(freq=...)``,
        # ``asfreq``) and ``attrs``.
        axes = f"{list(index.names)!r}:{index_dtypes!r}:{getattr(index, 'freq', None)!r}:"
        if type(value).__name__ == "DataFrame":
            dtypes = _pandas_dtype_keys(value.dtypes)
            schema = f"{list(value.columns)!r}:{list(value.columns.names)!r}:{dtypes!r}:{axes}"
        else:  # Series
            schema = f"{value.name!r}:{_pandas_dtype_key(value.dtype)!r}:{axes}"
        h = hashlib.sha256(schema.encode("utf-8"))
        if value.attrs:
            h.update(canonical_bytes(value.attrs, BUILTIN_CONTENT))
        _fold_pandas_values(h, value, pd)
        return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, NotImplementedError, pickle.PicklingError):
        logger.debug("Failed to hash pandas %s via hash_pandas_object", type(value).__name__)
        return None


def _fold_pandas_values(h: Any, value: Any, pd: Any) -> None:
    """Fold a frame's or series' values, then its index's, into *h*.

    Column by column and index level by index level, each in the form that
    keeps it exact and reads it fastest (`pandas_value_route`): numbers,
    booleans and dates from their own buffer, a nullable column from its
    values and its mask, an Arrow-backed column (pandas' ``str`` among them)
    from its Arrow buffers, a categorical by its codes (the categories are
    in the schema).

    ``hash_pandas_object`` keys an array of Python objects by ``str()`` of
    each: when one value is not a string it stringifies the whole column, so
    ``1`` and ``'1'``, ``True`` and ``'True'``, ``b'a'`` and ``'a'``, or a date
    and its ISO string hashed alike, and ``s * 2`` was served ``2`` for
    ``'11'``. The same held for an index of them. Such an array is keyed by
    its items (`_fold_object_items`) instead. Only what no route reads --
    periods, intervals, sparse columns -- keeps ``hash_pandas_object``'s
    per-value hash.
    """
    if type(value).__name__ == "DataFrame":
        rest = []
        column_buffer = _column_buffers(value)
        routes: dict[Any, str | None] = {}
        for pos, dtype in enumerate(value.dtypes):
            try:
                route = routes[dtype]
            except KeyError:
                route = routes[dtype] = pandas_value_route(dtype, pd)
            except TypeError:  # an unhashable dtype
                route = pandas_value_route(dtype, pd)
            if route is None:
                rest.append(pos)
                continue
            h.update(f"|{pos}|".encode())
            pieces = _COLUMN_PIECES.current
            key = column_memory_key(value, pos, route) if pieces is not None else None
            if key is not None:
                piece = pieces.find(key)
                if piece is not None:
                    h.update(piece)
                    continue
                into = _PieceRecorder(h)
            else:
                into = h
            buffer = column_buffer(pos, dtype) if route == "raw" else None
            if buffer is not None:
                fold_buffer(into, raw_view(buffer))
            else:
                _fold_pandas_array(into, value.iloc[:, pos], pd)
            if key is not None and into.piece is not None:
                pieces.note(key, into.piece)
        if rest:
            others = value if len(rest) == value.shape[1] else value.iloc[:, rest]
            h.update(b"|others|")
            fold_buffer(h, _np_array(pd.util.hash_pandas_object(others, index=False)))
    else:
        _fold_pandas_array(h, value, pd)
    index = value.index
    if isinstance(index, pd.MultiIndex):
        # Each level's distinct values, then which one each row holds.
        for level, codes in zip(index.levels, index.codes):
            h.update(b"|level|")
            _fold_pandas_array(h, level, pd)
            fold_buffer(h, _np_array(codes))
    elif isinstance(index, pd.RangeIndex):
        h.update(f"|range({index.start}, {index.stop}, {index.step})".encode())
    else:
        h.update(b"|index|")
        _fold_pandas_array(h, index, pd)
    if type(value).__name__ == "DataFrame":
        # The schema keys the column labels by ``repr``, which for an object
        # of the user's own is its address: fold the objects themselves.
        columns = value.columns
        levels = columns.levels if isinstance(columns, pd.MultiIndex) else [columns]
        if any(level.dtype == object and not immutable_labels(level._values) for level in levels):
            h.update(b"|columns|")
            _fold_pandas_array(h, columns, pd)


# What `pandas.api.types.infer_dtype` calls an array of labels no one can
# change in place: str, bytes, numbers, dates and times.
_IMMUTABLE_LABEL_KINDS = frozenset(
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
        "datetime",
        "datetime64",
        "date",
        "time",
        "timedelta",
        "timedelta64",
    }
)


def immutable_labels(values: Any) -> bool:
    """Whether an object-dtype pandas axis holds only immutable scalars, read
    by pandas in C. An index of ``Site`` objects changes with
    ``a.capacity = 100`` without any array or label ``repr`` changing."""
    from pandas.api.types import infer_dtype

    return infer_dtype(values, skipna=False) in _IMMUTABLE_LABEL_KINDS


def immutable_items(values: Any) -> bool:
    """Whether an object-dtype array holds only immutable scalars and missing
    values: `immutable_labels` with gaps allowed, as a column of strings
    has them. Read by pandas in C, about 10 ms per million items; a 2-d
    block of several such columns is read as one."""
    from pandas.api.types import infer_dtype

    flat = values.ravel(order="K") if values.ndim > 1 else values
    return infer_dtype(flat, skipna=True) in _IMMUTABLE_LABEL_KINDS


#: numpy dtype kinds whose buffer IS the values: bool, integers, floats,
#: complex numbers, datetimes and timedeltas.
_RAW_KINDS = frozenset("biufcmM")


def pandas_value_route(dtype: Any, pd: Any) -> str | None:
    """How `_fold_pandas_array` keys values of *dtype*: ``"raw"``,
    ``"datetimetz"``, ``"masked"``, ``"arrow"``, ``"objects"``, ``"codes"``,
    or None for ``hash_pandas_object``."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "codes"
    if isinstance(dtype, _numpy().dtype):
        kind = dtype.kind
        # Not an extended-precision float: its padding bytes hold garbage.
        if kind in _RAW_KINDS and dtype.itemsize <= (16 if kind == "c" else 8):
            return "raw"
        if str(dtype) == "object":
            return "objects"
        return None
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "datetimetz"
    if isinstance(dtype, pd.StringDtype):
        return "arrow" if str(getattr(dtype, "storage", "")).startswith("pyarrow") else "objects"
    if isinstance(dtype, getattr(pd, "ArrowDtype", ())):
        return "arrow"
    if type(dtype).__name__ in _MASKED_DTYPES:
        return "masked"
    return None


#: pandas' nullable dtypes: a numpy array of values beside a mask of the
#: missing ones.
_MASKED_DTYPES = frozenset(
    {
        *(f"{base}{bits}Dtype" for base in ("Int", "UInt") for bits in (8, 16, 32, 64)),
        "Float32Dtype",
        "Float64Dtype",
        "BooleanDtype",
    }
)


def _numpy() -> Any:
    import numpy

    return numpy


def _fold_pandas_array(h: Any, values: Any, pd: Any) -> None:
    """Fold one Series' or Index's values into *h* (`_fold_pandas_values`)."""
    route = pandas_value_route(values.dtype, pd)
    if route == "raw":
        fold_buffer(h, raw_view(_np_array(values)))
        return
    if route == "datetimetz":
        asi8 = getattr(held_values(values), "asi8", None)
        if asi8 is not None:
            fold_buffer(h, raw_view(asi8))
            return
        route = None
    elif route == "masked":
        array = held_values(values)
        data, mask = getattr(array, "_data", None), getattr(array, "_mask", None)
        if data is not None and mask is not None:
            # The values under a missing one are whatever was there: two equal
            # columns can differ in them, and only hash apart -- never alike.
            h.update(b"|masked|")
            fold_buffer(h, raw_view(data))
            fold_buffer(h, raw_view(mask))
            return
        route = None
    if route == "codes":
        fold_buffer(h, _np_array(values.codes if isinstance(values, pd.Index) else values.cat.codes))
    elif route == "objects":
        _fold_object_items(h, _np_array(values, object))
    elif route == "arrow":
        fold_arrow_column(h, held_values(values).__arrow_array__())
    else:
        fold_buffer(h, _np_array(pd.util.hash_pandas_object(values, index=False)))


class ColumnPieces:
    """What `_fold_pandas_values` asks while the frame memo hashes a frame
    (`column_pieces`): the bytes a column whose memory it knows folds into
    a frame's digest (``find``), and the bytes each column it read folded
    (``note``). `cash.decorator.arg_hashing.FrameDigests` answers only from
    a frame it can still vouch is unchanged, so a column a new frame shares
    with it (``df.assign(x=...)``, ``df[cols]``, a loop's ``matches.assign``)
    is not read again. The digest is the same either way: the same bytes
    go into it."""

    def find(self, key: tuple) -> bytes | None:  # pragma: no cover - interface
        return None

    def note(self, key: tuple, piece: bytes) -> None:  # pragma: no cover - interface
        return None


class _CurrentPieces(threading.local):
    current: ColumnPieces | None = None


#: The `ColumnPieces` the frame memo set on this thread while it hashes a
#: frame (`column_pieces`); None, the usual case, reads every column.
_COLUMN_PIECES = _CurrentPieces()


def column_pieces(pieces: ColumnPieces | None) -> ColumnPieces | None:
    """Make *pieces* the `ColumnPieces` this thread's frame hashes consult;
    returns the one it replaces, to be put back."""
    previous = _COLUMN_PIECES.current
    _COLUMN_PIECES.current = pieces
    return previous


#: Who knows the pieces of a frame it vouches is unchanged now
#: (`known_column_pieces`): the frame memo registers itself
#: (`cash.decorator.arg_hashing.FrameDigests.pieces_of`).
_PIECE_SOURCE: list[Callable[[Any], Any]] = []


def set_piece_source(source: Callable[[Any], Any]) -> None:
    """Make *source* what `known_column_pieces` asks."""
    _PIECE_SOURCE[:] = [source]


def known_column_pieces(frame: Any) -> Any:
    """What the columns of the pandas frame *frame* fold into a digest, as
    far as the frame memo can vouch for them now (`ColumnPieces`): an object
    whose ``get(column memory key)`` is the piece, a function of the
    column's content alone, or None. None when there is no memo."""
    if not _PIECE_SOURCE:
        return None
    try:
        return _PIECE_SOURCE[0](frame)
    except Exception:  # noqa: BLE001 - only a speedup: nothing is known
        return None


#: The most bytes a column's piece may hold: a big column's part of a
#: digest is a tag and a tree digest; one that adds its raw bytes is small
#: enough to read again.
_PIECE_MAX_BYTES = 512


class _PieceRecorder:
    """Passes every ``update`` on to *h* and keeps the bytes, while they
    stay under `_PIECE_MAX_BYTES` (``piece``, else None)."""

    __slots__ = ("_h", "_parts", "_size")

    def __init__(self, h: Any) -> None:
        self._h = h
        self._parts: list[bytes] | None = []
        self._size = 0

    def update(self, data: Any) -> None:
        self._h.update(data)
        if self._parts is None:
            return
        self._size += memoryview(data).nbytes
        if self._size > _PIECE_MAX_BYTES:
            self._parts = None
        else:
            self._parts.append(bytes(data))

    @property
    def piece(self) -> bytes | None:
        return b"".join(self._parts) if self._parts is not None else None


def column_memory_key(frame: Any, pos: int, route: str | None) -> tuple | None:
    """What identifies the memory column *pos* of *frame* reads from, for
    `ColumnPieces`: the address, shape, strides and type of its numpy data
    (a numpy block's row, the array under dates or strings held as Python
    objects), or the Arrow array it holds, with its dtype and route. None
    for any other column, which is read every time.

    Only a key: the memo that answers it vouches that the memory has not
    changed since, and holds what keeps it alive."""
    import numpy as np

    try:
        mgr = frame._mgr
        values = mgr.blocks[mgr.blknos[pos]].values
        dtype = str(frame.dtypes.iloc[pos])
        if type(values) is np.ndarray:
            if values.ndim != 2:
                return None
            array = values[mgr.blklocs[pos]]
        else:
            arrow = getattr(values, "_pa_array", None)
            if arrow is not None:
                return _arrow_memory_key(arrow, dtype, route)
            array = getattr(values, "_ndarray", None)
            if type(array) is not np.ndarray:
                return None
            if array.ndim == 2:  # dates kept as a 2-d block
                array = array[mgr.blklocs[pos]]
            elif array.ndim != 1:
                return None
        interface = array.__array_interface__
        return (
            "numpy",
            type(values).__name__,
            interface["data"][0],
            array.shape,
            array.strides,
            array.dtype.str,
            dtype,
            route,
        )
    except Exception:  # noqa: BLE001 - pandas internals changed: read the column
        return None


#: Chunks an Arrow column may have for `column_memory_key` to name them.
_KEYED_CHUNKS = 64


def _arrow_memory_key(arrow: Any, dtype: str, route: str | None) -> tuple | None:
    """`column_memory_key` of an Arrow-backed column: each chunk's type,
    offset, length and the address and size of each of its buffers. Arrow
    buffers are never written; pandas wraps the same ones anew in each
    frame that shares the column."""
    chunks = getattr(arrow, "chunks", None)
    if chunks is None:
        chunks = [arrow]
    if len(chunks) > _KEYED_CHUNKS:
        return None
    parts = tuple(
        (
            str(chunk.type),
            chunk.offset,
            len(chunk),
            tuple((buf.address, buf.size) if buf is not None else None for buf in chunk.buffers()),
        )
        for chunk in chunks
    )
    return ("arrow", parts, dtype, route)


def _column_buffers(frame: Any) -> Callable[[int, Any], Any]:
    """``buffer(pos, dtype)``: the numpy array holding column *pos* of
    *frame*, read from its block, or None when that column is not a plain
    numpy array of *dtype* (an extension array, a pandas without the
    internals read here). It is the array ``np.asarray(frame.iloc[:, pos])``
    gives, so the digest is the same, without a Series built per column: a
    2,000-column frame spent half its hash in ``iloc``.

    The array is a view into the block: read here, never handed on.
    """
    import numpy as np

    try:
        mgr = frame._mgr
        blocks, blknos, blklocs = mgr.blocks, mgr.blknos, mgr.blklocs
    except Exception:  # noqa: BLE001 - pandas internals changed: through iloc
        return lambda pos, dtype: None

    def buffer(pos: int, dtype: Any) -> Any:
        try:
            values = blocks[blknos[pos]].values
            if type(values) is not np.ndarray or values.ndim != 2:
                return None
            column = values[blklocs[pos]]
        except Exception:  # noqa: BLE001 - through iloc
            return None
        return column if column.dtype == dtype else None

    return buffer


def raw_view(array: Any) -> Any:
    """*array*'s buffer as something ``memoryview`` takes: datetimes and
    timedeltas, which the buffer protocol refuses, as their int64 counts
    (their unit is in the dtype the key holds)."""
    if array.dtype.kind in "mM":
        return array.view("i8")
    return array


def fold_arrow_column(h: Any, column: Any) -> None:
    """Fold a pyarrow Array or ChunkedArray's values into *h*, exactly.

    Numbers, dates and times by their values buffer, strings and binaries
    by each value's length and their data buffer, booleans byte by byte,
    and the positions of the missing values: read from each chunk's own
    offset, so a slice keys as its rows, and the same rows in any chunking
    key alike. Each part is hashed as one stream (`BulkHasher`), on several
    threads when it is big. A dictionary column is read by its values.
    Other types -- lists, structs, decimals -- are written as an Arrow IPC
    stream, which carries the validity bitmap and the values as they are,
    and each chunk's rows from its offset (`_fold_arrow_ipc`).

    The values under a missing one are whatever the buffer holds there: two
    equal columns can differ in them, and only hash apart, never alike.
    """
    import numpy as np
    import pyarrow as pa

    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    if pa.types.is_dictionary(column.type):
        # By the values: which index stands for which value is the
        # dictionary's order, which one process builds as it meets them.
        h.update(f"|dictionary:{column.type}|".encode())
        chunks = [chunk.dictionary_decode() for chunk in chunks]
        column = pa.chunked_array(chunks, type=column.type.value_type)
    kind, width = _arrow_layout(column.type)
    if kind is None:
        _fold_arrow_ipc(h, column)
        return
    nulls, lengths, values = BulkHasher(), BulkHasher(), BulkHasher()
    base = 0
    for chunk in chunks:
        n = len(chunk)
        if not n:
            continue
        if chunk.null_count:
            missing = np.flatnonzero(~chunk.is_valid().to_numpy(zero_copy_only=False)) + base
            nulls.update(missing.astype("<i8"))
        buffers = chunk.buffers()
        if kind == "fixed":
            values.update(memoryview(buffers[1])[chunk.offset * width : (chunk.offset + n) * width])
        elif kind == "bool":
            filled = chunk.fill_null(False) if chunk.null_count else chunk
            values.update(np.asarray(filled.to_numpy(zero_copy_only=False), dtype=np.bool_))
        else:  # offsets + data
            offsets = np.frombuffer(buffers[1], dtype=f"<i{width}", count=n + 1, offset=chunk.offset * width)
            lengths.update(np.diff(offsets))
            if buffers[2] is not None and offsets[-1] > offsets[0]:
                values.update(memoryview(buffers[2])[int(offsets[0]) : int(offsets[-1])])
        base += n
    h.update(f"|arrow:{column.type}:{kind}:{base}|".encode())
    h.update(nulls.digest() + lengths.digest() + values.digest())


def _arrow_layout(arrow_type: Any) -> tuple[str | None, int]:
    """How `fold_arrow_column` reads values of *arrow_type*: ``("fixed",
    bytes per value)``, ``("bool", 1)``, ``("offsets", bytes per offset)``,
    or ``(None, 0)`` for the IPC stream."""
    import pyarrow as pa

    t = pa.types
    if t.is_boolean(arrow_type):
        return "bool", 1
    if t.is_string(arrow_type) or t.is_binary(arrow_type):
        return "offsets", 4
    if t.is_large_string(arrow_type) or t.is_large_binary(arrow_type):
        return "offsets", 8
    if (
        t.is_integer(arrow_type)
        or t.is_floating(arrow_type)
        or t.is_timestamp(arrow_type)
        or t.is_date(arrow_type)
        or t.is_time(arrow_type)
        or t.is_duration(arrow_type)
    ):
        bits = arrow_type.bit_width
        if bits % 8 == 0:
            return "fixed", bits // 8
    return None, 0


def _fold_arrow_ipc(h: Any, column: Any) -> None:
    """Fold a pyarrow column into *h* as an Arrow IPC stream: the validity
    bitmap and the values as they are, in every type."""
    import pyarrow as pa

    table = pa.table({"v": column})
    h.update(b"|ipc|")
    with pa.ipc.new_stream(pa.PythonFile(_HashSink(h), mode="w"), table.schema) as writer:
        writer.write(table)


def _np_array(values: Any, dtype: Any = None) -> Any:
    """``np.asarray(values)``: a pandas value's array through ``__array__``,
    not ``to_numpy()``, which the decorator's frame memo watches for handles
    a caller could write through (`arg_hashing.watch_writable_handles`)."""
    import numpy as np

    return np.asarray(values, dtype=dtype)


def _np_bytes(values: Any) -> bytes:
    return _np_array(values).tobytes()


#: Python types a column of which is keyed by the numpy array it converts
#: to: its buffer holds every value exactly, and reads ~10x faster than
#: pickling them one by one. Exact types: an ``np.float64`` is a ``float``
#: subclass and stays on the pickled path.
_TYPED_ITEMS = {float: "<f8", bool: "?", int: "<i8"}


def _fold_object_items(h: Any, array: Any) -> None:
    """Fold a 1-d object array's items into *h*: as a typed buffer when
    every item is a ``float``, every one a ``bool`` or every one an ``int``
    that fits 64 bits (`_fold_typed_items`), else pickled
    (`_object_items_bytes`). Pickled items bigger than one leaf go in as
    their tree digest (`fold_buffer`), so what the column adds to a frame's
    digest is a few dozen bytes the frame memo can keep (`ColumnPieces`)."""
    items = array.tolist()
    if not _fold_typed_items(h, array, items):
        blob = _object_items_bytes(items)
        if len(blob) > DIRECT_MAX_BYTES:
            h.update(b"|items-tree|")
            fold_buffer(h, blob)
        else:
            h.update(blob)


def _fold_typed_items(h: Any, array: Any, items: list) -> bool:
    """Fold *items* (the items of the 1-d object *array*) into *h* as the
    numpy array of one `_TYPED_ITEMS` type they convert to; False, folding
    nothing, when they are not all of one such type."""
    kinds = set(map(type, items))
    if len(kinds) != 1:
        return False
    kind = kinds.pop()
    dtype = _TYPED_ITEMS.get(kind)
    if dtype is None:
        return False
    try:
        typed = array.astype(dtype)
    except (OverflowError, ValueError, TypeError):  # an int beyond 64 bits
        return False
    h.update(f"|items:{kind.__name__}|".encode())
    fold_buffer(h, typed)
    return True


def _fold_object_array(h: Any, value: Any) -> None:
    """Fold an object-dtype ndarray's items into *h*.

    Its buffer holds raw ``PyObject`` pointers, not content: identical
    content in fresh objects would never hash alike (permanent misses,
    unstable across processes) and a reused address could alias distinct
    content onto one key. So the items are read. A table of them (2-d, a
    ``DataFrame.to_numpy()`` of mixed columns) column by column, each column
    of floats, bools or ints as its typed buffer (`_fold_typed_items`) and
    the other columns pickled together, so an object held in two of them is
    still seen held twice.
    """
    if value.ndim == 2 and value.shape[0] and value.shape[1] > 1:
        rest = []
        for j in range(value.shape[1]):
            column = value[:, j]
            items = column.tolist()
            h.update(f"|col {j}|".encode())
            if not _fold_typed_items(h, column, items):
                h.update(b"|later|")
                rest.append(items)
        if rest:
            h.update(b"|rest|")
            h.update(_object_items_bytes(rest))
        return
    flat = value.reshape(-1)
    if not _fold_typed_items(h, flat, flat.tolist()):
        h.update(_object_items_bytes(value.tolist()))


def _object_items_bytes(items: list) -> bytes:
    """The bytes a list of Python objects keys on: plain data pickled as it
    is (`_plain_data`, C speed), anything else in its canonical form
    (`stable_key_repr`), so sets and dicts inside are in a stable order."""
    if _plain_data.is_plain(items):
        return b"P" + _plain_data.pickle_unshared(items)
    # The array or column holds each item besides *items*.
    tree = _plain_data.sharing(items, tree=True, held_twice_at=0)
    if tree is not None:
        return b"T" + _plain_data.pickle_unshared((items, tree[0]))
    return b"S" + canonical_bytes(items, BUILTIN_CONTENT)


def _pandas_dtype_keys(dtypes: Any) -> list[str]:
    """`_pandas_dtype_key` of each of *dtypes*, a numpy dtype worked out
    once however many columns have it. Only a numpy dtype: two categorical
    dtypes over the same categories in another order are equal, and keyed
    apart."""
    import numpy as np

    known: dict[Any, str] = {}
    keys = []
    for dtype in dtypes:
        if type(dtype) is not np.dtype and not isinstance(dtype, np.dtype):
            keys.append(_pandas_dtype_key(dtype))
            continue
        key = known.get(dtype)
        if key is None:
            key = known[dtype] = _pandas_dtype_key(dtype)
        keys.append(key)
    return keys


def _pandas_dtype_key(dtype: Any) -> str:
    """A pandas dtype as a key reads it: ``repr``, and all of a categorical.

    ``str`` is ``'category'`` for every categorical, and ``repr`` elides
    a long list of categories, so the categories (every one, by content) and
    the ``ordered`` flag are spelled out: ``get_dummies``, ``value_counts`` and ``cat.codes``
    read them, and two series of the same values over different categories
    were served each other's columns.
    """
    categories = getattr(dtype, "categories", None)
    if categories is None or getattr(dtype, "name", None) != "category":
        return repr(dtype)
    listed = hashlib.sha256(_object_items_bytes(categories.tolist())).hexdigest()
    return f"category:{dtype.ordered!r}:{_pandas_dtype_key(categories.dtype)}:{listed}"


def array_layout(value: Any) -> str:
    """The order *value*'s axes are laid out in memory: ``C``, ``F`` or ``K…``.

    The key used to fold in the raw strides (0fd2cb5), which separated C-
    from F-ordered arrays -- the point -- but also a strided VIEW from its
    contiguous copy. Those hold the same values in the same memory order,
    so no order-reading callee (``ravel(order='A'/'K')``, ``reshape``) tells
    them apart -- only ``.flags`` does, and a result computed FROM
    contiguity now shares an entry between the two, knowingly. What did
    tell them apart, on every run, was the cache itself. A function that
    returned ``arr[:, 0]`` handed its caller a view on the computing run and
    a contiguous copy on every restored one, so the caller's key changed
    between the two and its expensive step ran twice after every upstream
    edit.

    So: the axes of length > 1, ordered by |stride| from outermost in, with
    a broadcast (zero-stride) axis outermost. Identity is ``C``, reversed is
    ``F``; anything else spells the permutation. Stride MAGNITUDE and sign
    do not change what a memory-order read returns, so they stay out.

    Except for one flag. ``order='A'`` (``ravel``, ``reshape``, ``tobytes``,
    ``copy``) reads in Fortran order only when the array is F-CONTIGUOUS,
    and C order otherwise -- so an F-like strided view (``a.T[::2]``) and
    its F-contiguous copy read differently, and sharing ``F`` handed one the
    other's result. ``Fs`` is the F-like array that is not
    F-contiguous. Nothing else needs the flag: with two or more axes longer
    than 1, only an F-like layout can be F-contiguous, and ``order='A'``
    reads everything else in C order, as ``C`` and ``K…`` already imply.

    One case still re-keys once: an F-like but non-contiguous view is stored
    by pickle as a C-ordered copy, and it genuinely ravels differently from
    one, so the restored value must key apart. Safe direction.
    """
    axes = [(axis, stride) for axis, (n, stride) in enumerate(zip(value.shape, value.strides)) if n > 1]
    if len(axes) <= 1:
        return "C"
    outer_first = sorted(axes, key=lambda a: (-abs(a[1]) if a[1] else float("-inf"), a[0]))
    perm = tuple(axis for axis, _ in outer_first)
    natural = tuple(axis for axis, _ in axes)
    if perm == natural:
        return "C"
    if perm == natural[::-1]:
        return "F" if value.flags.f_contiguous else "Fs"
    return "K" + ",".join(map(str, perm))


def hash_numpy(value: Any) -> str | None:
    """Hash a numpy ndarray over its FULL contents.

    Correctness requires hashing every byte, not a sample: two large arrays
    that differ only outside a sampled window would otherwise collide and
    return a wrong cached result (a silent data-corruption bug, especially for
    the large ML/data arrays caching targets). Shape and dtype are folded in
    so a reshape or retype of the same bytes does not collide. Uses a
    zero-copy ``memoryview`` for contiguous arrays and a C-order copy
    otherwise; a big buffer is hashed on several threads (`fold_buffer`).
    An object array is keyed by its items (`_fold_object_array`).

    The LAYOUT is folded in too -- the order the axes sit in memory, see
    `array_layout` -- because the C-order fallback above erases it. Without
    it a C-ordered and an F-ordered array holding equal values hash
    identically, and a layout-sensitive callee is served the other one's
    result: measured, ``np.ravel(x, order='A')`` returned ``[0, 1, 2, …]``
    for an F-ordered input whose true answer is ``[0, 4, 8, 1, …]``.
    Normalising to C-order is right for value EQUALITY and wrong for a KEY.
    """
    try:
        h = hashlib.sha256(f"{value.shape}:{value.dtype}:{array_layout(value)}:".encode())
        if getattr(value.dtype, "hasobject", False):
            if value.dtype == object:
                _fold_object_array(h, value)
            else:  # a structured dtype with an object field
                h.update(_object_items_bytes(value.tolist()))
            return h.hexdigest()
        # A record dtype's padding holds whatever the memory held.
        value = packed_records(value)
        try:
            data = memoryview(raw_view(value))  # no copy if C-contiguous
        except (TypeError, ValueError):
            data = value.tobytes()  # a dtype the buffer protocol refuses
        fold_buffer(h, data)
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError, pickle.PicklingError):
        logger.debug("Failed to hash numpy ndarray")
        return None


def hash_polars(value: Any) -> str | None:
    """Hash a polars DataFrame, Series, or LazyFrame, schema included.

    Each column is read from its Arrow buffers (`fold_arrow_column`), with no
    copy for numbers, and the schema -- names and dtypes -- is folded in
    ahead of them, so an ``Int32`` and an ``Int64`` column holding the same
    numbers, or a renamed column, key apart. Polars' own ``hash_rows()`` is
    a 64-bit hash per row: not a key, and slower.

    An ``Object`` column is keyed by its items' content
    (`_object_items_bytes`). Polars hashes Object values with Python's
    ``hash()``, which panics on an unhashable one (an ndarray, a dict) and is
    no content key: ``hash(-1) == hash(-2)``, ``1``, ``1.0`` and ``True``
    share a hash, and a string's changes with ``PYTHONHASHSEED``.

    A ``LazyFrame`` is identified by ``serialize()``, never by ``explain()``.
    ``explain()`` renders the human-readable QUERY PLAN, and two frames over
    different in-memory data print identically -- both
    ``pl.DataFrame({"x": [1, 2, 3]}).lazy()`` and the same over
    ``[10, 20, 30]`` are ``DF ["x"]; PROJECT */1 COLUMNS``, so the second
    call was served the first's result. ``serialize()`` carries the plan
    *and* the data the plan closes over, and is byte-identical across
    processes, so persisted entries still hit after a restart. A plan
    ``serialize()`` refuses gets no built-in hash at all.

    KNOWN GAP: a plan that reads from an external source
    (``scan_csv``/``scan_parquet``/...) serializes the PATH, not the file's
    contents, so editing that file in place does not move the key. Closing
    that would mean collecting the frame to build a cache key, which defeats
    the point of a LazyFrame and can be arbitrarily expensive. Collect before
    passing, or name the file with ``file_depends_on=``.
    """
    try:
        import polars as pl

        if isinstance(value, pl.DataFrame):
            h = hashlib.sha256(f"{value.schema}:{value.height}:".encode("utf-8"))
            objects = [name for name, dtype in value.schema.items() if dtype == pl.Object]
            rest = value.drop(objects) if objects else value
            if rest.width:
                table = _polars_to_arrow(rest, pl)
                for name, column in zip(table.column_names, table.columns):
                    h.update(f"|{name!r}|".encode("utf-8"))
                    fold_arrow_column(h, column)
            for name in objects:
                h.update(f"|{name!r}|".encode("utf-8"))
                h.update(_object_items_bytes(value.get_column(name).to_list()))
            return h.hexdigest()
        if isinstance(value, pl.Series):
            h = hashlib.sha256(f"{value.name!r}:{value.dtype}:{len(value)}:".encode("utf-8"))
            if value.dtype == pl.Object:
                h.update(_object_items_bytes(value.to_list()))
            else:
                fold_arrow_column(h, _polars_to_arrow(value, pl))
            return h.hexdigest()
        if isinstance(value, pl.LazyFrame):
            try:
                return hashlib.sha256(value.serialize()).hexdigest()
            except Exception:  # noqa: BLE001 - polars raises its own types
                logger.debug("polars LazyFrame serialize() failed; no built-in hash for it")
                return None
    except (ImportError, TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash polars %s", type(value).__name__)
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        logger.debug("polars panicked hashing a %s: %s", type(value).__name__, exc)
    return None


def polars_signature(value: Any) -> tuple | None:
    """What must stay the same for a polars DataFrame's or Series' content
    hash to hold, read from its buffers' addresses, not their bytes: its
    schema (names, dtypes) and height, and for every chunk of every column
    the offset, the length and each Arrow buffer's address and size, as
    polars exports them without a copy. None for anything else, for a frame
    with an ``Object`` column (Python objects change in place), and when
    the export fails.

    Polars writes in place (``df[0, "a"] = 1``, ``s.scatter``, ``extend``,
    ``df.columns = [...]``) only into memory nothing else shares: with a
    copy of the frame held (``clone()``, which shares every buffer), each
    such write gives the frame new buffers at new addresses, the old ones
    still alive. So while the signature stays the same, no write through
    polars happened. A write past polars into memory it borrowed (a numpy
    array it was built over without a copy) is not seen: a caller trusts it
    only over a span no such write can fall in (`ArgFingerprints`).
    """
    try:
        import polars as pl

        frame = value.to_frame() if isinstance(value, pl.Series) else value
        if not isinstance(frame, pl.DataFrame):
            return None
        schema = tuple((name, repr(dtype)) for name, dtype in frame.schema.items())
        if any("Object" in dtype for _, dtype in schema):
            return None
        table = frame.to_arrow(compat_level=pl.CompatLevel.newest())
        columns = tuple(
            tuple(
                (chunk.offset, len(chunk), tuple(None if b is None else (b.address, b.size) for b in chunk.buffers()))
                for chunk in column.chunks
            )
            for column in table.columns
        )
        return (schema, frame.height, columns)
    except Exception:  # noqa: BLE001 - polars or pyarrow refused: no signature, hash it
        return None
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        return None


def _polars_to_arrow(value: Any, pl: Any) -> Any:
    """A polars DataFrame or Series as Arrow, strings as ``large_string``
    (the oldest compat level) rather than the views newer polars hands out,
    which `fold_arrow_column` would have to write as an IPC stream."""
    compat = getattr(pl, "CompatLevel", None)
    if compat is not None:
        try:
            return value.to_arrow(compat_level=compat.oldest())
        except TypeError:
            pass
    return value.to_arrow()


def held_objects(value: Any) -> list | None:
    """The Python objects a library value keeps in object storage, other
    than plain leaves; None when it keeps none.

    A numpy ``object`` array, a pandas ``object`` column and a polars
    ``Object`` column hold arbitrary objects -- a model, a function -- where
    no attribute walk reaches them: in the array buffer, in the blocks. Their
    content hashers pickle such an object by reference, so its code is in no
    key unless the code search is handed them here. Strings, numbers and
    dates, which fill most object columns, are left out at C speed.
    """
    family = builtin_family_of(type(value))
    columns: list = []
    try:
        if family == "numpy":
            if getattr(value.dtype, "hasobject", False):
                columns.append(value.ravel(order="K"))
        elif family == "pandas":
            frame = type(value).__name__ == "DataFrame"
            dtypes = list(value.dtypes) if frame else [value.dtype]
            # Asked of each distinct dtype once: ``str`` of a dtype per
            # column of a 2,000-column frame was most of a call's key.
            if any(str(dtype) == "object" for dtype in _distinct(dtypes)):
                for pos, dtype in enumerate(dtypes):
                    if str(dtype) == "object":
                        columns.append(_np_array(value.iloc[:, pos] if frame else value, object))
        elif family == "polars":
            import polars as pl

            if isinstance(value, pl.DataFrame):
                columns.extend(value.get_column(n).to_list() for n, dt in value.schema.items() if dt == pl.Object)
            elif isinstance(value, pl.Series) and value.dtype == pl.Object:
                columns.append(value.to_list())
    except Exception:  # a library's internals changed: nothing found
        logger.debug("Could not list the objects a %s holds", type(value).__name__, exc_info=True)
        return None
    found: list = []
    for column in columns:
        if not all(k in LEAF_TYPES for k in set(map(type, column))):
            found.extend(v for v in column if type(v) not in LEAF_TYPES)
    return found or None


def _distinct(items: list) -> list:
    """*items* without repeats, or all of them when one is unhashable."""
    try:
        return list(dict.fromkeys(items))
    except TypeError:
        return items


def is_native_panic(exc: BaseException) -> bool:
    """Is *exc* a panic raised out of a Rust extension (``pyo3``), such as
    polars'? It derives from ``BaseException``, so no ``except Exception``
    stops it, and a panic while building a key ended the user's call."""
    return type(exc).__name__ == "PanicException"


class _HashSink:
    """A write-only file that feeds what is written to it into a hash."""

    closed = False

    def __init__(self, h: Any) -> None:
        self._h = h

    def write(self, data: Any) -> int:
        self._h.update(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def hash_pyarrow(value: Any) -> str | None:
    """Hash a PyArrow Table, RecordBatch, Array or ChunkedArray by what it
    holds, schema or type included.

    Column by column (`fold_arrow_column`). The whole buffers are not the
    content: a slice (``t.slice(2, 2)``, each batch of ``to_batches()``)
    shares its parent's buffers and differs only in an offset they do not
    show, so every equal-length slice of one table would hash alike; each
    chunk is read from its offset. A dictionary column's buffers hold only
    the indices, so ``["red", "blue"]`` and ``["cat", "dog"]`` would hash
    alike too; its values are read.
    """
    try:
        import pyarrow as pa

        if isinstance(value, (pa.Table, pa.RecordBatch)):
            h = hashlib.sha256(f"{type(value).__name__}:{value.num_rows}:".encode())
            fold_buffer(h, value.schema.serialize())
            for column in value.columns:
                h.update(b"|column|")
                fold_arrow_column(h, column)
            return h.hexdigest()
        if isinstance(value, (pa.Array, pa.ChunkedArray)):
            h = hashlib.sha256(f"{type(value).__name__}:{len(value)}:{value.type}:".encode())
            fold_arrow_column(h, value)
            return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, MemoryError, NotImplementedError):
        logger.debug("Failed to hash PyArrow %s", type(value).__name__)
    return None


class ArrowDigests:
    """The digests of Arrow values nothing can change: ``id(value) ->
    (weak reference, digest)``.

    A PyArrow Table, RecordBatch, Array or ChunkedArray never changes what it
    holds -- every method returns a new one -- and the memory under it does
    not change either when Arrow allocated it (`arrow_data_owned`: read from
    a file, computed by Arrow, built from Python values): Arrow writes a
    buffer only while building it, and hands out its memory read-only
    (``to_numpy``, ``to_pandas``; polars copies before writing). So its
    digest holds for as long as the object lives: read once, then looked up.
    The weak reference tells a live object from a new one at the same
    address. Arrow data over memory someone else owns -- a numpy array
    (``pa.array(a)`` copies nothing), a Python buffer, polars' memory, a
    memory-mapped file -- can change under it, and is read every time.

    Not seen: a write through a ``pa.Buffer``'s own buffer protocol
    (``np.frombuffer(arr.buffers()[1])``, a buffer from
    ``pa.allocate_buffer`` written after an array was built over it), into
    memory Arrow treats as immutable; a hit sharing a stored Arrow copy relies
    on the same.
    """

    def __init__(self, maxsize: int) -> None:
        self._entries: LruMemo[int, tuple[weakref.ref, str]] = LruMemo(maxsize)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def digest(self, value: Any) -> str | None:
        """`hash_pyarrow` of *value*, remembered while it lives when its
        memory is Arrow's own."""
        key = id(value)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is value:
            return entry[1]
        digest = hash_pyarrow(value)
        if digest is None:
            return None
        try:
            from .backends.frame_sharing import arrow_data_owned

            if not arrow_data_owned(value):
                return digest
            entries = self._entries
            wref = weakref.ref(value, lambda _ref, key=key, entries=entries: entries.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup: read it next time
            return digest
        self._entries[key] = (wref, digest)
        return digest


#: The process's memo of unchangeable Arrow values' digests (`ArrowDigests`).
ARROW_DIGESTS = ArrowDigests(ARROW_VALUES)


def hash_modin(value: Any) -> str | None:
    """Hash a modin DataFrame or Series as the pandas one it converts to --
    schema included (`hash_pandas`)."""
    try:
        return hash_pandas(value._to_pandas() if hasattr(value, "_to_pandas") else value)
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash modin %s", type(value).__name__)
        return None


def hash_dask(value: Any) -> str | None:
    """Hash a dask collection by its task-graph keys, plus its schema.

    The keys carry a data-derived token. The schema is the collection's
    ``_meta``, the empty pandas frame or numpy array that stands for its
    columns and dtypes, hashed as that type is.
    """
    try:
        h = hashlib.sha256(str(value.__dask_keys__()).encode("utf-8"))
        meta = getattr(value, "_meta", None)
        if meta is not None:
            h.update(f":{builtin_hash(meta)}".encode("utf-8"))
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash dask object via __dask_keys__")
        return None


def hash_sparse(value: Any) -> str | None:
    """Hash a scipy sparse matrix or array by its content, format included.

    The type, shape and dtype, then the arrays its format keeps (`SPARSE_PARTS`,
    and a COO's coordinates), each hashed as an array (`hash_numpy`), so an
    explicit zero, an unsorted index or an ``int32`` against an ``int64``
    index keys apart: code reading ``.data`` or ``.indices`` sees them. A
    DOK or LIL matrix, whose entries live in Python dicts and lists, is keyed
    as the CSR matrix it converts to. Without this every sparse argument --
    a TF-IDF matrix, a one-hot encoding -- ran uncached.
    """
    try:
        t = type(value)
        h = hashlib.sha256(f"{t.__module__}.{t.__qualname__}:{value.shape}:{value.dtype}:{value.format}:".encode())
        if value.format in ("dok", "lil"):
            canon = value.tocsr()
            canon.sort_indices()
            arrays = [(name, getattr(canon, name)) for name in ("data", "indices", "indptr")]
        else:
            arrays = [(name, getattr(value, name)) for name in SPARSE_PARTS if hasattr(value, name)]
            coords = getattr(value, "coords", None)
            if coords is None and value.format == "coo":
                coords = (value.row, value.col)
            if coords is not None:
                arrays.extend((f"coords{i}", c) for i, c in enumerate(coords))
        for name, array in arrays:
            digest = hash_numpy(array)
            if digest is None:
                return None
            h.update(f"|{name}:{digest}".encode())
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError):
        logger.debug("Failed to hash scipy sparse %s", type(value).__name__)
        return None
