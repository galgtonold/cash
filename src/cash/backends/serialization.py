"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base) and `PickleSerializer`, which stores
every decorated result, pandas DataFrames included.
"""

from __future__ import annotations

import logging
import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .. import _cash_views, kept_state

logger = logging.getLogger(__name__)

__all__ = ["Serializer", "PickleSerializer", "RESTORE_ERRORS", "rebuild", "restore_value", "stored_reducers"]

#: Buffers at least this large are handed out of band by `serialize_split`;
#: smaller ones stay in the stream, where a separate part costs more than the
#: copy it saves.
OUT_OF_BAND_MIN_BYTES = 64 * 1024


class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""


class PickleSerializer(Serializer):
    """Serializer using Python's built-in pickle module.

    Protocol 5 writes large buffers (numpy arrays, DataFrame columns) in one
    piece: a 1M-row x 10-column float frame took 47 ms against 466 ms with
    the default protocol 4. It also stores DataFrames: Parquet was 5-40x
    slower to write and 2-20x slower to read, and changed what it could not
    store exactly (cell types, subclasses, ``attrs``). Its only gain was
    smaller files for low-variety columns, which the file backend's
    ``compress`` option recovers for most of them.

    A polars DataFrame or Series is pickled as its Arrow data
    (`_reduce_polars`), whose buffers protocol 5 hands out of band, and so
    is a pandas Arrow-backed column (`_reduce_arrow_array`).
    """

    def serialize(self, data: Any) -> bytes:
        return kept_state.dumps(data, protocol=5, extra=stored_reducers(for_write=True))

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def serialize_split(self, data: Any, *, copy: bool = True) -> tuple[bytes, list[Any]]:
        """The pickle stream and, apart from it, the large buffers it refers to.

        What the file backend writes (``entry_format.MAGIC_SPLIT``): a large
        array is copied once, here, instead of into the stream, then into a
        joined blob, then out of it. Copied because the write runs later, on
        another thread, and the caller may change the value in the meantime.

        ``copy=False`` hands each buffer out as a view of the value's own
        memory, for a value nobody changes until the write is done (the file
        backend's ``private`` values), where the copy would be pure cost.
        """
        buffers: list[Any] = []

        def take(buffer: pickle.PickleBuffer) -> bool:
            raw = buffer.raw()
            if raw.nbytes < OUT_OF_BAND_MIN_BYTES:
                return True  # in band
            if copy:
                buffers.append(raw.tobytes())
            else:
                _cash_views.note(raw)
                buffers.append(raw)
            return False

        stream = kept_state.dumps(
            data, protocol=5, buffer_callback=take, on_retry=buffers.clear, extra=stored_reducers(for_write=True)
        )
        return stream, buffers

    def deserialize_split(self, stream: Any, buffers: list) -> Any:
        """The value `serialize_split` split, using *buffers*' memory as it is."""
        return pickle.loads(stream, buffers=buffers)


#: What turning stored bytes back into an object raises when the entry is
#: damaged (truncated by a killed process or a full disk, overwritten) or
#: names a binding this process lacks (a class renamed, a module gone, a
#: ``__main__`` class from another run). Every backend reads such an entry as
#: a miss: the value is recomputed, never a reason to fail the caller.
RESTORE_ERRORS: tuple[type[BaseException], ...] = (
    pickle.PickleError,
    EOFError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    ImportError,
    OverflowError,
)


def restore_value(metadata: dict, payload: bytes) -> Any:
    """The value *payload* holds, rebuilt by the serializer *metadata* names.

    Raises one of `RESTORE_ERRORS` when it cannot be rebuilt.
    """
    serializer_cls = metadata.get("serializer_cls", PickleSerializer)
    return rebuild(serializer_cls().deserialize, payload)


def rebuild(load: Callable[..., Any], *args: Any) -> Any:
    """``load(*args)``, raising one of `RESTORE_ERRORS` for anything it raises.

    Unpickling runs the value's own code (``__setstate__``, a ``__reduce__``
    callable), which raises what it likes: torch's ``RuntimeError`` for a
    CUDA tensor read on a machine without a GPU, a library's own error class,
    ``RecursionError``. Such an entry cannot be restored HERE, which is a miss
    like a damaged one, not a crash of every call that reads it.
    """
    try:
        return load(*args)
    except RESTORE_ERRORS:
        raise
    except Exception as exc:
        raise pickle.UnpicklingError(f"rebuilding the stored value raised {type(exc).__name__}: {exc}") from exc


def stored_reducers(*, for_write: bool = False) -> dict | None:
    """The reducers cash pickles a value it stores with, by exact type
    (`kept_state.dumps` *extra*): `_reduce_polars` for polars' DataFrame
    and Series, `_reduce_arrow_array` for pandas' Arrow-backed arrays. Each
    only once its library is imported (a value cannot hold one before). A
    subclass keeps its own reduce.

    *for_write*: for the bytes `PickleSerializer` writes, also
    `_reduce_chunked_array` for an Arrow column of chunks. Not for the
    pickles a digest is taken over (`stored_pickle`): how an entry is
    written changes no key."""
    reducers: dict = {}
    pl = sys.modules.get("polars")
    frame, series = getattr(pl, "DataFrame", None), getattr(pl, "Series", None)
    if isinstance(frame, type) and isinstance(series, type):  # else no polars, or one still importing
        reducers[frame] = reducers[series] = _reduce_polars
    for arrow_class in _arrow_array_classes():
        reducers[arrow_class] = _reduce_arrow_array
    pd = sys.modules.get("pandas")
    string_array = getattr(getattr(pd, "arrays", None), "StringArray", None)
    if isinstance(string_array, type):
        reducers[string_array] = _reduce_string_array
    chunked = getattr(sys.modules.get("pyarrow"), "ChunkedArray", None)
    if for_write and isinstance(chunked, type):
        reducers[chunked] = _reduce_chunked_array
    return reducers or None


def _reduce_chunked_array(value: Any) -> Any:
    """An Arrow column whose chunks share buffers, each buffer pickled once
    (`_chunked_array`); any other column as pyarrow pickles it.

    The chunks of a polars text column (Arrow ``string_view``) each list
    every data buffer of the column, not only the ones their rows point
    into: pickled chunk by chunk, each buffer went into the entry once per
    chunk. A frame filtered on 32 threads has 32 chunks: 438 MB stored as
    10.3 GB on Windows, 39 s of writing beside the next cell. Rebuilt from
    the same buffers, at the same offsets, the column is the one stored.
    Never more than pyarrow writes: it pickles a sliced chunk's buffers
    whole too. A column of one chunk goes the same way: sorted, p08's
    text column is one chunk over 6,151 small buffers, which pyarrow
    pickled into the stream (`_packed`)."""
    import pyarrow as pa

    chunks = value.chunks
    kind = value.type
    if not chunks or kind.num_fields or pa.types.is_dictionary(kind) or isinstance(kind, pa.ExtensionType):
        # No chunk, or a type whose children or dictionary carry buffers
        # of their own that ``buffers()`` flattens: as pyarrow pickles it.
        return value.__reduce__()
    index: dict[tuple[int, int], int] = {}
    buffers: list = []
    layout: list[tuple[int, int, int, tuple[int | None, ...]]] = []
    listed = 0
    for chunk in chunks:
        refs: list[int | None] = []
        for buffer in chunk.buffers():
            if buffer is None:
                refs.append(None)
                continue
            listed += 1
            key = (buffer.address, buffer.size)
            if key not in index:
                index[key] = len(buffers)
                buffers.append(buffer)
            refs.append(index[key])
        layout.append((len(chunk), chunk.null_count, chunk.offset, tuple(refs)))
    small = [i for i, buffer in enumerate(buffers) if buffer.size < OUT_OF_BAND_MIN_BYTES]
    if len(small) > 1:
        return _packed_chunked_array, (value.type, *_packed(buffers, small), layout)
    if len(buffers) == listed:
        return value.__reduce__()  # nothing shared
    return _chunked_array, (value.type, buffers, layout)


#: Where `_packed` puts each buffer in its pack: at a multiple of this.
_PACK_ALIGN = 64


def _packed(buffers: list, small: list[int]) -> tuple[list, Any, list]:
    """*buffers* with the *small* ones copied into one buffer of their own:
    ``(buffers, pack, places)``, *places* naming for each buffer its index in
    *buffers* or its ``(offset, size)`` in *pack*.

    A buffer under `OUT_OF_BAND_MIN_BYTES` is pickled into the stream. p08's
    log frame, 1,280 chunks per column, was 708 MB of stream, copied as it
    grew: 6.6 s on Windows. One pack goes out of band, written once."""
    import pyarrow as pa

    picked = set(small)
    places: list = [None] * len(buffers)
    at = 0
    for i in small:
        places[i] = (at, buffers[i].size)
        at += -(-buffers[i].size // _PACK_ALIGN) * _PACK_ALIGN
    pack = bytearray(at)
    view = memoryview(pack)
    for i in small:
        start, size = places[i]
        view[start : start + size] = memoryview(buffers[i]).cast("B")
    kept = []
    for i, buffer in enumerate(buffers):
        if i not in picked:
            places[i] = len(kept)
            kept.append(buffer)
    return kept, pa.py_buffer(pack), places


def _packed_chunked_array(type_: Any, buffers: list, pack: Any, places: list, layout: list) -> Any:
    """The column `_reduce_chunked_array` stored with its small buffers in
    *pack* (`_packed`)."""
    import pyarrow as pa

    pack = pa.py_buffer(pack)
    whole = [buffers[place] if isinstance(place, int) else pack.slice(*place) for place in places]
    return _chunked_array(type_, whole, layout)


def _chunked_array(type_: Any, buffers: list, layout: list) -> Any:
    """The column `_reduce_chunked_array` stored: each chunk over the shared
    *buffers*, as ``(length, null count, offset, buffer indices)``."""
    import pyarrow as pa

    return pa.chunked_array(
        [
            pa.Array.from_buffers(type_, length, [None if i is None else buffers[i] for i in refs], nulls, offset)
            for length, nulls, offset, refs in layout
        ],
        type=type_,
    )


#: Python-string columns at least this long go out of band as one buffer
#: (`_reduce_string_array`): the size of `OUT_OF_BAND_MIN_BYTES` of pointers.
STRINGS_APART_MIN_ITEMS = OUT_OF_BAND_MIN_BYTES // 8


def _reduce_string_array(value: Any) -> Any:
    """A pandas column of Python strings (``StringDtype("python")``, pandas
    3's ``str`` without pyarrow) as pandas pickles it, but with its strings
    pickled into one buffer of their own, which protocol 5 hands out of
    band (`_string_array_from_items`).

    Out of band, the file tier keeps it in the chunk store by its content
    (`cash.backends.chunk_store`): the same column in another entry, as
    each result of ``h = df.assign(...)`` in a loop holds it, is stored
    once. A column the RAM tier keeps (`shared_blocks.is_kept`: its own,
    never written) is pickled once while it is kept (`_items_buffer`); the
    other entries holding it cost a digest of that buffer, not a pickle.
    """
    reduced = value.__reduce_ex__(5)
    try:
        func, args, state = reduced[0], reduced[1], reduced[2]
        if (
            len(reduced) != 3
            or type(state) is not tuple
            or len(state) < 2  # (dtype, strings), and pandas 2 adds the attributes
            or state[1] is not value._ndarray
            or len(state[1]) < STRINGS_APART_MIN_ITEMS
        ):
            return reduced
        items = _items_buffer(value)
    except Exception:  # not the layout this expects: pandas' own pickle
        logger.debug("pickling a %s the pandas way", type(value).__name__, exc_info=True)
        return reduced
    return _string_array_from_items, (func, args, state[0], pickle.PickleBuffer(items), state[2:])


def _string_array_from_items(func: Callable[..., Any], args: tuple, dtype: Any, items: Any, rest: tuple = ()) -> Any:
    """The column `_reduce_string_array` stored: built as pandas' own pickle
    builds it, with its strings read back from *items*. *rest* is what
    pandas' state holds after the strings: pandas 2 keeps the array's
    attributes there, pandas 3 nothing."""
    array = func(*args)
    array.__setstate__((dtype, pickle.loads(items), *rest))
    return array


class _KeptItems:
    """``id(values) -> (weak reference, bytes)``, the strings of columns the
    RAM tier keeps pickled once (`_items_buffer`), least recently used
    first, at most `_KEPT_ITEMS_MAX_BYTES` together."""

    def __init__(self) -> None:
        import threading
        from collections import OrderedDict

        self.lock = threading.Lock()
        self.entries: OrderedDict[int, tuple[Any, bytes]] = OrderedDict()
        self.nbytes = 0
        self.pickled = 0  # columns pickled, for the work-count tests

    def drop(self, key: int, ref: Any) -> None:
        with self.lock:
            found = self.entries.get(key)
            if found is not None and found[0] is ref:
                del self.entries[key]
                self.nbytes -= len(found[1])


_KEPT_ITEMS = _KeptItems()
#: What `_KeptItems` holds at most: past it, the oldest go.
_KEPT_ITEMS_MAX_BYTES = 256 * 1024**2


def _items_buffer(value: Any) -> bytes:
    """*value*'s strings, pickled: from `_KEPT_ITEMS` for a column the RAM
    tier keeps, which nothing writes while it lives."""
    import weakref

    from .shared_blocks import is_kept

    kept = is_kept(value)
    memo = _KEPT_ITEMS
    if kept:
        with memo.lock:
            found = memo.entries.get(id(value))
            if found is not None and found[0]() is value:
                memo.entries.move_to_end(id(value))
                return found[1]
    items = pickle.dumps(value._ndarray, protocol=5)
    memo.pickled += 1
    if kept and len(items) <= _KEPT_ITEMS_MAX_BYTES // 4:
        key = id(value)
        ref = weakref.ref(value, lambda _ref, key=key: memo.drop(key, _ref))
        with memo.lock:
            old = memo.entries.pop(key, None)
            if old is not None:
                memo.nbytes -= len(old[1])
            memo.entries[key] = (ref, items)
            memo.nbytes += len(items)
            while memo.nbytes > _KEPT_ITEMS_MAX_BYTES and memo.entries:
                _old_key, (_ref, old_items) = memo.entries.popitem(last=False)
                memo.nbytes -= len(old_items)
    return items


#: pandas' Arrow-backed array classes `_reduce_arrow_array` takes, found
#: once pandas has loaded them (`_arrow_array_classes`).
_ARROW_CLASSES: list[tuple[type, ...]] = []


def _arrow_array_classes() -> tuple[type, ...]:
    """pandas' ``ArrowExtensionArray`` and its subclasses in pandas
    (``ArrowStringArray``), each still pickled by the base class's
    ``__getstate__``/``__setstate__`` that `_reduce_arrow_array` stands in
    for; empty before pandas loaded them."""
    if _ARROW_CLASSES:
        return _ARROW_CLASSES[0]
    module = sys.modules.get("pandas.core.arrays.arrow.array")
    base = getattr(module, "ArrowExtensionArray", None)
    if not isinstance(base, type) or "pyarrow" not in sys.modules:
        return ()
    found: list[type] = []
    pending = [base]
    while pending:
        klass = pending.pop()
        pending.extend(klass.__subclasses__())
        if (
            klass.__module__.startswith("pandas.")
            and klass.__getstate__ is base.__getstate__
            and klass.__setstate__ is base.__setstate__
            and klass.__reduce_ex__ is object.__reduce_ex__
            and klass.__reduce__ is object.__reduce__
        ):
            found.append(klass)
    _ARROW_CLASSES.append(tuple(found))
    return _ARROW_CLASSES[0]


def _reduce_arrow_array(value: Any) -> Any:
    """A pandas Arrow-backed array (``str`` columns, ``ArrowDtype``) as its
    Arrow chunks, rebuilt by `_arrow_array_from_chunks`.

    pandas' own pickle joins the chunks first (``combine_chunks``), a copy
    of every value -- 0.8 s for the ten text columns of a 5.6M-row frame --
    which the writer then copied again. The chunks pickle as their Arrow
    buffers, which protocol 5 hands out of band as they are. A chunk that is
    a small slice of bigger buffers (``df.head()`` of a big frame) is
    copied down to its rows first (`_compact`), as pandas' join does; Arrow
    would pickle the whole buffers under it. pandas' own state (the dtype)
    is kept as it is, and restored by pandas' own ``__setstate__``.
    """
    try:
        chunked = value._pa_array
        chunks = [_compact(chunk) for chunk in chunked.chunks]
        arrow_type = chunked.type
    except Exception:  # not the layout this expects: pandas' own pickle
        logger.debug("pickling a %s the pandas way", type(value).__name__, exc_info=True)
        return value.__reduce_ex__(5)
    state = dict(value.__dict__)
    del state["_pa_array"]
    return _arrow_array_from_chunks, (type(value), chunks, arrow_type, state)


def _compact(chunk: Any) -> Any:
    """*chunk*, or a copy of its rows alone when the buffers it slices hold
    more than twice what it needs (and at least `OUT_OF_BAND_MIN_BYTES`
    more): those would all be written."""
    import pyarrow as pa

    needed = chunk.nbytes
    if chunk.get_total_buffer_size() - needed > max(needed, OUT_OF_BAND_MIN_BYTES):
        return pa.concat_arrays([chunk])
    return chunk


def _arrow_array_from_chunks(cls: type, chunks: list, arrow_type: Any, state: dict) -> Any:
    """The array `_reduce_arrow_array` stored: *chunks* of *arrow_type*,
    with pandas' own *state*, through pandas' own ``__setstate__``."""
    import pyarrow as pa

    array = cls.__new__(cls)
    array.__setstate__({**state, "_pa_array": pa.chunked_array(chunks, type=arrow_type)})
    return array


def _reduce_polars(value: Any) -> Any:
    """A polars DataFrame or Series as its Arrow data, rebuilt by
    `_polars_from_arrow`.

    Polars' own pickle writes the frame as one IPC blob inside the stream:
    for 80 MB, 190 ms to write and 90 ms to read. An Arrow table shares the
    frame's buffers, and protocol 5 hands them to the file tier apart from
    the stream, as it does a pandas frame's. Polars' own pickle stays for
    what Arrow cannot take: a dtype this pyarrow lacks, no pyarrow at all.
    An ``Object`` column is refused with `pickle.PicklingError`, so the
    value stays in RAM: polars refuses it too, polars 1 with a panic that
    no ``except Exception`` stops.
    """
    import polars as pl

    flags = value.flags
    if isinstance(value, pl.Series):
        dtype = value.dtype
        dtypes = [dtype]
    else:
        dtype = value.schema
        dtypes = list(dtype.values())
    if any(d == pl.Object for d in dtypes):
        raise pickle.PicklingError("polars does not pickle an Object column")
    try:
        arrow = value.to_arrow(compat_level=pl.CompatLevel.newest())
    except Exception:  # noqa: BLE001 - no pyarrow, a polars without CompatLevel, a dtype Arrow lacks
        return value.__reduce_ex__(5)
    name = value.name if isinstance(value, pl.Series) else None
    return _polars_from_arrow, (arrow, name, dtype, flags)


def rebuilt_polars(value: Any, copy_arrow: Callable[[Any], Any]) -> Any:
    """*value*, a polars DataFrame or Series, rebuilt as a disk entry
    rebuilds it from its Arrow data passed through *copy_arrow*; None when
    it is not stored as Arrow data (`_reduce_polars`)."""
    rebuild, args = _reduce_polars(value)
    if rebuild is not _polars_from_arrow:
        return None
    arrow, *rest = args
    return _polars_from_arrow(copy_arrow(arrow), *rest)


def _polars_from_arrow(arrow: Any, name: str | None, dtype: Any, flags: dict) -> Any:
    """The polars value `_reduce_polars` stored: from *arrow*, without a
    copy, cast back to its polars *dtype* (a schema for a DataFrame) where
    Arrow has none of its own (an Enum comes back a Categorical), and
    marked sorted where it was (*flags*, ``value.flags``)."""
    import polars as pl

    out = pl.from_arrow(arrow, rechunk=False)
    if name is None:
        if out.schema != dtype:
            out = out.cast(dict(dtype))
        sorted_columns = [_sorted(column, flags[column.name]) for column in out.get_columns()]
        if any(new is not old for new, old in zip(sorted_columns, out.get_columns())):
            out = pl.DataFrame(sorted_columns)
        return out
    if not isinstance(out, pl.Series):
        out = out.to_series()
    out = out.alias(name)
    if out.dtype != dtype:
        out = out.cast(dtype)
    return _sorted(out, flags)


def _sorted(series: Any, flags: dict) -> Any:
    """*series*, marked sorted as *flags* say it was."""
    if flags.get("SORTED_ASC"):
        return series.set_sorted()
    if flags.get("SORTED_DESC"):
        return series.set_sorted(descending=True)
    return series
