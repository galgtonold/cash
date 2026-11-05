"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base) and `PickleSerializer`, which stores
every decorated result, pandas DataFrames included.
"""

from __future__ import annotations

import logging
import pickle
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .. import _cash_views, kept_state
from . import block_digests, text_columns
from .datetime_columns import reduce_datetime_objects
from .no_file_writes import NoFileWrites, import_what_it_imported, no_file_writes

logger = logging.getLogger(__name__)

__all__ = ["Serializer", "PickleSerializer", "RESTORE_ERRORS", "rebuild", "restore_value", "stored_reducers"]

#: Buffers at least this large are handed out of band by `serialize_split`;
#: smaller ones stay in the stream, where a separate part costs more than the
#: copy it saves.
OUT_OF_BAND_MIN_BYTES = 64 * 1024


#: numpy's object arrays of dates and times as columns of numbers, every
#: other array as numpy pickles it (`datetime_columns`).
_DATETIME_OBJECTS = kept_state.Ahead(reduce_datetime_objects)


def _numpy_frombuffer() -> Callable[..., Any] | None:
    """numpy's own rebuild of an array pickled out of band (what
    ``ndarray.__reduce_ex__(5)`` names), or None when it names another."""
    np = sys.modules.get("numpy")
    try:
        rebuild = np.zeros(1).__reduce_ex__(5)[0]
    except Exception:  # noqa: BLE001 - an older numpy: arrays as before
        return None
    return rebuild if getattr(rebuild, "__name__", None) == "_frombuffer" else None


_FROMBUFFER: list[Callable[..., Any] | None] = []


def reduce_datetime64(array: Any) -> tuple | None:
    """A ``datetime64`` or ``timedelta64`` array's reduce with its data out
    of band, or None to pickle it as numpy does.

    numpy pickles such an array into the stream: its dtype exports no
    buffer, so protocol 5 cannot hand the data apart. A frame's date column
    was then written whole into every entry holding it, never shared in the
    chunk store as its other columns are: each result of ``h, _ =
    run(played, k)`` in a loop wrote ``played``'s dates again (7 MB a
    result for 900k rows). The same bytes go out of band as 8-byte integers
    and come back through numpy's own rebuild with the array's dtype: the
    array numpy would have loaded."""
    np = sys.modules.get("numpy")
    try:
        if (
            array.dtype.kind not in "mM"
            or np is None
            or type(array) is not np.ndarray
            or array.dtype.metadata is not None
            or array.nbytes < OUT_OF_BAND_MIN_BYTES
        ):
            return None
        if array.flags.c_contiguous:
            order = "C"
        elif array.flags.f_contiguous:
            order = "F"
        else:
            return None
        if not _FROMBUFFER:
            _FROMBUFFER.append(_numpy_frombuffer())
        rebuild = _FROMBUFFER[0]
        if rebuild is None:
            return None
        # A Fortran-ordered array's bytes are its transpose's in C order,
        # which every buffer consumer takes (numpy hands those over too).
        data = (array.T if order == "F" else array).view(np.int64)
        return rebuild, (pickle.PickleBuffer(data), array.dtype, array.shape, order)
    except Exception:  # noqa: BLE001 - anything unforeseen: pickled as numpy does
        return None


def _reduce_array_for_write(array: Any) -> tuple | None:
    """`reduce_datetime_objects`, else `reduce_datetime64`, else
    `reduce_text_objects`: what the bytes an entry is written as take for a
    numpy array."""
    reduced = reduce_datetime_objects(array)
    if reduced is None:
        reduced = reduce_datetime64(array)
    return reduced if reduced is not None else reduce_text_objects(array)


def reduce_text_objects(array: Any) -> tuple | None:
    """A numpy object array of strings and missing values (``None``,
    ``nan``, ``pd.NA``) with its strings in one buffer out of band
    (`text_columns`), rebuilt by `_text_objects`; None to pickle it as
    numpy does: another item, a small array, a string UTF-8 cannot hold.

    Pickled as numpy does, each string is an object of its own in the
    stream: a column of a million strings took longer to write than to
    read from its CSV."""
    np = sys.modules.get("numpy")
    try:
        if (
            np is None
            or type(array) is not np.ndarray
            or array.dtype != object
            or array.size < STRINGS_APART_MIN_ITEMS
            or array.ndim > 2
        ):
            return None
        encoded = text_columns.encode(array, missing_only=True)
    except Exception:  # anything unforeseen: pickled as numpy does
        logger.debug("pickling an object array the numpy way", exc_info=True)
        return None
    if encoded is None:
        return None
    return _text_objects, encoded.parts()


def _text_objects(*parts: Any) -> Any:
    """The object array `reduce_text_objects` stored."""
    return text_columns.decode(*parts)


#: `_DATETIME_OBJECTS`, and a ``datetime64`` array's data out of band, for
#: the bytes `PickleSerializer` writes only.
_ARRAYS_FOR_WRITE = kept_state.Ahead(_reduce_array_for_write)


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
            if _over_bytes(raw):
                buffers.append(raw)  # nothing can write it: no copy
            elif copy:
                # A block the frame memo vouches for is copied into an
                # array whose digest its content key gives (`_known_copy`).
                known = _known_copy(raw)
                buffers.append(known if known is not None else raw.tobytes())
            else:
                _cash_views.note(raw)
                buffers.append(raw)
            return False

        # The caller's value, pickled now on its thread: the blocks whose
        # bytes the frame memo vouches for are known by their content key.
        previous = _KNOWN.arrays
        _KNOWN.arrays = _blocks_known(data) if copy else None
        try:
            stream = kept_state.dumps(
                data, protocol=5, buffer_callback=take, on_retry=buffers.clear, extra=stored_reducers(for_write=True)
            )
        finally:
            _KNOWN.arrays = previous
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

    Nor may that code change a file (`no_file_writes`): a sink reopened with
    ``"w"`` in ``__setstate__`` emptied the user's file on every hit, where
    plain Python opened it once. Such an entry is a miss too.
    """
    value, failed, check = _rebuild_once(load, args)
    if check.refused is not None and import_what_it_imported(check):
        # A first import that writes its own caches: imported as the
        # program would import it, outside the check, then rebuilt again.
        value, failed, check = _rebuild_once(load, args)
    check.check()
    if failed is not None:
        if isinstance(failed, RESTORE_ERRORS):
            raise failed
        raise pickle.UnpicklingError(
            f"rebuilding the stored value raised {type(failed).__name__}: {failed}"
        ) from failed
    return value


def _rebuild_once(load: Callable[..., Any], args: tuple) -> tuple[Any, Exception | None, NoFileWrites]:
    """``load(*args)`` inside `no_file_writes`: ``(value, what it raised, the check)``."""
    with no_file_writes() as check:
        try:
            return load(*args), None, check
        except Exception as exc:  # noqa: BLE001 - handed to the caller
            return None, exc, check


def stored_reducers(*, for_write: bool = False) -> dict | None:
    """The reducers cash pickles a value it stores with, by exact type
    (`kept_state.dumps` *extra*): `_reduce_polars` for polars' DataFrame
    and Series, `_reduce_arrow_array` for pandas' Arrow-backed arrays, and
    for numpy's object arrays of dates and times, ahead of numpy's own
    reduce, their columns (`datetime_columns`). Each only once its library
    is imported (a value cannot hold one before). A subclass keeps its own
    reduce.

    *for_write*: for the bytes `PickleSerializer` writes, also
    `_reduce_chunked_array` for an Arrow column of chunks,
    `reduce_datetime64` for a ``datetime64`` array, and a column of Python
    strings as UTF-8 (`_reduce_text_for_write`, `reduce_text_objects`)
    in place of `_reduce_string_array`. Not for the
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
        reducers[string_array] = _reduce_text_for_write if for_write else _reduce_string_array
    ndarray = getattr(sys.modules.get("numpy"), "ndarray", None)
    if isinstance(ndarray, type):
        reducers[ndarray] = _ARRAYS_FOR_WRITE if for_write else _DATETIME_OBJECTS
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


def _reduce_text_for_write(value: Any) -> Any:
    """A pandas column of Python strings as `_reduce_string_array` keeps it,
    but with its strings joined into one UTF-8 buffer (`text_columns`),
    not pickled one by one, rebuilt by `_string_array_from_text`. Missing
    values keep their kind, and the array its dtype (storage and missing
    value), restored by pandas' own ``__setstate__``.

    A column whose strings UTF-8 cannot hold goes as
    `_reduce_string_array` does."""
    reduced = value.__reduce_ex__(5)
    try:
        func, args, state = reduced[0], reduced[1], reduced[2]
        if (
            len(reduced) != 3
            or type(state) is not tuple
            or len(state) < 2
            or state[1] is not value._ndarray
            or len(state[1]) < STRINGS_APART_MIN_ITEMS
            or getattr(state[0], "storage", None) != "python"
        ):
            return _reduce_string_array(value)
        objects = value._ndarray
        encoded = text_columns.encode_column(value, objects, _content_key(objects))
    except Exception:  # not the layout this expects
        logger.debug("pickling a %s as strings", type(value).__name__, exc_info=True)
        return _reduce_string_array(value)
    if encoded is None:
        return _reduce_string_array(value)
    return _string_array_from_text, (func, args, state[0], encoded.parts(), state[2:])


def _string_array_from_text(func: Callable[..., Any], args: tuple, dtype: Any, parts: tuple, rest: tuple = ()) -> Any:
    """The column `_reduce_text_for_write` stored: built as pandas' own
    pickle builds it, with its strings read back from *parts*."""
    array = func(*args)
    array.__setstate__((dtype, text_columns.decode(*parts), *rest))
    return array


class _Known(threading.local):
    """``(address, bytes) -> (array, content key)`` for the block arrays of
    the tables in a value being pickled on its caller's thread
    (`_blocks_known`), or None. By memory, not by array: numpy pickles a
    view of a block's array (a date column's integers), whose ``base`` is
    the memory's owner, not the block's array."""

    arrays: dict[tuple[int, int], tuple[Any, tuple]] | None = None


_KNOWN = _Known()


def _content_key(array: Any) -> tuple | None:
    """The content key naming the bytes of the block array *array* now:
    cash's own copy (`block_digests.content_key_of`), or a block of a
    table being written from its caller's thread (`_KNOWN`)."""
    known = _KNOWN.arrays
    if known and array.flags.c_contiguous:
        found = known.get((_address(array), array.nbytes))
        if found is not None:
            return found[1]
    return block_digests.content_key_of(array)


def _blocks_known(value: Any) -> dict[tuple[int, int], tuple[Any, tuple]]:
    """`_Known.arrays` of *value*: the blocks of the big pandas tables in it
    (or in the dicts, lists and tuples around them) whose bytes the frame
    memo vouches for now (`shared_blocks.content_keys`)."""
    pd = sys.modules.get("pandas")
    frame_type = getattr(pd, "DataFrame", None)
    if not isinstance(frame_type, type):
        return {}
    frames: list[Any] = []
    pending: list[tuple[Any, int]] = [(value, 0)]
    while pending and len(frames) < 64:
        item, depth = pending.pop()
        kind = type(item)
        if kind is frame_type:
            frames.append(item)
        elif depth < 3 and kind in (dict, list, tuple) and len(item) <= 1024:
            items = item.values() if kind is dict else item
            pending.extend((each, depth + 1) for each in items)
    known: dict[tuple[int, int], tuple[Any, tuple]] = {}
    if not frames:
        return known
    from .shared_blocks import content_keys

    for frame in frames:
        try:
            if not any(block.values.nbytes >= block_digests.MIN_BYTES for block in frame._mgr.blocks):
                continue
            keys = content_keys(frame)
            if not keys:
                continue
            for block in frame._mgr.blocks:
                key = keys.get(tuple(int(pos) for pos in block.mgr_locs.as_array))
                if key is not None:
                    # A content key is only for a contiguous block
                    # (`shared_blocks._layout`): its memory is its bytes.
                    array = getattr(block.values, "_ndarray", block.values)
                    known[(_address(array), array.nbytes)] = (array, key)
        except Exception:  # only a speedup: nothing known
            logger.debug("cash: a table's blocks unknown", exc_info=True)
    return known


def _known_copy(raw: memoryview) -> Any:
    """A copy of *raw* when it is all the memory of a block array in
    `_KNOWN`, made cash's own and recorded as holding the bytes the block's
    content key names (`block_digests.note_own`); None for any other
    buffer. The block's table holds its array while it is pickled, so no
    other memory is at that address."""
    known = _KNOWN.arrays
    if not known or raw.nbytes < block_digests.MIN_BYTES:
        return None
    found = known.get((_address(raw), raw.nbytes))
    if found is None:
        return None
    np = sys.modules["numpy"]
    copied = np.frombuffer(raw.tobytes(), dtype=np.uint8)
    block_digests.note_own(copied, found[1])
    return memoryview(copied)


def _address(buffer: Any) -> int:
    """Where *buffer*'s memory (a numpy array, or a contiguous buffer)
    starts."""
    np = sys.modules["numpy"]
    if isinstance(buffer, np.ndarray):
        return buffer.__array_interface__["data"][0]
    return np.frombuffer(buffer, dtype=np.uint8).__array_interface__["data"][0]


def _over_bytes(raw: memoryview) -> bool:
    """Is *raw* memory of a ``bytes`` object (a text column's buffers,
    `text_columns`), which nothing can write, so needs no copy?"""
    np = sys.modules.get("numpy")
    obj = raw.obj
    for _depth in range(8):
        if np is None or not isinstance(obj, np.ndarray):
            return type(obj) is bytes
        if obj.flags.writeable:
            return False
        obj = obj.base
    return False


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
        whole = chunked.chunks
        chunks = [_compact(chunk) for chunk in whole]
        arrow_type = chunked.type
    except Exception:  # not the layout this expects: pandas' own pickle
        logger.debug("pickling a %s the pandas way", type(value).__name__, exc_info=True)
        return value.__reduce_ex__(5)
    # The buffers written as they are, held by *chunked*: a table stored
    # again holding the same column is not hashed again (`block_digests`).
    block_digests.note_arrow(chunked, [chunk for chunk, kept in zip(whole, chunks) if kept is chunk])
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
