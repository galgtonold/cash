"""Columns of Python strings as two buffers, for the bytes an entry is
written as.

pandas 3 keeps text without pyarrow in a numpy array of ``str`` objects
(``StringDtype("python")``), and pickle writes those one object at a time:
p28's loop results pickled at about 11 MB/s on Windows, 158 MB in 14 s.
Such a column is written instead as its strings joined by ``"\\0"`` and
encoded as UTF-8, one buffer pickle hands out of band (`encode`). A column
whose rows hold the same objects again and again (a CSV reader makes each
team's name one object) is written as each object once and an array of
codes, as pickle's memo writes an object once and refers back to it: a hit
holds each one once, as the column did.

Exact or not at all. Every item that is not exactly a ``str`` (a missing
value -- ``nan``, ``None``, ``pd.NA`` -- or a ``str`` subclass) is kept
apart with its position and pickled as it is. A string holding ``"\\0"`` or
a lone surrogate, which UTF-8 cannot encode, leaves the column to pickle as
before (None).

The same column stored again costs nothing new: an encoding is kept by the
content key of the block it came from (`shared_blocks.content_keys`, which
names its bytes while the frame memo vouches for them) or, for a column the
RAM tier keeps and never writes (`shared_blocks.is_kept`), by the column
itself, and its buffers are bytes nothing can change, whose digest the
chunk store reads once (`block_digests.note_immutable`).

Only for writing: the pickles a key or a digest is taken over never see
this (`serialization.stored_reducers`).
"""

from __future__ import annotations

import operator
import sys
import threading
from collections import OrderedDict
from itertools import compress, repeat
from typing import Any

from .. import _weak
from . import block_digests

__all__ = ["Encoded", "decode", "encode", "encode_column"]

#: Joins the strings of a column; a column holding it is not encoded.
SEPARATOR = "\x00"
#: Columns of fewer items are written each item in turn.
_CODES_MIN_ITEMS = 1024
#: A column of strings each its own object, longer than this many
#: characters on average, is pickled as it is (`_long_strings`).
_PLAIN_ABOVE_MEAN_CHARS = 128
#: How many items `_long_strings` averages over at most (about).
_LENGTH_SAMPLE = 1024


class Encoded:
    """A column of strings as `encode` writes it: *count* items in *shape*
    (memory *order*), written as *items*: each object of the column once,
    picked for each row by *codes*, or each row's own (*codes* None). Of
    the items, the *strings* ``str`` ones joined and encoded (*data*), and
    the positions (*at*) and values (*others*) of the rest, or None."""

    __slots__ = ("__weakref__", "at", "codes", "count", "data", "nbytes", "order", "others", "shape", "strings")

    def __init__(
        self,
        shape: tuple,
        order: str,
        strings: int,
        data: Any,
        codes: Any,
        at: Any,
        others: list | None,
    ) -> None:
        self.shape = shape
        self.order = order
        self.count = 1
        for size in shape:
            self.count *= size
        self.strings = strings
        self.data = data
        self.codes = codes
        self.at = at
        self.others = others
        self.nbytes = sum(int(part.nbytes) for part in (data, codes, at) if part is not None) + 8 * len(others or ())

    def parts(self) -> tuple:
        """What a pickle carries, for `decode`."""
        return (self.shape, self.order, self.strings, self.data, self.codes, self.at, self.others)


def _immutable(array: Any) -> Any:
    """A read-only copy of *array* over a ``bytes`` object, which nothing
    can write: its digest is read once (`block_digests.note_immutable`) and
    a write needs no copy of it."""
    import numpy as np

    frozen = np.frombuffer(array.tobytes(), dtype=array.dtype)
    block_digests.note_immutable(frozen)
    return frozen


#: The types of the items an object array of text may hold besides ``str``
#: (`_is_missing` decides which of their values are missing ones).
_MISSING_KINDS = frozenset({str, type(None), float})


def _na_type(kind: type) -> bool:
    """Is *kind* pandas' ``NAType``?"""
    return kind.__name__ == "NAType" and kind.__module__.startswith("pandas")


def _is_missing(item: Any) -> bool:
    """Is *item* a missing value pandas uses: None, a float NaN, ``pd.NA``?"""
    if item is None:
        return True
    if type(item) is float:
        return item != item
    return _na_type(type(item))


def encode(objects: Any, *, missing_only: bool = False) -> Encoded | None:
    """The numpy object array *objects* as an `Encoded`, or None to pickle
    it as it is: a string holds ``"\\0"`` or cannot be encoded as UTF-8,
    or (*missing_only*) an item is neither a ``str`` nor a missing value,
    or more than half are not ``str``, or its strings, each its own
    object, are long (`_long_strings`).

    When most items are the same objects as others (read from a file,
    each team's name is one object held 900k times), each object is
    written once with a code per item, as pickle's memo writes it once and
    refers back to it: the items to write are those objects, in the order
    first met. Equal strings that are separate objects stay separate:
    finding them would hash every string, slower than pickling them."""
    import numpy as np

    order = "F" if objects.ndim > 1 and objects.flags.f_contiguous and not objects.flags.c_contiguous else "C"
    flat = objects.reshape(-1, order=order) if objects.ndim != 1 else objects
    count = len(flat)
    codes, first = _shared(flat)
    if codes is None and _long_strings(flat):
        return None
    items = (flat if first is None else flat.take(first)).tolist()
    kinds = set(map(type, items))
    at = others = None
    if missing_only and (str not in kinds or not all(map(_na_type, kinds - _MISSING_KINDS))):
        return None
    if kinds and kinds != {str}:
        is_str = np.fromiter(map(operator.is_, map(type, items), repeat(str)), dtype=bool, count=len(items))
        where = np.flatnonzero(~is_str)
        others = [items[i] for i in where.tolist()]
        if missing_only and (_apart(where, codes, count) * 2 > count or not all(map(_is_missing, others))):
            return None
        at = _immutable(where.astype(np.int64))
        items = list(compress(items, is_str))
    joined = SEPARATOR.join(items)
    if joined.count(SEPARATOR) != max(len(items) - 1, 0):
        return None  # a string holds the separator
    try:
        data = joined.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: pickled as it is
        return None
    data_array = np.frombuffer(data, dtype=np.uint8)
    block_digests.note_immutable(data_array)
    return Encoded(objects.shape, order, len(items), data_array, codes, at, others)


def _long_strings(flat: Any) -> bool:
    """Are the strings of the 1-d object array *flat* longer, on average
    over a sample of `_LENGTH_SAMPLE` items, than `_PLAIN_ABOVE_MEAN_CHARS`?

    Each its own object, long strings gain nothing from the buffer: pickle
    writes and reads one as a length and its bytes, where `encode` adds a
    list, a set of types, a join, a count of separators and an encode, and
    `decode` a decode, a split and an array. 50,000 unique review texts
    (66 MB) took 81 ms against plain pickle's 42 ms on Windows. On Linux,
    over synthetic columns of unique strings, it wrote 2.6x and read 2.2x
    slower at 1.3 KB a string, 1.3x and 1.5x slower at 190 characters, and
    wrote as fast near 128, while 900k strings of ~17 characters still
    wrote in 0.6x the time and read in the same."""
    count = len(flat)
    if not count:
        return False
    sample = flat[:: max(1, count // _LENGTH_SAMPLE)].tolist()
    lengths = [len(item) for item in sample if type(item) is str]
    return bool(lengths) and sum(lengths) > _PLAIN_ABOVE_MEAN_CHARS * len(lengths)


def _apart(where: Any, codes: Any, count: int) -> int:
    """How many of *count* items the items at *where* stand for: as many,
    or each as often as *codes* names it."""
    if codes is None:
        return len(where)
    import numpy as np

    return int(np.isin(codes, where).sum()) if len(where) else 0


def _shared(flat: Any) -> tuple[Any, Any]:
    """``(codes, first)`` for the 1-d object array *flat* whose items are
    mostly the same objects as others: each item's code, the number of its
    object in the order first met, and where each object is first met;
    else ``(None, None)``."""
    import numpy as np

    pd = sys.modules.get("pandas")
    count = len(flat)
    if count < _CODES_MIN_ITEMS or pd is None:
        return None, None
    pointers = _pointers(flat)
    codes, objects = pd.factorize(pointers, sort=False)
    if len(objects) * 2 > count:
        return None, None
    # Codes are numbered in the order first met: an object is first met
    # where the highest code so far goes up.
    highest = np.maximum.accumulate(codes)
    first = np.flatnonzero(np.concatenate(([True], highest[1:] > highest[:-1])))
    for dtype in (np.int8, np.int16, np.int32):
        if len(objects) <= np.iinfo(dtype).max:
            codes = codes.astype(dtype)
            break
    return _immutable(codes), first


def _pointers(flat: Any) -> Any:
    """Which object each item of the 1-d object array *flat* is, as an
    integer array: its pointers, read from its memory when it is
    contiguous, else each item's ``id``."""
    import ctypes

    import numpy as np

    if flat.flags.c_contiguous and flat.dtype.itemsize == np.dtype(np.intp).itemsize:
        start = flat.__array_interface__["data"][0]
        memory = (ctypes.c_char * flat.nbytes).from_address(start)
        return np.frombuffer(memory, dtype=np.intp).copy()
    return np.fromiter(map(id, flat.tolist()), dtype=np.intp, count=len(flat))


def decode(shape: tuple, order: str, strings: int, data: Any, codes: Any, at: Any, others: list | None) -> Any:
    """The numpy object array `encode` wrote. Raises ``ValueError`` or
    ``IndexError`` (`serialization.RESTORE_ERRORS`) for parts that do not
    fit together."""
    import numpy as np

    count = 1
    for size in shape:
        count *= size
    text = str(memoryview(data).cast("B"), "utf-8")
    parts = text.split(SEPARATOR) if strings else []
    if len(parts) != strings:
        raise ValueError("a stored text column does not hold the strings it names")
    found = np.array(parts, dtype=object) if parts else np.empty(0, dtype=object)
    if at is not None:
        at = np.asarray(at)
        if len(others) != len(at):
            raise ValueError("a stored text column does not hold the items it names")
        items = np.empty(strings + len(at), dtype=object)
        is_str = np.ones(len(items), dtype=bool)
        is_str[at] = False
        apart = np.empty(len(others), dtype=object)
        apart[:] = others
        items[at] = apart
        items[is_str] = found
        found = items
    flat = found if codes is None else found.take(np.asarray(codes))
    if flat.size != count:
        raise ValueError("a stored text column does not hold the items it names")
    return flat.reshape(shape, order=order) if len(shape) != 1 else flat


# ------------------------------------------------------- kept encodings


class _Kept:
    """Encodings kept for the next store of the same column, least recently
    used first, at most `_KEPT_MAX_BYTES` together: by content key
    (``by_key``) and, for a column the RAM tier keeps, by ``id`` with a
    weak reference to it (``by_id``)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entries: OrderedDict[Any, tuple[Any, Encoded]] = OrderedDict()
        self.nbytes = 0
        self.encoded = 0  # columns encoded, for the work-count tests

    def get(self, slot: Any, column: Any) -> Encoded | None:
        with self.lock:
            found = self.entries.get(slot)
            if found is None or (found[0] is not None and found[0]() is not column):
                return None
            self.entries.move_to_end(slot)
            return found[1]

    def put(self, slot: Any, ref: Any, encoded: Encoded) -> None:
        if encoded.nbytes > _KEPT_MAX_BYTES // 4:
            return
        with self.lock:
            old = self.entries.pop(slot, None)
            if old is not None:
                self.nbytes -= old[1].nbytes
            self.entries[slot] = (ref, encoded)
            self.nbytes += encoded.nbytes
            while self.nbytes > _KEPT_MAX_BYTES and self.entries:
                _slot, (_ref, gone) = self.entries.popitem(last=False)
                self.nbytes -= gone.nbytes

    def drop(self, slot: Any, ref: Any) -> None:
        with self.lock:
            found = self.entries.get(slot)
            if found is not None and found[0] is ref:
                del self.entries[slot]
                self.nbytes -= found[1].nbytes


_KEPT = _Kept()
#: What `_KEPT` holds at most: past it, the oldest go.
_KEPT_MAX_BYTES = 256 * 1024**2


def encode_column(column: Any, objects: Any, key: tuple | None) -> Encoded | None:
    """`encode` of *objects*, the strings of the pandas array *column*, with
    missing values of any kind: from `_KEPT` when this content (*key*, a
    content key naming *objects*' bytes now) or this column (one the RAM
    tier keeps) was encoded before."""
    from .shared_blocks import is_kept

    kept = key is None and is_kept(column)
    slot: Any = ("key", key) if key is not None else ("id", id(column)) if kept else None
    if slot is not None:
        found = _KEPT.get(slot, None if key is not None else column)
        if found is not None:
            return found
    encoded = encode(objects)
    _KEPT.encoded += 1
    if encoded is not None and slot is not None:
        ref = None
        if key is None:
            ref = _weak.ref(column, lambda ref, slot=slot: _KEPT.drop(slot, ref))
        _KEPT.put(slot, ref, encoded)
    return encoded
