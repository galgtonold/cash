"""Columns of the same content the RAM tier keeps once.

A loop's results share most of their columns with the table they were made
from: ``h = matches.assign(p=...)`` holds ``matches``' columns and adds one.
The tier copies each table it stores (`frame_sharing.copy_keeping_blocks`),
so fifteen iterations kept fifteen copies of the same seven columns.

A pandas block the tier has copied is the tier's own: no caller holds it,
and the tier never writes a value it stores (a hit is a copy, or a shallow
copy of frozen data). A block of the same content can therefore be put in
another stored table instead of a new copy. Content is named by the bytes
each column folded into the table's digest (`known_column_pieces`), which
the frame memo hands out only while it vouches that the table is unchanged
since it was hashed; with dtype, shape and memory order beside them. A
table whose digest is not known, or a column without a piece, is copied as
before.

A table no one hashed is matched by where its blocks came from: a cell that
rebinds a table and adds a column (``df["hour"] = ...``) stores the same
frame again, its other blocks the very arrays the tier copied a statement
before. The tier remembers which caller's array each stored block was
copied from (`keep`); when that array is stored again, its bytes are
compared with the stored block's (`_same_bytes`), and only an exact match
is taken. The caller may have written the array in place since, past
pandas (``s.array[0] = ...``); comparing reads both, writes nothing and
allocates nothing, where the copy it replaces wrote fresh pages.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any

from ..content_hashers import column_memory_key, known_column_pieces, pandas_value_route

logger = logging.getLogger(__name__)

__all__ = ["BlockReuse", "bytes_to_copy", "compared_count", "is_kept", "reused_count"]

#: ``content key -> weak reference`` to block values a stored table holds.
_KEPT: dict[tuple, Any] = {}
#: ``id(values) -> weak reference`` for the same values: what `is_kept` asks.
_KEPT_IDS: dict[int, Any] = {}
#: ``id(caller's block values) -> (weak reference to them, weak reference
#: to the stored block values copied from them)``: what `for_frame` asks of
#: a table no one hashed.
_COPIED_FROM: dict[int, tuple[Any, Any]] = {}
_LOCK = threading.Lock()


def is_kept(values: Any) -> bool:
    """Are *values* block values a stored table holds and `find` hands out,
    so the RAM tier's own: never written while they live?"""
    ref = _KEPT_IDS.get(id(values))
    return ref is not None and ref() is values


class _Counts:
    reused = 0
    reused_bytes = 0
    compared = 0
    compared_bytes = 0


def reused_count() -> tuple[int, int]:
    """Blocks put into a stored table instead of a copy, and their bytes."""
    return _Counts.reused, _Counts.reused_bytes


def compared_count() -> tuple[int, int]:
    """Caller's blocks compared with the stored block copied from them
    (`_same_bytes`), and the bytes of each side compared."""
    return _Counts.compared, _Counts.compared_bytes


def _layout(values: Any) -> tuple | None:
    """The type, dtype, shape and memory order of a block's *values*; None
    for values not shared: only numpy arrays and the pandas arrays over one
    that hold nothing else (dates, durations, strings as Python objects)."""
    import numpy as np
    import pandas as pd

    kind = type(values)
    if kind is np.ndarray:
        array = values
        extra: Any = None
    elif kind in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray):
        array = values._ndarray
        extra = repr(values.freq)
    elif kind is pd.arrays.StringArray:
        array = values._ndarray
        extra = None
    else:
        return None
    if type(array) is not np.ndarray or not (array.flags.c_contiguous or array.flags.f_contiguous):
        return None
    return (
        kind.__qualname__,
        str(values.dtype),
        array.shape,
        array.flags.c_contiguous,
        array.flags.f_contiguous,
        extra,
    )


class BlockReuse:
    """Finds, for the blocks of one table being stored, a block of the same
    content another stored table holds (`find`), and makes the copy's own
    blocks findable once it is stored (`keep`)."""

    __slots__ = ("_keys", "_sources")

    def __init__(self, keys: dict[tuple, tuple], sources: dict[tuple, tuple[Any, tuple]]) -> None:
        #: ``positions of a block's columns -> content key``.
        self._keys = keys
        #: ``positions -> (the caller's block values, their layout)`` for the
        #: blocks `_layout` knows: what `keep` records the stored copies as
        #: copied from.
        self._sources = sources

    @classmethod
    def for_frame(cls, frame: Any) -> BlockReuse | None:
        """None when *frame* is not a DataFrame, or none of its blocks has
        a layout a stored block can be taken for (`_layout`)."""
        if getattr(frame, "ndim", 0) != 2:
            return None
        keys = {}
        pieces = known_column_pieces(frame)
        if pieces is not None:
            plan = _plan(frame)
            # Asked only now: the memo counts who references a frame's arrays to
            # vouch for it, and `_plan` holds none of them any more.
            for positions, layout, memory in plan:
                found = [pieces.get(key) for key in memory]
                if all(piece is not None for piece in found):
                    keys[positions] = (layout, tuple(found))
        # Taken after the memo was asked: they hold the frame's arrays.
        sources = _sources(frame)
        return cls(keys, sources) if keys or sources else None

    def find(self, block: Any) -> Any:
        """Values a stored table holds with the content of the caller's
        *block*, or None."""
        positions = tuple(block.mgr_locs.as_array)
        key = self._keys.get(positions)
        values = None
        if key is not None:
            with _LOCK:
                ref = _KEPT.get(key)
            values = ref() if ref is not None else None
        if values is None:
            values = self._copied_before(block, positions)
        if values is not None:
            _Counts.reused += 1
            _Counts.reused_bytes += int(getattr(values, "nbytes", 0))
        return values

    def _copied_before(self, block: Any, positions: tuple) -> Any:
        """The stored values copied from *block*'s own values, when their
        bytes are still the same; else None."""
        source = self._sources.get(positions)
        if source is None or source[0] is not block.values:
            return None
        with _LOCK:
            refs = _COPIED_FROM.get(id(block.values))
        if refs is None or refs[0]() is not block.values:
            return None
        stored = refs[1]()
        if stored is None or _layout(stored) != source[1]:
            return None
        return stored if _same_bytes(block.values, stored) else None

    def keep(self, copied: Any) -> None:
        """Make the blocks of *copied*, the stored copy of the table, the
        ones `find` hands out for their content, and for the caller's
        blocks they were copied from."""
        try:
            for block in copied._mgr.blocks:
                positions = tuple(block.mgr_locs.as_array)
                layout = _layout(block.values)
                values = block.values
                source = self._sources.get(positions)
                if source is not None and source[1] == layout and source[0] is not values:
                    _remember_copy(source[0], values)
                key = self._keys.get(positions)
                # The copy keeps each block's content and layout
                # (`copy_keeping_blocks`); checked, not trusted.
                if key is None or layout != key[0]:
                    continue
                with _LOCK:
                    ref = _KEPT.get(key)
                    if ref is not None and ref() is not None:
                        continue
                    ref = weakref.ref(values, lambda _ref, key=key, at=id(values): _drop(key, at, _ref))
                    _KEPT[key] = ref
                    _KEPT_IDS[id(values)] = ref
        except Exception:  # only a saving: nothing kept
            logger.debug("cash: stored blocks not kept for reuse", exc_info=True)


def bytes_to_copy(frame: Any) -> int:
    """About the bytes storing the pandas *frame* copies: its blocks, less
    those the tier copied from these very arrays and still keeps (taken
    again when their bytes still match, `BlockReuse.find`) and Arrow data
    (shared when Arrow allocated it, `frame_sharing.arrow_owned`). An
    estimate for the store's cost, never a promise of what is shared."""
    total = 0
    for block in frame._mgr.blocks:
        values = block.values
        if getattr(values, "_pa_array", None) is not None:
            continue
        with _LOCK:
            refs = _COPIED_FROM.get(id(values))
        if refs is not None and refs[0]() is values and refs[1]() is not None:
            continue
        total += int(getattr(values, "nbytes", 0))
    return total


def _sources(frame: Any) -> dict[tuple, tuple[Any, tuple]]:
    """``positions -> (values, layout)`` for each block of *frame* whose
    bytes `_same_bytes` can compare."""
    sources = {}
    try:
        for block in frame._mgr.blocks:
            layout = _layout(block.values)
            if layout is not None and _raw(block.values).dtype.kind not in "OV":
                sources[tuple(int(pos) for pos in block.mgr_locs.as_array)] = (block.values, layout)
    except Exception:  # pandas internals changed: copy it all
        logger.debug("cash: a table's blocks unknown, copied", exc_info=True)
        return {}
    return sources


def _remember_copy(source: Any, stored: Any) -> None:
    """Record that the stored block values *stored* were copied from the
    caller's *source*, for as long as both live."""
    at = id(source)
    made: list[tuple[Any, Any]] = []

    def forget(_ref: Any) -> None:
        with _LOCK:
            if made and _COPIED_FROM.get(at) is made[0]:
                del _COPIED_FROM[at]

    try:
        entry = (weakref.ref(source, forget), weakref.ref(stored, forget))
    except TypeError:  # values that take no weak reference: not remembered
        return
    made.append(entry)
    with _LOCK:
        _COPIED_FROM[at] = entry


def _raw(values: Any) -> Any:
    """The numpy array a block's *values* keep their data in (`_layout`)."""
    return getattr(values, "_ndarray", values)


#: Elements `_same_bytes` compares at a time: what it allocates is this many
#: booleans, whatever the size of the block.
_COMPARE_CHUNK = 1 << 20


def _same_bytes(values: Any, stored: Any) -> bool:
    """Do the block values *values* and *stored*, of one `_layout`, hold the
    same bytes? Read in memory order, a chunk at a time."""
    import numpy as np

    a, b = _raw(values), _raw(stored)
    try:
        a = np.ravel(a, order="K").view(np.uint8)
        b = np.ravel(b, order="K").view(np.uint8)
    except (TypeError, ValueError):  # no plain bytes to compare
        return False
    if a.size != b.size or not (a.flags.c_contiguous and b.flags.c_contiguous):
        return False
    _Counts.compared += 1
    _Counts.compared_bytes += int(a.size)
    if a.size % 8 == 0:
        a, b = a.view(np.uint64), b.view(np.uint64)
    out = np.empty(min(a.size, _COMPARE_CHUNK), dtype=bool)
    for start in range(0, a.size, _COMPARE_CHUNK):
        stop = min(start + _COMPARE_CHUNK, a.size)
        part = out[: stop - start]
        np.equal(a[start:stop], b[start:stop], out=part)
        if not part.all():
            return False
    return True


def _plan(frame: Any) -> list[tuple[tuple, tuple, list[tuple]]]:
    """``(positions, layout, column memory keys)`` for each block of *frame*
    `_layout` knows. Holds no reference to an array once it returns."""
    import pandas as pd

    plan = []
    try:
        dtypes = frame.dtypes
        for block in frame._mgr.blocks:
            layout = _layout(block.values)
            if layout is None:
                continue
            positions = tuple(int(pos) for pos in block.mgr_locs.as_array)
            memory = [column_memory_key(frame, pos, pandas_value_route(dtypes.iloc[pos], pd)) for pos in positions]
            if all(key is not None for key in memory):
                plan.append((positions, layout, memory))
    except Exception:  # pandas internals changed: copy it all
        logger.debug("cash: a table's blocks unknown, copied", exc_info=True)
        return []
    return plan


def _drop(key: tuple, at: int, ref: Any) -> None:
    with _LOCK:
        if _KEPT.get(key) is ref:
            del _KEPT[key]
        if _KEPT_IDS.get(at) is ref:
            del _KEPT_IDS[at]
