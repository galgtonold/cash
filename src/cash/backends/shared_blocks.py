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
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any

from ..content_hashers import column_memory_key, known_column_pieces, pandas_value_route

logger = logging.getLogger(__name__)

__all__ = ["BlockReuse", "is_kept", "reused_count"]

#: ``content key -> weak reference`` to block values a stored table holds.
_KEPT: dict[tuple, Any] = {}
#: ``id(values) -> weak reference`` for the same values: what `is_kept` asks.
_KEPT_IDS: dict[int, Any] = {}
_LOCK = threading.Lock()


def is_kept(values: Any) -> bool:
    """Are *values* block values a stored table holds and `find` hands out,
    so the RAM tier's own: never written while they live?"""
    ref = _KEPT_IDS.get(id(values))
    return ref is not None and ref() is values


class _Counts:
    reused = 0
    reused_bytes = 0


def reused_count() -> tuple[int, int]:
    """Blocks put into a stored table instead of a copy, and their bytes."""
    return _Counts.reused, _Counts.reused_bytes


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

    __slots__ = ("_keys",)

    def __init__(self, keys: dict[tuple, tuple]) -> None:
        #: ``positions of a block's columns -> content key``.
        self._keys = keys

    @classmethod
    def for_frame(cls, frame: Any) -> BlockReuse | None:
        """None when no block of the DataFrame *frame* has a content the
        frame memo vouches for."""
        if getattr(frame, "ndim", 0) != 2:
            return None
        pieces = known_column_pieces(frame)
        if pieces is None:
            return None
        plan = _plan(frame)
        # Asked only now: the memo counts who references a frame's arrays to
        # vouch for it, and `_plan` holds none of them any more.
        keys = {}
        for positions, layout, memory in plan:
            found = [pieces.get(key) for key in memory]
            if all(piece is not None for piece in found):
                keys[positions] = (layout, tuple(found))
        return cls(keys) if keys else None

    def find(self, block: Any) -> Any:
        """Values a stored table holds with the content of the caller's
        *block*, or None."""
        key = self._keys.get(tuple(block.mgr_locs.as_array))
        if key is None:
            return None
        with _LOCK:
            ref = _KEPT.get(key)
        values = ref() if ref is not None else None
        if values is not None:
            _Counts.reused += 1
            _Counts.reused_bytes += int(getattr(values, "nbytes", 0))
        return values

    def keep(self, copied: Any) -> None:
        """Make the blocks of *copied*, the stored copy of the table, the
        ones `find` hands out for their content."""
        try:
            for block in copied._mgr.blocks:
                key = self._keys.get(tuple(block.mgr_locs.as_array))
                # The copy keeps each block's content and layout
                # (`copy_keeping_blocks`); checked, not trusted.
                if key is None or _layout(block.values) != key[0]:
                    continue
                values = block.values
                with _LOCK:
                    ref = _KEPT.get(key)
                    if ref is not None and ref() is not None:
                        continue
                    ref = weakref.ref(values, lambda _ref, key=key, at=id(values): _drop(key, at, _ref))
                    _KEPT[key] = ref
                    _KEPT_IDS[id(values)] = ref
        except Exception:  # only a saving: nothing kept
            logger.debug("cash: stored blocks not kept for reuse", exc_info=True)


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
