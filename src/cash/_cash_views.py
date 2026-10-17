"""The views of an array's memory that cash itself holds while it writes.

A store hands a table's large arrays to the background write as views of
their own memory (`PickleSerializer.serialize_split` with ``copy=False``),
for a value nothing changes until the write is done. Each view is one more
reference to the array, and the frame memo
(`cash.decorator.arg_hashing.frame_borrows_its_data`) reads any reference it
cannot account for as someone's handle: while a 50 MB result of a cached
call was being written, the statement's checks right after read it in full
again, twice in each pass of p04's grid loop. A view noted here is cash's,
and is counted as such until it is gone.

A view's reference lasts as long as any view made from it (a slice shares
its buffer), so the count may drop before the reference does: the memo then
sees one reference too many, and reads the table, as it did before.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

__all__ = ["held", "note"]

#: ``id(array) -> views of it cash holds``.
_HELD: dict[int, int] = {}
_LOCK = threading.Lock()


def note(view: memoryview) -> None:
    """Count *view*, a view cash keeps of ``view.obj``'s memory, until it is
    collected."""
    key = id(view.obj)
    with _LOCK:
        _HELD[key] = _HELD.get(key, 0) + 1
    weakref.finalize(view, _release, key)


def _release(key: int) -> None:
    with _LOCK:
        left = _HELD.get(key, 0) - 1
        if left > 0:
            _HELD[key] = left
        else:
            _HELD.pop(key, None)


def held(array: Any) -> int:
    """How many references to *array* the views cash holds make."""
    return _HELD.get(id(array), 0)
