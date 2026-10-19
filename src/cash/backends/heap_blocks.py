"""Big arrays the RAM tier keeps, out of glibc's heap.

glibc serves an allocation from a free hole of its main heap when one fits,
however big. A 47 MB table the caller drops (``s = signal(close, lb)`` in a
loop) leaves such a hole, and the next call's temporaries of the same size
reuse it. The RAM tier keeping that table, or a copy it made into such a
hole, takes the hole away for good: the temporaries are mapped fresh, and
page-faulted in again, on every later call. p04's backtests faulted 48,924
pages a call under cash against 24,462 plain, 3 s over 32 calls.

So, on Linux with glibc, an array of at least `MIN_BYTES` the tier copies is
copied into a mapping of its own (`fresh_copy`), and a table it would keep
uncopied is copied so when its data sits in the heap (`in_heap`): the
caller's hole is freed with the caller's table, as without cash. A mapping
is returned to the system when the entry goes. Elsewhere big blocks get
mappings of their own anyway, and nothing here runs.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import sys
import threading
from collections.abc import Iterator
from typing import Any

__all__ = ["MIN_BYTES", "fresh_copy", "in_heap", "per_store"]

#: Smallest array kept out of the heap: glibc's largest dynamic mmap
#: threshold. A smaller block glibc may put in the heap itself anyway.
MIN_BYTES = 32 << 20

_GLIBC: list[bool] = []
_STORE = threading.local()


def _glibc() -> bool:
    if not _GLIBC:
        ok = False
        if sys.platform.startswith("linux"):
            try:
                ok = (os.confstr("CS_GNU_LIBC_VERSION") or "").startswith("glibc")
            except (ValueError, OSError, AttributeError):
                ok = False
        _GLIBC.append(ok)
    return _GLIBC[0]


@contextlib.contextmanager
def per_store() -> Iterator[None]:
    """One store: `in_heap` reads the heap's range at most once inside."""
    outer = getattr(_STORE, "heap", None)
    _STORE.heap = _UNREAD
    try:
        yield
    finally:
        _STORE.heap = outer


_UNREAD = object()


def _heap_range() -> tuple[int, int] | None:
    """The main heap's ``(start, end)``, read once per store; None when it
    cannot be read, or outside a store."""
    held = getattr(_STORE, "heap", None)
    if held is None:
        return None
    if held is _UNREAD:
        held = _read_heap_range()
        _STORE.heap = held
    return held or None


def _read_heap_range() -> tuple[int, int] | tuple[()]:
    """From the first ``[heap]`` line to the last: ``madvise`` (numpy's
    huge pages) splits the heap into several."""
    starts, ends = [], []
    try:
        with open("/proc/self/maps", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.rstrip().endswith("[heap]"):
                    start, end = line.split(None, 1)[0].split("-")
                    starts.append(int(start, 16))
                    ends.append(int(end, 16))
    except (OSError, ValueError):
        return ()
    return (min(starts), max(ends)) if starts else ()


def in_heap(array: Any) -> bool:
    """Is *array* a view of at least `MIN_BYTES` of glibc's main heap?
    False when that cannot be told, which keeps today's path."""
    if not _glibc():
        return False
    import numpy as np

    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    if root.nbytes < MIN_BYTES:
        return False
    heap = _heap_range()
    if heap is None:
        return False
    try:
        address = root.__array_interface__["data"][0]
    except (AttributeError, KeyError, TypeError):
        return False
    return heap[0] <= address < heap[1]


def fresh_copy(array: Any) -> Any | None:
    """A copy of the numpy array *array* in a mapping of its own, in its
    memory order, or None for one to copy the usual way: not on glibc,
    smaller than `MIN_BYTES`, of objects, a subclass, or neither C nor F
    contiguous (the usual copy keeps any other layout), or outside a store
    (`per_store`): a hit's copy is the caller's, as a computed value is. The
    mapping starts zeroed, so a record dtype's padding is zero as
    ``_copy_values`` leaves it."""
    if getattr(_STORE, "heap", None) is None or not _glibc():
        return None
    import numpy as np

    if type(array) is not np.ndarray or array.nbytes < MIN_BYTES or array.dtype.hasobject:
        return None
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        return None
    try:
        memory = mmap.mmap(-1, array.nbytes)
    except (OSError, ValueError):
        return None
    copied = np.ndarray(array.shape, dtype=array.dtype, buffer=memory, order=order)
    np.copyto(copied, array, casting="no")
    return copied
