"""RAM-tier reads the decorator's fast hit path held back, written in.

A fast hit (`cash.decorator.hit_memo`) is served by the C extension without
calling into the RAM tier. What an ordinary read records there -- the
entry's ``last_access`` and ``access_count``, and its place in the eviction
order (`InMemoryBackend._touch`) -- is counted in C instead and written in
here, before anything reads it: an eviction, a metadata read.
"""

from __future__ import annotations

from ._native import cwatch as _cw

_take = getattr(_cw, "take_touches", None)


def flush() -> None:
    """Write every held-back read into its RAM tier."""
    if _take is None:
        return
    for ram, md, key, reads, last in _take():
        with ram._lock:
            if ram._store.get(key, (None,))[0] is not md:
                continue  # the entry went or was replaced meanwhile
            md["last_access"] = last
            md["access_count"] = md.get("access_count", 0) + reads
            ram._touch(key)
