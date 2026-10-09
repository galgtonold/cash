"""A decorated function's result comes back from the RAM tier as it was
returned, never as the bytes the tier keeps part of it in.

A dict holding big parsed records beside something else is kept with the
records as marshal bytes, read out by each hit's copy. When that copy could
not be made -- the store's copy had turned an object into one no copy takes
again -- the hit handed back the tier's own dict, wrapper and all.
"""

from __future__ import annotations

import threading

N = 5_000


class _BecomesALock:
    """Copied into the tier as a lock, which no later copy can copy."""

    def __reduce__(self):
        return threading.Lock, ()


def test_records_beside_what_cannot_be_copied_again_come_back_as_records(cash_instance):
    calls = []

    @cash_instance.cache
    def load():
        calls.append(1)
        return {"records": [{"id": i, "tags": ["a"]} for i in range(N)], "handle": _BecomesALock()}

    first = load()
    second = load()
    assert len(calls) == 1, "the second call is a hit"
    assert type(second["records"]) is list
    assert second["records"] == first["records"]
    assert [r["id"] for r in second["records"][:3]] == [0, 1, 2]
