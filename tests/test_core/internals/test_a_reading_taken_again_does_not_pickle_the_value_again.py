"""The RAM tier takes a slow unpickling reading again by reading the same
pickle again, not by pickling the value a second time.

A reading of half the compute or more is taken twice and the faster one
kept (`_may_refuse`). The second copy pickled the whole value again only to
time its unpickling: a loop over a list of 10,000 sessions of
``(action, datetime)`` pairs stored that list beside its loop variables, and
the second pickle was half of a 2 s store, for 0.06 s of unpickling.
"""

from __future__ import annotations

import time

from cash.backends.memory_backend import InMemoryBackend

#: ``__reduce__`` and load calls, in order.
EVENTS: list[str] = []
#: Seconds each load sleeps: one-off stalls first, then ``LOAD_S``.
STALLS: list[float] = []


def _load(n: int) -> Row:
    EVENTS.append("load")
    time.sleep(STALLS.pop(0) if STALLS else 0.0)
    return Row(n)


class Row:
    def __init__(self, n: int) -> None:
        self.n = n

    def __reduce__(self):
        EVENTS.append("reduce")
        return (_load, (self.n,))


def _payload() -> dict:
    # Over a MiB, so the tier times its unpickling, beside one object of a
    # class of its own.
    return {"variables": {"row": Row(7), "rows": [[i, ("click", float(i))] for i in range(40_000)]}, "stdout": ""}


def test_a_slow_reading_is_taken_again_from_the_same_pickle():
    EVENTS.clear()
    STALLS[:] = [0.05]
    backend = InMemoryBackend()

    backend.set("k", _payload(), {"execution_time": 0.01})

    assert EVENTS == ["reduce", "load", "load"], "the reading was not taken again, or the value was pickled again"
    hit = backend.get("k")[1]
    assert hit["variables"]["row"].n == 7
    assert hit["variables"]["rows"][-1] == [39_999, ("click", 39_999.0)]


def test_a_fast_reading_is_not_taken_again():
    EVENTS.clear()
    STALLS.clear()
    backend = InMemoryBackend()

    backend.set("k", _payload(), {"execution_time": 100.0})

    assert EVENTS == ["reduce", "load"]
