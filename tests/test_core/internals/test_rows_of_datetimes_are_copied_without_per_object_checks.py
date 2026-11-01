"""The RAM tier copies rows of ``(action, datetime)`` pairs without asking
every pair and every datetime what kind of object it is, and pickles a
datetime without a Python call of its own.

Its copy is a pickle round trip whose ``persistent_id`` hook picks out
arrays, frames and kept bytes. The hook ran all its checks for each of the
280,000 tuples and datetimes of 10,000 parsed sessions, and each datetime
went through a Python reduce that had nothing to part: a third of the
store of a list that a loop over it kept beside its loop variables. The
checks are asked once per type, and a datetime keeps the C reduce, which
writes the same bytes.
"""

from __future__ import annotations

import datetime
import pickle
import sys

from cash import kept_state
from cash.backends.memory_backend import InMemoryBackend

ROWS = 2_000


def _sessions() -> list:
    start = datetime.datetime(2020, 1, 1)
    return [[uid, [("click", start + datetime.timedelta(seconds=uid + j)) for j in range(5)]] for uid in range(ROWS)]


def _calls_by_name(run) -> dict[str, int]:
    """Python calls made by *run*, by function name."""
    counts: dict[str, int] = {}

    def profile(frame, event, _arg):
        if event == "call":
            counts[frame.f_code.co_name] = counts.get(frame.f_code.co_name, 0) + 1

    sys.setprofile(profile)
    try:
        run()
    finally:
        sys.setprofile(None)
    return counts


def _lines_in(name: str, run) -> tuple[int, int]:
    """Calls of, and lines run in, the functions called *name* while *run* runs."""
    calls = lines = 0

    def local(frame, event, _arg):
        nonlocal lines
        if event == "line":
            lines += 1
        return local

    def trace(frame, event, _arg):
        nonlocal calls
        if frame.f_code.co_name != name:
            return None
        calls += 1
        return local

    sys.settrace(trace)
    try:
        run()
    finally:
        sys.settrace(None)
    return calls, lines


def test_a_datetime_is_pickled_by_its_own_reduce():
    rows = _sessions()
    counts = _calls_by_name(lambda: kept_state.dumps(rows, pickle.HIGHEST_PROTOCOL))
    assert counts.get("reduce", 0) < 10, counts.get("reduce")


def test_the_bytes_are_the_ones_pickle_writes():
    tz = datetime.timezone(datetime.timedelta(hours=2))
    values = [
        datetime.datetime(2020, 1, 2, 3, 4, 5, 6),
        datetime.datetime(2020, 1, 2, tzinfo=tz),
        datetime.date(2020, 1, 2),
        datetime.time(3, 4, tzinfo=datetime.timezone.utc),
        datetime.timedelta(days=1, microseconds=5),
    ]
    for protocol in (2, pickle.HIGHEST_PROTOCOL):
        assert kept_state.dumps(values, protocol) == pickle.dumps(values, protocol)
        assert kept_state.dumps(values, protocol, keyed=True) == pickle.dumps(values, protocol)


class Stamp(datetime.datetime):
    pass


def test_a_datetime_subclass_keeps_its_attributes():
    stamp = Stamp(2020, 1, 2)
    stamp.source = "log"
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"s": [stamp]}}, {"execution_time": 1.0})

    hit = backend.get("k")[1]["variables"]["s"][0]
    assert hit == stamp and hit.source == "log"


def test_each_row_is_only_looked_up_in_the_memo():
    rows = _sessions()
    # As a loop leaves them: its variable is the last row's list, held in
    # the list too, so the entry is copied through pickle.
    entry = {"variables": {"uid": ROWS - 1, "actions": rows[-1][1], "data": rows}, "stdout": ""}
    backend = InMemoryBackend()
    calls, lines = _lines_in("persistent_id", lambda: backend.set("k", entry, {"execution_time": 1.0}))

    assert calls > 2 * ROWS, "the copy did not go through the hook: the test measures nothing"
    assert lines / calls < 7, f"{lines / calls:.1f} lines a call"
    assert backend.get("k")[1]["variables"]["data"] == rows
