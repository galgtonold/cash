"""A list the RAM copy made already is not walked before a flat-first pickle.

On Python 3.10 under Windows (``FLAT_FIRST``) a value is walked on the heap
before the pickler sees it, to know it is shallow enough for the small C
stack there. The walk opened every list, dict and tuple, and asked the copy's
persistent id only of the other objects: an entry naming ``data``, a log of
3,000 sessions the copy had already made (and would put in place by its
id), had every record and datetime of the log walked and asked about, 24,000
calls on each store and each hit. A container is asked first, as the pickler
asks it, and one taken whole is not opened.
"""

from __future__ import annotations

import datetime
import sys

import pytest

from cash import kept_state
from cash.backends.memory_backend import InMemoryBackend

SESSIONS = 3_000


def _log() -> list:
    start = datetime.datetime(2020, 1, 1)
    return [
        [uid, [(f"a{j % 7}", start + datetime.timedelta(seconds=uid * 10 + j)) for j in range(8)]]
        for uid in range(SESSIONS)
    ]


def _hook_calls(run) -> int:
    """Calls of the RAM copy's persistent id while *run* runs."""
    calls = [0]

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code.co_name == "persistent_id":
            calls[0] += 1

    sys.setprofile(profile)
    try:
        run()
    finally:
        sys.setprofile(None)
    return calls[0]


@pytest.mark.parametrize("flat_first", [False, True])
def test_the_log_is_not_walked_on_a_store_or_a_hit(monkeypatch, flat_first):
    monkeypatch.setattr(kept_state, "FLAT_FIRST", flat_first)
    data = _log()
    entry = {"variables": {"uid": data[-1][0], "actions": data[-1][1], "data": data}, "stdout": ""}
    backend = InMemoryBackend()
    stored = _hook_calls(lambda: backend.set("k", entry, {"execution_time": 0.05}))
    served = _hook_calls(lambda: backend.get("k"))
    assert stored < SESSIONS and served < SESSIONS, (stored, served)


def test_the_copy_still_shares_the_list_two_names_hold(monkeypatch):
    monkeypatch.setattr(kept_state, "FLAT_FIRST", True)
    data = _log()
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"actions": data[-1][1], "data": data}, "stdout": ""}, {"execution_time": 0.05})
    hit = backend.get("k")[1]["variables"]
    assert hit["data"] == data
    assert hit["actions"] is hit["data"][-1][1]
    assert hit["data"] is not data


def test_a_value_deeper_than_the_pickler_may_follow_is_still_pickled_flat(monkeypatch):
    """The positive control: a chain the copy has not made is walked, found
    deep, and pickled flat, and comes back whole."""
    monkeypatch.setattr(kept_state, "FLAT_FIRST", True)
    chain: list = []
    for _ in range(500):
        chain = [chain]
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"chain": chain}, "stdout": ""}, {"execution_time": 0.05})
    got = backend.get("k")[1]["variables"]["chain"]
    depth = 0
    while got:
        got = got[0]
        depth += 1
    assert depth == 500
