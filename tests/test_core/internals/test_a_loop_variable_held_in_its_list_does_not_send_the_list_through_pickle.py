"""A loop's entry that names its variable before the list the variable is
part of is copied a container at a time, as when it names the list first.

``for uid, actions in data:`` leaves ``actions``, the last session's list of
``(name, datetime)`` pairs, which ``data`` holds too. The RAM tier copies
such a list on its own first (a new list over the same pairs), and the
level-at-a-time copy of ``data`` then gave up on meeting it again: the whole
log went through pickle, every pair and datetime one at a time, on every
store and every hit. That copy now puts the list already made in its place,
as pickle through the same memo does; the two names still share one list.
"""

from __future__ import annotations

import datetime
import sys

from cash.backends.memory_backend import InMemoryBackend

SESSIONS = 3_000


def _log() -> list:
    start = datetime.datetime(2020, 1, 1)
    return [
        [uid, [(f"a{j % 7}", start + datetime.timedelta(seconds=uid * 10 + j)) for j in range(8)]]
        for uid in range(SESSIONS)
    ]


def _pickled_objects(run) -> int:
    """Calls of the RAM copy's pickle hook while *run* runs: one per object
    pickled."""
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


def _entry(data: list, variable_first: bool) -> dict:
    names = {"uid": data[-1][0], "actions": data[-1][1], "k": 3}
    variables = {**names, "data": data} if variable_first else {"data": data, **names}
    return {"variables": variables, "stdout": "0.5\n"}


def test_the_list_is_not_pickled_whichever_name_comes_first():
    data = _log()
    for variable_first in (True, False):
        backend = InMemoryBackend()
        entry = _entry(data, variable_first)
        stored = _pickled_objects(lambda: backend.set("k", entry, {"execution_time": 0.05}))
        served = _pickled_objects(lambda: backend.get("k"))
        assert stored < SESSIONS and served < SESSIONS, (variable_first, stored, served)


def test_the_two_names_still_share_one_list():
    data = _log()
    backend = InMemoryBackend()
    backend.set("k", _entry(data, variable_first=True), {"execution_time": 0.05})

    for _ in range(2):  # the store's copy, then the plan the first hit made
        hit = backend.get("k")[1]["variables"]
        assert hit["data"] == data
        assert hit["actions"] is hit["data"][-1][1]
        assert hit["data"] is not data and hit["actions"] is not data[-1][1]
        hit["actions"].append(("x", None))
        assert data[-1][1][-1] != ("x", None), "the hit wrote into the stored list"


def test_a_list_holding_lists_held_by_another_name_is_still_shared():
    rows = [[i, [i, i + 1]] for i in range(2_000)]
    last = rows[-1]  # a list holding a list: not taken as it is
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"last": last, "rows": rows}, "stdout": ""}, {"execution_time": 0.05})

    hit = backend.get("k")[1]["variables"]
    assert hit["rows"] == rows
    assert hit["last"] is hit["rows"][-1]
