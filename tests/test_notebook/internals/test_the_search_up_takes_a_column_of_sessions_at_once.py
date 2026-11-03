"""The search for the variables holding a statement's outputs
(`shared_objects.referring_names`) takes the lists of a frame's column of
sessions in a few passes, not a step and a budget check each: 87,000
sessions of ``(action, time)`` records cost a second a statement there, and
the search ran out of time before it looked up (student_8)."""

from __future__ import annotations

import datetime

import pytest

from cash.notebook import shared_objects

pd = pytest.importorskip("pandas")

SESSIONS = 5_000


def _sessions() -> list:
    t0 = datetime.datetime(2020, 1, 1)
    return [[(f"a{s % 11}", t0 + datetime.timedelta(seconds=s + a)) for a in range(3)] for s in range(SESSIONS)]


def _budget_checks(monkeypatch) -> list[int]:
    checks = [0]
    check = shared_objects.check_walk_budget

    def counted() -> None:
        checks[0] += 1
        check()

    monkeypatch.setattr(shared_objects, "check_walk_budget", counted)
    return checks


def test_the_lists_of_a_column_are_taken_at_once(monkeypatch):
    logs = _sessions()
    df = pd.DataFrame({"actions": logs})
    checks = _budget_checks(monkeypatch)
    names, complete = shared_objects.referring_names([df], {"df": df, "logs": logs}, 30.0, lambda name: name == "df")
    assert complete
    assert names == {"logs"}
    assert checks[0] < SESSIONS // 10


def test_a_session_list_bound_to_a_name_is_found():
    logs = _sessions()
    df = pd.DataFrame({"actions": logs})
    one = logs[1234]
    names, complete = shared_objects.referring_names([df], {"df": df, "one": one}, 30.0, lambda name: name == "df")
    assert complete
    assert names == {"one"}


def test_a_column_holding_nothing_named_finds_no_name():
    df = pd.DataFrame({"actions": _sessions()})
    names, complete = shared_objects.referring_names([df], {"df": df}, 30.0, lambda name: name == "df")
    assert complete
    assert names == set()
