"""The walks below an output pass over values, and over tuples holding
values alone, a container's worth at a time, not a step each.

A loop over 87,000 sessions left ``deltas``, a list of 1.3 million floats,
and ``log``, a session of ``(action, time)`` records held by the list of
them all. Finding which outputs share an object (`owners_sharing`) asked
each float whether it was a value, 4 s; the share walk asked each record of
each session whether it was a node, 2 s. What they find is what a step
each found.
"""

from __future__ import annotations

import datetime

from cash.notebook import shared_objects

T0 = datetime.datetime(2019, 10, 1)


def _sessions(n: int) -> list:
    return [(f"@u{i}", [(f"a{j}", T0 + datetime.timedelta(seconds=j)) for j in range(20)]) for i in range(n)]


def _value_types() -> tuple:
    return shared_objects.VALUE_TYPES + shared_objects.library_value_types()


def _counting(monkeypatch, name: str) -> list:
    asked: list = []
    real = getattr(shared_objects, name)

    def counting(*args, **kwargs):
        asked.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shared_objects, name, counting)
    return asked


def test_the_ids_below_a_list_of_floats_are_found_without_a_step_each(monkeypatch):
    deltas = [i / 7 for i in range(50_000)]
    asked = _counting(monkeypatch, "is_value")
    assert shared_objects._reached_ids(deltas, _value_types()) == {id(deltas)}
    assert len(asked) < 10


def test_the_ids_below_records_are_each_record_and_what_holds_it(monkeypatch):
    inner = ("b", 2)
    value = [1.0, ("a", 1), [inner, ("c", 3)], ("d", [4])]
    found = shared_objects._reached_ids(value, _value_types())
    assert found == {id(value), id(value[1]), id(value[2]), id(inner), id(value[2][1]), id(value[3]), id(value[3][1])}


def _descend(root) -> dict:
    """What the walk below *root* takes, as the anchor check walks a list
    the statement left as it was (`_take_back_holding`)."""
    nodes: dict = {}
    shared_objects._descend([root], "", nodes, {}, {}, [], _value_types())
    return nodes


def test_sessions_of_records_are_walked_a_session_at_a_time(monkeypatch):
    logs = _sessions(2_000)
    asked = _counting(monkeypatch, "_is_node")
    nodes = _descend(logs)
    assert len(asked) < 2_000 * 20
    # Each session and its list: no record is a node.
    assert set(nodes) == {*map(id, logs), *(id(session[1]) for session in logs)}


def test_a_session_whose_records_hold_a_list_is_walked_below():
    logs = _sessions(100)
    held = [1]
    logs[7][1].append(("x", held))
    nodes = _descend(logs)
    assert id(held) in nodes and id(logs[7][1][-1]) in nodes
    assert id(logs[7][1][0]) not in nodes


def test_long_sessions_of_records_held_twice_are_not_read_as_trees(monkeypatch):
    """An explode leaves each record held by a frame's column too: a
    session of 32 records or more was read for the records held beyond
    their list, a C walk and a Python pass per session."""
    logs = [(f"@u{i}", [(f"a{j}", T0 + datetime.timedelta(seconds=j)) for j in range(40)]) for i in range(500)]
    column = [record for _user, session in logs for record in session]
    asked = []
    real = shared_objects._plain_data.held_beyond_parents

    def counting(*args, **kwargs):
        asked.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shared_objects._plain_data, "held_beyond_parents", counting)
    nodes: dict = {}
    shared_objects._descend([logs], "", nodes, {}, {}, [], _value_types(), trees=set())
    assert len(asked) < 10
    assert set(nodes) == {*map(id, logs), *(id(session[1]) for session in logs)}
    del column
