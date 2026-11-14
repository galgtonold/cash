"""Plain records are cleared of ids a level at a time, whatever their count.

``sessions = parse_log(path)`` gives a list of dicts of lists of dicts of
strings and numbers. The identity check walked it an object at a time,
half a second a statement, and past its count of objects gave up and ran
the statement every time. Exact dicts, lists and tuples over immutable
leaves can hold no weak reference, none can be weakly referenced, and none
is the user's code: only their ints and strings could find an object by
its ``id()``. They are read a level at a time (`_plain_data.tree_walk`),
and the walk object by object runs only when one could be an id.

Positive controls: an ``id()`` in a field, in a key, or spelled in hex is
still found, and a value holding anything else is walked as before.
"""

from __future__ import annotations

import types

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import identity_held_reason


def _sessions(n: int) -> list[dict]:
    return [
        {"id": k, "user": f"u{k % 9}", "actions": [{"name": f"a{a}", "at": k * 10 + a, "ok": True} for a in range(3)]}
        for k in range(n)
    ]


@pytest.fixture
def walks(monkeypatch) -> list[int]:
    """How many times the walk object by object looked at a value."""
    ran: list[int] = []
    real = identity_holders._Walk.run

    def run(self, roots):
        if not self.quiet:
            ran.append(len(roots))
        return real(self, roots)

    monkeypatch.setattr(identity_holders._Walk, "run", run)
    return ran


def test_records_past_the_walks_count_are_cleared_without_it(walks):
    sessions = _sessions(identity_holders._MAX_SEEN // 4)
    assert identity_held_reason("sessions", sessions, {"sessions": sessions}) is None
    assert walks == []


def test_an_id_in_a_field_is_found(walks):
    target = [1, 2, 3]
    sessions = _sessions(500)
    sessions[-1]["actions"][1]["ref"] = id(target)
    held = identity_held_reason("sessions", sessions, {"sessions": sessions, "target": target})
    assert held is not None and held.kind == "an id()", held
    assert walks


def test_an_id_as_a_key_is_found():
    target = [1, 2, 3]
    registry = {"by_id": {id(target): "target"}, "rows": _sessions(50)}
    held = identity_held_reason("registry", registry, {"registry": registry, "target": target})
    assert held is not None and held.kind == "an id()", held


def test_an_id_spelled_in_hex_is_found():
    target = [1, 2, 3]
    sessions = _sessions(50)
    sessions[7]["user"] = hex(id(target))
    held = identity_held_reason("sessions", sessions, {"sessions": sessions, "target": target})
    assert held is not None and held.kind == "an id()", held


@pytest.mark.parametrize(
    "odd",
    [bytearray(b"x"), {1, 2}, types.SimpleNamespace(v=1)],
    ids=["a bytearray", "a set", "an object"],
)
def test_a_value_holding_anything_else_is_walked(walks, odd):
    sessions = _sessions(50)
    sessions[3]["actions"][0]["extra"] = odd
    assert identity_held_reason("sessions", sessions, {"sessions": sessions}) is None
    assert walks == [1]


def test_an_object_of_the_users_past_the_count_is_still_given_up_on(monkeypatch):
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 100)
    sessions = _sessions(500)
    sessions[0]["extra"] = types.SimpleNamespace(v=1)
    held = identity_held_reason("sessions", sessions, {"sessions": sessions})
    assert held is not None and "more objects than cash looks through" in held
