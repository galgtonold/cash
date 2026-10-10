"""The share check gives up when it would take longer than the statement is worth.

A loop variable that is an element of a list of 40,000 sessions holds an
object of the outputs, so the check walks the whole list to find that holder:
6 s after a loop that ran 0.1 s. Given a budget, the walk raises instead, and
the statement is refused (it re-runs each time) rather than cached late.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import WalkBudgetExceeded, share_group, walk_budget


def _namespace(n: int = 1500) -> dict:
    sessions = [{"actions": [{"name": "a"}, {"name": "b"}]} for _ in range(n)]
    return {"sessions": sessions, "session": sessions[-1], "actions": sessions[-1]["actions"]}


def test_a_spent_budget_stops_the_walk():
    ns = _namespace()
    values = {"session": ns["session"], "actions": ns["actions"]}
    with pytest.raises(WalkBudgetExceeded), walk_budget(0.0):
        share_group(["session", "actions"], values, ns)


def test_with_time_to_spare_the_holder_is_found():
    ns = _namespace()
    values = {"session": ns["session"], "actions": ns["actions"]}
    with walk_budget(60.0):
        holders, shared = share_group(["session", "actions"], values, ns)
    assert set(holders) == {"sessions"}
    assert not shared


def test_no_budget_set_means_no_limit():
    ns = _namespace()
    values = {"session": ns["session"], "actions": ns["actions"]}
    holders, _shared = share_group(["session", "actions"], values, ns)
    assert set(holders) == {"sessions"}
