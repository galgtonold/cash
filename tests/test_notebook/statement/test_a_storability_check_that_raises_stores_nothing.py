"""A result whose storability check raises is not stored.

A call asks whether its result may be stored. Answering "yes" when that check
raises would store the one value it could not judge -- a closure that keeps
state, a figure pyplot points at -- and hand it back on a hit. Refusing to
store only means the call runs again.
"""

from __future__ import annotations

import time

from cash.notebook import call_entries
from cash.notebook.call_interception import CallSite


def _raises(*_args, **_kwargs):
    raise RuntimeError("the check itself failed")


def slow_list():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return [1, 2, 3]


def test_the_call_is_run_again_not_served(call_unit_harness, monkeypatch):
    monkeypatch.setattr(call_entries, "holds_a_closure_with_state", _raises)
    unit = call_unit_harness(lineage={"slow_list": "hash-slow-list"}, user_ns={})
    site = CallSite(source="slow_list()", free_names=frozenset({"slow_list"}), occurrence_index=0)

    unit.begin_statement()
    unit.wrap(slow_list, site)()
    unit.begin_statement()
    unit.wrap(slow_list, site)()

    assert [e["cache_hit"] for e in unit.drain()] == [False, False], "a result the check could not judge was stored"
