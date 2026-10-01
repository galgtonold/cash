"""What a statement's cached calls stand for, counted once.

A statement's own work is its cost less the calls the cache holds
(``CallUnit.cached_compute_s``). A call made inside another cached call -- a
callee calling the function it was handed, cached at its own site -- is part
of the outer call's time: counted again, the calls could outweigh the whole
statement and price its own work at nothing.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook.call_interception import CallSite
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def unit_and_sites(call_unit_harness):
    def slow(v):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return [v]

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "slow": slow})
    inner = unit.wrap(slow, CallSite(source="slow(x)", free_names=frozenset({"slow", "x"}), occurrence_index=0))
    return unit, inner


def test_a_call_inside_a_stored_call_is_counted_once(unit_and_sites):
    unit, inner = unit_and_sites

    def outer(v):
        return inner(v) + [0]

    wrapped = unit.wrap(outer, CallSite(source="outer(x)", free_names=frozenset({"outer", "x"}), occurrence_index=0))
    assert wrapped(1) == [1, 0]
    events = unit.drain()
    assert [e["stored"] for e in events] == [True, True], "both calls must be cached for this to test anything"
    inner_s, outer_s = (e["execution_time"] for e in events)
    assert inner_s >= ABOVE_PERSISTENCE_FLOOR_S * 0.9
    # Against the times the calls measured, not the sleep: a slow runner
    # oversleeps by more than any fixed margin.
    assert unit.cached_compute_s == pytest.approx(outer_s), (
        f"counted {unit.cached_compute_s:.3f}s for an outer call of {outer_s:.3f}s holding an inner one of {inner_s:.3f}s"
    )


def test_a_call_inside_one_the_cache_does_not_hold_still_counts(unit_and_sites):
    """The outer call returns its argument, so it is not stored: it runs again,
    and the inner call it makes is served."""
    unit, inner = unit_and_sites

    def outer(v):
        inner(1)
        return v

    arg = [5]
    wrapped = unit.wrap(outer, CallSite(source="outer(y)", free_names=frozenset({"outer", "y"}), occurrence_index=0))
    assert wrapped(arg) is arg
    assert [e["stored"] for e in unit.drain()] == [True, False]
    assert unit.cached_compute_s >= ABOVE_PERSISTENCE_FLOOR_S * 0.9


def test_a_hit_counts_its_recorded_cost(unit_and_sites):
    unit, inner = unit_and_sites
    inner(1)
    before = unit.cached_compute_s
    inner(1)
    assert unit.drain()[-1]["cache_hit"]
    assert unit.cached_compute_s - before == pytest.approx(before, rel=0.01)
    assert unit.cached_restore_s >= 0.0
