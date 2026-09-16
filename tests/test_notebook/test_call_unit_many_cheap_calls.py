"""Many cheap calls to one site in one statement stop being cached.

A call in a comprehension is made once per element; caching each one costs a
key, a lookup, a store and a file tracker -- ~14 ms a call around a function
reading one small file, which made r23s4's 5,030-file read go 8.4 s -> 71.6 s.
Past ``_GUARD_AFTER_CALLS`` calls, cheap ones are timed plain on a few samples
and, when caching costs more than ``_OVERHEAD_FACTOR`` times the work, the
rest of the statement's calls to the site run plain. Expensive calls are never
timed plain, and a new statement run starts over.

Overhead is made deterministic here by slowing the lookup.
"""
from __future__ import annotations

import time

import pytest

from cash.notebook import call_unit as cu
from cash.notebook.call_interception import CallSite

SITE = CallSite(source="work(v + 0)", free_names=frozenset({"work"}), occurrence_index=0,
                computed_arg_positions=(0,), local_arg_positions=(0,))
N = cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES + 40


@pytest.fixture
def slow_lookup(monkeypatch):
    real = cu.CallUnit._lookup
    lookups: list[str] = []

    def lookup(self, key):
        lookups.append(key)
        time.sleep(0.002)
        return real(self, key)
    monkeypatch.setattr(cu.CallUnit, "_lookup", lookup)
    return lookups


def test_many_cheap_calls_stop_being_cached(call_unit_harness, slow_lookup):
    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(lambda v: v * 2, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_AFTER_CALLS, "cheap calls kept being looked up"


def test_a_new_statement_run_starts_over(call_unit_harness, slow_lookup):
    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(lambda v: v * 2, SITE)
    for i in range(N):
        wrapped(i)
    unit.begin_statement()
    slow_lookup.clear()

    wrapped(0)

    assert len(slow_lookup) == 1


def test_expensive_calls_are_never_run_plain_to_measure_them(call_unit_harness, monkeypatch):
    """Timing a plain sample of a slow call would repeat the slow work; a
    call over the cheap bar is not sampled at all."""
    monkeypatch.setattr(cu, "_GUARD_CHEAP_BELOW_S", 0.001)
    ran: list[int] = []

    def work(v):
        ran.append(v)
        time.sleep(0.005)          # over the (lowered) cheap bar and the store floor
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)
    for i in range(N):
        wrapped(i)
    ran.clear()
    unit.begin_statement()

    for i in range(N):
        wrapped(i)

    assert ran == [], "an expensive call was run again instead of served"


def test_calls_a_hit_could_not_beat_stop_being_cached(call_unit_harness, slow_lookup):
    """Round 25 (r25s5): ``add_features(g)`` over 360 groups, ~5 ms a call,
    cost ~8 ms more a call to cache and still under 4x its work, so it kept
    being cached. Keying and looking it up alone cost as much as the call: a
    hit would never have been faster. Measured on the same samples, a site
    whose key and lookup cost at least the call runs plain."""
    def work(v):
        time.sleep(0.0015)         # under the 2 ms lookup; well under 4x the miss
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_AFTER_CALLS, "calls no hit could beat kept being looked up"
