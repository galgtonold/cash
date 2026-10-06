"""A call site whose calls the cache will not serve stops paying for a key.

A call the cache has refused (it changed an argument, or drew random numbers)
runs plain on every later call with the same key, but only after its key was
built. Such a call never reported what it cost, so the guard that sends a
cheap site to plain never fired: ``re.search`` writes ``re._cache``, its key
hashed that cache each time, and a 6 s loop over a log took minutes.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_unit as cu
from cash.notebook.call_interception import CallSite

SITE = CallSite(
    source="grow(v)",
    free_names=frozenset({"grow"}),
    occurrence_index=0,
    computed_arg_positions=(0,),
    local_arg_positions=(0,),
)
N = cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES + 40


def test_a_refused_site_is_judged_after_a_few_calls(call_unit_harness):
    unit = call_unit_harness(lineage={"grow": "g"}, user_ns={})
    built: list[int] = []
    real = unit._keys.key

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    unit._keys.key = counting

    def grow(items):
        items.append(1)
        return len(items)

    wrapped = unit.wrap(grow, SITE)
    assert [wrapped([0]) for _ in range(N)] == [2] * N
    assert len(built) < N, f"every one of {N} calls built its key"
    assert len(built) <= cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES + 1


@pytest.mark.parametrize("calls", [N])
def test_a_refused_call_still_runs_and_returns_its_value(call_unit_harness, calls):
    unit = call_unit_harness(lineage={"grow": "g"}, user_ns={})
    seen: list[int] = []

    def grow(items):
        items.append(1)
        seen.append(len(items))
        return len(items)

    wrapped = unit.wrap(grow, SITE)
    assert [wrapped([0]) for _ in range(calls)] == [2] * calls
    assert len(seen) == calls
