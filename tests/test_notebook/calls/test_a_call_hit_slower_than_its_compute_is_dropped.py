"""A call-cache hit that costs more than the compute it saves is dropped.

`net_returns(orders, w)` hits took ~10 s each against ~4 s
of compute, and the badge reported "4/4 hit". The call cache had no
restore-cost check at all -- the statement path has one -- and nothing
compared a hit's real cost with what it saved. Measured on the hit now:
a loss drops the entry and the site runs plain for the rest of the session.

The costs are spent on the clock the call unit times itself with, not
slept: measured in wall-clock, a loaded runner stretched the 20 ms compute
past the 0.3 s hit, and the hit was rightly kept.
"""

import pytest

import cash.notebook.call_unit as cu
from cash.notebook.call_interception import CallSite


class FakeClock:
    """The clock ``call_unit`` reads, moved only by :meth:`spend`."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def spend(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(cu, "_perf_counter", fake)
    return fake


def _unit(call_unit_harness, calls, clock, monkeypatch, compute_s):
    """A call site whose every run of ``work`` costs *compute_s*: spent
    around the body, which must not touch the clock itself (a call that
    changes an object it reaches is never stored)."""
    # Not `calls.append` in the body: a hit replays that write to its closure.
    note = calls.append

    def work(k):
        note(k)
        return k * 2

    real_call = cu.call_capturing_output

    def timed_call(fn, args, kwargs):
        clock.spend(compute_s)
        return real_call(fn, args, kwargs)

    monkeypatch.setattr(cu, "call_capturing_output", timed_call)
    unit = call_unit_harness(lineage={"k": "hash-2"}, user_ns={"k": 2, "work": work})
    site = CallSite(source="work(k)", free_names=frozenset({"work", "k"}), occurrence_index=0)
    return unit, unit.wrap(work, site)


def _slow_lookups(unit, clock, monkeypatch, seconds):
    """Every backend read of *unit*'s cache costs *seconds*."""
    backend = unit._cash.backend
    real_get = backend.get

    def slow_get(key, *a, **k):
        clock.spend(seconds)
        return real_get(key, *a, **k)

    monkeypatch.setattr(backend, "get", slow_get)


def _entries(unit):
    return [m["key"] for m in unit._cash.backend.list_entries() or ()]


def test_a_hit_slower_than_its_compute_is_dropped(call_unit_harness, clock, monkeypatch):
    calls: list[int] = []
    unit, wrapped = _unit(call_unit_harness, calls, clock, monkeypatch, compute_s=0.02)
    assert wrapped(2) == 4 and calls == [2]
    assert len(_entries(unit)) == 1, "the compute was not stored"
    _slow_lookups(unit, clock, monkeypatch, 0.3)  # a hit that costs far more than 20 ms
    assert wrapped(2) == 4
    assert calls == [2], "the second call should still have been a hit"
    assert _entries(unit) == [], "the losing hit's entry was kept"
    assert wrapped(2) == 4
    assert calls == [2, 2], "a hit that cost 0.3 s to save 0.02 s was served again"


def test_a_hit_that_pays_is_kept(call_unit_harness, clock, monkeypatch):
    calls: list[int] = []
    unit, wrapped = _unit(call_unit_harness, calls, clock, monkeypatch, compute_s=0.3)
    assert wrapped(2) == 4
    _slow_lookups(unit, clock, monkeypatch, 0.02)  # a hit that costs far less than 0.3 s
    for _ in range(3):
        assert wrapped(2) == 4
    assert calls == [2]
    assert len(_entries(unit)) == 1
