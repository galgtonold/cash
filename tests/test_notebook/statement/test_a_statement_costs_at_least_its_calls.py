"""A statement is never recorded as cheaper than a cached call inside it.

A statement's cost was its wall time less cash's tax inside it, the file
tracker's seconds and the call unit's overhead. Part of that tax was counted
twice or belonged to the call: recording the read of the call's own entry
during its lookup was in both the tracker's seconds and the call's overhead,
and a read recorded inside the call is in the time the call's entry records.
So ``b = shifted(a) + [1]`` over a 0.2 s call was recorded 0.3-0.5 ms under
the call on Windows, and a hit on it credited less than a hit on the call.

These tests drive the accounting with a fake clock, so no sleep's precision
decides them.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_unit as call_unit_module
from cash.notebook.call_interception import CallSite
from cash.notebook.statement.call_routing import CashMarks, statement_price
from cash.tracking import file_tracker


class _Clock:
    """A perf counter that moves only when told to."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(call_unit_module, "_perf_counter", fake)
    return fake


@pytest.fixture
def record_reads(monkeypatch):
    """``record_reads(s)``: the file tracker spends *s* seconds recording a read."""

    def spend(seconds: float) -> None:
        monkeypatch.setattr(file_tracker, "_tracking_seconds", file_tracker.tracking_seconds() + seconds)

    return spend


def _marks(unit) -> CashMarks:
    return CashMarks(
        file_tracker.tracking_seconds(),
        unit,
        unit.overhead_s,
        unit.hits_saved_s,
        unit.cached_compute_s,
        unit.cached_restore_s,
        unit.reads_seq,
        unit.computed_s,
        unit.tracking_in_calls_s,
    )


def _since(before: CashMarks, unit) -> CashMarks:
    now = _marks(unit)
    return CashMarks(*(a - b if isinstance(a, float) else a for a, b in zip(now, before, strict=True)))


def _costly_lookup(unit, clock, record_reads, *, seconds: float, reading: float) -> None:
    """Make every lookup take *seconds*, *reading* of them recording the
    read of the entry, as a disk tier's lookup does."""
    lookup = unit._entries.lookup

    def slow_lookup(key):
        clock.now += seconds
        record_reads(reading)
        return lookup(key)

    unit._entries.lookup = slow_lookup


def test_the_tax_around_and_inside_a_call_leaves_the_call_whole(call_unit_harness, clock, record_reads):
    """``b = shifted(a) + [1]``: the lookup records a read of the call's
    entry, the call records a read of its own, and the ``+ [1]`` is quick."""

    def shifted(x):
        clock.advance(0.2)
        record_reads(0.0003)
        return [x]

    unit = call_unit_harness(lineage={"a": "h"}, user_ns={"a": 1, "shifted": shifted})
    _costly_lookup(unit, clock, record_reads, seconds=0.001, reading=0.0004)
    wrapped = unit.wrap(
        shifted, CallSite(source="shifted(a)", free_names=frozenset({"shifted", "a"}), occurrence_index=0)
    )

    before, started = _marks(unit), clock.now
    assert wrapped(1) + [1] == [1, 1]
    clock.now += 0.00005  # the statement's own work
    price = statement_price(clock.now - started, _since(before, unit))

    (event,) = unit.drain()
    assert event["stored"] and event["execution_time"] == pytest.approx(0.2)
    assert price.cost >= event["execution_time"], price
    assert price.cost == pytest.approx(0.2 + 0.00005)
    # Its own work is the + [1], and cash's time around the call is tax.
    assert price.store_cost == pytest.approx(0.00005 + unit.cached_restore_s)
    assert price.tax == pytest.approx(0.001), "the read recorded in the lookup is counted once"


def test_a_call_under_the_floor_is_the_statements_work(call_unit_harness, clock, record_reads):
    """A call the cache keeps in RAM only is left in what storing the
    statement saves, all of it."""

    def quick(x):
        clock.advance(0.02)
        record_reads(0.0003)
        return [x]

    unit = call_unit_harness(lineage={"a": "h"}, user_ns={"a": 1, "quick": quick})
    _costly_lookup(unit, clock, record_reads, seconds=0.001, reading=0.0004)
    wrapped = unit.wrap(quick, CallSite(source="quick(a)", free_names=frozenset({"quick", "a"}), occurrence_index=0))

    before, started = _marks(unit), clock.now
    wrapped(1)
    price = statement_price(clock.now - started, _since(before, unit))

    (event,) = unit.drain()
    assert event["execution_time"] == pytest.approx(0.02)
    assert unit.cached_compute_s == 0.0, "the call is under the persistence floor"
    assert price.store_cost >= event["execution_time"], price


def test_a_call_inside_a_call_is_part_of_its_time(call_unit_harness, clock, record_reads):
    """Keying and looking up an inner call is in the outer call's measured
    time, which the outer call's entry records: it is not taken off again."""

    def inner(v):
        clock.advance(0.2)
        return [v]

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "inner": inner})
    _costly_lookup(unit, clock, record_reads, seconds=0.001, reading=0.0004)
    wrapped_inner = unit.wrap(
        inner, CallSite(source="inner(x)", free_names=frozenset({"inner", "x"}), occurrence_index=0)
    )

    def outer(v):
        clock.advance(0.2)
        return wrapped_inner(v) + [0]

    wrapped = unit.wrap(outer, CallSite(source="outer(x)", free_names=frozenset({"outer", "x"}), occurrence_index=0))
    before, started = _marks(unit), clock.now
    wrapped(1)
    price = statement_price(clock.now - started, _since(before, unit))

    outer_s = next(e["execution_time"] for e in unit.drain() if e["call_source"] == "outer(x)")
    assert outer_s == pytest.approx(0.401), "the inner call's lookup is in the outer call's time"
    assert unit.computed_s - before.computed == pytest.approx(outer_s)
    assert price.cost == pytest.approx(outer_s)


def test_a_served_call_is_credited_what_it_stood_for(call_unit_harness, clock, record_reads):
    def shifted(x):
        clock.advance(0.2)
        return [x]

    unit = call_unit_harness(lineage={"a": "h"}, user_ns={"a": 1, "shifted": shifted})
    wrapped = unit.wrap(
        shifted, CallSite(source="shifted(a)", free_names=frozenset({"shifted", "a"}), occurrence_index=0)
    )
    wrapped(1)
    _costly_lookup(unit, clock, record_reads, seconds=0.001, reading=0.0004)

    before, started = _marks(unit), clock.now
    wrapped(1)
    price = statement_price(clock.now - started, _since(before, unit))

    assert unit.drain()[-1]["cache_hit"]
    assert price.cost == pytest.approx(0.2)
    assert price.tax == pytest.approx(0.001), "the read recorded in the lookup is counted once"


@pytest.mark.parametrize(
    ("wall", "spent", "cost", "store_cost", "tax"),
    [
        # Nothing routed: the wall time less the tracker's seconds.
        (1.0, CashMarks(0.25, None, 0.0, 0.0, 0.0, 0.0, 0), 0.75, 0.75, 0.25),
        # A 0.2 s call with 3 ms of overhead around it, of which 0.4 ms was
        # recording a read, and 0.3 ms recorded inside the call.
        (0.2031, CashMarks(0.0007, None, 0.003, 0.0, 0.2, 0.0, 0, 0.2, 0.0007), 0.2001, 0.0001, 0.003),
        # A tax measured over the wall time never makes the call cheaper.
        (0.201, CashMarks(0.0, None, 0.005, 0.0, 0.2, 0.0, 0, 0.2, 0.0), 0.2, 0.0, 0.005),
        # A hit: what it stood in for, plus the statement's own work.
        (0.0021, CashMarks(0.0, None, 0.002, 0.5, 0.5, 0.01, 0), 0.5001, 0.0101, 0.002),
    ],
)
def test_the_price_of_a_statement(wall, spent, cost, store_cost, tax):
    price = statement_price(wall, spent)
    assert price.cost == pytest.approx(cost)
    assert price.store_cost == pytest.approx(store_cost)
    assert price.tax == pytest.approx(tax)
