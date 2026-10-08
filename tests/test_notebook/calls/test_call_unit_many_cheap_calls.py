"""Many cheap calls to one site in one statement stop being cached.

A call in a comprehension is made once per element; caching each one costs a
key, a lookup, a store and a file tracker -- ~14 ms a call around a function
reading one small file, which made a real 5,030-file read go 8.4 s -> 71.6 s.
Once the calls show that caching them costs more than ``_OVERHEAD_FACTOR``
times the work -- after ``_GUARD_MIN_CALLS`` calls with no hit when that is
plain, past ``_GUARD_AFTER_CALLS`` in any case -- cheap ones are timed plain on
a few samples and, when the samples agree, the rest of the statement's calls to
the site run plain. Expensive calls are never timed plain, and a new statement
run starts over.

The guard's clock is a fake one here, moved only by the work and the lookup
by exact amounts. They slept once: on Windows, Python 3.10's ``time.sleep`` ends
on a system-timer tick, so a 1.5 ms call and a 2 ms lookup both waited ~2 ms
or ~15.6 ms, whichever the timer resolution was at the time, and the verdict
turned on the tick rather than on the costs the tests set out.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_unit as cu
from cash.notebook.call_entries import CallEntries
from cash.notebook.call_interception import CallSite

SITE = CallSite(
    source="work(v + 0)",
    free_names=frozenset({"work"}),
    occurrence_index=0,
    computed_arg_positions=(0,),
    local_arg_positions=(0,),
)
N = cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES + 40


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


@pytest.fixture
def slow_lookup(monkeypatch, clock):
    """Every lookup costs exactly 2 ms."""
    real = CallEntries.lookup
    lookups: list[str] = []

    def lookup(self, key):
        lookups.append(key)
        clock.spend(0.002)
        return real(self, key)

    monkeypatch.setattr(CallEntries, "lookup", lookup)
    return lookups


def test_many_cheap_calls_stop_being_cached(call_unit_harness, slow_lookup):
    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(lambda v: v * 2, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_MIN_CALLS, "cheap calls kept being looked up"


def test_a_new_statement_run_starts_over(call_unit_harness, slow_lookup):
    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(lambda v: v * 2, SITE)
    for i in range(N):
        wrapped(i)
    unit.begin_statement()
    slow_lookup.clear()

    wrapped(0)

    assert len(slow_lookup) == 1


def test_expensive_calls_are_never_run_plain_to_measure_them(call_unit_harness, monkeypatch, clock):
    """Timing a plain sample of a slow call would repeat the slow work; a
    call over the cheap bar is not sampled at all."""
    monkeypatch.setattr(cu, "_GUARD_CHEAP_BELOW_S", 0.001)
    ran: list[int] = []
    # Not `ran.append` in the body: a hit replays that write to its closure.
    note = ran.append

    def work(v):
        note(v)
        clock.spend(0.005)  # over the (lowered) cheap bar and the store floor
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


def test_calls_a_hit_could_not_beat_stop_being_cached(call_unit_harness, slow_lookup, clock):
    """``add_features(g)`` over 360 groups, ~5 ms a call,
    cost ~8 ms more a call to cache and still under 4x its work, so it kept
    being cached. Keying and looking it up alone cost as much as the call: a
    hit would never have been faster. Measured on the same samples, a site
    whose key and lookup cost at least the call runs plain."""

    def work(v):
        clock.spend(0.0015)  # under the 2 ms lookup; well under 4x the miss
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_MIN_CALLS, "calls no hit could beat kept being looked up"


def test_one_stalled_sample_does_not_keep_the_site_cached(call_unit_harness, slow_lookup, clock):
    """The test above failed on a macOS runner with 90 lookups for 50: the
    five samples were timed, and the site stayed cached. A stall inside one
    sample (the process descheduled, a garbage collection) is enough to lift
    the mean of five past the bar. The verdict reads their median: the
    samples here are 30, 1.5, 1.5, 1.5 and 1.5 ms, a mean of 7.2 ms that a
    2 ms lookup is under three quarters of, and a median of 1.5 ms that it
    is not."""
    ran: list[int] = []
    # Not `ran.append` in the body: a hit replays that write to its closure.
    note = ran.append

    def work(v):
        note(v)
        # The first call run plain to time it comes after the cached ones.
        stalled = len(ran) == cu._GUARD_MIN_CALLS + 1
        clock.spend(0.03 if stalled else 0.0015)
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_MIN_CALLS, "one slow sample kept the site cached"


def test_a_stall_in_the_first_calls_does_not_decide(call_unit_harness, slow_lookup, clock):
    """A call worth caching -- 10 ms of work behind a 2 ms lookup -- whose
    first call stalls for 300 ms. The mean of the first five calls through
    the cache is over four times their work; their median is not, and the
    site is not judged until ``_GUARD_AFTER_CALLS``, where the samples keep
    it cached. Those samples are still looked up (see
    ``test_the_calls_of_the_re_check_are_still_looked_up_and_stored``)."""
    ran: list[int] = []
    note = ran.append

    def work(v):
        note(v)
        clock.spend(0.31 if len(ran) == 1 else 0.01)
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == N, "a stall ran a call worth caching plain"


def test_an_early_verdict_to_cache_is_judged_again_later(call_unit_harness, slow_lookup, clock):
    """The first calls look plain (0.1 ms of work behind a 2 ms lookup, each
    stored with the cost floor off), but the plain samples taken then are
    slow: the site stays cached, and is timed again on fresh samples at
    ``_GUARD_AFTER_CALLS`` -- calls still looked up -- after which it runs
    plain."""
    ran: list[int] = []
    note = ran.append
    first_samples = range(cu._GUARD_MIN_CALLS + 1, cu._GUARD_MIN_CALLS + cu._PLAIN_SAMPLES + 1)

    def work(v):
        note(v)
        clock.spend(0.02 if len(ran) in first_samples else 0.0001)
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    unit._cash.config.call_cost_floor_seconds = 0.0
    wrapped = unit.wrap(work, SITE)

    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES
    # Its later calls can skip the wrapper (``CallCache.resolve``).
    assert SITE in unit.plain_sites


def test_every_call_is_counted_including_the_plain_ones(call_unit_harness, slow_lookup):
    """The badge read ``sub-call read_doc(p): 5220/5225`` for
    5,225 files, ``296/301``, ``495/500``: every count five short. The calls
    the guard ran plain -- its samples and the rest of the run -- were never
    logged, so they vanished from the denominator. They are counted now, and
    marked as run plain, in one record per site."""
    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(lambda v: v * 2, SITE)
    for i in range(N):
        wrapped(i)

    events = unit.drain()
    assert sum(e["calls"] for e in events) == N, f"{sum(e['calls'] for e in events)} of {N} calls logged"
    plain = [e for e in events if e.get("ran_plain")]
    assert sum(e["calls"] for e in plain) == N - cu._GUARD_MIN_CALLS
    assert len(plain) == 1, "the plain-run calls of one site are kept as one record"
    assert all(not e["cache_hit"] for e in plain)


def test_the_sub_call_line_says_how_many_ran_plain():
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    event = {
        "func_name": "m.work",
        "call_source": "work(v)",
        "occurrence_index": 0,
        "intercepted": True,
        "cache_key": None,
        "execution_time": 0.001,
        "time_saved": 0.0,
    }
    calls = [dict(event, cache_hit=True) for _ in range(3)] + [
        dict(event, cache_hit=False, ran_plain=True) for _ in range(5)
    ]
    out = render_text(
        build_interactive_badge(
            [
                {
                    "status": "COMPUTED",
                    "code": "out = [work(v) for v in xs]",
                    "execution_time": 0.1,
                    "decorator_calls": calls,
                }
            ]
        )
    )
    assert "sub-call work(v): 3/8 hit" in out, out
    assert "5 run plain" in out, out


def test_one_record_for_many_plain_calls_counts_as_all_of_them():
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    event = {
        "func_name": "m.work",
        "call_source": "work(v)",
        "occurrence_index": 0,
        "intercepted": True,
        "cache_key": None,
        "time_saved": 0.0,
        "cache_hit": False,
    }
    calls = [
        dict(event, execution_time=0.001, calls=1),
        dict(event, execution_time=0.4, calls=2000, ran_plain=True),
    ]
    out = render_text(
        build_interactive_badge(
            [
                {
                    "status": "COMPUTED",
                    "code": "out = [work(v) for v in xs]",
                    "execution_time": 0.5,
                    "decorator_calls": calls,
                }
            ]
        )
    )
    assert "2000 run plain" in out, out
    assert "0/2001 hit" in out, out


def test_a_row_whose_calls_were_served_says_what_they_saved():
    """``EXECUTED: results[name] = evaluate(...)  (0.03s)``
    for a model fit served from the cache -- the word EXECUTED and 0.03 s, and
    only the footer's "3/4 cached" said otherwise. The statement did run; the
    row says what its cached calls saved."""
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    hit = {
        "func_name": "m.evaluate",
        "call_source": "evaluate(name)",
        "occurrence_index": 0,
        "intercepted": True,
        "cache_key": "call:ab",
        "cache_hit": True,
        "execution_time": 0.0,
        "time_saved": 3.02,
    }
    out = render_text(
        build_interactive_badge(
            [
                {
                    "status": "COMPUTED",
                    "code": "results[name] = evaluate(name)",
                    "execution_time": 0.03,
                    "decorator_calls": [hit],
                }
            ]
        )
    )
    row = next(line for line in out.splitlines() if "results[name]" in line)
    assert "saved 3.02s" in row, out


def test_calls_the_cache_cannot_keep_run_plain(call_unit_harness, slow_lookup, clock):
    """1 ms calls behind a 0.2 ms key and lookup: worth caching by cost, but
    under the cost floor, so no miss is stored and nothing can be served.
    1,000 of them paid ~0.4 ms each for keys no call could hit, +0.5 s. They
    run plain after ``_GUARD_MIN_CALLS``."""
    lookups = []
    real = CallEntries.lookup

    def cheap_lookup(self, key):
        lookups.append(key)
        clock.spend(0.0002)
        return real(self, key)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CallEntries, "lookup", cheap_lookup)

        def work(v):
            clock.spend(0.001)
            return v * 2

        unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
        wrapped = unit.wrap(work, SITE)
        assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(lookups) == cu._GUARD_MIN_CALLS, len(lookups)


def test_calls_the_cache_keeps_stay_cached(call_unit_harness, slow_lookup, clock):
    """The same, over the cost floor: each miss is stored, a later run can be
    served, and the site stays cached."""

    def work(v):
        clock.spend(0.01)
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)
    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]

    assert len(slow_lookup) == N


def test_the_calls_of_the_re_check_are_still_looked_up_and_stored(call_unit_harness, slow_lookup, clock):
    """The re-check at ``_GUARD_AFTER_CALLS`` used to run its sample calls
    plain -- neither looked up nor stored, whatever they cost -- so the 51st to
    55th calls of ``[work(i) for i in range(60)]``, 0.6 s each, ran again on
    every re-run while the other calls were served. They go through the cache
    now, and a second run of the statement serves every call."""
    slow = range(cu._GUARD_AFTER_CALLS, cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES)

    def work(v):
        clock.spend(0.6 if v in slow else 0.01)
        return v * 2

    unit = call_unit_harness(lineage={"work": "w"}, user_ns={})
    wrapped = unit.wrap(work, SITE)
    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]
    unit.drain()

    unit.begin_statement()
    assert [wrapped(i) for i in range(N)] == [i * 2 for i in range(N)]
    events = unit.drain()
    served = sum(e["calls"] for e in events if e["cache_hit"])
    computed = sorted({e["call_source"] for e in events if not e["cache_hit"]})
    assert served == N, f"{N - served} calls ran again on a re-run ({computed})"
