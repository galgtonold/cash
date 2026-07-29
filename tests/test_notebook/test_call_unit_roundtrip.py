"""A call unit stores and restores through the statement backend (CAS-243).

``CallCache.resolve`` used to delegate to ``@cash.cache`` -- the decorator,
which keys a call by pickling every argument. ``CallUnit`` is the replacement:
it keys through ``call_cache_key`` (the statement key builder, under the
``"call"`` namespace) and reads/writes ``Cash.backend`` directly, the same
``get``/``set`` calls the statement path itself uses. These tests exercise
that round-trip end to end, independent of ``CallCache``/AST rewriting.
"""
from __future__ import annotations

import time

from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import CallUnit


def _site(source="slow(x)", names=("slow", "x")):
    return CallSite(source=source, free_names=frozenset(names), occurrence_index=0)


def test_second_call_with_same_lineage_is_a_hit(call_unit_harness):
    calls = []

    def slow(v):
        calls.append(v)
        time.sleep(0.05)
        return v * 2

    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns={"x": 5, "slow": slow})
    wrapped = unit.wrap(slow, _site())

    assert wrapped(5) == 10
    assert wrapped(5) == 10
    assert calls == [5], "second call should have been served from cache"
    assert [e["cache_hit"] for e in unit.call_log] == [False, True]


def test_changed_argument_lineage_recomputes(call_unit_harness):
    calls = []

    def slow(v):
        calls.append(v)
        time.sleep(0.05)
        return v * 2

    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns={"x": 5, "slow": slow})
    assert unit.wrap(slow, _site())(5) == 10

    unit.set_lineage({"x": "hash-9"})
    assert unit.wrap(slow, _site())(9) == 18
    assert calls == [5, 9]


def test_a_cheap_call_is_never_stored(call_unit_harness):
    calls = []

    def cheap(v):
        calls.append(v)
        return v

    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns={"x": 5, "cheap": cheap})
    wrapped = unit.wrap(cheap, _site(source="cheap(x)", names=("cheap", "x")))

    wrapped(5)
    wrapped(5)
    assert calls == [5, 5], "below the cost floor, nothing should be cached"


def test_a_computed_argument_is_hashed_into_the_key(call_unit_harness):
    """The half of the key ``arg_digests`` exists for: a non-Name argument.

    ``site.computed_arg_positions`` marks position 0 as needing its live value
    hashed -- unlike the bare-Name tests above, two different values passed at
    the SAME call site must land in two different cache entries.
    """
    calls = []

    def slow(v):
        calls.append(v)
        time.sleep(0.05)
        return v * 2

    site = CallSite(
        source="slow(next(it))",
        free_names=frozenset({"slow", "it"}),
        occurrence_index=0,
        computed_arg_positions=(0,),
    )
    unit = call_unit_harness(lineage={}, user_ns={"slow": slow})
    wrapped = unit.wrap(slow, site)

    assert wrapped(5) == 10
    assert wrapped(9) == 18
    assert wrapped(5) == 10
    assert calls == [5, 9], "two distinct computed-argument values must not collapse to one key"


def test_a_mismatched_digest_count_runs_uncached(call_unit_harness):
    """``call_cache_key`` refuses (returns None) on a digest/position mismatch;
    the thunk must treat that as "run uncached", not crash or mis-key.
    """
    calls = []

    def slow(v):
        calls.append(v)
        time.sleep(0.05)
        return v * 2

    # Two computed positions declared, but wrap()'s _invoke will only ever see
    # one live argument -- `_arg_digests` silently drops the out-of-range
    # position, so `len(arg_digests)` (1) will never match
    # `len(computed_arg_positions)` (2), and `call_cache_key` returns None.
    site = CallSite(
        source="slow(a, b)",
        free_names=frozenset({"slow"}),
        occurrence_index=0,
        computed_arg_positions=(0, 1),
    )
    unit = call_unit_harness(lineage={}, user_ns={"slow": slow})
    wrapped = unit.wrap(slow, site)

    assert wrapped(5) == 10
    assert wrapped(5) == 10
    assert calls == [5, 5], "a refused key must never be silently treated as a hit"
    assert unit.call_log == [], "an uncached call must not be recorded as a cache event"
