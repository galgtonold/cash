"""The ledger must fire on real waste and stay silent on everything else.

A warning that cries wolf gets filtered permanently, and then the one case
that mattered is filtered too. So the controls here matter more than the
positive case: most of these tests assert SILENCE, and each pins a distinct
way a naive implementation would produce a confident, wrong diagnosis.
"""
from __future__ import annotations

import pytest

from cash.effectiveness import MIN_OBSERVATIONS, EffectivenessLedger


def _drive(ledger, *, overhead, body, calls, hit=True, name="f"):
    """Record *calls* identical calls; return the first message emitted."""
    first = None
    for _ in range(calls):
        msg = ledger.record(name, overhead_seconds=overhead,
                            body_seconds=body, was_hit=hit)
        if msg and first is None:
            first = msg
    return first


def test_it_fires_on_the_case_that_motivated_it():
    """390ms of hashing to avoid 11ms of work, over enough calls to matter."""
    ledger = EffectivenessLedger()
    msg = _drive(ledger, overhead=0.390, body=0.011, calls=10, name="summarise")

    assert msg is not None, "the pathological case did not warn"
    assert "summarise" in msg
    assert "register_hasher" in msg, "the warning must name a remedy that keeps caching"


def test_it_stays_quiet_below_the_cumulative_threshold():
    """The anti-spam control, and the reason the threshold is in seconds.

    A 100x ratio on work nobody notices is not worth a warning. Same
    lopsided ratio as the test above -- only the accumulated total differs,
    which is exactly the discriminator being pinned.
    """
    ledger = EffectivenessLedger()
    msg = _drive(ledger, overhead=0.010, body=0.0001, calls=5, name="tiny")

    assert msg is None, f"warned over {5 * 0.010:.2f}s of total overhead: {msg}"


def test_a_function_with_an_expensive_tail_is_not_flagged():
    """The control that stops a confident, wrong diagnosis.

    Usually 5ms, occasionally 30s. Caching it is correct, and comparing
    against the MEAN body time would flag it anyway. The comparison is
    against the largest body time observed, so it stays quiet.
    """
    ledger = EffectivenessLedger()
    for i in range(30):
        body = 30.0 if i == 7 else 0.005
        msg = ledger.record("occasionally_slow", overhead_seconds=0.100,
                            body_seconds=body, was_hit=True)
        assert msg is None, f"flagged a function with a 30s tail on call {i}: {msg}"


def test_it_stays_quiet_when_caching_genuinely_wins():
    ledger = EffectivenessLedger()
    msg = _drive(ledger, overhead=0.002, body=5.0, calls=50, name="worth_it")
    assert msg is None, msg


def test_it_warns_once_and_then_shuts_up():
    ledger = EffectivenessLedger()
    name = "noisy"
    messages = [ledger.record(name, overhead_seconds=0.390,
                              body_seconds=0.011, was_hit=True)
                for _ in range(200)]
    assert len([m for m in messages if m]) == 1, "warned more than once"


def test_a_call_with_unknown_body_time_is_ignored_entirely():
    """Entries written before body time was recorded must not skew the verdict.

    Counting their overhead while crediting no saving would bias straight
    towards warning -- the wrong direction for a diagnostic.
    """
    ledger = EffectivenessLedger()
    for _ in range(50):
        msg = ledger.record("legacy", overhead_seconds=0.500,
                            body_seconds=None, was_hit=True)
        assert msg is None, msg


def test_a_miss_pays_overhead_without_earning_a_saving():
    """A miss ran the body, so caching returned nothing for that overhead.

    Asserted on the accounting rather than on whether a warning fires,
    because those are different questions -- see the test below.
    """
    hits, misses = EffectivenessLedger(), EffectivenessLedger()
    _drive(hits, overhead=0.050, body=0.060, calls=60, hit=True)
    _drive(misses, overhead=0.050, body=0.060, calls=60, hit=False)

    assert hits._ledgers["f"].saved_seconds == pytest.approx(0.060 * 60)
    assert misses._ledgers["f"].saved_seconds == 0.0
    assert misses._ledgers["f"].overhead_seconds == pytest.approx(0.050 * 60)


def test_a_function_that_simply_never_hits_is_out_of_scope():
    """Deliberately silent, and the distinction is worth keeping.

    60 misses at 50ms of overhead is 3s of pure waste -- past the
    threshold -- but the diagnosis is NOT "the key costs more than the
    work". Compute here is 60ms against 50ms of overhead, so if this
    function ever hit, caching would pay. The problem is that it never
    hits, which has a different cause (arguments that are always new) and
    a different remedy.

    Warning "cache is costing more than it saves" here would be a
    confident, wrong diagnosis. Never-hits may deserve its own notice one
    day; it is not this one.
    """
    ledger = EffectivenessLedger()
    msg = _drive(ledger, overhead=0.050, body=0.060, calls=60, hit=False)
    assert msg is None, msg


def test_no_verdict_before_there_is_a_distribution():
    """One observation is not a tail. Pinned so the threshold cannot be met
    by a single enormous call before anything is known about the function."""
    ledger = EffectivenessLedger()
    msg = ledger.record("once", overhead_seconds=99.0, body_seconds=0.001, was_hit=True)
    assert msg is None
    assert MIN_OBSERVATIONS > 1


def test_functions_are_accounted_separately():
    ledger = EffectivenessLedger()
    _drive(ledger, overhead=0.390, body=0.011, calls=10, name="bad")
    good = _drive(ledger, overhead=0.001, body=2.0, calls=10, name="good")
    assert good is None, good


@pytest.mark.parametrize("threshold", [0.5, 2.0, 10.0])
def test_the_threshold_is_the_knob_that_decides(threshold):
    """Waste just under the bar is silent; just over it speaks."""
    ledger = EffectivenessLedger(waste_threshold_seconds=threshold)
    per_call, calls = 0.100, 8
    quiet = _drive(ledger, overhead=per_call, body=0.001,
                   calls=int(threshold / per_call) - 1, name="under")
    assert quiet is None, quiet

    loud = _drive(EffectivenessLedger(waste_threshold_seconds=threshold),
                  overhead=per_call, body=0.001,
                  calls=int(threshold / per_call) + calls, name="over")
    assert loud is not None
