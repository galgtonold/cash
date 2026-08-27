"""End to end: the decorator warns when caching costs more than it saves.

``tests/test_effectiveness_ledger.py`` pins the decision rule in isolation.
This pins the wiring -- that the numbers reaching that rule are the right
ones, which is where this feature was one silent mistake away from being
useless: ``CacheMetadata.execution_time`` is measured from the top of the
wrapper and so INCLUDES the cache-key hashing. Judging caching by that
number compares the overhead against itself, and the warning could never
fire. Hence ``body_seconds``, and hence this test.

The discriminating case is the last one: a large argument where the work
genuinely dominates. If the warning fired there it would really be saying
"your argument is big", which is useless advice.

Thresholds are injected rather than waited out. At the shipped 2s bar this
file would need ~30 seconds of real hashing to say anything.

The overhead/body RATIO is owned by this file rather than borrowed from
pandas. An earlier version passed a DataFrame to a function that summed a
column, which is the real-world shape -- and it went red on macos-3.13 with
"expected exactly one warning, got 0", because whether hashing a frame costs
more than summing it depends on the host. Nothing was wrong: on that machine
the pathology genuinely was not present, so warning would have been the bug.

Both sides are now plain Python loops, so the ratio is a property of this
file and the same everywhere. Realism about the pandas case lives where it
was measured, in ``cash/effectiveness.py``.
"""
from __future__ import annotations

import warnings

import pytest

from cash.core import Cash
from cash.effectiveness import CUMULATIVE_WASTE_SECONDS, EffectivenessLedger
from cash.exceptions import CashCacheIneffectiveWarning

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


class Payload:
    """An argument whose hashing cost this file controls."""

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = n


def _costly_hash(payload: "Payload") -> str:
    """Deterministic, and deliberately thousands of times the body below.

    Pure bytecode on both sides, so a faster machine scales both and the
    ratio -- the thing the warning actually judges -- does not move.
    """
    acc = 0
    for i in range(120_000):
        acc = (acc * 31 + i) & 0xFFFFFFFF
    return f"{payload.n}-{acc}"


@pytest.fixture
def frame():
    return pd.DataFrame({"a": np.arange(300_000, dtype="float64")})


# Injected low on purpose, and lower than it needs to be. Measured here, 12
# calls accumulate ~80ms of waste; at a 50ms bar that is 1.6x of headroom and
# a machine twice as fast fails. At 10ms it is 8x, which is the difference
# between a threshold and a coin flip. The ratio side has ~7400x and needs no
# such care -- it is the absolute side that moves with the host.
def _cash(tmp_path, threshold=0.01):
    cash = Cash(cache_dir=str(tmp_path), register_magic=False)
    cash._effectiveness = EffectivenessLedger(waste_threshold_seconds=threshold)
    return cash


def _run(cached, arg, calls=12):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(calls):
            cached(arg)
        return [w for w in caught
                if issubclass(w.category, CashCacheIneffectiveWarning)]


def test_it_warns_when_the_key_costs_more_than_the_work(tmp_path):
    cash = _cash(tmp_path)
    cash.register_hasher(Payload, _costly_hash)

    @cash.cache
    def trivial(payload):
        return payload.n + 1

    found = _run(trivial, Payload(7))

    assert len(found) == 1, f"expected exactly one warning, got {len(found)}"
    text = str(found[0].message)
    assert "trivial" in text
    assert "register_hasher" in text, "must offer a fix that keeps caching"


def test_it_stays_quiet_when_the_work_dominates(tmp_path):
    """The control that makes the warning mean something.

    Identical argument and identical hashing cost -- only the amount of
    work differs, and it is the same kind of loop, so the comparison holds
    on any machine. A warning here would be reporting "your argument is
    expensive to hash", which is not the same claim and not useful.
    """
    cash = _cash(tmp_path)
    cash.register_hasher(Payload, _costly_hash)

    @cash.cache
    def dominant(payload):
        acc = 0
        for i in range(120_000 * 8):        # ~8x the hasher, same bytecode
            acc = (acc * 31 + i) & 0xFFFFFFFF
        return acc + payload.n

    assert _run(dominant, Payload(7)) == [], "warned about a function worth caching"


def test_the_body_time_reaches_the_entry(tmp_path, frame):
    """``body_seconds`` must be stored, or a HIT cannot know what it saved.

    Without it every hit reports an unknown body time, the ledger ignores
    the call, and the warning silently never fires -- a failure that leaves
    no trace at all.
    """
    cash = _cash(tmp_path)

    @cash.cache
    def summarise(d):
        return float(d["a"].sum())

    summarise(frame)
    ledger = cash._effectiveness._ledgers
    assert ledger, "the first (miss) call recorded nothing"
    (led,) = ledger.values()
    assert led.body_samples, "no body time observed on the miss"
    assert all(t > 0 for t in led.body_samples)


def test_a_quiet_function_is_never_accounted_as_waste(tmp_path):
    """A cheap argument has no measurable overhead, so nothing to report.

    This one runs at the SHIPPED threshold rather than the injected one,
    because its claim is about real use: a trivial function must never trip
    the warning in a user's process. It also has to. 200 calls of `n + 1`
    accumulate ~5ms of overhead on an idle machine and more under xdist
    contention, which cleared the 10ms injected bar and turned this red --
    the injected threshold buys headroom for the loud tests by spending it
    here. Against the real 2s bar the margin is ~400x.
    """
    cash = _cash(tmp_path, threshold=CUMULATIVE_WASTE_SECONDS)

    @cash.cache
    def add(n):
        return n + 1

    assert _run(add, 7, calls=200) == []


def test_the_warning_does_not_repeat(tmp_path):
    """Same owned ratio as the positive case -- for the same reason.

    This asserted exactly one warning from a DataFrame-hashing function,
    which is host-dependent in precisely the way that turned
    test_it_warns_when_the_key_costs_more_than_the_work red on macos-3.13.
    It would have been next.
    """
    cash = _cash(tmp_path)
    cash.register_hasher(Payload, _costly_hash)

    @cash.cache
    def trivial(payload):
        return payload.n + 1

    assert len(_run(trivial, Payload(7), calls=40)) == 1
