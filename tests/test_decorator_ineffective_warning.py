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
"""
from __future__ import annotations

import warnings

import pytest

from cash.core import Cash
from cash.effectiveness import EffectivenessLedger
from cash.exceptions import CashCacheIneffectiveWarning

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture
def frame():
    return pd.DataFrame({"a": np.arange(300_000, dtype="float64")})


def _cash(tmp_path, threshold=0.05):
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


def test_it_warns_when_the_key_costs_more_than_the_work(tmp_path, frame):
    cash = _cash(tmp_path)

    @cash.cache
    def summarise(d):
        return float(d["a"].sum())

    found = _run(summarise, frame)

    assert len(found) == 1, f"expected exactly one warning, got {len(found)}"
    text = str(found[0].message)
    assert "summarise" in text
    assert "register_hasher" in text, "must offer a fix that keeps caching"


def test_it_stays_quiet_when_the_work_dominates(tmp_path, frame):
    """The control that makes the warning mean something.

    Same large argument, same hashing cost -- only the amount of work
    differs. A warning here would be reporting argument size, not
    ineffectiveness.
    """
    cash = _cash(tmp_path)

    @cash.cache
    def expensive(d):
        col = d["a"].to_numpy()
        return float(sum(np.sin(col).sum() + np.cos(col).sum() for _ in range(40)))

    assert _run(expensive, frame) == [], "warned about a function worth caching"


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
    """A cheap argument has no measurable overhead, so nothing to report."""
    cash = _cash(tmp_path)

    @cash.cache
    def add(n):
        return n + 1

    assert _run(add, 7, calls=200) == []


def test_the_warning_does_not_repeat(tmp_path, frame):
    cash = _cash(tmp_path)

    @cash.cache
    def summarise(d):
        return float(d["a"].sum())

    assert len(_run(summarise, frame, calls=40)) == 1
