"""What a self-check finds (``CASH_VERIFY=1`` and the switches under it) is
raised to the caller of the cached function, not turned into "this call
runs uncached".

A key build that raises where cash did not expect it is reported as
KEY-BUILD-FAILED and the call runs without the cache: right for a user's
unhashable object, wrong for a check that found a memo disagreeing with the
slow path. As a warning, the finding passed every test whose result stayed
right, so a suite run with the checks on could be green over a broken memo.
"""

from __future__ import annotations

import warnings

import pytest

from cash import _verify, value_watch
from cash.analysis import fact_store, report_store, scan_store, text_store, tree_memo
from cash.decorator import class_memo, hit_memo, state_memo
from cash.decorator.runtime import KeyBuilder
from cash.exceptions import CashCacheIneffectiveWarning

pytestmark = pytest.mark.core

FINDINGS = [
    hit_memo.HitMemoMismatch,
    state_memo.StateMemoMismatch,
    class_memo.ClassMemoMismatch,
    tree_memo.TreeMemoMismatch,
    fact_store.StoredFactMismatch,
    scan_store.StoredScanMismatch,
    text_store.StoredTextMismatch,
    report_store.StoredReportMismatch,
    value_watch.WatchMismatch,
]


def _failing_build(monkeypatch, error):
    def build(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(KeyBuilder, "build", build)


@pytest.mark.parametrize("finding", FINDINGS, ids=lambda f: f.__name__)
def test_a_finding_raised_in_the_key_build_is_raised_to_the_caller(disk_cash, monkeypatch, finding):
    assert issubclass(finding, _verify.Mismatch) and issubclass(finding, AssertionError)

    @disk_cash.cache
    def double(x):
        return 2 * x

    _failing_build(monkeypatch, finding("the remembered answer differs"))
    with pytest.raises(finding, match="the remembered answer differs"):
        double(3)


def test_any_other_error_in_the_key_build_still_runs_the_call_uncached(disk_cash, monkeypatch):
    @disk_cash.cache
    def double(x):
        return 2 * x

    _failing_build(monkeypatch, RuntimeError("an object that cannot be read"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert double(3) == 6
    assert any(
        issubclass(w.category, CashCacheIneffectiveWarning) and "KEY-BUILD-FAILED" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]
