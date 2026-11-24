"""A helper's globals fold remembered in one kind of key build is served
only to a build that keeps the same records (`helper_fold_memo`).

A real call's key build keeps a drift-guard watch (`CAPTURE_WATCH`) and a
state ledger (`STATE_LEDGER`); ``explain()`` keeps the ledger only, and the
key ``CASH_HITPATH_VERIFY=1`` builds behind a fast hit keeps neither. A
fold recorded in a build that kept no watch left an empty watch in the
memo, so a real call that reused it watched nothing of what the helper
read; and a fold recorded by a real call, folded again behind a fast hit,
came back with no watch or ledger, which verify mode took for a different
fold (``HitMemoMismatch: a remembered globals fold of thai_words differs
from folding it again`` under PyThaiNLP's tokenizer tests). The key itself
was the same either way.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator import helper_fold_memo, hit_memo, state_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the helper fold memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

FILENAME = "words.txt"


def _read_name(x):
    return len(FILENAME) * x


def through_helper(x):
    return _read_name(x)


@pytest.fixture
def c(tmp_path, monkeypatch):
    # The helper's fold is asked for on every key build, not the segment's.
    monkeypatch.setattr(state_memo, "ENABLED", False)
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _spec(c):
    registry = c._runner._registry
    return registry.cached[next(n for n in registry.cached if n.endswith("through_helper"))]


def test_a_real_call_after_explain_still_watches_what_the_helper_read(c):
    f = c.cache(through_helper)
    assert f(2) == 18
    served = helper_fold_memo.STATS["served"]
    for _ in range(3):
        f.explain(2)  # the second fold of the helper is recorded here
    _built, watch, _ledger = c._runner._keys.resolve(_spec(c), (2,), {})
    assert helper_fold_memo.STATS["served"] > served, "the helper's fold was never served from the memo"
    assert "FILENAME" in watch, sorted(watch)


@pytest.mark.filterwarnings("error::cash.exceptions.CashCacheIneffectiveWarning")
def test_verify_mode_folds_a_fast_hits_helpers_again_without_a_mismatch(c, monkeypatch):
    monkeypatch.setattr(hit_memo, "VERIFY", True)
    f = c.cache(through_helper)
    verified = hit_memo.VERIFIED[0]
    served = helper_fold_memo.STATS["served"]
    for _ in range(6):
        assert f(2) == 18
    assert hit_memo.VERIFIED[0] > verified, "no fast hit was verified"
    assert helper_fold_memo.STATS["served"] > served, "the helper's fold was never served from the memo"
