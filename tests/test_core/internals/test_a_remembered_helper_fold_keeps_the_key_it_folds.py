"""A helper's globals fold served from its memo (`helper_fold_memo`) leaves
the key build as folding it again would, so the key does not move.

A helper that reads a class folds the class's data and notes the class as
folded for the rest of the key (`CLASSES_FOLDED`); another helper reading
it, or the class reached as code, then does not fold it a second time. A remembered fold that
left that note out had the class folded again by the next step, so the
same call keyed differently once the memo served it: it ran again,
reported as "a global it reads changed", in every process (nilearn's
``unmask``, whose helpers read and build the same image classes).
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator import helper_fold_memo, hit_memo, state_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the helper fold memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


class _zweights:
    """Named to sort after the helpers below: the first helper's fold meets
    it first, and the second finds it folded."""

    factor = 100

    def __init__(self):
        self.scale = 2


def _breads(x):
    return x * _zweights.factor


def _make():
    return _zweights()


class Box:
    def __init__(self, v):
        self.v = v


def through_class(box):
    RUNS["n"] += 1
    return _breads(box.v) + _make().scale


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.mark.parametrize("whole_segment", [True, False], ids=["state-memo", "helper-memo-only"])
def test_repeated_calls_keep_one_key(c, monkeypatch, whole_segment):
    if not whole_segment:
        # The helper's fold is asked for on every call, served once remembered.
        monkeypatch.setattr(state_memo, "ENABLED", False)
    f = c.cache(through_class)
    runs = RUNS["n"]
    keys = []
    for _ in range(6):
        assert f(Box(3)) == 302
        keys.append(f.explain(Box(3)).cache_key)
    assert len(set(keys)) == 1, keys
    assert RUNS["n"] == runs + 1, "a call with nothing changed ran again"


def test_the_fold_is_served_and_the_class_still_keyed(c, monkeypatch):
    """The positive control: the memo does serve the helper's fold, and an
    edit to the class's data still reaches the key."""
    monkeypatch.setattr(state_memo, "ENABLED", False)
    f = c.cache(through_class)
    for _ in range(4):
        f(Box(3))
    served = helper_fold_memo.STATS["served"]
    runs = RUNS["n"]
    assert f(Box(3)) == 302
    assert helper_fold_memo.STATS["served"] > served, "the helper's fold was not served from the memo"
    assert RUNS["n"] == runs
    monkeypatch.setattr(_zweights, "factor", 7)
    assert f(Box(3)) == 23
    assert RUNS["n"] == runs + 1
