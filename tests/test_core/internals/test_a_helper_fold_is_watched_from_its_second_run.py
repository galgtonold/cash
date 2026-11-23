"""A helper's globals fold is watched and recorded (`helper_fold_memo`)
from the second time the same step runs, and not while the cached
function's state memo captures the segment around it, which then serves
the next call. In a fresh process a library's functions run a few times
each, and a watch costs about as much as the fold it would save.

Only when the work is done moves: every edit still reaches the key.
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
SCALE = 3


def _helper(x):
    return x * SCALE


class Box:
    def __init__(self, v):
        self.v = v


def through_helper(box):
    RUNS["n"] += 1
    return _helper(box.v)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _counts():
    return dict(helper_fold_memo.STATS)


def test_the_first_fold_is_not_watched_the_second_is(c, monkeypatch):
    monkeypatch.setattr(state_memo, "ENABLED", False)
    f = c.cache(through_helper)
    before = _counts()
    assert f(Box(2)) == 6
    after_first = _counts()
    # (A call may fold a step more than once: only its first run is
    # left unwatched.)
    deferred = after_first["deferred"] - before["deferred"]
    assert deferred > 0
    assert after_first["recorded"] - before["recorded"] <= deferred, "a fold was watched on its first run"
    f(Box(2))
    assert _counts()["recorded"] - before["recorded"] == deferred, "a second fold was not watched"
    served = _counts()["served"]
    runs = RUNS["n"]
    assert f(Box(2)) == 6
    assert _counts()["served"] > served and RUNS["n"] == runs
    monkeypatch.setitem(globals(), "SCALE", 5)
    assert f(Box(2)) == 10
    assert RUNS["n"] == runs + 1


def test_a_segment_the_state_memo_captures_records_no_helper_fold(c, monkeypatch):
    f = c.cache(through_helper)
    before = _counts()
    reused = state_memo.STATS["reused"]
    for _ in range(5):
        assert f(Box(2)) == 6
    assert state_memo.STATS["reused"] > reused, "the state memo served nothing"
    assert _counts()["left_to_state_memo"] > before["left_to_state_memo"]
    assert _counts()["recorded"] == before["recorded"]
    runs = RUNS["n"]
    monkeypatch.setitem(globals(), "SCALE", 7)
    assert f(Box(2)) == 14
    assert RUNS["n"] == runs + 1
    assert f(Box(2)) == 14
    assert RUNS["n"] == runs + 1
