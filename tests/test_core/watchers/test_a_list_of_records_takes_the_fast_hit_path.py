"""A list of records is served from the fast hit path while it is unchanged.

A cached function called again and again with one big list of small objects
built its whole key on every hit: the list is no plain data, so it was
walked into the key as it is, and the fast hit path had nothing to key the
call by. 200,000 records took ~250 ms a hit for a body of 7 ms. The
decorator now gives such a list a token of its own once the watchers watch
it (`value_watch.tag`); the fast path keys the call by that token for as
long as the watchers prove the list unchanged, and any change to it, its
items or their class sends the next call down the ordinary path again.
"""

from __future__ import annotations

import time

import pytest

from cash import Cash, value_watch
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Tagged:
    def __init__(self, x):
        self.x = x


def total(points):
    RUNS["n"] += 1
    return sum(p.x + p.y for p in points)


def total_tagged(items):
    RUNS["n"] += 1
    return sum(t.x for t in items)


@pytest.fixture
def c(tmp_path, watching):
    RUNS["n"] = 0
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _points(n=500):
    return [Point(i, i * 0.5) for i in range(n)]


def served_fast(f, *args):
    before_hits, before_runs = hit_memo.status()["fast_hits"], RUNS["n"]
    result = f(*args)
    return result, hit_memo.status()["fast_hits"] > before_hits and RUNS["n"] == before_runs


def warm(f, *args):
    """Call *f* until the memo serves it (see the remembered-hit tests)."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result, fast = served_fast(f, *args)
        if fast:
            return result
    raise AssertionError(f"{f.__name__} was never served from the memo")


def _expected(points):
    return sum(p.x + p.y for p in points)


def test_an_unchanged_list_of_records_is_served_fast(c, watching):
    f = c.cache(total)
    watching["points"] = points = _points()
    assert warm(f, points) == _expected(points)
    assert served_fast(f, points) == (_expected(points), True)
    assert RUNS["n"] == 1


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda ps: setattr(ps[3], "x", 1000), id="an item's attribute"),
        pytest.param(lambda ps: ps.__setitem__(3, Point(1000, 0)), id="an item replaced"),
        pytest.param(lambda ps: ps.append(Point(7, 7)), id="an item added"),
        pytest.param(lambda ps: ps.pop(), id="an item removed"),
        pytest.param(lambda ps: setattr(ps[3], "z", 1), id="an attribute added"),
        pytest.param(lambda ps: ps.reverse(), id="the order"),
    ],
)
def test_a_change_to_the_list_or_its_items_is_seen(c, watching, change):
    f = c.cache(total)
    watching["points"] = points = _points()
    warm(f, points)
    change(points)
    result, fast = served_fast(f, points)
    assert not fast
    assert result == _expected(points)
    assert warm(f, points) == _expected(points)


def test_an_equal_copy_answers_the_same(c, watching):
    f = c.cache(total)
    watching["points"] = points = _points()
    warm(f, points)
    runs = RUNS["n"]
    watching["copy"] = copy = _points()
    assert f(copy) == _expected(points)
    assert RUNS["n"] == runs, "an equal list is the same key"
    copy[0].x = 99
    assert f(copy) == _expected(copy)


def test_a_rebound_name_with_a_new_list_is_keyed_by_it(c, watching):
    f = c.cache(total)
    watching["points"] = points = _points()
    warm(f, points)
    watching["points"] = other = _points(400)
    assert f(other) == _expected(other)


def test_a_registered_hasher_keeps_the_list_on_the_ordinary_path(c, watching):
    """The walk ran the hasher, whose reads no watcher sees."""
    scale = {"by": 1}
    c.register_hasher(Tagged, lambda t: str(t.x * scale["by"]))
    f = c.cache(total_tagged)
    watching["items"] = items = [Tagged(i) for i in range(500)]
    for _ in range(6):
        result, fast = served_fast(f, items)
        assert result == sum(range(500))
        assert not fast
    assert value_watch._cw.info(items) is None or 2 not in value_watch._cw.info(items)["kinds"]


def test_every_fast_hit_builds_the_key_the_ordinary_path_builds(tmp_path, watching, monkeypatch):
    """``CASH_HITPATH_VERIFY=1``: each fast hit builds the key again and
    raises `HitMemoMismatch` when it differs."""
    monkeypatch.setattr(hit_memo, "VERIFY", True)
    RUNS["n"] = 0
    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(total)
    watching["points"] = points = _points()
    verified = hit_memo.VERIFIED[0]
    assert warm(f, points) == _expected(points)
    points[0].y = 5
    assert warm(f, points) == _expected(points)
    assert hit_memo.VERIFIED[0] >= verified + 2
