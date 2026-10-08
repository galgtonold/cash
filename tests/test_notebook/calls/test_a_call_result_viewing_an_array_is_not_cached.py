"""An intercepted call whose result is a numpy view of an array it was given
or of a global array -- or holds such views -- is not served from the call
cache.

A hit hands back a deserialised array that owns its memory, so ``w =
window(a, 2)`` followed by ``w[:] = 5`` no longer writes into ``a``
(bug-hunt-5 AL-03). A view is a different object from its base, so the
identity check alone did not see it; the result is now taken to hold the
arrays on its ``.base`` chain.
"""

from __future__ import annotations

import time

import pytest

import cash
from cash.notebook.call_interception import CallSite
from cash.notebook.shared_objects import holds_part_of
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

BIG = np.zeros(10)


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def window(arr, i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return arr[i : i + 3]


def win(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return BIG[i : i + 3]


def split(arr):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return np.split(arr, 2)


def doubled(arr):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return arr * 2


def grid():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return np.arange(10.0).reshape(2, 5)


def _hits(call_cache, fn, *args):
    call_cache.set_sites([CallSite(source=f"{fn.__name__}(x)", free_names=frozenset({fn.__name__}), occurrence_index=0)])
    cached = call_cache.resolve(fn)
    results = [cached(*args), cached(*args)]
    return results, [e["cache_hit"] for e in call_cache.drain_call_log()]


@pytest.mark.parametrize(
    ("fn", "args", "base"),
    [
        (window, (np.zeros(10), 2), lambda args: args[0]),
        (win, (2,), lambda args: BIG),
        (split, (np.zeros(10),), lambda args: args[0]),
    ],
    ids=["view-of-an-argument", "view-of-a-global", "list-of-views"],
)
def test_a_view_is_a_view_of_the_same_array_again(call_cache, fn, args, base):
    results, hits = _hits(call_cache, fn, *args)
    assert hits == [False, False], "a view was served as a detached copy"
    for result in results:
        views = result if isinstance(result, list) else [result]
        assert all(np.shares_memory(view, base(args)) for view in views)


@pytest.mark.parametrize("fn", [doubled, grid], ids=["new-array", "view-of-its-own-array"])
def test_an_array_of_its_own_is_still_cached(call_cache, fn):
    """Control: a new array, or a view of one only the result holds."""
    args = (np.ones(4),) if fn is doubled else ()
    results, hits = _hits(call_cache, fn, *args)
    assert hits == [False, True]
    assert np.array_equal(results[0], results[1])


def test_two_views_of_one_array_hold_part_of_each_other():
    a = np.zeros(10)
    assert holds_part_of(a[2:5], [a[5:]])
    assert holds_part_of([a[:2]], [{"x": a}])
    assert not holds_part_of(a.copy(), [a])
