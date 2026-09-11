"""A call that changes an argument in place is not stored.

Round 19 (r19s2): ``a -= a.mean()``, ``a *= 2``, ``np.clip(..., out=a)``,
``rng.shuffle(a)``, ``a[mask] = 0`` on an argument. The cold call changed the
caller's array; every warm call returned the stored result and left it as it
was, so everything downstream differed from an uncached run. The runtime
check saw the change, warned once, and stored anyway.

Decided 2026-09-11: such a call is not stored -- it runs every time, which is
what the code means -- and the warning names the argument. Only an object can
change this way: an int or str rebound inside the body is invisible to the
caller and must not trigger it.
"""
from __future__ import annotations

import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]
np = pytest.importorskip("numpy")


def _in_place_minus_mean(a):
    a -= a.mean()
    return 0


def _in_place_double(a):
    a *= 2
    return 0


def _clip_out(a):
    np.clip(a, -1, 1, out=a)
    return 0


def _shuffle(a):
    np.random.default_rng(0).shuffle(a)
    return 0


def _mask_assign(a):
    a[a > 1] = 1
    return 0


IDIOMS = [_in_place_minus_mean, _in_place_double, _clip_out, _shuffle, _mask_assign]


@pytest.mark.parametrize("body", IDIOMS, ids=lambda f: f.__name__.lstrip("_"))
def test_a_call_that_changes_an_array_argument_runs_every_time(tmp_path, body):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    def f(a):
        runs.append(1)
        return body(a)

    cached = c.cache(f)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            a = np.array([3.0, -2.0, 0.5, 4.0])
            expected = a.copy()
            body(expected)                        # what the uncached call does to it
            cached(a)
            assert np.array_equal(a, expected), "the caller's array was left unchanged"
    assert len(runs) == 3, "a result was stored and served without changing the argument"
    named = [w for w in rec if "'a'" in str(w.message) and "in place" in str(w.message)]
    assert named, "no warning named the argument"


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.append(99),
    lambda rows: rows.sort(),
], ids=["list-append", "list-sort"])
def test_a_list_argument_changed_by_a_library_call_is_not_stored(tmp_path, mutate):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def f(rows):
        runs.append(1)
        mutate(rows)            # through a callable: the source scan cannot see it
        return len(rows)

    for _ in range(2):
        rows = [3, 1, 2]
        f(rows)
        assert rows != [3, 1, 2]
    assert len(runs) == 2


def test_an_immutable_argument_rebound_in_the_body_still_caches(tmp_path):
    """Control: `n -= 1` rebinds a local; the caller's int cannot change."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def countdown(n, label):
        runs.append(1)
        total = 0
        while n > 0:
            total += n
            n -= 1
        label += "!"
        return total, label

    assert countdown(4, "x") == (10, "x!")
    assert countdown(4, "x") == (10, "x!")
    assert len(runs) == 1


def test_an_array_argument_that_is_only_read_still_caches(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def centred_sum(a):
        runs.append(1)
        return float((a - a.mean()).sum())

    a = np.array([1.0, 2.0, 3.0])
    centred_sum(a)
    centred_sum(a)
    assert len(runs) == 1


def test_assume_safe_is_the_opt_out(tmp_path):
    """An audited function (`assume_safe=True`) whose change to its argument is
    incidental still stores, as before."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache(assume_safe=True)
    def f(a):
        runs.append(1)
        a.sort()
        return float(a[0])

    f(np.array([3.0, 1.0]))
    f(np.array([3.0, 1.0]))
    assert len(runs) == 1
