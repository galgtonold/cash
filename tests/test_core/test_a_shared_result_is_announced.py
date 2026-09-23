"""A result that shares state with something the caller holds says so.

Found while attacking the decorator before round 26: on the computing run these
behave like plain Python, and on a HIT they quietly stop:

* ``return base[lo:hi]`` -- the numpy view stops sharing memory with ``base``,
* ``return Wrapper(rows)`` -- the result no longer holds the caller's list,
* ``return CONFIG`` -- the caller's write no longer reaches the module global.

A hit hands back a value rebuilt from the stored bytes, so this is what caching
such a result means. Cash cannot tell whether the caller relies on the sharing,
so it says what will differ and lets the user waive it (``assume_safe=True``).
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")

from cash import Cash
from cash.backends import InMemoryBackend

CONFIG: dict = {}


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def _warnings_of(fn, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(*args)
    return [str(w.message) for w in seen if "CACHE-RESULT-SHARED" in str(w.message)]


def test_a_view_of_an_argument_is_announced(cash):
    @cash.cache
    def window(base, lo, hi):
        return base[lo:hi]

    said = _warnings_of(window, np.arange(5), 1, 4)
    assert said, "no warning for a result sharing memory with an argument"
    assert "base" in said[0], said[0]


def test_a_result_holding_an_argument_is_announced(cash):
    @cash.cache
    def wrap(rows, tag):
        return {"data": rows, "tag": tag}

    assert _warnings_of(wrap, [1, 2, 3], "t")


def test_a_returned_module_global_is_announced(cash):
    @cash.cache
    def config(n):
        return CONFIG

    assert _warnings_of(config, 1)


def test_assume_safe_waives_it(cash):
    @cash.cache(assume_safe=True)
    def window(base, lo, hi):
        return base[lo:hi]

    assert not _warnings_of(window, np.arange(5), 1, 4)


def test_an_independent_result_says_nothing(cash):
    @cash.cache
    def copied(base, lo, hi):
        return base[lo:hi].copy()

    @cash.cache
    def rebuilt(rows, tag):
        return {"data": list(rows), "tag": tag}

    @cash.cache
    def snapshot(n):
        return dict(CONFIG)

    assert not _warnings_of(copied, np.arange(5), 1, 4)
    assert not _warnings_of(rebuilt, [1, 2, 3], "t")
    assert not _warnings_of(snapshot, 1)
