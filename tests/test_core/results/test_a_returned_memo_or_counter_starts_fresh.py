"""A result that keeps state between calls is not handed back from the cache.

A factory that returns a fresh memo (``functools.cache``) or a counter closure
gives the caller an object whose state grows as it is used. Plainly, every call
of the factory makes a new, empty one. A hit used to hand back the one the
first call made, so the second caller's memo already held the first caller's
entries and the counter carried on from where the first caller left it.

A plain namespace with no such state is still served from the cache.
"""

from __future__ import annotations

import functools
import warnings
from types import SimpleNamespace

import pytest

from cash import Cash


@pytest.fixture
def cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / "c"), register_magic=False)


def _double(x):
    return 2 * x


def test_a_returned_memo_starts_empty(cash):
    @cash.cache
    def tools(n):
        return SimpleNamespace(f=functools.cache(_double), n=n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = tools(1)
        first.f(3)
        first.f(4)
        second = tools(1)

    assert second.f.cache_info().currsize == 0


def test_a_returned_counter_starts_at_one(cash):
    @cash.cache
    def make_counter(start):
        count = start

        def bump():
            nonlocal count
            count += 1
            return count

        return bump

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = make_counter(0)
        first()
        first()
        second = make_counter(0)

    assert second() == 1


def test_a_plain_namespace_is_still_served(cash):
    runs = []

    @cash.cache
    def settings(n):
        runs.append(n)
        return SimpleNamespace(n=n, name="x")

    assert settings(1).n == 1
    assert settings(1).n == 1
    assert runs == [1]
