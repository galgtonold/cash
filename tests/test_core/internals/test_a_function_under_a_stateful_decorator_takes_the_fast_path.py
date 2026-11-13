"""A cached function whose own ``def`` carries a library-style decorator is
served from its memo.

``@cache(level=10)`` over a function, where ``cache`` is an object of a
library that hands the function back unchanged (or wraps it), ran once, when
the function was defined: the key of the function cash wraps never reads
what the decorator line names. The fast-path scan walked it anyway, met a
counter deep in the decorator's machinery that no watcher can see, and kept
every call on the ordinary path, about 300 us a hit for a function that
runs in 40.
"""

from __future__ import annotations

import itertools

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}
SCALE = 2


class _Maker:
    """Code generation machinery with a live counter, as a decorator
    library keeps one."""

    count = itertools.count()


class _CacheLevels:
    def __init__(self, level):
        self.level = level

    def __call__(self, level):
        def wrap(func):
            next(_Maker.count)
            return func

        return wrap


cache = _CacheLevels(10)


@cache(level=10)
def scaled(n):
    RUNS["n"] += 1
    return n * SCALE


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _served_fast(f, *args):
    for _ in range(4):
        result = f(*args)
    before, runs = hit_memo.status()["fast_hits"], RUNS["n"]
    assert f(*args) == result
    assert RUNS["n"] == runs, "not served"
    assert hit_memo.status()["fast_hits"] > before, "the memo never served it"
    return result


def test_a_function_under_a_stateful_decorator_is_served_from_the_memo(c):
    assert _served_fast(c.cache(scaled), 5) == 10


def test_a_global_its_body_reads_is_still_watched(c, monkeypatch):
    f = c.cache(scaled)
    assert _served_fast(f, 5) == 10
    monkeypatch.setitem(globals(), "SCALE", 3)
    assert f(5) == 15
