"""A function that calls numpy is served from its memo too.

Two things kept every such function on the ordinary path, ~500 us a hit
against ~0.7 us: ``np.sum`` and ``np.mean`` are a numpy C wrapper
(``_ArrayFunctionDispatcher``) the scan refused, and the installed version
of a library the call reaches, a part of every such key, was taken for an
environment read. The wrapper's implementation is fixed when it is made, and
is watched as a function; the version is read once per process. Each test
first checks the memo serves, then moves something the key reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


SCALE = 2
REDUCE = np.mean


def scaled(n):
    RUNS["n"] += 1
    return float(np.sum(np.arange(n))) * SCALE


def reduced(n):
    RUNS["n"] += 1
    return float(REDUCE(np.arange(n)))


def total(n):
    RUNS["n"] += 1
    return float(np.sum(np.arange(n)))


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


def test_a_numpy_reader_is_served_from_the_memo(c):
    assert _served_fast(c.cache(total), 5) == 10.0


def test_a_global_it_reads_set_again_is_seen(c, monkeypatch):
    f = c.cache(scaled)
    assert _served_fast(f, 5) == 20.0
    monkeypatch.setitem(globals(), "SCALE", 3)
    assert f(5) == 30.0


def test_a_numpy_wrapper_held_in_a_global_is_watched_as_its_function(c, monkeypatch):
    """``REDUCE = np.mean``: the wrapper is scanned (its implementation),
    and the global holding it is watched."""
    f = c.cache(reduced)
    assert _served_fast(f, 5) == 2.0
    monkeypatch.setitem(globals(), "REDUCE", np.sum)
    assert f(5) == 10.0
