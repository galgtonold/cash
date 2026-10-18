"""A hit whose entry carries a trace is still served from the fast path,
and only while what the trace names is unchanged.

On 3.12+ every stored entry carries a trace of the user code its call ran
(`cash.decorator.code_trace`), checked on each ordinary hit. The fast path
(`cash.decorator.hit_memo`) skips that check, so it watches what the trace
names instead (`hit_memo.scan_trace`): a function held in a table, run by the
call but found by no name in its code, keeps its fast hits, and an edit of
that function is a miss.
"""

from __future__ import annotations

import sys

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on Python 3.12+"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


def double(x):
    return x * 2


def triple(x):
    return x * 3


HANDLERS = {"scale": double}


def dispatch(x):
    RUNS["n"] += 1
    return HANDLERS["scale"](x)


@pytest.fixture(autouse=True)
def _put_back():
    code = double.__code__
    RUNS["n"] = 0
    yield
    double.__code__ = code
    HANDLERS["scale"] = double


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def served_fast(f, *args):
    before_hits, before_runs = hit_memo.status()["fast_hits"], RUNS["n"]
    result = f(*args)
    return result, hit_memo.status()["fast_hits"] > before_hits and RUNS["n"] == before_runs


def warm(f, *args):
    for _ in range(8):
        result, fast = served_fast(f, *args)
        if fast:
            return result
    raise AssertionError(f"{f.__name__}{args} was never served from the memo")


def test_a_traced_entry_is_served_from_the_fast_path(c):
    f = c.cache(dispatch)
    assert warm(f, 4) == 8
    for _ in range(100):
        assert served_fast(f, 4) == (8, True)


def test_editing_the_traced_function_recomputes(c):
    f = c.cache(dispatch)
    assert warm(f, 4) == 8
    runs = RUNS["n"]
    double.__code__ = triple.__code__
    assert f(4) == 12
    assert RUNS["n"] == runs + 1


def test_swapping_the_function_the_table_holds_recomputes(c):
    f = c.cache(dispatch)
    assert warm(f, 4) == 8
    runs = RUNS["n"]
    HANDLERS["scale"] = triple
    assert f(4) == 12
    assert RUNS["n"] == runs + 1
