"""A lambda or a closure passed to a cached function is cached, keyed by its code
and by what it captured.

``apply(lambda d: d * 2, xs)`` ran uncached with KEY-UNHASHABLE-ARG: pickle
cannot write a function it does not find by name. It is keyed as a function of
the user's passed by name is (its code, the globals that code reads), plus the
values its closure captured, as a helper's are: ``make(3)`` and ``make(4)``
share one code and key apart.
"""

from __future__ import annotations

import textwrap
import warnings

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from tests._scripts import run_python

pytestmark = pytest.mark.core

K = 5


def make(k):
    def g(x):
        return x * k

    return g


def make_lookup(table):
    def g(x):
        return table[x]

    return g


class Session:
    def __init__(self, n):
        self.n = n

    def __reduce__(self):
        raise TypeError("a live session cannot be pickled")


@pytest.fixture
def apply(disk_cash):
    @disk_cash.cache
    def apply(f, xs):
        return [f(x) for x in xs]

    return apply


def _runs(fn):
    return fn.cache_info()["misses"]


def _quiet(fn, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args)
    codes = [str(w.message) for w in rec if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert not codes, codes
    return out


def test_a_lambda_hits(apply):
    assert _quiet(apply, lambda x: x * 2, [1, 2]) == [2, 4]
    assert _quiet(apply, lambda x: x * 2, [1, 2]) == [2, 4]
    assert _runs(apply) == 1


def test_another_lambda_misses(apply):
    assert apply(lambda x: x * 2, [1, 2]) == [2, 4]
    assert apply(lambda x: x * 3, [1, 2]) == [3, 6]
    assert _runs(apply) == 2


def test_a_closure_keys_by_what_it_captured(apply):
    assert _quiet(apply, make(3), [1]) == [3]
    assert _quiet(apply, make(3), [1]) == [3]
    assert apply(make(4), [1]) == [4]
    assert _runs(apply) == 2


def test_a_captured_dict_changed_in_place_misses(apply):
    table = {1: "a"}
    assert apply(make_lookup(table), [1]) == ["a"]
    table[1] = "b"
    assert apply(make_lookup(table), [1]) == ["b"]


def test_a_global_the_lambda_reads_is_in_the_key(apply):
    global K
    assert apply(lambda x: x + K, [1]) == [6]
    K = 7
    try:
        assert apply(lambda x: x + K, [1]) == [8]
    finally:
        K = 5


def test_a_lambda_default_is_in_the_key(apply):
    assert apply(lambda x, k=2: x * k, [1]) == [2]
    assert apply(lambda x, k=3: x * k, [1]) == [3]


def test_a_lambda_inside_a_list_is_keyed(disk_cash):
    @disk_cash.cache
    def apply_all(fs, x):
        return [f(x) for f in fs]

    assert _quiet(apply_all, [lambda v: v + 1, make(2)], 3) == [4, 6]
    assert _quiet(apply_all, [lambda v: v + 1, make(2)], 3) == [4, 6]
    assert apply_all([lambda v: v + 1, make(5)], 3) == [4, 15]
    assert apply_all.cache_info()["hits"] == 1


def test_a_closure_capturing_what_cannot_be_hashed_still_runs_uncached(apply):
    session = Session(1)

    def locked(x):
        return x * session.n

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert apply(locked, [1]) == [1]
        assert apply(locked, [1]) == [1]
    assert apply.cache_info()["hits"] == 0
    assert any(issubclass(w.category, CashCacheIneffectiveWarning) for w in rec)


def test_a_lambda_argument_hits_in_a_new_process(tmp_path):
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import cash

            def make(k):
                return lambda x: x * k

            @cash.cache
            def apply(f, xs):
                return [f(x) for x in xs]

            apply(lambda x: x + 1, [1])
            apply(make(3), [1])
            info = apply.cache_info()
            print(info["hits"], info["misses"])
            """
        ),
        encoding="utf-8",
    )
    assert run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()[-1] == "0 2"
    assert run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()[-1] == "2 0"
