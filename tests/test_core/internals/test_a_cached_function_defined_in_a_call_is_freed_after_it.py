"""A cached function defined inside another function goes when plain
Python would let it go.

``@cash.cache`` on a helper defined where it is used (inside a function, a
factory called per request) made a new function, its wrappers and cash's
state for it on every call of the outer function, and kept all of them
for the life of the process: the table that finds a wrapper's remembered
hits held the state, and the state held the function. A long-running
loop grew by 18 KB per call.
"""

from __future__ import annotations

import asyncio
import gc
import types
import weakref

import pytest

ROUNDS = 60


def _alive(code_name: str) -> int:
    gc.collect()
    # By exact type: ``isinstance`` asks an object for its class, and a
    # ``weakref.proxy`` another test left behind, its object gone, raises.
    return sum(1 for o in gc.get_objects() if type(o) is types.FunctionType and o.__code__.co_name == code_name)


def _grows(code_name: str, once) -> int:
    """How many more functions named *code_name* are alive after `ROUNDS`
    calls of *once* than after the first few."""
    for i in range(5):
        once(i)
    before = _alive(code_name)
    for i in range(5, 5 + ROUNDS):
        once(i)
    return _alive(code_name) - before


@pytest.fixture(params=["disk", "memory"])
def cash(request, disk_cash, cash_instance):
    return disk_cash if request.param == "disk" else cash_instance


def test_a_helper_defined_and_called_in_a_function(cash):
    def process(i):
        @cash.cache
        def score_inner(x):
            return x * 2

        return [score_inner(v) for v in (1, 2, 3)]

    assert process(0) == [2, 4, 6]
    assert _grows("score_inner", process) <= 1


def test_a_helper_closing_over_a_value_that_differs_per_call(cash):
    def process(i):
        scale = i + 1

        @cash.cache
        def scaled_inner(x):
            return x * scale

        return scaled_inner(3)

    assert process(4) == 15
    assert _grows("scaled_inner", process) <= 1


def test_a_helper_decorated_and_never_called(cash):
    def process(i):
        @cash.cache
        def unused_inner(x):
            return x

        return i

    assert _grows("unused_inner", process) <= 1


def test_a_helper_that_calls_a_module_level_cached_function(cash):
    @cash.cache
    def base(x):
        return x + 1

    def process(i):
        @cash.cache
        def calling_inner(x):
            return base(x) * 2

        return calling_inner(i % 3), calling_inner(i % 3)

    assert process(1) == (4, 4)
    assert _grows("calling_inner", process) <= 1
    assert base(1) == 2


def test_an_async_helper_defined_in_a_function(cash):
    def process(i):
        @cash.cache
        async def awaited_inner(x):
            return x * 3

        return asyncio.run(awaited_inner(2))

    assert process(0) == 6
    assert _grows("awaited_inner", process) <= 1


def test_the_same_functions_decorated_again_and_again(cash):
    def make(k):
        def made_inner(x):
            return x + k

        return made_inner

    plain = [make(k) for k in range(4)]

    def process(i):
        return [cash.cache(f)(i % 2) for f in plain]

    assert process(1) == [1, 2, 3, 4]
    # The four functions are the test's own; nothing else of that name appears.
    assert _grows("made_inner", process) == 0
    gc.collect()
    wrappers = [
        o for o in gc.get_objects() if type(o) is types.FunctionType and getattr(o, "__wrapped__", None) in plain
    ]
    assert len(wrappers) <= 3 * len(plain), f"{len(wrappers)} wrappers of 4 functions are alive"


def test_a_caller_still_finds_a_live_callees_remembered_hits(cash):
    """The control for the weak table: a module-level callee's state is
    found while its wrapper lives, so its miss still drops what a caller
    remembered."""
    calls = []

    @cash.cache
    def inner(x):
        calls.append(("inner", x))
        return x + 1

    @cash.cache
    def outer(x):
        calls.append(("outer", x))
        return inner(x) * 2

    for _ in range(6):
        assert outer(1) == 4
    gc.collect()
    for _ in range(6):
        assert outer(1) == 4
    assert calls.count(("outer", 1)) == 1
    inner.cache_clear()
    assert outer(1) == 4


def test_the_count_passes_over_a_dead_proxy_another_test_left():
    class Gone:
        pass

    gone = Gone()
    left = [weakref.proxy(gone)]  # in a list the collector lists
    del gone
    assert _alive("no_function_has_this_name") == 0
    assert left
