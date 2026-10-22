"""Which calls read a memo's counters, which callables cannot reach one but
by code the trace sees, and which call sites may call something else next
time (`code_trace.reads_counters`, `harmless`, `callee_varies`)."""

from __future__ import annotations

import dis
import functools
import operator
import sys

import pytest

from cash.decorator.code_trace import (
    HANDS_A_CALLABLE,
    HANDS_A_COUNTER,
    callee_varies,
    harmless,
    reads_counters,
    what_is_handed,
)

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="what ran is recorded on Python 3.12+ only")

M = sys.monitoring.MISSING if sys.version_info >= (3, 12) else None


def _memo():
    # Made per test: a memo at the top of a test module is one of the
    # user's to every other test in the process.
    @functools.cache
    def memo(n):
        return n

    return memo


@pytest.mark.parametrize(
    "call",
    [
        lambda m: (m.cache_info, M),
        lambda m: (m.cache_clear, M),
        lambda m: (type(m).__dict__["cache_info"], m),
        lambda m: (map, m.cache_info),
        lambda m: (operator.methodcaller, "cache_info"),
        lambda m: (operator.methodcaller("cache_info"), m),
        lambda m: (operator.attrgetter("cache_clear"), m),
        lambda m: (functools.partial(m.cache_info), M),
        lambda m: (functools.partial(getattr, m, "cache_info"), M),
    ],
)
def test_a_counter_call(call):
    assert reads_counters(*call(_memo()))


@pytest.mark.parametrize(
    "call",
    [
        lambda m: (m, 3),
        lambda m: (len, [1]),
        lambda m: ([].append, 1),
        lambda m: (str.join, "-"),
        lambda m: (print, "info"),
    ],
)
def test_no_counter_call(call):
    assert not reads_counters(*call(_memo()))


def _py():
    return 1


@pytest.mark.parametrize("fn", [_py, len, abs, [].append, str.join, dict, int, None])
def test_harmless(fn):
    assert harmless(_memo() if fn is None else fn)


@pytest.mark.parametrize(
    "fn",
    [getattr, map, sorted, functools.reduce, operator.methodcaller("x"), functools.partial(len), None],
)
def test_not_harmless(fn):
    assert not harmless(_memo().cache_info if fn is None else fn)


def _sites(fn):
    calls = [i for i in dis.get_instructions(fn) if i.opname in ("CALL", "CALL_KW", "CALL_FUNCTION_EX")]
    return [callee_varies(fn.__code__, i.offset) for i in calls]


def _loop(gs):
    for g in gs:
        g()


def _arg(g):
    return g()


def _fixed(xs, obj):
    return len(xs), obj.method(), abs(1)


def _item(fs):
    return fs[0]()


def _result(f):
    return getattr(f, "x")()


def test_a_callee_bound_in_the_code_varies():
    assert _sites(_loop) == [True]
    assert _sites(_arg) == [True]
    assert _sites(_item) == [True]


def test_a_global_or_an_attribute_does_not():
    # ``obj.method()`` on a handed object is named by its attribute: the
    # watch lets it go once the method is found on the object's class.
    assert _sites(_fixed) == [False, "method", False]


def test_a_call_whose_result_is_called_varies():
    assert _sites(_result) == [False, True]


def _handed(fn, frame_locals):
    call = [i for i in dis.get_instructions(fn) if i.opname in ("CALL", "CALL_KW")][-1]

    class Frame:
        f_locals = frame_locals
        f_globals: dict = {}

    return what_is_handed(fn.__code__, call.offset, Frame)


def _lib(x, **kw):
    return functools.reduce(x, kw)


def _hand(a, key):
    return _lib(a, key=key)


@pytest.mark.parametrize(
    "a, key, handed",
    [
        # Data alone (``np.mean(a)``, ``os.path.basename(s)``): the site is let go.
        ("a/b", [1.0, 2.0], None),
        ([1, 2], float, None),
        ("x", None, None),
        # Something callable may be swapped for a counter by a later call.
        ([1], len, HANDS_A_CALLABLE),
        ([len], 1, HANDS_A_CALLABLE),
    ],
)
def test_what_a_library_call_is_handed(a, key, handed):
    assert _handed(_hand, {"a": a, "key": key}) is handed


def test_a_counter_handed_to_a_library_call():
    m = _memo()
    assert _handed(_hand, {"a": [m.cache_info], "key": 1}) is HANDS_A_COUNTER
    assert _handed(_hand, {"a": 1, "key": "cache_clear"}) is HANDS_A_COUNTER
