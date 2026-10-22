"""Which calls read a memo's counters, which callables cannot reach one but
by code the trace sees, and which call sites may call something else next
time (`code_trace.reads_counters`, `harmless`, `callee_varies`)."""

from __future__ import annotations

import dis
import functools
import operator
import sys
import weakref

import pytest

from cash.decorator import code_trace
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


class _Box:
    pass


_BOX = _Box()


def _each_later(*xs):
    out = None
    for x in xs:
        done = weakref.finalize(_BOX, x)  # one site: data first, then the counter
        if x is None:
            done.detach()
        else:
            out = done()
    return out


@pytest.mark.skipif(not code_trace.on(), reason="what ran is recorded on Python 3.12+ only")
def test_a_site_handing_data_first_and_a_counter_next_is_seen():
    # The library calls what it is handed where the trace does not see. The
    # site was let go once its first call handed data (None), so the counter
    # the second call handed went unseen and the statement was served.
    m = _memo()
    m(1)
    rec = code_trace.recorder_or_reason()
    unit = rec.begin("call")  # a notebook call, its memos about
    unit.counters_watched = True
    try:
        info = _each_later(None, m.cache_info)
    finally:
        rec.end(unit)
    assert info.currsize == 1
    assert unit.counters_read


def _library_calls(a):
    import os.path

    operator.neg(3)
    operator.neg(a)
    operator.neg(os.path.sep)
    operator.neg(-a + 1)
    operator.neg(f"{a}")
    operator.neg([1, (2, "x")])
    operator.neg(a.real)
    operator.neg(len(a))
    operator.neg(*a)
    return operator.neg(a or 1)


@pytest.mark.parametrize(
    "index, data",
    [
        (0, True),  # a literal
        (1, False),  # a name: data now, a counter next time
        (2, True),  # a library module's attribute
        (3, True),  # arithmetic
        (4, True),  # an f-string
        (5, True),  # a display of literals
        (6, False),  # an attribute of what a name holds
        (7, False),  # a call's result
        (8, False),  # a starred name
        (9, False),  # ``a or 1`` hands ``a`` on as it is
    ],
)
def test_a_site_is_let_go_only_when_its_text_hands_on_data(index, data):
    calls = [i for i in dis.get_instructions(_library_calls) if i.opname in ("CALL", "CALL_KW", "CALL_FUNCTION_EX")]
    calls = [c for c in calls if c.positions is not None and c.positions.col_offset in (4, 11)]

    class Frame:
        f_locals = {"a": 2}
        f_globals = {"operator": operator}

    import os

    Frame.f_locals["os"] = os
    assert code_trace.hands_data_only(_library_calls.__code__, calls[index].offset, Frame) is data


def test_an_attribute_of_a_users_module_is_not_data(tmp_path, monkeypatch):
    (tmp_path / "users_settings_mod.py").write_text("FN = None\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    mod = importlib.import_module("users_settings_mod")
    try:

        def uses():
            return operator.neg(users_settings_mod.FN)  # noqa: F821 - read from the frame below

        call = [i for i in dis.get_instructions(uses) if i.opname == "CALL"][-1]

        class Frame:
            f_locals: dict = {}
            f_globals = {"users_settings_mod": mod}

        assert code_trace.hands_data_only(uses.__code__, call.offset, Frame) is False
    finally:
        sys.modules.pop("users_settings_mod", None)
