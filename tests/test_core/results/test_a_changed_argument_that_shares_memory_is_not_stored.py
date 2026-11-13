"""A call that changes an argument in place is not stored when that
argument shares memory with another argument or with a value the function
reads outside its arguments.

The key hashes each argument by its content alone, so ``f(a, a[::-1])``
keys like ``f(a, b)`` with ``b`` an equal, independent array, though the
body sees ``b`` change as it writes ``a``. A stored result then came back
for the independent pair, and a hit writing ``a`` back left the view as no
plain run does. The same holds for a global the function reads that is,
or shares memory with, the argument it changes.
"""

from __future__ import annotations

import functools
import importlib
import sys
import warnings

import numpy as np
import pytest

needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it a call that changes its "
    "argument in place runs every time, with a warning",
)


def _view_sum(a, b):
    a += 1
    return float(b.sum())


def _sorted_head(a, b):
    a.sort()
    return float(b[0])


GLOBAL = np.array([3.0, 1.0, 2.0])


def _reads_global(a):
    a.sort()
    return float(GLOBAL[0])


def _quiet(f, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args)


def test_arguments_sharing_memory_give_what_plain_runs_give(disk_cash):
    """An independent pair first, then the shared ones; each call checked
    against the plain function on fresh data."""
    cases = [
        (_view_sum, lambda: (lambda a: (a, a[::-1]))(np.arange(3.0))),
        (_view_sum, lambda: (lambda a: (a, a[::-1].copy()))(np.arange(3.0))),
        (_sorted_head, lambda: (lambda a: (a, a[:2]))(np.array([3.0, 1.0, 2.0]))),
        (_sorted_head, lambda: (lambda a: (a, a[:2].copy()))(np.array([3.0, 1.0, 2.0]))),
    ]
    cached = {body: disk_cash.cache(body) for body in (_view_sum, _sorted_head)}
    for body, make in cases * 2:
        plain_args = make()
        plain = body(*plain_args)
        args = make()
        got = _quiet(cached[body], *args)
        assert got == plain, (body.__name__, got, plain)
        assert [x.tolist() for x in args] == [x.tolist() for x in plain_args]


def test_a_global_that_is_the_changed_argument_is_read_as_a_plain_run_reads_it(disk_cash):
    """A copy of the global first, stored; then the global itself, which
    keys alike."""
    cached = disk_cash.cache(_reads_global)
    seen, plain = [], []
    for use_global in (False, True, True):
        for out, f in ((plain, _reads_global), (seen, lambda a: _quiet(cached, a))):
            GLOBAL[:] = [3.0, 1.0, 2.0]
            out.append(f(GLOBAL if use_global else GLOBAL.copy()))
    GLOBAL[:] = [3.0, 1.0, 2.0]
    assert plain == [3.0, 1.0, 1.0]
    assert seen == plain


def test_the_warning_says_the_call_runs_every_time(disk_cash):
    """Not the line saying a hit writes the argument back: no hit does."""
    cached = disk_cash.cache(_sorted_head)
    a = np.array([3.0, 1.0, 2.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(a, a[:2])
    text = " ".join(str(w.message) for w in caught)
    assert "runs every time" in text, text
    assert "a hit writes it back" not in text, text


# -- held deeper, or reached through other code --------------------------------


def _first_held(a, held):
    a[0] = 99.0
    return float(np.asarray(_dig(held)).sum())


def _dig(held):
    while not isinstance(held, np.ndarray):
        held = held["v"] if isinstance(held, dict) else held.v if hasattr(held, "v") else held[0]
    return held


class _Holder:
    def __init__(self, v):
        self.v = v


HOLDINGS = {
    "a dict value": lambda view: {"v": view},
    "a tuple element": lambda view: (view,),
    "two lists down": lambda view: [[view]],
    "an attribute": _Holder,
}


@pytest.mark.parametrize("how", list(HOLDINGS))
def test_a_view_held_inside_another_argument_is_seen(disk_cash, how):
    """Stored from an independent pair, then handed a view of the changed
    array inside the other argument: a plain run sums the changed view."""
    wrap = HOLDINGS[how]
    cached = disk_cash.cache(_first_held)
    _quiet(cached, np.arange(4.0), wrap(np.arange(2.0)))
    for _ in range(2):
        a = np.arange(4.0)
        plain_a = np.arange(4.0)
        plain = _first_held(plain_a, wrap(plain_a[:2]))
        assert _quiet(cached, a, wrap(a[:2])) == plain == 100.0


G_STATIC = np.arange(4.0)
G_METHOD = np.arange(4.0)
G_LRU = np.arange(4.0)
G_BY_NAME = np.arange(4.0)


class _Reader:
    @staticmethod
    def static():
        return float(G_STATIC.sum())

    def method(self):
        return float(G_METHOD.sum())


_READER = _Reader()


@functools.lru_cache
def _lru():
    return G_LRU


def _via_static(a):
    a[0] = 99.0
    return _Reader.static()


def _via_method(a):
    a[0] = 99.0
    return _READER.method()


def _via_lru(a):
    a[0] = 99.0
    return float(_lru().sum())


def _via_globals(a):
    a[0] = 99.0
    return float(globals()["G_BY_NAME"].sum())


INDIRECT = {
    "a staticmethod": (_via_static, G_STATIC),
    "a method": (_via_method, G_METHOD),
    "an lru_cache helper": (_via_lru, G_LRU),
    "globals()": (_via_globals, G_BY_NAME),
}


def _stored_then_global(cached, body, global_array):
    _quiet(cached, np.arange(4.0))
    global_array[:] = np.arange(4.0)
    plain = body(global_array)
    global_array[:] = np.arange(4.0)
    got = _quiet(cached, global_array)
    global_array[:] = np.arange(4.0)
    return got, plain


@pytest.mark.parametrize("how", list(INDIRECT))
def test_a_global_read_through_other_code_is_seen(disk_cash, how):
    body, global_array = INDIRECT[how]
    got, plain = _stored_then_global(disk_cash.cache(body), body, global_array)
    assert got == plain == 105.0


def test_a_global_of_another_module_is_seen(disk_cash, tmp_path, monkeypatch):
    (tmp_path / "held_elsewhere.py").write_text(
        "import numpy as np\nM = np.arange(4.0)\n\ndef total():\n    return float(M.sum())\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("held_elsewhere")

    def body(a):
        a[0] = 99.0
        return module.total()

    got, plain = _stored_then_global(disk_cash.cache(body), body, module.M)
    assert got == plain == 105.0


@needs_trace
def test_the_warning_names_the_parameter(disk_cash):
    """``np.put`` changes ``a`` where reading the source does not see it,
    so the warning is the one the run gives."""

    def body(a, b):
        np.put(a, 0, 99.0)
        return float(b.sum())

    cached = disk_cash.cache(body)
    a = np.arange(4.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(a, a[:2])
    text = " ".join(str(w.message) for w in caught)
    assert "shares memory with the argument 'b'" in text, text
