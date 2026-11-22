"""A ``functools.lru_cache`` helper is keyed by the function it wraps.

The wrapper has no code of its own, only ``__wrapped__``: it was taken for
an opaque callable and KEY-OPAQUE-CALLABLE said an edit to it would NOT
invalidate (stingray's helpers), while the trace of the call did catch the
edit. Keyed by the wrapped function's code, the warning is gone and an edit
misses with or without a trace. A callable with no code to read at all
still warns.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]

HELPERS = """\
import functools


@functools.lru_cache(maxsize=None)
def coef(n):
    return n * {k}


def uses(x):
    return x + coef(3)
"""

CALLER = """\
import {helpers}


def through_a_helper(x):
    return {helpers}.uses(x)


def called(x):
    return {helpers}.coef(x) + 1


def handed(x, fn):
    return fn(x)
"""


def _opaque(caught):
    return [str(w.message) for w in caught if "KEY-OPAQUE-CALLABLE" in str(w.message)]


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("how", ["through_a_helper", "called", "handed"])
def test_an_edit_to_an_lru_cache_helper_misses_without_a_warning(tmp_path, monkeypatch, request, how, trace):
    monkeypatch.setenv("CASH_TRACE", trace)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    helpers = f"_lru_helpers_{tmp_path.name}"
    caller = f"_lru_caller_{tmp_path.name}"
    # Imported here, so gone again after: a later test in the worker that
    # looks for memos in the user's modules would find the helper's.
    request.addfinalizer(lambda: [sys.modules.pop(name, None) for name in (helpers, caller)])
    for name in (helpers, caller):
        monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{helpers}.py").write_text(HELPERS.format(k=2), encoding="utf-8")
    (tmp_path / f"{caller}.py").write_text(CALLER.format(helpers=helpers), encoding="utf-8")
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    module = importlib.import_module(caller)
    f = c.cache(getattr(module, how))

    def call():
        if how == "handed":
            return f(1, sys.modules[helpers].coef)
        return f(1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = call()
        assert call() == first
        assert f.cache_info()["hits"] == 1
        (tmp_path / f"{helpers}.py").write_text(HELPERS.format(k=5), encoding="utf-8")
        importlib.reload(sys.modules[helpers])
        assert call() != first, "the edited helper was served the old result"
        assert f.cache_info()["hits"] == 1
    assert _opaque(caught) == []


class _Opaque:
    """Control: a callable whose behaviour has no Python code to key."""

    __call__ = staticmethod(abs)


def _takes(x, fn):
    return fn(x)


def test_a_callable_with_no_code_still_warns(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    f = c.cache(_takes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f(-1, _Opaque())
    assert _opaque(caught)
