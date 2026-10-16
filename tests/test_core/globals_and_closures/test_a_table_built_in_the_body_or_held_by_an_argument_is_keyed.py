"""A function picked from a table the body builds, or from a table an
argument holds, is in the key: editing it, or a helper it calls, recomputes.

``handlers = {"a": fa, "b": fb}; handlers[name](x)`` names ``fa`` and ``fb``
in the body, so the key folds them as globals the body reads.
``router.table[name](x)`` and ``table[name](x)`` reach the functions through
an argument, and an argument that holds a function is keyed by its code.
Each shape is checked with an edit of the picked function (its code
replaced) and with a new definition of a helper that function calls.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tests._scripts import run_python

_SOURCE = """
def g(x):
    return x + 1

def g_edited(x):
    return x + 50

def fa(x):
    return g(x)

def fb(x):
    return x + 100

def other(x):
    return -x

class Router:
    def __init__(self):
        self.table = {"a": fa, "b": fb}

def local_dict(name, x):
    handlers = {"a": fa, "b": fb}
    return handlers[name](x)

def local_get(name, x):
    handlers = {"a": fa, "b": fb}
    return handlers.get(name, fb)(x)

def local_list(name, x):
    steps = [fa, fb]
    return steps[0 if name == "a" else 1](x)

def argument_attribute(router, name, x):
    return router.table[name](x)

def argument_table(table, name, x):
    return table[name](x)

def argument_get(table, name, x):
    return table.get(name, fb)(x)
"""

_SHAPES = ["local_dict", "local_get", "local_list", "argument_attribute", "argument_table", "argument_get"]


def _module(tmp_path, monkeypatch, name):
    (tmp_path / f"{name}.py").write_text(_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _call(module, shape):
    if shape == "argument_attribute":
        return lambda cached: cached(module.Router(), "a", 0)
    if shape in ("argument_table", "argument_get"):
        return lambda cached: cached({"a": module.fa, "b": module.fb}, "a", 0)
    return lambda cached: cached("a", 0)


@pytest.mark.parametrize("shape", _SHAPES)
def test_editing_the_picked_function_recomputes(cash_instance, tmp_path, monkeypatch, shape):
    module = _module(tmp_path, monkeypatch, f"_cash_dispatch_fn_{shape}")
    cached, call = cash_instance.cache(getattr(module, shape)), _call(module, shape)
    assert call(cached) == 1
    assert call(cached) == 1

    monkeypatch.setattr(module.fa, "__code__", module.other.__code__)
    assert call(cached) == 0, "the edit to the function the table picks served the old result"


@pytest.mark.parametrize("shape", _SHAPES)
def test_editing_a_helper_the_picked_function_calls_recomputes(cash_instance, tmp_path, monkeypatch, shape):
    module = _module(tmp_path, monkeypatch, f"_cash_dispatch_helper_{shape}")
    cached, call = cash_instance.cache(getattr(module, shape)), _call(module, shape)
    assert call(cached) == 1
    assert call(cached) == 1

    # Rebound, as a reload or a new definition of it leaves the module.
    monkeypatch.setattr(module, "g", module.g_edited)
    assert call(cached) == 50, "the edit to a helper of the picked function served the old result"


_SCRIPT = """\
import warnings; warnings.simplefilter("ignore")
import cash
from dispatch_impl import Router, fa, fb

@cash.cache
def local_dict(name, x):
    handlers = {"a": fa, "b": fb}
    return handlers[name](x)

@cash.cache
def argument_attribute(router, name, x):
    return router.table[name](x)

print("R", local_dict("a", 0), argument_attribute(Router(), "a", 0))
"""

_IMPL = """\
def g(x):
    return x + {g}

def fa(x):
    return g(x) + {fa}

def fb(x):
    return x + 100

class Router:
    def __init__(self):
        self.table = {{"a": fa, "b": fb}}
"""


@pytest.mark.slow
def test_an_edit_in_another_module_recomputes_in_the_next_process(tmp_path):
    """The user's case: the implementations live in a module that is edited
    between runs, and the next process builds the key afresh (``.cash`` is
    kept, or the test is vacuous)."""

    def run(g, fa):
        (tmp_path / "dispatch_impl.py").write_text(_IMPL.format(g=g, fa=fa), encoding="utf-8")
        (tmp_path / "main.py").write_text(_SCRIPT, encoding="utf-8")
        return run_python("main.py", cwd=tmp_path).stdout.strip().splitlines()[-1]

    assert run(g=1, fa=0) == "R 1 1"
    assert run(g=1, fa=0) == "R 1 1"
    assert run(g=1, fa=10) == "R 11 11", "an edit to the picked function served the old result"
    assert run(g=5, fa=10) == "R 15 15", "an edit to a helper of the picked function served the old result"
