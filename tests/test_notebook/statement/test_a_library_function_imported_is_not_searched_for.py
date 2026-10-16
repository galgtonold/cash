"""Importing a library's function does not search the heap for its holders.

scikit-learn wraps ``load_iris`` and ``train_test_split`` in its parameter
check: each is a closure, and a closure carries its cells, so the share
check after ``from sklearn.datasets import load_iris`` looked for every
object holding them (`gc.get_referrers`, once per level) and found the
module: 2 s on E11, next to an import of 0.1 s. A function its module binds
by its own name is pickled and copied by that name, and a restore hands
back the very same function; it carries nothing (`_found_by_name`).
"""

from __future__ import annotations

import functools
import sys
import types

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import VALUE_TYPES, is_value
from tests._cell_driver import run_cash_cell


def _checked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@pytest.fixture
def library():
    module = types.ModuleType("cash_test_wrapped_lib")

    def load(n):
        return list(range(n))

    load.__module__, load.__qualname__ = module.__name__, "load"
    module.load = _checked(load)
    sys.modules[module.__name__] = module
    yield module
    del sys.modules[module.__name__]


def test_a_wrapped_library_function_carries_nothing(library):
    assert library.load.__closure__
    assert is_value(library.load, VALUE_TYPES)


def test_a_closure_not_found_by_its_name_still_carries_its_cells():
    store: list = []
    hook = (lambda lst: lambda v: lst.append(v))(store)
    assert not is_value(hook, VALUE_TYPES)
    named = _checked(lambda: store)
    named.__module__, named.__qualname__ = "__main__", "named"
    sys.modules["__main__"].named = named
    try:
        assert not is_value(named, VALUE_TYPES), "a notebook's own function is stored by value"
    finally:
        del sys.modules["__main__"].named


def test_importing_one_searches_for_no_holder(cash_magics, statement_processor, library, monkeypatch):
    searched = []
    real = shared_objects._holders_from_referrers
    monkeypatch.setattr(shared_objects, "_holders_from_referrers", lambda *a, **k: searched.append(1) or real(*a, **k))
    run_cash_cell(cash_magics, "from cash_test_wrapped_lib import load")
    assert searched == []
    run_cash_cell(cash_magics, "rows = load(3)")
    assert statement_processor.shell.user_ns["rows"] == [0, 1, 2]
