"""The look through a result for a function whose closure holds another part
of it (STORE-PARTS-SHARED) has a time, and runs no code of the value's.

It had neither: a million small dicts took 5.5 s where pickling them took
1.6 s, and an object whose ``__dict__`` is a property making a new child at
each read (a lazy tree) was read until the property stopped. Pickle's own
walk now goes through the builtin containers; past twice what the work took
(at least a second) the result is not stored, and STORE-PARTS-UNCHECKED says
so. ``__dict__`` is read from the slot Python keeps, never through the
class's own property.
"""

from __future__ import annotations

import collections
import warnings
from typing import NamedTuple

import numpy as np

from cash import Cash, closure_sharing
from cash.backends.persistence_policy import PersistencePolicy
from cash.closure_sharing import closures_sharing
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

MADE: list = []


class Lazy:
    @property
    def __dict__(self):
        child = Lazy()
        MADE.append(child)
        return {"child": child}


class Box:
    def __init__(self, held):
        self.held = held


class Pair(NamedTuple):
    data: object
    fn: object


def _codes(caught) -> list:
    return [getattr(w.message, "code", None) for w in caught]


def test_a_computed_dict_is_not_read():
    MADE.clear()
    assert closures_sharing({"result": [Lazy(), Lazy()]}, 5.0) == (([], True))
    assert MADE == []


def test_a_function_sharing_a_part_is_found_inside_a_named_tuple_an_ordered_dict_and_a_cycle():
    arr = np.zeros(3)
    box = Box(None)
    box.held = box  # holds itself: met once
    found = closures_sharing(
        {
            "pair": Pair(arr, lambda: arr),
            "ordered": collections.OrderedDict(a=box, f=lambda: box),
            "plain": [np.ones(2), lambda: 1.0],
        },
        5.0,
    )
    assert found.finished
    assert found.names == ["ordered", "pair"]


def test_a_value_nested_deeper_than_pickle_goes_is_still_looked_through():
    arr = np.zeros(3)
    deep: list = [arr]
    for _ in range(5000):
        deep = [deep]
    found = closures_sharing({"result": (deep, lambda: arr)}, 5.0)
    assert found.finished and found.names == ["result"]


def _many():
    return [Box(i) for i in range(50_000)]


def test_past_its_time_the_result_is_returned_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(closure_sharing, "CLOSURE_CHECK_FLOOR_S", 0.0)
    monkeypatch.setattr(closure_sharing, "CLOSURE_CHECK_FACTOR", 0.0)
    c = Cash(cache_dir=str(tmp_path / "cache"), backend="memory", register_magic=False)
    many = c.cache(_many)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first, second = many(), many()
    assert len(first) == len(second) == 50_000
    assert _codes(caught).count("STORE-PARTS-UNCHECKED") == 1
    assert many.cache_info()["hits"] == 0


def test_in_a_notebook_past_its_time_the_statement_runs_again(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    # The look for other names sharing an object is not what is tested.
    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 60.0)
    monkeypatch.setattr(closure_sharing, "CLOSURE_CHECK_FLOOR_S", 0.0)
    monkeypatch.setattr(closure_sharing, "CLOSURE_CHECK_FACTOR", 0.0)
    cells = ["class B:\n    def __init__(self, i):\n        self.i = i", "bs = [B(i) for i in range(50_000)]"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert "STORE-PARTS-UNCHECKED" in _codes(caught)
    assert len(cash_magics.shell.user_ns["bs"]) == 50_000


LAZY_CELL = """
MADE = []
class Lazy:
    @property
    def __dict__(self):
        if len(MADE) >= 5000:
            return {}
        child = Lazy()
        MADE.append(child)
        return {'child': child}
"""


def test_in_a_notebook_a_computed_dict_is_never_read(cash_magics, monkeypatch):
    """Not by the closure look, nor by the size estimates and the lineage
    tag, which read ``__dict__`` the same way."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [LAZY_CELL, "n = Lazy()", "k = len(MADE)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["k"] == 0
