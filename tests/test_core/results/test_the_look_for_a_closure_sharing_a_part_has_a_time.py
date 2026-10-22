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


class Slotted:
    __slots__ = ("fn",)


def test_a_function_in_slots_an_object_array_a_frame_column_a_partial_or_a_method_is_found():
    import dataclasses
    import functools
    import types

    import pandas as pd

    @dataclasses.dataclass(slots=True)
    class Held:
        fn: object

    arr = np.zeros(3)
    slotted = Slotted()
    slotted.fn = lambda: arr
    holders = {
        "slots": slotted,
        "dataclass_slots": Held(lambda: arr),
        "object_array": np.array([lambda: arr, None], dtype=object),
        "frame_column": pd.DataFrame({"f": [lambda: arr, None]}),
        "series": pd.Series([lambda: arr, None]),
        "partial": functools.partial(lambda: arr),
        "bound_method": types.MethodType(lambda self: arr, object()),
    }
    for name, holder in holders.items():
        found = closures_sharing({"a": arr, "g": holder}, 5.0)
        assert found == (["g"], True), name


def test_plain_data_with_no_function_anywhere_is_not_walked():
    """Ten million strings are typed in bulk, never remembered one by one:
    the check finishes well inside its floor."""
    import time

    words = " ".join(map(str, range(10_000_000))).split()
    start = time.process_time()
    found = closures_sharing({"words": words, "rows": [{"a": i} for i in range(100_000)]}, 1.0)
    assert found == ([], True)
    assert time.process_time() - start < 1.0


OBJECT_HOLDERS = {
    "object_array": "np.array([lambda: arr, None], dtype=object)",
    "frame_column": "pd.DataFrame({'f': [lambda: arr, None]})",
}


def test_in_a_notebook_an_object_array_or_frame_column_holding_a_closure_keeps_the_write_through_a_view(
    cash_magics, monkeypatch, tmp_path
):
    """``g`` holds a function over ``a``: a write through a view of ``a``
    after an edit reaches what ``g`` returns, so ``g`` is not kept by
    reference with ``a`` copied apart from it."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    monkeypatch.syspath_prepend(str(tmp_path))
    for kind, holder in OBJECT_HOLDERS.items():
        module = f"closure_holder_{kind}"
        (tmp_path / f"{module}.py").write_text(
            "import numpy as np\nimport pandas as pd\n"
            f"def mk():\n    arr = np.zeros(10)\n    return arr, {holder}\n"
            "def call(g):\n    return (g[0] if isinstance(g, np.ndarray) else g['f'][0])()\n",
            encoding="utf-8",
        )
        cells = [f"from {module} import mk, call", "a, g = mk()", "v = a[2:5]", "v[:] = 7", "r = float(call(g).sum())"]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == 21.0, kind
        cells[3] = "v[:] = 9"
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == 27.0, kind
