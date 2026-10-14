"""A label written in place after the store reaches no hit.

The RAM tier copies a table it cannot freeze (a nullable or categorical
column) block by block, on the store and on every hit. Its labels it kept
as pandas' deep copy keeps them: views of the original's. ``df.index.array``,
``df.columns.values`` and ``s.cat.categories.array`` write a label array in
place, so such a write -- to the caller's table after the store, or to a
hit -- changed the stored entry and every later hit. Each copy now holds
labels of its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends.memory_backend import InMemoryBackend


def _nullable(index):
    return pd.DataFrame({"a": pd.array([1, 2, None, 4], dtype="Int64"), "b": np.arange(4.0)}, index=index)


TABLES = {
    "nullable column": lambda: _nullable(pd.Index(np.arange(100, 104))),
    "nullable series": lambda: pd.Series(pd.array([1, 2, None, 4], dtype="Int64"), index=pd.Index(list("wxyz"))),
    "dated index": lambda: _nullable(pd.date_range("2024-01-01", periods=4)),
    "categorical column": lambda: pd.DataFrame(
        {"c": pd.Categorical(list("abca"), categories=pd.Index(list("abc"), dtype=object))},
        index=pd.Index(np.arange(4.0)),
    ),
    "in a tuple": lambda: (_nullable(pd.Index(list("wxyz"), dtype=object)), {"rows": 4}),
}


def _frame(value):
    return value[0] if isinstance(value, tuple) else value


def _labels(value) -> list:
    frame = _frame(value)
    seen = [frame.index.tolist(), frame.tolist() if frame.ndim == 1 else frame.columns.tolist()]
    if frame.ndim == 2 and "c" in frame:
        seen.append(frame["c"].cat.categories.tolist())
    return seen


def _write_labels(value) -> None:
    frame = _frame(value)
    frame.index.array[0] = frame.index[1]
    if frame.ndim == 2:
        frame.columns.values[0] = "edited"
        if "c" in frame:
            frame["c"].cat.categories.array[0] = "edited"


@pytest.mark.parametrize("name", TABLES)
@pytest.mark.parametrize("who", ["the caller", "a hit"])
def test_a_label_written_in_place_after_the_store_reaches_no_hit(name, who):
    value = TABLES[name]()
    ram = InMemoryBackend()
    ram.set("k", value, {"execution_time": 10.0})
    _write_labels(value if who == "the caller" else ram.get("k")[1])
    assert _labels(ram.get("k")[1]) == _labels(TABLES[name]())


def test_a_table_copied_in_full_is_the_ram_tiers_own():
    ram = InMemoryBackend()
    ram.set("k", _nullable(pd.Index(np.arange(100, 104))), {"execution_time": 10.0})
    assert ram.holds_own_copy("k")
