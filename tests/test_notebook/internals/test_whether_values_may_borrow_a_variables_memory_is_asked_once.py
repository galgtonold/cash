"""Whether the outputs and holders a statement stores may borrow a
variable's memory is asked once for all of them.

``v3 = df.iloc[3:]`` over an object column holds df's cells, so df and
every view taken before it are stored with it, and each was asked
`borrowed_memory_names` on its own: one scan of the namespace a holder,
quadratic over a cell of views. `may_borrow_from_a_variable` asks once,
and its "no" is each value's "no": every value's candidates are among
those it finds, none of the values' own objects left out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.notebook import borrowed_memory
from cash.notebook.borrowed_memory import borrowed_memory_names, may_borrow_from_a_variable
from cash.notebook.statement import output_refusals
from cash.notebook.statement.derivation_edges import is_uncacheable_alias


def _views():
    df = pd.DataFrame({"x": np.arange(1000.0), "o": [{"i": i} for i in range(1000)]})
    ns = {"df": df, **{f"v{i}": df.iloc[i:] for i in range(20)}}
    return ns, list(ns)


def _a_series_over_an_array():
    arr = np.arange(10.0)
    ns = {"arr": arr, "s": pd.Series(arr, copy=False), "t": pd.Series(np.arange(3.0))}
    return ns, ["s", "t"]


def _a_view_and_its_base():
    a = np.arange(10.0)
    ns = {"a": a, "v": a[2:], "w": a[5:]}
    return ns, ["v", "w", "a"]


def _an_array_in_a_dict():
    data = {"x": np.arange(10.0)}
    ns = {"data": data, "s": pd.Series(data["x"], copy=False), "u": pd.Series(np.arange(4.0))}
    return ns, ["s", "u"]


def _polars_over_an_array():
    pl = pytest.importorskip("polars")
    arr = np.arange(10.0)
    ns = {"arr": arr, "p": pl.Series("p", arr), "q": pl.Series("q", [1.0, 2.0])}
    return ns, ["p", "q"]


def _nothing_to_borrow():
    ns = {"rows": [{"a": 1}, {"a": 2}], "n": 3, "f": pd.DataFrame({"a": [1, 2]})}
    return ns, ["rows", "f"]


CASES = pytest.mark.parametrize(
    "make",
    [
        _views,
        _a_series_over_an_array,
        _a_view_and_its_base,
        _an_array_in_a_dict,
        _polars_over_an_array,
        _nothing_to_borrow,
    ],
)


@CASES
@pytest.mark.parametrize("containers", [None, set(), {"data"}])
def test_no_for_all_of_them_is_no_for_each(make, containers):
    ns, names = make()
    values = [ns[name] for name in names]
    if not may_borrow_from_a_variable(values, ns, containers=containers):
        assert all(not borrowed_memory_names([value], ns, containers=containers) for value in values)
    assert may_borrow_from_a_variable(values, ns, containers=containers) >= any(
        borrowed_memory_names([value], ns, containers=containers) for value in values
    )


@CASES
@pytest.mark.parametrize("inputs", [(), ("data",)])
def test_the_refusal_is_the_one_each_value_asked_on_its_own_gets(make, inputs, monkeypatch):
    ns, names = make()
    outputs = set(names)
    trimmed = output_refusals._value_refusal(outputs, ns, ns, (), inputs)
    monkeypatch.setattr(output_refusals, "may_borrow_from_a_variable", lambda *args, **kwargs: True)
    assert output_refusals._value_refusal(outputs, ns, ns, (), inputs) == trimmed
    for name in names:
        assert is_uncacheable_alias(ns[name], ns, (), inputs) == is_uncacheable_alias(
            ns[name], ns, (), inputs, may_borrow=may_borrow_from_a_variable([ns[n] for n in names], ns, set(inputs))
        )


def test_a_cell_of_views_scans_the_namespace_once(monkeypatch):
    ns, names = _views()
    scans = []
    real = borrowed_memory._candidates
    monkeypatch.setattr(borrowed_memory, "_candidates", lambda *args: scans.append(1) or real(*args))
    assert output_refusals._value_refusal(set(names), ns, ns) is None
    assert len(scans) == 1
