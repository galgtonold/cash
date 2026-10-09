"""Frames, arrays and tables are keyed by their buffers, column by column.

Numbers, booleans and dates are read from their own buffer, a nullable
column from its values and mask, an Arrow-backed column (pandas' ``str``,
polars, pyarrow) from its Arrow buffers, and an object column of floats,
bools or ints as the typed array it converts to. Whatever the route, every
value is in the key: a change anywhere misses, equal content hits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.content_hashers import builtin_hash

pa = pytest.importorskip("pyarrow")


def _frame(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f": rng.random(n),
            "i": rng.integers(0, 100, n),
            "b": rng.random(n) > 0.5,
            "d": pd.date_range("2020-01-01", periods=n, freq="h"),
            "tz": pd.date_range("2020-01-01", periods=n, freq="h", tz="Europe/Berlin"),
            "td": pd.to_timedelta(np.arange(n), unit="s"),
            "m": pd.array([1, None] * (n // 2), dtype="Int64"),
            "s": pd.array([f"name{i}" for i in range(n)], dtype="string[pyarrow]"),
            "o": pd.Series([float(i) for i in range(n)], dtype=object),
        }
    )


EDITS = {
    "f": lambda d: d.loc.__setitem__((3, "f"), d.loc[3, "f"] + 1e-12),
    "i": lambda d: d.loc.__setitem__((3, "i"), d.loc[3, "i"] + 1),
    "b": lambda d: d.loc.__setitem__((3, "b"), not d.loc[3, "b"]),
    "d": lambda d: d.loc.__setitem__((3, "d"), pd.Timestamp("1999-01-01")),
    "tz": lambda d: d.loc.__setitem__((3, "tz"), pd.Timestamp("1999-01-01", tz="Europe/Berlin")),
    "td": lambda d: d.loc.__setitem__((3, "td"), pd.Timedelta(seconds=999)),
    "m": lambda d: d.loc.__setitem__((1, "m"), 0),  # a missing value filled
    "s": lambda d: d.loc.__setitem__((3, "s"), "name3x"),
    "o": lambda d: d.loc.__setitem__((3, "o"), 3.5),
}


@pytest.mark.parametrize("column", sorted(EDITS))
def test_a_change_in_any_column_moves_the_key(column):
    frame = _frame()
    before = builtin_hash(frame)
    assert before is not None
    assert builtin_hash(frame.copy()) == before, "an equal frame keyed apart"
    EDITS[column](frame)
    assert builtin_hash(frame) != before, f"a change in {column!r} kept the key"


def test_a_missing_value_and_a_value_key_apart():
    a = pd.Series(pd.array([1, None, 3], dtype="Int64"))
    b = pd.Series(pd.array([1, 0, 3], dtype="Int64"))
    assert builtin_hash(a) != builtin_hash(b)
    s = pd.Series(pd.array(["a", None, "c"], dtype="string[pyarrow]"))
    e = pd.Series(pd.array(["a", "", "c"], dtype="string[pyarrow]"))
    assert builtin_hash(s) != builtin_hash(e)


def test_text_keys_by_its_values_not_its_chunks():
    """Two text columns holding the same values in different Arrow chunks are
    one column to the code reading them."""
    whole = pd.arrays.ArrowStringArray(pa.chunked_array([["a", "bb", None, "dddd"]]))
    pieces = pd.arrays.ArrowStringArray(pa.chunked_array([["a"], ["bb", None], ["dddd"]]))
    assert builtin_hash(pd.Series(whole)) == builtin_hash(pd.Series(pieces))
    moved = pd.arrays.ArrowStringArray(pa.chunked_array([["a", "b"], ["b", None, "dddd"]]))
    assert builtin_hash(pd.Series(moved)) != builtin_hash(pd.Series(whole)), "a boundary between values moved"


def test_a_slice_of_text_keys_as_its_rows():
    base = pa.chunked_array([["a", "b", "c", "d"]])
    sliced = pd.Series(pd.arrays.ArrowStringArray(base.slice(2, 2)))
    fresh = pd.Series(pd.arrays.ArrowStringArray(pa.chunked_array([["c", "d"]])))
    assert builtin_hash(sliced) == builtin_hash(fresh)
    first = pd.Series(pd.arrays.ArrowStringArray(base.slice(0, 2)))
    assert builtin_hash(first) != builtin_hash(sliced)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1.0, 2.0], [1, 2]),  # floats against ints
        ([True, False], [1, 0]),  # bools against ints
        ([1.0, 2.0], [1.0, np.float64(2.0)]),  # a numpy float is not a float
        ([2**70, 1], [2**70 + 1, 1]),  # an int beyond 64 bits
        ([float("nan"), 1.0], [float("inf"), 1.0]),
        ([0.0, 1.0], [-0.0, 1.0]),
    ],
)
def test_object_columns_keep_every_type_and_value(a, b):
    left, right = np.array(a, dtype=object), np.array(b, dtype=object)
    assert builtin_hash(left) != builtin_hash(right)
    assert builtin_hash(left) == builtin_hash(np.array(list(a), dtype=object))
    assert builtin_hash(pd.Series(left)) != builtin_hash(pd.Series(right))


def test_an_object_table_keys_by_every_cell():
    """``DataFrame.to_numpy()`` of floats and dummies: an object table read
    column by column."""
    table = np.empty((4, 3), dtype=object)
    table[:, 0] = [0.5, 1.5, 2.5, 3.5]
    table[:, 1] = [True, False, True, False]
    table[:, 2] = ["a", "b", "c", "d"]
    before = builtin_hash(table)
    for row, col, value in [(2, 0, 9.5), (1, 1, True), (3, 2, "z"), (0, 1, 1)]:
        changed = table.copy()
        changed[row, col] = value
        assert builtin_hash(changed) != before, (row, col, value)
    assert builtin_hash(table.T.copy()) != before, "a transposed table keyed alike"


def test_polars_columns_key_by_their_values():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(
        {"a": [1, 2, 3], "f": [0.5, None, 1.5], "s": ["x", None, "zz"], "c": pl.Series(["u", "v", "u"], dtype=pl.Categorical)}
    )
    before = builtin_hash(frame)
    assert builtin_hash(frame.clone()) == before
    for changed in (
        frame.with_columns(pl.col("a") + 1),
        frame.with_columns(pl.col("f").fill_null(0.0)),
        frame.with_columns(pl.col("s").fill_null("")),
        frame.with_columns(pl.Series("c", ["u", "u", "u"], dtype=pl.Categorical)),
        frame.with_columns(pl.col("a").cast(pl.Int32)),
        frame.slice(1, 2),
    ):
        assert builtin_hash(changed) != before
    assert builtin_hash(frame["s"]) != builtin_hash(frame["s"].fill_null(""))
    assert builtin_hash(frame.slice(1, 2)) == builtin_hash(pl.DataFrame(frame.slice(1, 2).to_dicts(), schema=frame.schema))


def test_pyarrow_tables_key_by_their_rows():
    table = pa.table({"x": [1, 2, 3, 4], "s": ["a", "b", "c", "d"], "l": [[1], [2], [3], [4]]})
    assert builtin_hash(table.slice(0, 2)) != builtin_hash(table.slice(2, 2))
    assert builtin_hash(table.slice(2, 2)) == builtin_hash(pa.table({"x": [3, 4], "s": ["c", "d"], "l": [[3], [4]]}))
    red = pa.table({"c": pa.array(["red", "blue"]).dictionary_encode()})
    cat = pa.table({"c": pa.array(["cat", "dog"]).dictionary_encode()})
    assert builtin_hash(red) != builtin_hash(cat)
