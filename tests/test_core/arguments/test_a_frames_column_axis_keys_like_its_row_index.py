"""A DataFrame's column axis is keyed by the rule its row index is keyed by.

The columns of a frame are an index: a ``CategoricalIndex`` with its
category order (``pd.crosstab`` and ``pivot_table`` make one from a
categorical column, and ``sort_index(axis=1)`` follows that order), a
``MultiIndex`` with the levels a filter left unused (``columns.levels``
shows them), an index of ``int32`` or ``int64``, a ``DatetimeIndex`` with a
unit and a ``freq``. The key held only the list of the labels and the axis
names, so each pair below -- equal labels, equal cells -- shared an entry
and the second frame was served the first one's answer. A series name or
an axis name is keyed as a value, not by what it prints.
"""

from __future__ import annotations

import pytest

from cash.content_hashers import hash_pandas
from cash.value_hash import compute_hash

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


def _over(columns, cells=None):
    cells = [list(range(len(columns)))] if cells is None else cells
    return pd.DataFrame(cells, columns=columns)


def _sizes_table(order):
    df = pd.DataFrame(
        {"shop": ["a", "a", "b", "b"], "size": pd.Categorical(["S", "L", "M", "S"], categories=order, ordered=True)}
    )
    return pd.crosstab(df["shop"], df["size"])


def _filtered():
    wide = _over(pd.MultiIndex.from_tuples([("north", "x"), ("south", "x"), ("south", "y")]), [[1, 2, 3]])
    return wide.loc[:, wide.columns.get_level_values(0) != "north"]


class Label:
    """Two labels that print alike and are not equal."""

    def __init__(self, unit):
        self.unit = unit

    def __repr__(self):
        return "Label()"

    def __hash__(self):
        return hash(self.unit)

    def __eq__(self, other):
        return type(other) is Label and other.unit == self.unit


def _pairs():
    days = pd.date_range("2020-01-01", periods=3, freq="D")
    return {
        "category order": (_sizes_table(["L", "M", "S"]), _sizes_table(["S", "M", "L"])[["L", "M", "S"]]),
        "categorical or plain labels": (_over(pd.CategoricalIndex(["a", "b"])), _over(pd.Index(["a", "b"]))),
        "int64 or int32 labels": (_over(pd.Index([1, 2], dtype="int64")), _over(pd.Index([1, 2], dtype="int32"))),
        "int64 or object labels": (_over(pd.Index([1, 2], dtype="int64")), _over(pd.Index([1, 2], dtype=object))),
        "a range or its numbers": (_over(pd.RangeIndex(3)), _over(pd.Index([0, 1, 2]))),
        "datetime unit": (_over(days.as_unit("ns")), _over(days.as_unit("s"))),
        "freq": (_over(days), _over(pd.DatetimeIndex(list(days)))),
        "level dtype": (
            _over(pd.MultiIndex.from_arrays([pd.Index([1, 2], dtype="int64"), ["x", "y"]])),
            _over(pd.MultiIndex.from_arrays([pd.Index([1, 2], dtype="int32"), ["x", "y"]])),
        ),
        "unused level": (
            _filtered(),
            _over(pd.MultiIndex.from_tuples([("south", "x"), ("south", "y")]), [[2, 3]]),
        ),
        "level order": (
            _over(pd.MultiIndex(levels=[["a", "b"], ["x"]], codes=[[0, 1], [0, 0]])),
            _over(pd.MultiIndex(levels=[["b", "a"], ["x"]], codes=[[1, 0], [0, 0]])),
        ),
        "series names printing alike": (pd.Series([1], name=Label("kg")), pd.Series([1], name=Label("lb"))),
        "index names printing alike": (
            pd.Series([1], index=pd.Index([0], name=Label("kg"))),
            pd.Series([1], index=pd.Index([0], name=Label("lb"))),
        ),
        "column axis names printing alike": (
            _over(pd.Index(["a"], name=Label("kg"))),
            _over(pd.Index(["a"], name=Label("lb"))),
        ),
    }


PAIRS = _pairs()


@pytest.mark.parametrize("name", list(PAIRS))
def test_frames_alike_in_labels_and_cells_key_apart(name):
    first, second = PAIRS[name]
    if hasattr(first, "columns"):
        # The positive control: by labels and cells these are one frame.
        assert list(first.columns) == list(second.columns) and (first.to_numpy() == second.to_numpy()).all()
    assert hash_pandas(first) != hash_pandas(second)
    assert compute_hash(first) != compute_hash(second)  # the notebook's value hash


@pytest.mark.parametrize("name", list(PAIRS))
def test_a_copy_keys_alike(name):
    first, _ = PAIRS[name]
    assert hash_pandas(first) == hash_pandas(first.copy()) is not None


def test_sorting_the_columns_follows_the_category_order(disk_cash):
    @disk_cash.cache
    def columns_in_order(table):
        return list(table.sort_index(axis=1).columns)

    alphabetical, by_size = PAIRS["category order"]
    assert columns_in_order(alphabetical) == ["L", "M", "S"]
    assert columns_in_order(by_size) == ["S", "M", "L"]


def test_the_levels_a_filter_left_are_read(disk_cash):
    @disk_cash.cache
    def groups(table):
        return list(table.columns.levels[0])

    filtered, rebuilt = PAIRS["unused level"]
    assert filtered.equals(rebuilt)
    assert groups(filtered) == ["north", "south"]
    assert groups(rebuilt) == ["south"]


def test_a_big_frames_column_freq_dropped_in_place_is_seen(disk_cash):
    """A frame big enough for its digest to be remembered between calls:
    the remembered digest is not served once the columns' ``freq`` is
    cleared on the same index object."""

    @disk_cash.cache
    def step(table):
        return table.columns.freq is None

    table = pd.DataFrame(np.zeros((70_000, 16)), columns=pd.date_range("2020-01-01", periods=16, freq="D"))
    assert [step(table), step(table)] == [False, False]
    table.columns.freq = None
    assert step(table) is True
