"""What a pandas value carries beside its labels and values is in its key.

A categorical's categories and ``ordered`` flag, the column axis's own name,
an index's ``freq`` and ``attrs`` all change what code reading the value
returns, and none of them showed in the key: ``str(dtype)`` is
``'category'`` for every categorical, and the values hash says nothing of the
rest. Each pair below was served the first value's result.
"""

from __future__ import annotations

import pytest

from cash.object_hashing import hash_pandas

pd = pytest.importorskip("pandas")

LEVELS = pd.CategoricalDtype(["low", "mid", "high"])


def _named_columns():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    frame.columns.name = "feature"
    return frame


def _with_attrs(value):
    value = value.copy()
    value.attrs["unit"] = "kg"
    return value


PAIRS = {
    "unused categories": (
        pd.Series(["low", "high"], dtype=LEVELS),
        pd.Series(["low", "high"], dtype=LEVELS).cat.remove_unused_categories(),
    ),
    "ordered flag": (
        pd.Series(["low", "high"], dtype=LEVELS),
        pd.Series(["low", "high"], dtype=pd.CategoricalDtype(["low", "mid", "high"], ordered=True)),
    ),
    "many categories": (
        pd.Series(["c0"], dtype=pd.CategoricalDtype([f"c{i}" for i in range(500)])),
        pd.Series(["c0"], dtype=pd.CategoricalDtype([f"c{i}" for i in range(500) if i != 250])),
    ),
    "categorical frame column": (
        pd.DataFrame({"c": pd.Series(["low"], dtype=LEVELS)}),
        pd.DataFrame({"c": pd.Series(["low"], dtype=pd.CategoricalDtype(["low", "high"]))}),
    ),
    "categorical index": (
        pd.Series([1], index=pd.CategoricalIndex(["low"], categories=["low", "mid"])),
        pd.Series([1], index=pd.CategoricalIndex(["low"], categories=["low", "high"])),
    ),
    "column axis name": (pd.DataFrame({"a": [1], "b": [2]}), _named_columns()),
    "index freq": (
        pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D")),
        pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"])),
    ),
    "series attrs": (pd.Series([1.0]), _with_attrs(pd.Series([1.0]))),
    "frame attrs": (pd.DataFrame({"a": [1]}), _with_attrs(pd.DataFrame({"a": [1]}))),
}


@pytest.mark.parametrize("left, right", list(PAIRS.values()), ids=list(PAIRS))
def test_metadata_reaches_the_key(left, right):
    assert hash_pandas(left) is not None
    assert hash_pandas(left) != hash_pandas(right)


@pytest.mark.parametrize("left, right", list(PAIRS.values()), ids=list(PAIRS))
def test_an_equal_copy_still_keys_alike(left, right):
    assert hash_pandas(right) == hash_pandas(right.copy())


def test_the_decorator_serves_each_its_own_columns(tmp_path):
    from cash import Cash, FileBackend

    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def one_hot_columns(s):
        return list(pd.get_dummies(s).columns)

    @c.cache
    def long_form_columns(df):
        return list(df.melt().columns)

    full = pd.Series(["low", "high"], dtype=LEVELS)
    assert one_hot_columns(full) == ["low", "mid", "high"]
    assert one_hot_columns(full.cat.remove_unused_categories()) == ["low", "high"]
    assert long_form_columns(pd.DataFrame({"a": [1], "b": [2]})) == ["variable", "value"]
    assert long_form_columns(_named_columns()) == ["feature", "value"]
