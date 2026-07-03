"""CAS-106: pandas argument hashing must cover schema labels, not just values.

``hash_pandas_object`` hashes row values + index values but not column names,
``Series.name``, or index names — so renamed frames/series and empty frames of
any schema collided into one cache entry and returned each other's results.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_column_rename_does_not_collide(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    @c.cache
    def agg(df):
        calls["n"] += 1
        return df.sum().to_dict()

    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = df1.rename(columns={"a": "b"})
    assert agg(df1) == {"a": 3}
    assert agg(df2) == {"b": 3}
    assert calls["n"] == 2, "renamed-column frame must not share a cache entry"


def test_series_rename_does_not_collide(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def describe(s):
        return {s.name: float(s.sum())}

    s1 = pd.Series([1, 2], name="alpha")
    assert describe(s1) == {"alpha": 3.0}
    assert describe(s1.rename("beta")) == {"beta": 3.0}


def test_index_name_does_not_collide(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def idxname(df):
        return df.index.name

    df1 = pd.DataFrame({"a": [1, 2]})
    assert idxname(df1) is None
    assert idxname(df1.rename_axis("rowid")) == "rowid"


def test_empty_frames_with_different_schemas_do_not_collide(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def cols(df):
        return df.columns.tolist()

    assert cols(pd.DataFrame({"a": []})) == ["a"]
    assert cols(pd.DataFrame({"b": []})) == ["b"]
    assert cols(pd.DataFrame({"x": [], "y": []})) == ["x", "y"]


def test_identical_frames_still_hit(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    @c.cache
    def agg(df):
        calls["n"] += 1
        return float(df["v"].sum())

    df = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.Index([10, 20], name="id"))
    assert agg(df) == 3.0
    assert agg(df.copy()) == 3.0
    assert calls["n"] == 1, "equal frames (incl. schema) must share one entry"


def test_multiindex_frames_distinct_and_stable(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    @c.cache
    def total(df):
        calls["n"] += 1
        return float(df["v"].sum())

    idx1 = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["g", "i"])
    idx2 = pd.MultiIndex.from_tuples([("b", 1), ("b", 2)], names=["g", "i"])
    assert total(pd.DataFrame({"v": [1.0, 2.0]}, index=idx1)) == 3.0
    assert total(pd.DataFrame({"v": [5.0, 6.0]}, index=idx2)) == 11.0
    assert total(pd.DataFrame({"v": [1.0, 2.0]}, index=idx1)) == 3.0
    assert calls["n"] == 2
