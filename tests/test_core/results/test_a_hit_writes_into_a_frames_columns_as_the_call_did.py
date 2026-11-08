"""A hit puts a frame's stored state back the way the call changed it.

A call that writes into a column (``df.iloc[0, 0] = 1``) leaves the
column's array in place, so a name for it (``h = df["a"].array``) still
reads and writes the frame. A call that gives a column a new array
(``df["a"] = df["a"] + 1``) leaves such a name with the old one. A hit
gave every column a new array, so after a hit ``h`` no longer reached the
frame where a plain run's did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _write_into(df):
    df.iloc[0, 0] = df.iloc[0, 0] + 1
    return 1


def _new_array(df):
    df["a"] = df["a"] + 1
    return 1


def test_a_name_for_a_column_sees_what_a_plain_run_shows(disk_cash):
    """Both ways in one test: the column given a new array is the control."""

    def run(f):
        df = pd.DataFrame({"a": np.zeros(3), "b": ["x", "y", "z"]})
        h = df["a"].array
        f(df)
        h[1] = 7.0
        return df["a"].tolist()

    for body in (_write_into, _new_array):
        cached = disk_cash.cache(body)
        plain = run(body)
        seen = [run(cached) for _ in range(3)]
        assert cached.cache_info()["hits"] == 2
        assert seen == [plain] * 3, (body.__name__, plain, seen)


def _iloc_whole(df):
    df.iloc[:, 0] = np.arange(4.0)
    return 1


def _loc_whole(df):
    df.loc[:, "a"] = np.arange(4.0)
    return 1


def _loc_cell(df):
    df.loc[0, "a"] = df["a"].iloc[1]
    return 1


def _fill(df):
    df.fillna(df["a"].iloc[1], inplace=True)
    return 1


#: ``(body, frame maker, value written through the handle)``: writes the
#: call makes into a column, which plain pandas makes into the column's own
#: array, so a name for it sees them. Cash's own hold on the frame's blocks
#: made pandas copy the column first in the call that was stored.
WRITES = {
    "iloc, the whole column": (_iloc_whole, lambda: pd.DataFrame({"a": np.zeros(4)}), 7.0),
    "loc, the whole column": (_loc_whole, lambda: pd.DataFrame({"a": np.zeros(4)}), 7.0),
    "loc, a cell of a two-column block": (
        _loc_cell,
        lambda: pd.DataFrame(np.arange(8.0).reshape(4, 2), columns=["b", "a"]),
        7.0,
    ),
    "a string column": (_loc_cell, lambda: pd.DataFrame({"a": pd.array(list("abcd"), dtype="string")}), "q"),
    "a categorical column": (_loc_cell, lambda: pd.DataFrame({"a": pd.Categorical(list("abab"))}), "a"),
    "a tz column": (
        _loc_cell,
        lambda: pd.DataFrame({"a": pd.date_range("2020-01-01", periods=4, tz="UTC")}),
        pd.Timestamp("2022-01-01", tz="UTC"),
    ),
    "fillna into Int64": (_fill, lambda: pd.DataFrame({"a": pd.array([None, 1, 2, 3], dtype="Int64")}), 7),
}


@pytest.mark.parametrize("how", list(WRITES))
def test_a_write_into_a_column_reaches_a_name_for_it_after_a_hit(disk_cash, how):
    body, make, sentinel = WRITES[how]

    def run(f):
        df = make()
        h = df["a"].array
        f(df)
        seen = [str(v) for v in h]
        h[1] = sentinel
        return seen, [str(v) for v in df["a"]]

    cached = disk_cash.cache(body)
    plain = run(body)
    cached(make())  # stored from a frame no name of a column's array is kept for
    got = [run(cached) for _ in range(2)]
    assert cached.cache_info()["hits"] == 2
    assert got == [plain] * 2, (plain, got)


def test_a_frame_another_frame_shares_columns_with_is_left_as_plain_pandas_leaves_both(disk_cash):
    """Stored from a frame nothing shared; then handed one another frame
    shares its block with, which pandas copies before writing: writing
    into the block in place would change the other frame too."""

    def run(f):
        df = pd.DataFrame({"a": np.zeros(4)})
        other = df[["a"]]
        f(df)
        return df["a"].tolist(), other["a"].tolist()

    cached = disk_cash.cache(_iloc_whole)
    cached(pd.DataFrame({"a": np.zeros(4)}))
    assert [run(cached) for _ in range(2)] == [run(_iloc_whole)] * 2
