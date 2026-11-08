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
