"""Parts of a result that share memory are found wherever the result keeps
them, and a pair no one can write through is stored.

The check skipped namedtuples and slots objects, stopped after 1,024 items
of whatever came first (a long list of numbers beside ``a, a[2:5]``), and
did not look at a table and its column, which pandas 2 without
copy-on-write shares. It refused ``df, df["y"].to_numpy()``, a read-only
view of a frame that never writes it: ordinary model-fitting code.
"""

from __future__ import annotations

import collections
import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from cash.decorator.result_parts import split_by_a_hit

Pair = collections.namedtuple("Pair", "raw window")


@dataclasses.dataclass(slots=True)
class SlottedPair:
    raw: object
    window: object


def _served_after_a_write(cash, build):
    """What ``window[0]`` reads after ``raw[2] = -1``, on three calls."""
    load = cash.cache(build)
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            r = load()
            r.raw[2] = -1.0
            seen.append(float(r.window[0]))
    return seen


@pytest.mark.parametrize("holder", [Pair, SlottedPair], ids=["a namedtuple", "a slots dataclass"])
def test_an_array_and_a_view_of_it_in_a_holder_without_a_dict_are_not_stored(cash_instance, holder):
    def build():
        a = np.arange(10.0)
        return holder(a, a[2:5])

    a = np.arange(10.0)
    assert split_by_a_hit(holder(a, a[2:5])) is not None
    assert _served_after_a_write(cash_instance, build) == [-1.0, -1.0, -1.0], "a hit handed the parts back apart"


def test_an_array_and_a_view_of_it_after_a_long_list_are_found():
    a = np.arange(10.0)
    assert split_by_a_hit(([0.0] * 1100, a, a[2:5])) is not None


def test_a_table_and_its_column_are_not_stored_without_copy_on_write(cash_instance):
    if not pd.__version__.startswith("2.") or pd.options.mode.copy_on_write is not False:
        pytest.skip("pandas 2 without copy-on-write: a column is a view its table shares")
    runs = []

    def build():
        runs.append(1)
        df = pd.DataFrame({"x": np.arange(10.0)})
        return df, df["x"]

    load = cash_instance.cache(build)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            df, s = load()
            s.iloc[0] = -1.0
            assert df["x"].iloc[0] == -1.0, "a hit handed the table and its column back apart"


def test_a_table_and_a_read_only_view_of_its_column_are_stored():
    if pd.__version__.startswith("2.") and pd.options.mode.copy_on_write is not True:
        pytest.skip("needs pandas copy-on-write")
    df = pd.DataFrame({"x": np.arange(10.0), "z": np.arange(10.0), "y": np.arange(10.0) * 2})
    assert split_by_a_hit((df, df["y"].to_numpy())) is None
    assert split_by_a_hit((df, df[["x", "z"]].to_numpy(), df["y"].to_numpy())) is None


def test_a_polars_frame_and_its_zero_copy_column_are_stored():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": np.arange(10.0)})
    view = df["x"].to_numpy()
    with pytest.raises(ValueError):
        view.flags.writeable = True  # no one can write it
    assert split_by_a_hit((df, view)) is None


def test_two_read_only_views_of_one_array_are_stored():
    a = np.arange(10.0)
    a.flags.writeable = False
    assert split_by_a_hit((a, a[2:5])) is None  # no one writes through either
    assert split_by_a_hit((a, np.arange(10.0)[2:5], a[1:])) is None


def test_a_read_only_view_beside_the_writeable_array_is_not_stored():
    a = np.arange(10.0)
    view = a[2:5]
    view.flags.writeable = False
    assert split_by_a_hit((a, view)) == ("result[0]", "result[1]")


def test_a_private_slot_is_looked_into():
    class Holder:
        __slots__ = ("__raw", "window")

        def __init__(self, a):
            self.__raw = a
            self.window = a[2:5]

    a = np.arange(10.0)
    assert split_by_a_hit(Holder(a)) == ("result.__raw", "result.window")
