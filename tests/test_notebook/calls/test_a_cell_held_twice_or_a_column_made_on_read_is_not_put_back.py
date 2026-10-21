"""A served call puts back only cells it can tell apart, and only in
columns of Python objects.

``sel1(df)`` with one list in both of ``df``'s rows has the key of a frame
holding two equal lists there. The entry said "the argument's first row",
the first place its address was found, and a hit on the second frame put
back row 0's list where the call had taken row 1's. A cell an argument
holds in two places is not linked: the call is refused, as any result
holding cells something else holds is.

A categorical, period or interval column is read as an object array
built for the read: its "cells" were made then, and a hit put that plain
object array back over the column, so a categorical came back as object;
the arrays freed between reads left addresses another read reused, and a
period frame's links failed on every hit, running the call each time.
Only numpy's object dtype is read now.
"""

from __future__ import annotations

import time

import pandas as pd

from cash.notebook.call_interception import CallSite


def sel1(d):
    time.sleep(0.12)
    return d[d["n"] == 1]


def flt(d):
    time.sleep(0.12)
    return d[d["n"] % 2 == 1]


def cols(d):
    time.sleep(0.12)
    return d[["p2", "n"]]


def _twice(call_unit_harness, fn, first, second=None):
    unit = call_unit_harness(lineage={fn.__name__: f"hash-{fn.__name__}", "arg": "hash-arg"}, user_ns={"arg": first})
    site = CallSite(source=f"{fn.__name__}(arg)", free_names=frozenset({fn.__name__, "arg"}), occurrence_index=0)
    out = []
    for arg in (first, first if second is None else second):
        unit.begin_statement()
        out.append(unit.wrap(fn, site)(arg))
    return [e["cache_hit"] for e in unit.drain()], out


def test_a_list_in_two_rows_is_not_put_back_for_either(call_unit_harness):
    shared = [1]
    first = pd.DataFrame({"o": [shared, shared], "n": [0, 1]})
    second = pd.DataFrame({"o": [[1], [1]], "n": [0, 1]})
    del shared
    _hits, (_w1, w2) = _twice(call_unit_harness, sel1, first, second)
    assert w2["o"].iloc[0] is second["o"].iloc[1], "row 0's list was put back for row 1"


def test_a_categorical_column_keeps_its_dtype(call_unit_harness):
    df = pd.DataFrame({"c": pd.Categorical([(1, 2), (3, 4)] * 50), "n": range(100)})
    _hits, (_first, second) = _twice(call_unit_harness, flt, df)
    assert str(second["c"].dtype) == "category"


def test_a_frame_of_period_columns_is_served(call_unit_harness):
    n = 3000
    df = pd.DataFrame(
        {
            "p1": pd.period_range("2020-01-01", periods=n, freq="D"),
            "p2": pd.period_range("1990-01-01", periods=n, freq="D"),
            "n": range(n),
        }
    )
    hits, (_first, second) = _twice(call_unit_harness, cols, df)
    assert hits == [False, True]
    assert str(second["p2"].dtype) == "period[D]" and (second["p2"] == df["p2"]).all()
