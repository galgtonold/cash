"""A served call's categorical column of Python objects has its argument's
own categories, as the call's did.

``flt(df)`` returning ``df[mask]`` keeps the column's dtype, the very
object ``df``'s column has, and with it the objects its categories are: a
later ``df['c'].iloc[1].v.append(5)`` shows through the result. A hit's
result was a copy, which held copies of them. The entry records the column
whose dtype the result shares, and a hit gives the result that dtype again.
"""

from __future__ import annotations

import time

import pandas as pd

from cash.notebook.call_interception import CallSite


class K:
    def __init__(self, v):
        self.v = [v]


def flt(d):
    time.sleep(0.12)
    return d[d["n"] % 2 == 1]


def test_the_hit_holds_the_arguments_objects(call_unit_harness):
    df = pd.DataFrame({"c": pd.Categorical([K(i) for i in range(3)] * 10), "n": range(30)})
    unit = call_unit_harness(lineage={"flt": "hash-flt", "df": "hash-df"}, user_ns={"df": df})
    site = CallSite(source="flt(df)", free_names=frozenset({"flt", "df"}), occurrence_index=0)
    results = []
    for _ in range(2):
        unit.begin_statement()
        results.append(unit.wrap(flt, site)(df))
    assert [e["cache_hit"] for e in unit.drain()] == [False, True]
    served = results[1]
    assert served["c"].dtype is df["c"].dtype
    assert served["c"].iloc[0] is df["c"].iloc[1]
    assert served["n"].tolist() == list(range(1, 30, 2))
