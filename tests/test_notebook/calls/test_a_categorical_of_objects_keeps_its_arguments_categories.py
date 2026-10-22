"""A served call's categorical column of Python objects has its argument's
own categories, as the call's did.

``flt(df)`` returning ``df[mask]`` keeps the column's dtype, the very
object ``df``'s column has, and with it the objects its categories are: a
later ``df['c'].iloc[1].v.append(5)`` shows through the result. A hit's
result was a copy, which held copies of them. The entry records the column
whose dtype the result shares, and a hit gives the result that dtype again.

pandas 2 builds a dtype of its own around the same categories, so the hit
checked for the dtype itself failed every time and ran the call bare: the
statement around it took the result for its own and stored a copy, served
after a restart. A hit now checks the categories, and a call whose cells
cannot be put back runs as a miss. The categories must be the stored ones
position by position: a code names a position.
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
    assert served["c"].dtype.categories is df["c"].dtype.categories
    assert served["c"].iloc[0] is df["c"].iloc[1]
    assert served["n"].tolist() == list(range(1, 30, 2))


def test_categories_in_another_order_run_the_call(call_unit_harness):
    first = pd.DataFrame({"c": pd.Categorical([K(i) for i in range(3)] * 4), "n": range(12)})
    ks = [K(i) for i in range(3)]
    second = pd.DataFrame({"c": pd.Categorical([ks[i % 3] for i in range(12)], categories=ks[::-1]), "n": range(12)})
    # One lineage for both, so the key cannot tell them apart.
    unit = call_unit_harness(lineage={"flt": "hash-flt", "df": "hash-df"}, user_ns={"df": first})
    site = CallSite(source="flt(df)", free_names=frozenset({"flt", "df"}), occurrence_index=0)
    results = []
    for df in (first, second):
        unit.begin_statement()
        results.append(unit.wrap(flt, site)(df))
    assert [e["cache_hit"] for e in unit.drain()] == [False, False]
    assert all(a is b for a, b in zip(results[1]["c"], second["c"].iloc[1::2]))
