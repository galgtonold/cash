"""A call's result is checked against what its arguments hold in a frame's
object column, and that check has a budget.

``mk(df)`` returning ``Namespace(x=df['d'].iloc[0])`` hands back the very
list the frame's object column holds: a stored copy would no longer be
it. Neither side's walk read a frame's cells.

Reading a big argument down to compare is paid on every call that is not
served: a namespace of a million dicts cost 3.5 s each time. A result that
holds nothing anything else holds is not compared at all, and a comparison
that runs past the statement share check's budget gives up: the call is
not stored, and a slow call says so.
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd

from cash.notebook.call_interception import CallSite
from cash.notebook.statement import processor

RECT = [1]
DF = pd.DataFrame({"d": [RECT, [2]]})
BIG = argparse.Namespace(rows=[{"i": i} for i in range(300_000)])


def cell(d):
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return argparse.Namespace(x=d["d"].iloc[0])


def fresh(src):
    time.sleep(0.05)
    return argparse.Namespace(rows=[{"i": i} for i in range(1000)], n=len(src.rows))


def part(src):
    time.sleep(0.12)  # over the compute floor: a refusal it pays for is told
    return argparse.Namespace(row=src.rows[-1])


def _twice(call_unit_harness, fn, arg, name):
    unit = call_unit_harness(lineage={name: "hash-" + name, "arg": "hash-arg"}, user_ns={"arg": arg})
    site = CallSite(source=f"{name}(arg)", free_names=frozenset({name, "arg"}), occurrence_index=0)
    unit.begin_statement()
    unit.wrap(fn, site)(arg)
    unit.begin_statement()
    result = unit.wrap(fn, site)(arg)
    return [e["cache_hit"] for e in unit.drain()], result


def test_a_frame_cell_the_result_holds_stays_the_frames(call_unit_harness):
    hits, w = _twice(call_unit_harness, cell, DF, "cell")
    assert hits == [False, False]
    assert w.x is RECT


def test_a_fresh_result_is_served_without_reading_a_big_argument(call_unit_harness):
    hits, w = _twice(call_unit_harness, fresh, BIG, "fresh")
    assert hits == [False, True]
    assert w.n == len(BIG.rows)


def test_a_comparison_past_its_budget_does_not_store_and_says_so(call_unit_harness, monkeypatch):
    # A budget the walk over BIG cannot meet on any machine.
    monkeypatch.setattr(processor, "share_check_budget", lambda cost: 0.001)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hits, w = _twice(call_unit_harness, part, BIG, "part")
    assert hits == [False, False]
    assert w.row is BIG.rows[-1]
    assert sum("NOTEBOOK-ANALYSIS-BUDGET" in str(c.message) for c in caught) == 1, [str(c.message) for c in caught]
