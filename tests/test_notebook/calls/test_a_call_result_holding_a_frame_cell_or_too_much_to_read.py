"""A call's result is checked against what its arguments hold in a frame's
object column, and that check has a budget.

``mk(df)`` returning ``Namespace(x=df['d'].iloc[0])`` hands back the very
list the frame's object column holds: a stored copy would no longer be
it. Neither side's walk read a frame's cells. Read all, a million-row
frame cost seconds in every share check: only the cells something else
holds too are read. ``flt(df)`` returning ``df[mask]`` holds ``df``'s own
cells: a hit puts them back. When something else holds them too, the call
runs every time and says so.

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
INNER = [1]
NESTED = pd.DataFrame({"d": [[INNER], [2]]})  # the cell holding INNER is the frame's alone
BIG = argparse.Namespace(rows=[{"i": i} for i in range(300_000)])
TOKENS = pd.DataFrame({"tokens": [["a", str(i)] for i in range(1000)], "n": range(1000)})


def cell(d):
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return argparse.Namespace(x=d["d"].iloc[0])


def inner(d):
    time.sleep(0.05)
    return argparse.Namespace(x=d["d"].iloc[0][0])


def fresh(src):
    time.sleep(0.05)
    return argparse.Namespace(rows=[{"i": i} for i in range(1000)], n=len(src.rows))


def part(src):
    time.sleep(0.12)  # over the compute floor: a refusal it pays for is told
    return argparse.Namespace(row=src.rows[-1])


def flt(d):
    time.sleep(0.12)
    return d[d["n"] % 2 == 0]


def tokens(n):
    time.sleep(0.05)
    return pd.DataFrame({"tokens": [["a", str(i)] for i in range(n)]})


def _twice(call_unit_harness, fn, arg, name):
    """Two statements calling *fn*, neither of them stored itself: a call
    refused for the cells it shares then says so."""
    unit = call_unit_harness(lineage={name: "hash-" + name, "arg": "hash-arg"}, user_ns={"arg": arg})
    site = CallSite(source=f"{name}(arg)", free_names=frozenset({name, "arg"}), occurrence_index=0)
    unit.begin_statement()
    unit.wrap(fn, site)(arg)
    unit.settle_cells_shared(stored=False)
    unit.begin_statement()
    result = unit.wrap(fn, site)(arg)
    unit.settle_cells_shared(stored=False)
    return [e["cache_hit"] for e in unit.drain()], result


def test_a_frame_cell_the_result_holds_stays_the_frames(call_unit_harness):
    hits, w = _twice(call_unit_harness, cell, DF, "cell")
    assert hits == [False, False]
    assert w.x is RECT


def test_an_object_inside_a_cell_only_the_frame_holds_stays_the_frames(call_unit_harness):
    hits, w = _twice(call_unit_harness, inner, NESTED, "inner")
    assert hits == [False, False]
    assert w.x is INNER


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


def test_a_result_holding_an_arguments_frame_cells_is_served_holding_them(call_unit_harness):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hits, w = _twice(call_unit_harness, flt, TOKENS, "flt")
    assert hits == [False, True]
    assert w["tokens"].iloc[1] is TOKENS["tokens"].iloc[2]
    assert w["tokens"].iloc[499] is TOKENS["tokens"].iloc[998]
    assert w["n"].tolist() == list(range(0, 1000, 2))
    assert not [str(c.message) for c in caught if "NOTEBOOK-CELLS-SHARED" in str(c.message)]


def test_a_result_holding_cells_something_else_holds_too_runs_every_time_and_says_so(call_unit_harness):
    held = TOKENS["tokens"].tolist()  # the cells, held outside the frames too
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hits, w = _twice(call_unit_harness, flt, TOKENS, "flt")
    assert hits == [False, False]
    assert w["tokens"].iloc[1] is held[2]
    told = [str(c.message) for c in caught if "NOTEBOOK-CELLS-SHARED" in str(c.message)]
    assert len(told) == 1, [str(c.message) for c in caught]
    assert "shares cells with argument d" in told[0]


def test_a_fresh_frame_of_lists_is_served(call_unit_harness):
    hits, w = _twice(call_unit_harness, tokens, 1000, "tokens")
    assert hits == [False, True]
    assert w["tokens"].iloc[7] == ["a", "7"]


#: Cells holding a list only the frame holds: more than values, read by the
#: share check, and still put back.
BOXED = pd.DataFrame({"b": [[[i]] for i in range(200)], "n": range(200)})


def pair(d):
    time.sleep(0.12)
    return d[d["n"] % 2 == 0], len(d)


def test_a_result_holding_cells_that_hold_more_than_values_is_served_holding_them(call_unit_harness):
    hits, w = _twice(call_unit_harness, flt, BOXED, "flt")
    assert hits == [False, True]
    assert w["b"].iloc[3] is BOXED["b"].iloc[6]
    assert w["b"].iloc[3][0] is BOXED["b"].iloc[6][0]


def test_cells_are_put_back_in_a_frame_a_tuple_holds_and_from_a_keyword_argument(call_unit_harness):
    unit = call_unit_harness(lineage={"pair": "hash-pair", "arg": "hash-arg"}, user_ns={"arg": TOKENS})
    site = CallSite(source="pair(d=arg)", free_names=frozenset({"pair", "arg"}), occurrence_index=0)
    for _ in range(2):
        unit.begin_statement()
        w, n = unit.wrap(pair, site)(d=TOKENS)
    assert [e["cache_hit"] for e in unit.drain()] == [False, True]
    assert n == 1000
    assert w["tokens"].iloc[5] is TOKENS["tokens"].iloc[10]


def test_a_cell_that_cannot_be_put_back_runs_the_call(call_unit_harness, monkeypatch):
    from cash.notebook import call_unit, frame_cells

    monkeypatch.setattr(call_unit, "relink_cells", lambda *args: frame_cells.RELINK_FAILED)
    _hits, w = _twice(call_unit_harness, flt, TOKENS, "flt")
    assert w["tokens"].iloc[1] is TOKENS["tokens"].iloc[2]
