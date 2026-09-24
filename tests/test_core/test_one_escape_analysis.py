"""The decorator's escape analysis is the flow-sensitive one alone.

Two analyses decided whether a mutated local was the function's own object:
a flow-insensitive pass in ``purity_analyzer`` (a name bound only to fresh
allocations anywhere) and the flow-sensitive ``purity_flow`` pass, with
their answers ORed and their tables of "returns a new object" spellings
copied from one file to the other. The flow pass decides every case the
other did, so it is the only one; these pin the cases, both ways.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cash.analysis.purity_analyzer import ISSUE_SCOPE_MUTATION, PurityAnalyzer

SRC = Path(__file__).resolve().parents[2] / "src" / "cash"

GLOBAL_ROWS: list = []


def accumulates_in_a_loop(r):
    rows = []
    for i in r:
        rows.append(i)
    return rows


def fills_a_dict_in_a_while(n):
    d = {}
    while n:
        d[n] = n
        n -= 1
    return d


def mutates_through_an_alias():
    x = []
    y = x
    y.append(1)
    return y


def fresh_on_both_branches(c):
    if c:
        x = []
    else:
        x = {}
    x.clear()
    return x


def sorts_a_copy(rows):
    s = sorted(rows)
    s.reverse()
    return s


def appends_to_a_parameter(rows):
    rows.append(1)
    return rows


def rebinds_to_a_parameter(rows):
    x = []
    x = rows
    x.append(1)
    return x


def appends_to_a_global():
    GLOBAL_ROWS.append(1)


def appends_to_each_element(groups):
    x = []
    for x in groups:
        x.append(1)


@pytest.mark.parametrize(
    "fn",
    [accumulates_in_a_loop, fills_a_dict_in_a_while, mutates_through_an_alias, fresh_on_both_branches, sorts_a_copy],
    ids=lambda f: f.__name__,
)
def test_a_local_the_function_made_is_its_own(fn):
    kinds = [i.kind for i in PurityAnalyzer().analyze(fn).issues]
    assert ISSUE_SCOPE_MUTATION not in kinds, kinds


@pytest.mark.parametrize(
    "fn",
    [appends_to_a_parameter, rebinds_to_a_parameter, appends_to_a_global, appends_to_each_element],
    ids=lambda f: f.__name__,
)
def test_someone_elses_object_is_still_reported(fn):
    issues = PurityAnalyzer().analyze(fn).issues
    assert issues, "a mutation of an object the function did not make went unreported"


def test_the_fresh_allocation_tables_live_in_one_place():
    text = (SRC / "analysis" / "purity_analyzer.py").read_text(encoding="utf-8")
    assert not re.search(r"^_FRESH_\w+\s*[:=]", text, re.MULTILINE)
