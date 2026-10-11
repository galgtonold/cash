"""Names sharing an object are still linked when the share check runs out of time.

The share check after a statement stops at a time budget (twice the
statement's cost, at least a second); past it the statement re-runs each
time. Which other variables hold its outputs is then not known from that
check, so they are looked for from the outputs up their referrers instead,
and a later change through either name still moves the other. When even
that is cut short, NOTEBOOK-SHARE-UNCHECKED says so.
"""

from __future__ import annotations

import warnings

import pytest

from cash.notebook import shared_objects
from cash.notebook.statement import output_refusals, processor
from tests._cell_driver import run_cash_cell


@pytest.fixture
def no_budget(monkeypatch):
    """The share check runs out of time on every statement, however small."""

    def spent(*_args, **_kwargs):
        raise shared_objects.WalkBudgetExceeded

    monkeypatch.setattr(processor, "share_group", spent)
    monkeypatch.setattr(output_refusals, "share_group", spent)
    # The search that stands in has time to finish on a loaded machine.
    monkeypatch.setattr(processor, "SHARE_FALLBACK_FLOOR_S", 30.0)


@pytest.mark.parametrize(
    "link, change, other",
    [
        ("data = raw", "raw.append(4)", "data"),
        ("config = {'deep': {'features': raw}}", "raw.append(4)", "config"),
        ("config = {'deep': {'features': raw}}", "config['deep']['features'].append(4)", "raw"),
        ("item = rows[1]", "item.append(4)", "rows"),
    ],
    ids=["alias", "held deep in a dict", "changed deep through the dict", "taken out of a list"],
)
def test_a_change_moves_the_other_name_past_the_budget(cash_magics, no_budget, link, change, other):
    cells = ["raw = [1]\nrows = [[1], [2]]", link, change]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells[:2]:
            run_cash_cell(cash_magics, cell, cells=cells)
        before = cash_magics.tracking_state.variable_lineage[other]
        run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.tracking_state.variable_lineage[other] != before
    assert not [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-SHARE-UNCHECKED"]


def test_the_climb_finds_a_holder_and_what_the_root_holds():
    inner = [1]
    root = {"x": inner}
    ns = {"root": root, "inner": inner, "outer": {"a": [root]}, "other": [2]}
    names, complete = shared_objects.referring_names([root], ns, 5.0, skip_name=lambda n: n == "root")
    assert complete
    assert names == {"inner", "outer"}


def test_a_spent_budget_is_reported_incomplete():
    root = [[i] for i in range(5000)]
    names, complete = shared_objects.referring_names([root], {"root": root}, 0.0)
    assert not complete


def test_an_incomplete_search_warns_once(cash_magics, no_budget, monkeypatch):
    monkeypatch.setattr(shared_objects, "_REFERRER_DEPTH", 0)
    cells = ["raw = [1]", "data = raw"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
        run_cash_cell(cash_magics, cells[1], cells=cells)
    codes = [getattr(w.message, "code", None) for w in caught]
    assert codes.count("NOTEBOOK-SHARE-UNCHECKED") == 1
