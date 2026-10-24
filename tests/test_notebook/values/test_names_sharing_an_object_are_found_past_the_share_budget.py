"""Names sharing an object are still linked when the share check runs out of time.

The share check after a statement stops at a time budget (twice the
statement's cost, at least a second); past it the statement re-runs each
time. Which other variables hold its outputs is then not known from that
check, so they are looked for from the outputs, and what they hold, up their
referrers instead,
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
    # (SHARE_FALLBACK_FLOOR_S is bound as a default argument, so patching the
    # module constant would not reach it; the analysis budget caps it too.)
    monkeypatch.setattr(processor.StatementProcessor, "_fallback_seconds", lambda *_args: 30.0)


@pytest.mark.parametrize(
    "link, change, other",
    [
        ("data = raw", "raw.append(4)", "data"),
        ("config = {'deep': {'features': raw}}", "raw.append(4)", "config"),
        ("config = {'deep': {'features': raw}}", "config['deep']['features'].append(4)", "raw"),
        ("item = rows[1]", "item.append(4)", "rows"),
        ("head = rows[:1]", "head[0].append(4)", "rows"),
        ("head = recs[:1]", "head[0].x = 4", "recs"),
    ],
    ids=[
        "alias",
        "held deep in a dict",
        "changed deep through the dict",
        "taken out of a list",
        "a slice copy",
        "a slice copy of records",
    ],
)
def test_a_change_moves_the_other_name_past_the_budget(cash_magics, no_budget, link, change, other):
    cells = [
        "import argparse\nraw = [1]\nrows = [[1], [2]]\nrecs = [argparse.Namespace(x=i) for i in range(3)]",
        link,
        change,
    ]
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


def test_the_climb_finds_a_holder_of_what_the_root_holds():
    """``head = points[:2]``: nothing holds the new list, but ``points``
    holds its items."""
    points = [[i] for i in range(10)]
    head = points[:2]
    names, complete = shared_objects.referring_names([head], {"points": points, "head": head}, 5.0)
    assert complete
    assert "points" in names


@pytest.mark.parametrize(
    "link",
    ["head = recs[:100]", "head = list(recs)", "import copy\nhead = copy.copy(recs)"],
    ids=["a slice copy of many records", "a list of them", "a copy of the list"],
)
def test_a_search_cut_short_links_the_output_to_what_it_read(cash_magics, no_budget, monkeypatch, link):
    """The search from the outputs up is cut short, finding nothing: ``head``
    may hold what ``recs`` holds, and a change through it still moves ``recs``."""
    monkeypatch.setattr(processor, "referring_names", lambda *_args, **_kwargs: (set(), False))
    cells = ["import argparse\nrecs = [argparse.Namespace(x=i) for i in range(300)]", link, "head[0].x = 4"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells[:2]:
            run_cash_cell(cash_magics, cell, cells=cells)
        before = cash_magics.tracking_state.variable_lineage["recs"]
        run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.tracking_state.variable_lineage["recs"] != before
    assert [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-SHARE-UNCHECKED"]


def test_a_share_check_cut_short_links_the_output_to_what_it_read(cash_magics):
    """The share check itself gives up (``head = recs[:65]``: more records
    than one climb asks about at once, in a list too wide to read down):
    ``head`` is still linked to ``recs``, and a cached read of ``recs``
    below sees a change through ``head``."""
    n = shared_objects._DOWN_WIDEST + 1000
    cells = [
        f"import argparse\nrecs = [argparse.Namespace(x=i) for i in range({n})]",
        f"head = recs[:{shared_objects._MAX_REFERRER_TARGETS + 1}]",
        "s = sum(r.x for r in recs)",
        "head[7].x = -3",
        "s = sum(r.x for r in recs)  # after",
    ]
    cash_magics.cash_on("")
    cash_magics.cash_persist("on")
    cash_magics.badges.mode = "off"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["s"] == sum(range(n)) - 10
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["s"] == sum(range(n)) - 10


def test_a_climb_past_its_object_limit_is_reported_incomplete(monkeypatch):
    monkeypatch.setattr(shared_objects, "_REFERRER_OBJECTS", 100)
    points = [[i] for i in range(1000)]
    head = points[:]
    names, complete = shared_objects.referring_names([head], {"points": points, "head": head}, 30.0)
    assert not complete


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
    warned = [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-SHARE-UNCHECKED"]
    # At the cell, as cash's other notebook warnings are, not at ipykernel's frame.
    assert warned[0].filename == "<cash>", warned[0].filename


def test_the_shell_is_left_out_of_the_check(cash_magics, no_budget, monkeypatch):
    """``ip = get_ipython()`` binds an object that holds the whole session:
    the check ran out of time on it and warned on a harmless line."""
    monkeypatch.setattr(shared_objects, "_REFERRER_DEPTH", 0)
    shell = cash_magics.shell
    shell.user_ns["get_ipython"] = lambda: shell
    cells = ["ip = get_ipython()", "same = ip"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert shell.user_ns["same"] is shell
    assert not [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-SHARE-UNCHECKED"]
