"""A value with no lineage is walked once per statement, and again after
anything may have changed it in place.

``records = [Rec(i) for i in range(1_500_000)]``, then the cell defining
``Rec`` runs again: ``records`` loses its lineage (its objects are of the
old class), so nothing remembers what walking it for the notebook
functions it holds found (`held_functions.held_notebook_code`). The
upstream check asked twice, the reads log, the key and the checks before
the statement runs once each: six walks of every object for ``n =
len(records)``, 5.6 s where plain Python takes 2 ms. The walks before one
statement now share the first one's answer. Code that runs between them (the
statement before, a loop's header, a statement the upstream check runs
again) is seen: the share ends where it runs.
"""

from __future__ import annotations

import pytest

from cash.notebook import held_functions
from cash.notebook.upstream.checker import UpstreamChecker, UpstreamResult
from tests._cell_driver import run_cash_cell

CLASS = """
class Rec:
    def __init__(self, i):
        self.i = i
        self.x = i * 0.5
"""
BUILD = "records = [Rec(i) for i in range(2000)]"
HELPER = """
def scaled(v):
    return v * factor
"""


@pytest.fixture
def walks(monkeypatch, mock_shell):
    """What each walk of the value ``records`` names found: the names of
    the functions and classes it met, or None when it found none."""
    seen: list[set[str] | None] = []
    original = held_functions._walk

    def counting(value, namespace, check_at=None):
        try:
            found = original(value, namespace, check_at)
        except Exception:
            if value is mock_shell.user_ns.get("records"):
                seen.append(None)
            raise
        if value is mock_shell.user_ns.get("records"):
            seen.append({getattr(obj, "__name__", "?") for obj in (*found[0], *found[1])})
        return found

    monkeypatch.setattr(held_functions, "_walk", counting)
    return seen


@pytest.fixture
def unlineaged(cash_magics, mock_shell):
    """``records``, its class defined again since it was built: no lineage."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "factor = 3")
    run_cash_cell(cash_magics, HELPER)
    run_cash_cell(cash_magics, CLASS)
    run_cash_cell(cash_magics, BUILD)
    run_cash_cell(cash_magics, CLASS)
    assert "records" not in cash_magics.tracking_state.variable_lineage
    return cash_magics


def test_a_statement_reading_a_value_without_a_lineage_walks_it_once(unlineaged, walks, mock_shell):
    for k in range(3):
        before = len(walks)
        run_cash_cell(unlineaged, f"n{k} = len(records) + {k}")
        assert mock_shell.user_ns[f"n{k}"] == 2000 + k
        # Once for the cell's check and its statement together; six times before.
        assert len(walks) - before == 1, walks[before:]
    assert "records" not in unlineaged.tracking_state.variable_lineage


def test_a_change_in_place_by_the_statement_before_is_seen(unlineaged, walks, mock_shell):
    run_cash_cell(unlineaged, "records.append(scaled)\nn = len(records)\nv = records[-1](2)")
    assert mock_shell.user_ns["n"] == 2001
    assert mock_shell.user_ns["v"] == 6
    # The cell's check and its first statement walked it before the append;
    # the statements after it walk it again, and find the function.
    assert walks[0] is None or "scaled" not in walks[0]
    assert walks[-1] is not None and "scaled" in walks[-1], walks


def test_a_change_in_place_by_a_cell_before_is_seen(unlineaged, walks, mock_shell):
    run_cash_cell(unlineaged, "n0 = len(records)")
    run_cash_cell(unlineaged, "records.append(scaled)")
    before = len(walks)
    run_cash_cell(unlineaged, "n1 = len(records)")
    assert mock_shell.user_ns["n1"] == 2001
    assert len(walks) > before
    assert walks[-1] is not None and "scaled" in walks[-1], walks


def test_a_loop_header_that_changes_the_value_is_seen_by_its_body(unlineaged, walks, mock_shell):
    run_cash_cell(unlineaged, "for f in (records.append(scaled) or [1]):\n    m = len(records)")
    assert mock_shell.user_ns["m"] == 2001
    assert walks[-1] is not None and "scaled" in walks[-1], walks


def test_a_change_the_upstream_check_makes_in_place_is_seen(unlineaged, walks, mock_shell, monkeypatch):
    """What the check restores or runs again above the cell may change the
    value in place: the cell's statement then walks it again."""
    original = UpstreamChecker._bring_up_to_date

    def restoring(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # As a restore into the live list would.
        mock_shell.user_ns["records"].append(mock_shell.user_ns["scaled"])
        return UpstreamResult(result[0], result[1] + 1e-6, result[2])

    monkeypatch.setattr(UpstreamChecker, "_bring_up_to_date", restoring)
    run_cash_cell(unlineaged, "n = len(records)")
    assert mock_shell.user_ns["n"] == 2001
    assert len(walks) == 2, walks
    assert walks[-1] is not None and "scaled" in walks[-1], walks


def test_nothing_is_shared_or_kept_after_the_cell(unlineaged, walks):
    """The share holds the user's objects: kept, one would read as another
    holder to a later share check."""
    run_cash_cell(unlineaged, "n = len(records)")
    assert not held_functions._SHARED
    assert not held_functions._SHARING[0]
