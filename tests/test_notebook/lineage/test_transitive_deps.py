"""
Test transitive dependency invalidation through multiple cells.

This tests the scenario:
1. Cell 1: a = 5
2. Cell 2: b = a + 3
3. Cell 3: c = b * 2

When we change Cell 1 to a = 7, Cell 3 should get updated value.
"""

import pytest

from tests._cell_driver import run_cash_cell


def test_three_cell_cascade():
    """Test that changing a affects b affects c across 3 cells.

    NOTE: This test is xfail because transitive upstream re-execution
    requires a real notebook file to read the upstream cells from, and
    ``run_cash_cell`` (used in unit tests) runs a cell with no notebook
    behind it, so there is nothing upstream to resolve.
    See test_notebook_integration/upstream/test_edit_chains.py for proper integration tests.
    """
    pytest.xfail(
        "Transitive re-execution requires _execute_cell + notebook file, "
        "not run_cash_cell. Covered by integration tests."
    )


def test_direct_dependency_invalidation(cash_magics, mock_shell):
    """Test that changing an input variable invalidates the cache."""

    # Cell 1: x = 10
    cell1 = "x = 10"
    run_cash_cell(cash_magics, cell1)
    assert mock_shell.user_ns["x"] == 10, "x should be 10"

    # Cell 2: y = x + 5 (should be 15)
    cell2 = "y = x + 5"
    run_cash_cell(cash_magics, cell2)
    assert mock_shell.user_ns["y"] == 15, "y should be 15 (10 + 5)"

    # Change x directly in namespace (simulating upstream change)
    mock_shell.user_ns["x"] = 20

    # Update the lineage hash for x to reflect the change
    if hasattr(cash_magics, "tracking_state") and "x" in cash_magics.tracking_state.variable_lineage:
        # Force lineage change by removing x's lineage (simulating new value)
        cash_magics.tracking_state.lineage.discard("x")

    # Re-run Cell 2: y should be recalculated to 25
    run_cash_cell(cash_magics, cell2)
    assert mock_shell.user_ns["y"] == 25, "y should be recalculated to 25 after x changed to 20"
