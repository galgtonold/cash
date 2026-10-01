"""Characterization tests pinning the NotebookSimulator boundary contract.

These tests exist solely to detect behavioral drift during the simulator
split (docs/superpowers/plans/2026-05-16-notebook-simulator-split.md).
They MUST stay green through every extraction task. Do not weaken them
to make a refactor easier — fix the refactor instead.
"""

from __future__ import annotations

import copy

from cash.analysis.mutation_effects import CellEffects
from tests._cell_driver import run_cash_cell

# ---------------------------------------------------------------------------
# Shared fixture (defined locally — magics_fixture has no shared conftest)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _snapshot_tracking_state(simulator):
    """Deep-copy the TrackingState fields the simulator can write.

    The simulator writes through its ``tracking_state``.
    """
    return {
        "variable_lineage": dict(simulator.tracking_state.variable_lineage),
        "executed_cell_codes": copy.deepcopy(simulator.tracking_state.executed_cell_codes),
        "executed_cell_hashes": copy.deepcopy(simulator.tracking_state.executed_cell_hashes),
        "executed_input_lineages": copy.deepcopy(simulator.tracking_state.executed_input_lineages),
    }


# ---------------------------------------------------------------------------
# Characterization tests
# ---------------------------------------------------------------------------


def test_clean_notebook_no_changes_returns_empty_plan(cash_magics):
    """A notebook with no modifications produces no re-execution work."""
    run_cash_cell(cash_magics, "x = 1")
    run_cash_cell(cash_magics, "y = x + 1")

    simulator = cash_magics._upstream_checker.simulator
    before = _snapshot_tracking_state(simulator)

    stmts, restored, total_t = simulator.simulate_upstream(
        current_cell_idx=2,
        notebook_cells=["x = 1", "y = x + 1", "z = y"],
        required_inputs={"y"},
        effects=CellEffects(outputs=frozenset({"z"})),
    )

    assert stmts == []
    assert restored == []
    assert total_t == 0.0
    after = _snapshot_tracking_state(simulator)
    assert after == before, "clean simulation must not mutate tracking_state"


def test_modified_upstream_cell_schedules_reexecution(cash_magics):
    """Editing an upstream cell flags dependent stmts for re-run."""
    run_cash_cell(cash_magics, "x = 1")
    run_cash_cell(cash_magics, "y = x + 1")

    simulator = cash_magics._upstream_checker.simulator

    # Present a modified version of cell 0 to the simulator so it detects
    # a code change.
    stmts, restored, _t = simulator.simulate_upstream(
        current_cell_idx=2,
        notebook_cells=["x = 99", "y = x + 1", "z = y"],
        required_inputs={"y"},
        effects=CellEffects(outputs=frozenset({"z"})),
    )

    # At least one of the upstream statements must be scheduled.
    assert len(stmts) > 0, f"expected upstream stmt to be re-scheduled, got {stmts!r}"
    # The modified statement itself (or its dependent) must appear.
    assert any("x" in s for s in stmts), f"expected a statement involving 'x' in re-execution plan, got {stmts!r}"


def test_simulate_upstream_return_types(cash_magics):
    """simulate_upstream always returns (list, list, float)."""
    run_cash_cell(cash_magics, "data = [1, 2, 3]")

    simulator = cash_magics._upstream_checker.simulator

    stmts, restored, t = simulator.simulate_upstream(
        current_cell_idx=1,
        notebook_cells=["data = [1, 2, 3]", "x = data[0]"],
        required_inputs={"data"},
        effects=CellEffects(outputs=frozenset({"x"})),
    )

    # No upstream modification — early-exit returns the empty triple.
    assert stmts == []
    assert restored == []
    assert t == 0.0


def test_reset_caches_clears_simulator_state(cash_magics):
    """reset_caches() empties the simulator-owned caches."""
    run_cash_cell(cash_magics, "x = 1")

    simulator = cash_magics._upstream_checker.simulator

    # Warm up the simulator so caches are populated.
    simulator.simulate_upstream(
        current_cell_idx=1,
        notebook_cells=["x = 1", "y = x"],
        required_inputs={"x"},
        effects=CellEffects(outputs=frozenset({"y"})),
    )

    simulator.reset_caches()

    assert simulator.cache.entries == []
    assert simulator.cache.cell_hashes == {}
