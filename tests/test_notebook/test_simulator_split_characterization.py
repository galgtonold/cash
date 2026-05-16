"""Characterization tests pinning the NotebookSimulator boundary contract.

These tests exist solely to detect behavioral drift during the simulator
split (docs/superpowers/plans/2026-05-16-notebook-simulator-split.md).
They MUST stay green through every extraction task. Do not weaken them
to make a refactor easier — fix the refactor instead.
"""
from __future__ import annotations

import copy
import pytest
from unittest.mock import MagicMock

from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


# ---------------------------------------------------------------------------
# Shared fixture (defined locally — magics_fixture has no shared conftest)
# ---------------------------------------------------------------------------

class MockShell(Configurable):
    """Minimal mock IPython shell."""

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def magics_fixture():
    """Provide CashMagics + shell + backend (3-tuple, matching repo convention)."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _snapshot_tracking_state(simulator):
    """Deep-copy the parts of tracking_state the simulator can write."""
    state = simulator  # simulator *is* the TrackingState proxy (shares dicts)
    return {
        "variable_lineage": copy.deepcopy(simulator.variable_lineage),
        "executed_cell_codes": copy.deepcopy(simulator.executed_cell_codes),
        "executed_cell_hashes": copy.deepcopy(simulator.executed_cell_hashes),
        "executed_input_lineages": copy.deepcopy(simulator.executed_input_lineages),
    }


# ---------------------------------------------------------------------------
# Characterization tests
# ---------------------------------------------------------------------------

def test_clean_notebook_no_changes_returns_empty_plan(magics_fixture):
    """A notebook with no modifications produces no re-execution work."""
    magics, shell, _backend = magics_fixture
    magics.cash("", "x = 1")
    magics.cash("", "y = x + 1")

    simulator = magics._upstream_checker.simulator
    before = _snapshot_tracking_state(simulator)

    stmts, restored, total_t = simulator._simulate_and_find_changes(
        current_cell_idx=2,
        notebook_cells=["x = 1", "y = x + 1", "z = y"],
        required_inputs={"y"},
        current_cell_outputs={"z"},
    )

    assert stmts == []
    assert restored == []
    assert total_t == 0.0
    after = _snapshot_tracking_state(simulator)
    assert after == before, "clean simulation must not mutate tracking_state"


def test_modified_upstream_cell_schedules_reexecution(magics_fixture):
    """Editing an upstream cell flags dependent stmts for re-run."""
    magics, shell, _backend = magics_fixture
    magics.cash("", "x = 1")
    magics.cash("", "y = x + 1")

    simulator = magics._upstream_checker.simulator

    # Present a modified version of cell 0 to the simulator so it detects
    # a code change.
    stmts, restored, _t = simulator._simulate_and_find_changes(
        current_cell_idx=2,
        notebook_cells=["x = 99", "y = x + 1", "z = y"],
        required_inputs={"y"},
        current_cell_outputs={"z"},
    )

    # At least one of the upstream statements must be scheduled.
    assert len(stmts) > 0, (
        f"expected upstream stmt to be re-scheduled, got {stmts!r}"
    )
    # The modified statement itself (or its dependent) must appear.
    assert any("x" in s for s in stmts), (
        f"expected a statement involving 'x' in re-execution plan, got {stmts!r}"
    )


def test_simulate_and_find_changes_return_types(magics_fixture):
    """_simulate_and_find_changes always returns (list, list, float)."""
    magics, shell, _backend = magics_fixture
    magics.cash("", "data = [1, 2, 3]")

    simulator = magics._upstream_checker.simulator

    stmts, restored, t = simulator._simulate_and_find_changes(
        current_cell_idx=1,
        notebook_cells=["data = [1, 2, 3]", "x = data[0]"],
        required_inputs={"data"},
        current_cell_outputs={"x"},
    )

    # Return-type contract must hold regardless of outcome.
    assert isinstance(stmts, list), f"stmts must be list, got {type(stmts)}"
    assert isinstance(restored, list), f"restored must be list, got {type(restored)}"
    assert isinstance(t, float), f"total_t must be float, got {type(t)}"
    assert t >= 0.0, "restore time must be non-negative"


def test_reset_caches_clears_simulator_state(magics_fixture):
    """reset_caches() empties all three simulator-owned caches."""
    magics, shell, _backend = magics_fixture
    magics.cash("", "x = 1")

    simulator = magics._upstream_checker.simulator

    # Warm up the simulator so caches are populated.
    simulator._simulate_and_find_changes(
        current_cell_idx=1,
        notebook_cells=["x = 1", "y = x"],
        required_inputs={"x"},
        current_cell_outputs={"y"},
    )

    simulator.reset_caches()

    assert simulator._simulation_cache == []
    assert simulator._simulation_cell_hashes == {}
    assert simulator._ast_cache == {}
