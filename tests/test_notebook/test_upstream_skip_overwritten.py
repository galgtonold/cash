"""
Unit test: Backwards scan skips earlier definitions when variable is fully redefined.

Tests the specific backwards-scan optimization in UpstreamChecker._simulate_and_find_changes
where outputs of scheduled-for-execution statements are removed from needed_vars,
preventing unnecessary cascading to earlier statements.
"""
import pytest
from unittest.mock import MagicMock, patch

from cash.notebook.upstream import UpstreamChecker
from cash.notebook._protocols import TrackingState
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def upstream_checker():
    """Provide UpstreamChecker instance for testing."""
    backend = InMemoryBackend()
    cash_inst = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    checker = UpstreamChecker(shell, cash_instance=cash_inst, debug=True)
    checker.set_tracking_state(TrackingState())
    yield checker, shell, backend
    backend.clear()


class TestSkipOverwrittenVarUnit:
    """Unit tests for the backwards scan optimization."""

    @patch('cash.notebook.upstream.get_notebook_cells')
    @patch('cash.notebook.upstream.get_notebook_cells_with_ids')
    def test_fully_redefined_var_stops_cascade(self, mock_cells_ids, mock_cells, upstream_checker):
        """
        If x is produced by stmt A then fully redefined by stmt B,
        only stmt B (and its deps) should be scheduled, not stmt A.

        Notebook:
          Cell 0: x = 'first'        (earlier definition - should be SKIPPED)
          Cell 1: x = 'second'       (full redefinition - should be scheduled)
          Cell 2: x                   (current cell needing x)
        """
        checker, shell, backend = upstream_checker

        cells = [
            "x = 'first'\nprint('cell0_ran')",
            "x = 'second'",
            "x",
        ]
        mock_cells.return_value = cells
        mock_cells_ids.return_value = [
            (f"cell_{i}", cell) for i, cell in enumerate(cells)
        ]

        # x is required but missing from memory (simulating kernel restart)
        result, restored_info, restore_time = checker.simulator._simulate_and_find_changes(
            current_cell_idx=2,
            notebook_cells=cells,
            required_inputs={'x'},
            current_cell_outputs=set(),
        )

        # Only the second definition should be scheduled (not the first)
        assert len(result) == 1, (
            f"Expected 1 statement to execute, got {len(result)}: {result}"
        )
        assert "x = 'second'" in result[0], (
            f"Expected 'x = second' to be scheduled, got: {result}"
        )
        # The first definition (with print) should NOT be scheduled
        assert not any("first" in s for s in result), (
            f"Earlier definition should NOT be scheduled: {result}"
        )

    @patch('cash.notebook.upstream.get_notebook_cells')
    @patch('cash.notebook.upstream.get_notebook_cells_with_ids')
    def test_mutation_still_needs_earlier_def(self, mock_cells_ids, mock_cells, upstream_checker):
        """
        If x is first defined, then mutated (x['a'] = ...), the mutation
        needs x as input so the definition should still be scheduled.

        Notebook:
          Cell 0: x = {'b': 1}       (definition - NEEDED because mutation depends on it)
          Cell 1: x['a'] = 10        (mutation of x - x is both input and output)
          Cell 2: x                   (current cell)
        """
        checker, shell, backend = upstream_checker

        cells = [
            "x = {'b': 1}",
            "x['a'] = 10",
            "x",
        ]
        mock_cells.return_value = cells
        mock_cells_ids.return_value = [
            (f"cell_{i}", cell) for i, cell in enumerate(cells)
        ]

        result, restored_info, restore_time = checker.simulator._simulate_and_find_changes(
            current_cell_idx=2,
            notebook_cells=cells,
            required_inputs={'x'},
            current_cell_outputs=set(),
        )

        # Both statements should be scheduled: definition AND mutation
        assert len(result) == 2, (
            f"Expected 2 statements (definition + mutation), got {len(result)}: {result}"
        )

    @patch('cash.notebook.upstream.get_notebook_cells')
    @patch('cash.notebook.upstream.get_notebook_cells_with_ids')
    def test_try_except_skipped_when_later_full_redef(self, mock_cells_ids, mock_cells, upstream_checker):
        """
        Reproduces the exact bug from the issue:
        - try/except defines x (with side effects)
        - later x = {'b': 123} fully redefines it
        - even later x['a'] = f(0) mutates it

        Only the chain from x = {'b': 123} onwards should execute.
        """
        checker, shell, backend = upstream_checker

        cells = [
            # Cell 0: try/except that defines x (should be SKIPPED)
            (
                "try:\n"
                "    print('hi')\n"
                "    asdf = 235\n"
                "    raise ValueError('test')\n"
                "    x = 123\n"
                "except ValueError as e:\n"
                "    print(f'error: {e}')"
            ),
            # Cell 1: Define c
            "c = 122",
            # Cell 2: Define f and fully redefine x, then mutate x
            (
                "def f(a):\n"
                "    return c + a\n"
                "\n"
                "x = {'b': 123}\n"
                "x['a'] = f(0)"
            ),
            # Cell 3: Use x (current cell)
            "x",
        ]
        mock_cells.return_value = cells
        mock_cells_ids.return_value = [
            (f"cell_{i}", cell) for i, cell in enumerate(cells)
        ]

        result, restored_info, restore_time = checker.simulator._simulate_and_find_changes(
            current_cell_idx=3,
            notebook_cells=cells,
            required_inputs={'x'},
            current_cell_outputs=set(),
        )

        # The try/except block should NOT be in the execution list
        for stmt in result:
            assert "ValueError" not in stmt, (
                f"try/except block should NOT be scheduled: {stmt}"
            )
            assert "asdf" not in stmt, (
                f"try/except block should NOT be scheduled: {stmt}"
            )

        # c = 122 and def f(a) and x = {'b': 123} and x['a'] = f(0) should be scheduled
        stmt_texts = " ".join(result)
        assert "c = 122" in stmt_texts, f"c = 122 should be scheduled, got: {result}"
        assert "def f(a)" in stmt_texts, f"def f(a) should be scheduled, got: {result}"
        assert "x = {'b': 123}" in stmt_texts, f"x = {{'b': 123}} should be scheduled, got: {result}"
