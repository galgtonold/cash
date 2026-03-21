"""
Test that upstream execution works for unsaved cells (not yet on disk).

Bug: When a user creates a new cell and executes it without saving the notebook,
the cell isn't in the .ipynb file on disk. The upstream checker couldn't find it,
returned None, and skipped the entire upstream check — leaving required variables
unresolved (NameError).

Fix: When the current cell can't be found in the notebook but has required inputs
missing from memory, treat ALL saved notebook cells as upstream and run the full
simulation. This works because an unsaved cell must logically come after everything
already saved to disk.
"""

import pytest

pytestmark = pytest.mark.upstream


class TestUnsavedCellUpstream:
    """Verify upstream resolution works for cells not yet saved to disk."""

    def test_unsaved_cell_resolves_upstream_variable(self, nb_runner):
        """
        Simulate an unsaved cell scenario by:
        1. Creating a notebook with cells defining a variable
        2. Running all cells to populate cache
        3. Restarting kernel (clearing memory)
        4. Appending a new cell that uses the variable (simulates unsaved)
        5. Running just the new cell — it should resolve upstream deps
        """
        nb_runner.create_notebook([
            # Cell 1: Define a class and create instance
            (
                "class Test:\n"
                "    def __init__(self):\n"
                "        self.x = 1\n"
                "        self.y = 2\n"
                "\n"
                "a = Test()\n"
                "a.x = 126"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Now add a new cell that uses 'a' (this simulates the unsaved cell scenario)
        # After reset, memory is cleared but the notebook file has cell 1
        nb_runner.reset_cash_state()
        nb_runner.add_cell("print(a.x)")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "126" in output, f"Expected '126' in output, got: {output}"

    def test_unsaved_cell_with_simple_variable(self, nb_runner):
        """
        Simple case: saved cells define x=42, unsaved cell prints x.
        """
        nb_runner.create_notebook([
            "x = 42",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        nb_runner.reset_cash_state()
        nb_runner.add_cell("print(f'x is {x}')")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "x is 42" in output, f"Expected 'x is 42' in output, got: {output}"

    def test_unsaved_cell_with_chain_dependency(self, nb_runner):
        """
        Unsaved cell needs y, which depends on x from an earlier saved cell.
        Both should be resolved through upstream execution.
        """
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 3",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        nb_runner.reset_cash_state()
        nb_runner.add_cell("print(f'y is {y}')")
        nb_runner.run_cell(3)
        output = nb_runner.get_output(3)
        assert "y is 30" in output, f"Expected 'y is 30' in output, got: {output}"

    def test_no_upstream_when_inputs_already_available(self, nb_runner):
        """
        If the unsaved cell's inputs are already in memory, no upstream
        execution should be needed (fast path).
        """
        nb_runner.create_notebook([
            "x = 99",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # DON'T reset — x is still in memory
        nb_runner.add_cell("print(f'x is {x}')")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "x is 99" in output, f"Expected 'x is 99' in output, got: {output}"
