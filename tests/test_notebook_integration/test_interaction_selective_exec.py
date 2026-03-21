"""Batch 132 – Selective cell execution interaction tests.

Tests that exercise running specific cells out of order,
skipping cells, and running only subsets of cells.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestSelectiveCellExecution:
    """Run only specific cells."""

    def test_run_only_first_and_last(self, nb_runner):
        """Run first cell, then jump to last cell (skip middle).
        Upstream should detect and re-execute middle cells."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # Skip cell 2, run cell 3 — upstream should ensure y exists
        nb_runner.run_cell(3)
        assert "z = 21" in nb_runner.get_output(3)

    def test_run_cells_in_reverse(self, nb_runner):
        """Run cells in reverse order."""
        nb_runner.create_notebook([
            "a = 5",
            "b = a + 1",
            "c = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        # Run in reverse: cell 3 first (upstream should find and run 1, 2)
        nb_runner.run_cell(3)
        assert "c = 7" in nb_runner.get_output(3)

    def test_run_middle_only(self, nb_runner):
        """Run only a middle cell."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2\nprint(f'y = {y}')",
            "z = y + 1",
        ])
        nb_runner.start_kernel()
        # Run only cell 2 — upstream should ensure x exists
        nb_runner.run_cell(2)
        assert "y = 84" in nb_runner.get_output(2)


class TestRerunAfterEdit:
    """Re-run specific cells after editing."""

    def test_edit_and_rerun_single(self, nb_runner):
        """Edit a cell, re-run only that cell."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100")
        # Only re-run cell 2 — upstream should detect x changed
        nb_runner.run_cell(2)
        assert "y = 101" in nb_runner.get_output(2)

    def test_edit_middle_rerun_last(self, nb_runner):
        """Edit middle cell, re-run only last cell."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_cell(3)
        assert "c = 11" in nb_runner.get_output(3)


class TestPartialExecution:
    """Execute parts of a notebook."""

    def test_run_first_three_then_add_fourth(self, nb_runner):
        """Run first 3 cells, then add and run a 4th."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 5",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2, 3])

        # Add a 4th cell and run it
        nb_runner.set_cell_source(3, "z = y + 5\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 25" in nb_runner.get_output(3)

    def test_run_subset_edit_run_rest(self, nb_runner):
        """Run subset, edit, run remaining."""
        nb_runner.create_notebook([
            "a = 5",
            "b = a + 1",
            "c = b * 2",
            "d = c + 3\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2])

        # Edit cell 1 before running rest
        nb_runner.set_cell_source(1, "a = 50")
        nb_runner.run_cells([3, 4])
        # Upstream should detect a changed and re-execute
        assert "d = " in nb_runner.get_output(4)


class TestRepeatExecution:
    """Execute cells multiple times."""

    def test_run_same_cell_twice(self, nb_runner):
        """Run the same cell twice — idempotent result."""
        nb_runner.create_notebook([
            "x = 42",
            "result = x * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        # Run cell 2 again (should be cached/skipped)
        nb_runner.run_cell(2)
        assert "result = 84" in nb_runner.get_output(2)

    def test_run_all_twice(self, nb_runner):
        """Run all cells twice — should be idempotent."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)
