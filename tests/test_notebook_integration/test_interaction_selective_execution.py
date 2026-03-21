"""Batch 147 – Selective cell execution interaction tests.

Tests where users run cells out of order, skip cells,
run subsets of cells, and verify cache consistency.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(45)]


class TestRunSubsetOfCells:
    """Run only some cells, not all."""

    def test_run_first_and_last_skip_middle(self, nb_runner):
        """Run cells 1 and 3, skip cell 2."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2  # skipped on first pass",
            "z = x + 5\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 3])
        # z only depends on x, not y
        assert "z = 15" in nb_runner.get_output(3)

    def test_run_all_then_rerun_single_edited_cell(self, nb_runner):
        """Run all, edit one cell, run only that cell."""
        nb_runner.create_notebook([
            "a = 1",
            "b = 2",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Edit cell 1 and rerun only cell 3
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cells([1, 3])
        assert "c = 102" in nb_runner.get_output(3)

    def test_incremental_cell_execution(self, nb_runner):
        """Run cells one at a time."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 3",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        nb_runner.run_cell(3)
        assert "z = 16" in nb_runner.get_output(3)


class TestRerunAfterEdit:
    """Rerun specific cells after editing."""

    def test_edit_cell_rerun_only_downstream(self, nb_runner):
        """Edit cell 1, rerun only downstream cells."""
        nb_runner.create_notebook([
            "base = 10",
            "mid = base + 5",
            "final = mid * 2\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 30" in nb_runner.get_output(3)

        # Edit cell 1, then run cells 1-3
        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_cells([1, 2, 3])
        assert "final = 210" in nb_runner.get_output(3)

    def test_edit_middle_rerun_from_there(self, nb_runner):
        """Edit middle cell, rerun from middle to end."""
        nb_runner.create_notebook([
            "x = 3",
            "y = x + 7",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cells([2, 3])
        assert "z = 206" in nb_runner.get_output(3)


class TestRepeatedSingleCellRuns:
    """Run same cell repeatedly with edits."""

    def test_edit_and_rerun_same_cell_many_times(self, nb_runner):
        """Edit and rerun the same output cell."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        for multiplier in [2, 3, 5]:
            nb_runner.set_cell_source(
                2, f"result = x * {multiplier}\nprint(f'result = {{result}}')"
            )
            nb_runner.run_cell(2)
            assert f"result = {10 * multiplier}" in nb_runner.get_output(2)
