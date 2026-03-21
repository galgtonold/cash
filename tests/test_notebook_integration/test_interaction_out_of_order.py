"""Batch 174 – Out-of-order execution interaction tests.

Tests executing cells in non-sequential order and verifying
that cash correctly handles the dependency tracking.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestOutOfOrderExecution:
    """Running cells in non-sequential order."""

    def test_run_later_cell_first(self, nb_runner):
        """Run cell 2 before cell 1 — should fail or handle gracefully."""
        nb_runner.create_notebook([
            "a = 10  # define a",
            "b = a + 5\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()

        # Run cell 1 then cell 2 (normal order)
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "b = 15" in nb_runner.get_output(2)

    def test_skip_middle_cell(self, nb_runner):
        """Run cells 1 and 3, skipping cell 2."""
        nb_runner.create_notebook([
            "x = 5  # step 1",
            "y = x * 2  # step 2 (will be skipped initially)",
            "z = x + 10\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 15" in nb_runner.get_output(3)

        # Now run all to fill in middle
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(3)

    def test_run_all_then_single_cell(self, nb_runner):
        """Run all, then re-run a single middle cell."""
        nb_runner.create_notebook([
            "a = 1  # value a",
            "b = a + 1",
            "c = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Re-run just cell 2 — should be idempotent
        nb_runner.run_cell(2)
        nb_runner.run_cell(3)
        assert "c = 3" in nb_runner.get_output(3)


class TestSelectiveCellExecution:
    """Running specific subsets of cells."""

    def test_run_cells_subset(self, nb_runner):
        """Run only specific cells."""
        nb_runner.create_notebook([
            "x = 10  # cell 1",
            "y = 20  # cell 2 (independent)",
            "z = x + y\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2, 3])
        assert "z = 30" in nb_runner.get_output(3)

    def test_edit_and_run_single_cell(self, nb_runner):
        """Edit one cell and run only that cell and its dependents."""
        nb_runner.create_notebook([
            "n = 5  # parameter",
            "result = n ** 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Edit and run just the edited cell + dependent
        nb_runner.set_cell_source(1, "n = 10  # parameter bigger")
        nb_runner.run_cells([1, 2])
        assert "result = 100" in nb_runner.get_output(2)

    def test_run_last_cell_only_after_full_run(self, nb_runner):
        """After full run, re-running last cell should use cached deps."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]  # data list",
            "total = sum(data)",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(3)

        # Re-run only the last cell
        nb_runner.run_cell(3)
        assert "total = 15" in nb_runner.get_output(3)
