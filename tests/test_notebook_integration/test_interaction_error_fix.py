"""Batch 146 – Error handling and recovery interaction tests.

Tests where code errors occur, user fixes them, and caching
should properly handle the error-recovery workflow.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestErrorThenFixCells:
    """Introduce error, then fix it."""

    def test_name_error_then_fix(self, nb_runner):
        """Cell causes NameError, fix it, run again."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x + undefined_var\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix the cell
        nb_runner.set_cell_source(2, "result = x + 5\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_type_error_then_fix(self, nb_runner):
        """Cell causes TypeError, fix it."""
        nb_runner.create_notebook([
            "val = 'hello'",
            "result = val + 10\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "result = val + str(10)\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        assert "result = hello10" in nb_runner.get_output(2)

    def test_index_error_then_fix(self, nb_runner):
        """Cell causes IndexError, fix it."""
        nb_runner.create_notebook([
            "items = [1, 2, 3]",
            "val = items[10]\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(2, "val = items[2]\nprint(f'val = {val}')")
        nb_runner.run_cell(2)
        assert "val = 3" in nb_runner.get_output(2)


class TestErrorInMiddleChain:
    """Error in middle of dependency chain."""

    def test_error_in_cell2_fix_continue(self, nb_runner):
        """Error in cell 2 of 3, fix, continue."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x / 0  # will error",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(2, "y = x * 2")
        nb_runner.run_cells([2, 3])
        assert "z = 11" in nb_runner.get_output(3)


class TestFixSourceThenRerunCells:
    """Fix the source cell that caused downstream error."""

    def test_fix_upstream_data(self, nb_runner):
        """Downstream fails because of bad data, fix data."""
        nb_runner.create_notebook([
            "data = []  # empty causes error",
            "avg = sum(data) / len(data)\nprint(f'avg = {avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "avg = 20.0" in nb_runner.get_output(2)
