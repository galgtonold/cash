"""Batch 116 – Error handling + cell edit interaction tests.

Tests that exercise error scenarios: syntax errors, runtime errors,
exception recovery, and cell edits to fix errors.

Note: nbclient raises CellExecutionError for cell errors, so we must
use try/except to handle expected errors gracefully.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestSyntaxErrorRecovery:
    """Introduce syntax error, then fix it."""

    def test_fix_syntax_error(self, nb_runner):
        """Cell has syntax error, fix it and re-run."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x *",  # Syntax error (incomplete expression)
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)  # Cell 1 should work

        # Cell 2 should fail with syntax error
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix the error and add print
        nb_runner.set_cell_source(2, "y = x * 2\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_syntax_error_does_not_corrupt_state(self, nb_runner):
        """Syntax error in one cell shouldn't corrupt other cached values."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 43" in nb_runner.get_output(2)

        # Introduce syntax error in cell 1
        nb_runner.set_cell_source(1, "x = ")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix cell 1
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)


class TestRuntimeErrorRecovery:
    """Runtime errors (NameError, TypeError, etc.) and recovery."""

    def test_fix_name_error(self, nb_runner):
        """NameError due to undefined variable, fix by defining it."""
        nb_runner.create_notebook([
            "y = undefined_var + 1",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix: define the variable first
        nb_runner.set_cell_source(
            1, "undefined_var = 10\ny = undefined_var + 1\nprint(f'y = {y}')"
        )
        nb_runner.run_cell(1)
        assert "y = 11" in nb_runner.get_output(1)

    def test_fix_type_error(self, nb_runner):
        """TypeError, fix by correcting the operation."""
        nb_runner.create_notebook([
            "x = 'hello'",
            "y = x + 1",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix: change to string concatenation
        nb_runner.set_cell_source(2, "y = x + ' world'\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = hello world" in nb_runner.get_output(2)

    def test_fix_zero_division(self, nb_runner):
        """ZeroDivisionError, fix by changing divisor."""
        nb_runner.create_notebook([
            "x = 10\ny = 0",
            "z = x / y",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix: change y and add print
        nb_runner.set_cell_source(1, "x = 10\ny = 2")
        nb_runner.set_cell_source(2, "z = x / y\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 5.0" in nb_runner.get_output(2)

    def test_fix_index_error(self, nb_runner):
        """IndexError, fix by using valid index."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "val = data[10]",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "val = data[2]\nprint(f'val = {val}')")
        nb_runner.run_cell(2)
        assert "val = 3" in nb_runner.get_output(2)


class TestErrorThenSuccess:
    """Error in first run, success in subsequent runs."""

    def test_error_then_fix_sequence(self, nb_runner):
        """Error -> fix -> success sequence."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x / 0",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "y = x * 2\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_success_then_error_then_fix(self, nb_runner):
        """Success -> introduce error -> fix -> success."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Introduce error
        nb_runner.set_cell_source(2, "y = x / 0")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 30" in nb_runner.get_output(2)

    def test_error_in_middle_of_chain(self, nb_runner):
        """Error in middle cell, fix it, downstream should work."""
        nb_runner.create_notebook([
            "x = 10",
            "y = 1 / 0",  # Error
            "z = 999\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x + 5")
        nb_runner.run_all()
        assert "z = 999" in nb_runner.get_output(3)


class TestErrorAfterRestart:
    """Error handling after kernel restart."""

    def test_restart_after_error(self, nb_runner):
        """Error, restart, run clean."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Introduce error
        nb_runner.set_cell_source(2, "y = x / 0")
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Restart and fix
        nb_runner.shutdown()
        nb_runner.set_cell_source(2, "y = x * 5\nprint(f'y = {y}')")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 50" in nb_runner.get_output(2)
