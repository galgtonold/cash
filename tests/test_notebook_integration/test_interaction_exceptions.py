"""Batch 123 – Exception handling code + cell edit interaction tests.

Tests that exercise try/except blocks, error recovery code paths,
and how cash handles errors and recovers across cell edits.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestTryExceptEdits:
    """Try/except blocks + cell edits."""

    def test_fix_error_in_cell(self, nb_runner):
        """Cell has an error, fix it, re-run."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x / 0\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Fix the error
        nb_runner.set_cell_source(2, "y = x / 2\nprint(f'y = {y}')")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 5.0" in nb_runner.get_output(2)

    def test_try_except_catches_error(self, nb_runner):
        """Try/except catches error, edit to change handling."""
        nb_runner.create_notebook([
            "x = 0",
            "try:\n    result = 10 / x\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        # Fix the root cause
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "result = 2.0" in nb_runner.get_output(2)

    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler logic."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "try:\n    val = data[10]\nexcept IndexError:\n    val = 'out of range'\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = out of range" in nb_runner.get_output(2)

        # Change to a valid index
        nb_runner.set_cell_source(
            2,
            "try:\n    val = data[1]\nexcept IndexError:\n    val = 'out of range'\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = 2" in nb_runner.get_output(2)


class TestErrorRecoveryChain:
    """Error in middle of chain, fix and propagate."""

    def test_error_in_middle_fix_and_continue(self, nb_runner):
        """Error in middle cell, fix it, run the rest."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 'string'  # TypeError",
            "c = b * 2\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError, match="TypeError"):
            nb_runner.run_all()

        # Fix the error
        nb_runner.set_cell_source(2, "b = a + 5")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(3)

    def test_introduce_error_then_fix(self, nb_runner):
        """Working code → introduce error → fix it."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 21" in nb_runner.get_output(3)

        # Introduce an error
        nb_runner.set_cell_source(2, "y = undefined_var * 2")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Fix it back
        nb_runner.set_cell_source(2, "y = x * 3")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 31" in nb_runner.get_output(3)


class TestConditionalErrorHandling:
    """Conditional error handling + edits."""

    def test_conditional_with_error_branch(self, nb_runner):
        """Edit condition to switch between error and success paths."""
        nb_runner.create_notebook([
            "mode = 'safe'",
            "if mode == 'safe':\n    result = 42\nelse:\n    result = 1 / 0\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        # Switch to unsafe mode
        nb_runner.set_cell_source(1, "mode = 'unsafe'")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

        # Back to safe
        nb_runner.set_cell_source(1, "mode = 'safe'")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

    def test_guard_clause_edit(self, nb_runner):
        """Edit guard clause that prevents errors."""
        nb_runner.create_notebook([
            "values = [1, 2, 0, 4]",
            "safe = [v for v in values if v != 0]\nresult = sum(10 / v for v in safe)\nprint(f'result = {result:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10/1 + 10/2 + 10/4 = 10 + 5 + 2.5 = 17.5
        assert "result = 17.50" in nb_runner.get_output(2)

        # Edit to remove the zero
        nb_runner.set_cell_source(1, "values = [1, 2, 5, 4]")
        nb_runner.run_all()
        # 10/1 + 10/2 + 10/5 + 10/4 = 10 + 5 + 2 + 2.5 = 19.5
        assert "result = 19.50" in nb_runner.get_output(2)
