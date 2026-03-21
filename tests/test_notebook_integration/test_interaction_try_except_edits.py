"""Batch 166 – Try/except/finally interaction tests.

Tests editing code within try/except blocks, changing exception
types, and modifying finally clauses.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.control, pytest.mark.timeout(90)]


class TestTryExceptEdits:
    """Editing try/except patterns."""

    def test_edit_try_body(self, nb_runner):
        """Edit the code inside a try block."""
        nb_runner.create_notebook([
            "x = 10  # divisor",
            "try:\n    result = 100 // x\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Make it divide by zero
        nb_runner.set_cell_source(1, "x = 0  # divisor zero")
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler to return a different fallback."""
        nb_runner.create_notebook([
            "data = 'not_a_number'  # bad data",
            "try:\n    val = int(data)\nexcept ValueError:\n    val = 0\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 0" in nb_runner.get_output(2)

        # Change fallback value
        nb_runner.set_cell_source(
            2,
            "try:\n    val = int(data)\nexcept ValueError:\n    val = -999\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = -999" in nb_runner.get_output(2)

    def test_fix_error_then_rerun(self, nb_runner):
        """Fix code that was raising an exception."""
        nb_runner.create_notebook([
            "nums = [1, 2, 3]  # data list",
            "try:\n    val = nums[10]\nexcept IndexError:\n    val = 'out_of_bounds'\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = out_of_bounds" in nb_runner.get_output(2)

        # Fix the index
        nb_runner.set_cell_source(
            2,
            "try:\n    val = nums[2]\nexcept IndexError:\n    val = 'out_of_bounds'\nprint(f'val = {val}')",
        )
        nb_runner.run_all()
        assert "val = 3" in nb_runner.get_output(2)


class TestFinallyEdits:
    """Editing finally clauses."""

    def test_edit_finally_action(self, nb_runner):
        """Edit what happens in a finally block."""
        nb_runner.create_notebook([
            "status = 'unknown'  # init status",
            "try:\n    result = 42\nfinally:\n    status = 'done'\nprint(f'status = {status}, result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "status = done" in nb_runner.get_output(2)
        assert "result = 42" in nb_runner.get_output(2)

        # Change finally action
        nb_runner.set_cell_source(
            2,
            "try:\n    result = 42\nfinally:\n    status = 'complete'\nprint(f'status = {status}, result = {result}')",
        )
        nb_runner.run_all()
        assert "status = complete" in nb_runner.get_output(2)

    def test_add_finally_clause(self, nb_runner):
        """Add a finally clause to existing try/except."""
        nb_runner.create_notebook([
            "cleanup_done = False  # cleanup flag",
            "try:\n    x = 100\nexcept Exception:\n    x = 0\nprint(f'x = {x}, cleanup = {cleanup_done}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)
        assert "cleanup = False" in nb_runner.get_output(2)

        # Add finally
        nb_runner.set_cell_source(
            2,
            "try:\n    x = 100\nexcept Exception:\n    x = 0\nfinally:\n    cleanup_done = True\nprint(f'x = {x}, cleanup = {cleanup_done}')",
        )
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)
        assert "cleanup = True" in nb_runner.get_output(2)
