"""Batch 234 – Error handling interaction edit tests.

Tests editing cells that change error-handling behavior: adding/removing
try/except blocks, changing raised exceptions, etc.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestErrorHandlingEdits:
    """Editing error-handling patterns."""

    def test_edit_add_try_except(self, nb_runner):
        """Add a try/except to a cell that previously had no error handling."""
        nb_runner.create_notebook([
            "data = {'a': 1, 'b': 2}",
            "val = data.get('c', -1)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = -1" in nb_runner.get_output(2)

        # Edit to use try/except instead of .get()
        nb_runner.set_cell_source(2, "try:\n    val = data['c']\nexcept KeyError:\n    val = 'missing'\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = missing" in nb_runner.get_output(2)

    def test_edit_fix_error_to_success(self, nb_runner):
        """Edit a cell from one that raises to one that succeeds."""
        nb_runner.create_notebook([
            "x = 0",
            "try:\n    result = 10 / x\nexcept ZeroDivisionError:\n    result = float('inf')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = inf" in nb_runner.get_output(2)

        # Fix the error by changing x
        nb_runner.set_cell_source(1, "x = 2")
        nb_runner.run_all()
        assert "result = 5.0" in nb_runner.get_output(2)

    def test_edit_change_default_value(self, nb_runner):
        """Edit the default/fallback value in error handling."""
        nb_runner.create_notebook([
            "items = [10, 20, 30]",
            "try:\n    val = items[5]\nexcept IndexError:\n    val = 0\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 0" in nb_runner.get_output(2)

        # Change default value
        nb_runner.set_cell_source(2, "try:\n    val = items[5]\nexcept IndexError:\n    val = -999\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = -999" in nb_runner.get_output(2)

    def test_edit_remove_error_condition(self, nb_runner):
        """Edit data so error condition no longer triggers."""
        nb_runner.create_notebook([
            "values = []",
            "try:\n    avg = sum(values) / len(values)\nexcept ZeroDivisionError:\n    avg = 0\nprint(f'avg = {avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg = 0" in nb_runner.get_output(2)

        # Add data so division works
        nb_runner.set_cell_source(1, "values = [10, 20, 30]")
        nb_runner.run_all()
        assert "avg = 20" in nb_runner.get_output(2)
