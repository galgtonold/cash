"""Batch 144 – Conditional logic and boolean pattern interaction tests.

Tests where users edit conditional logic (if/elif/else),
boolean variables, and branch selection patterns.
"""

import pytest

pytestmark = [pytest.mark.control, pytest.mark.stress, pytest.mark.timeout(45)]


class TestIfElseEdits:
    """If/else editing patterns."""


    def test_edit_branch_logic(self, nb_runner):
        """Edit the branch logic itself."""
        nb_runner.create_notebook([
            "x = 10",
            "result = 'positive' if x > 0 else 'non-positive'\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = positive" in nb_runner.get_output(2)

        # Change the logic
        nb_runner.set_cell_source(
            2,
            "result = 'even' if x % 2 == 0 else 'odd'\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = even" in nb_runner.get_output(2)


class TestBooleanEdits:
    """Boolean flag editing patterns."""

    def test_toggle_boolean_flag(self, nb_runner):
        """Toggle a boolean flag."""
        nb_runner.create_notebook([
            "debug = True",
            "data = [1, 2, 3]\nif debug:\n    data = [x * 10 for x in data]",
            "total = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)

        # Toggle debug off
        nb_runner.set_cell_source(1, "debug = False")
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(3)

    def test_multiple_boolean_flags(self, nb_runner):
        """Multiple boolean flags controlling logic."""
        nb_runner.create_notebook([
            "normalize = True\nround_result = True",
            "raw = [10, 20, 30]\nif normalize:\n    vals = [x / max(raw) for x in raw]\nelse:\n    vals = raw",
            "if round_result:\n    vals = [round(v, 1) for v in vals]",
            "print(f'vals = {vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "vals = " in output

        # Turn off normalize
        nb_runner.set_cell_source(1, "normalize = False\nround_result = True")
        nb_runner.run_all()
        assert "vals = [10, 20, 30]" in nb_runner.get_output(4)


class TestNestedConditionalEdits:
    """Nested conditional patterns with edits."""

    def test_edit_outer_condition(self, nb_runner):
        """Edit outer condition in nested if."""
        nb_runner.create_notebook([
            "mode = 'fast'\nverbose = True",
            "if mode == 'fast':\n    result = 100\nelif mode == 'slow':\n    result = 1\nelse:\n    result = 10",
            "if verbose:\n    msg = f'Mode: {mode}, Result: {result}'\nelse:\n    msg = str(result)\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Mode: fast, Result: 100" in nb_runner.get_output(3)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'slow'\nverbose = True")
        nb_runner.run_all()
        assert "Mode: slow, Result: 1" in nb_runner.get_output(3)

    def test_edit_inner_condition(self, nb_runner):
        """Edit inner condition flag."""
        nb_runner.create_notebook([
            "mode = 'fast'\nverbose = True",
            "if mode == 'fast':\n    result = 100\nelse:\n    result = 1",
            "if verbose:\n    msg = f'Result: {result}'\nelse:\n    msg = 'done'\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Result: 100" in nb_runner.get_output(3)

        # Turn off verbose
        nb_runner.set_cell_source(1, "mode = 'fast'\nverbose = False")
        nb_runner.run_all()
        assert "done" in nb_runner.get_output(3)
