"""Batch 238 – Conditional logic and branching edit tests.

Tests editing cells with conditional logic to switch between
branches and verify cache handles the change correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.control, pytest.mark.timeout(90)]


class TestConditionalBranchEdits:
    """Editing conditional branching patterns."""

    def test_edit_if_condition_flip(self, nb_runner):
        """Flip an if condition from True to False."""
        nb_runner.create_notebook([
            "threshold = 50\nvalue = 75",
            "if value > threshold:\n    label = 'above'\nelse:\n    label = 'below'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = above" in nb_runner.get_output(2)

        # Change value to be below threshold
        nb_runner.set_cell_source(1, "threshold = 50\nvalue = 25")
        nb_runner.run_all()
        assert "label = below" in nb_runner.get_output(2)

    def test_edit_elif_chain(self, nb_runner):
        """Edit value to hit a different elif branch."""
        nb_runner.create_notebook([
            "score = 85",
            "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelif score >= 70:\n    grade = 'C'\nelse:\n    grade = 'F'\nprint(f'grade = {grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(2)

        # Change to grade A
        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(2)

    def test_edit_ternary_expression(self, nb_runner):
        """Edit a ternary expression's condition."""
        nb_runner.create_notebook([
            "x = 10",
            "label = 'positive' if x > 0 else 'non-positive'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = positive" in nb_runner.get_output(2)

        # Change to negative
        nb_runner.set_cell_source(1, "x = -5")
        nb_runner.run_all()
        assert "label = non-positive" in nb_runner.get_output(2)

    def test_edit_nested_condition(self, nb_runner):
        """Edit a nested if/else pattern."""
        nb_runner.create_notebook([
            "age = 25\nhas_license = True",
            "if age >= 18:\n    if has_license:\n        status = 'can drive'\n    else:\n        status = 'needs license'\nelse:\n    status = 'too young'\nprint(f'status = {status}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "status = can drive" in nb_runner.get_output(2)

        # Remove license
        nb_runner.set_cell_source(1, "age = 25\nhas_license = False")
        nb_runner.run_all()
        assert "status = needs license" in nb_runner.get_output(2)
