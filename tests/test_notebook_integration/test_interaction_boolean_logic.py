"""Batch 263 – Boolean logic and condition edits.

Tests boolean operations, conditions, short-circuit edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBooleanLogicEdits:
    """Boolean logic patterns with edit propagation."""

    def test_compound_condition_edit(self, nb_runner):
        """Edit compound boolean condition."""
        nb_runner.create_notebook([
            "data = [15, 25, 35, 45, 55, 65, 75]",
            "lo = 20\nhi = 60",
            "in_range = [x for x in data if lo <= x <= hi]\nprint(f'in_range = {in_range}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "in_range = [25, 35, 45, 55]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "lo = 30\nhi = 70")
        nb_runner.run_all()
        assert "in_range = [35, 45, 55, 65]" in nb_runner.get_output(3)

    def test_any_all_edit(self, nb_runner):
        """Edit data, any/all results change."""
        nb_runner.create_notebook([
            "scores = [80, 90, 70, 85, 95]",
            "all_pass = all(s >= 60 for s in scores)\nany_perfect = any(s == 100 for s in scores)\nprint(f'all_pass={all_pass} any_perfect={any_perfect}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_pass=True any_perfect=False" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "scores = [80, 90, 50, 85, 100]")
        nb_runner.run_all()
        assert "all_pass=False any_perfect=True" in nb_runner.get_output(2)

    def test_boolean_function_edit(self, nb_runner):
        """Edit function with boolean logic."""
        nb_runner.create_notebook([
            "def classify(n):\n    if n > 0:\n        return 'positive'\n    elif n < 0:\n        return 'negative'\n    return 'zero'",
            "labels = [classify(x) for x in [-5, 0, 5]]\nprint(f'labels = {labels}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "labels = ['negative', 'zero', 'positive']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def classify(n):\n    if n >= 0:\n        return 'non-negative'\n    return 'negative'",
        )
        nb_runner.run_all()
        assert "labels = ['negative', 'non-negative', 'non-negative']" in nb_runner.get_output(2)
