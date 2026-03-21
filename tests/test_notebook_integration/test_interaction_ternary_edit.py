"""Batch 278 – Ternary and inline conditional edit propagation.

Tests ternary expressions and inline conditionals.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTernaryEdits:
    """Ternary/inline conditional patterns."""

    def test_ternary_condition_edit(self, nb_runner):
        """Edit condition in ternary expression."""
        nb_runner.create_notebook([
            "threshold = 50",
            "value = 75",
            "label = 'high' if value > threshold else 'low'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = high" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "threshold = 80")
        nb_runner.run_all()
        assert "label = low" in nb_runner.get_output(3)

    def test_chained_ternary_edit(self, nb_runner):
        """Edit value with chained ternary."""
        nb_runner.create_notebook([
            "score = 85",
            "grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'F'\nprint(f'grade = {grade}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "score = 65")
        nb_runner.run_all()
        assert "grade = F" in nb_runner.get_output(2)

    def test_list_comp_with_conditional_edit(self, nb_runner):
        """Edit filter condition in list comprehension."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "limit = 5",
            "filtered = [x for x in data if x > limit]\nprint(f'filtered = {filtered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered = [6, 7, 8, 9, 10]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "limit = 8")
        nb_runner.run_all()
        assert "filtered = [9, 10]" in nb_runner.get_output(3)
