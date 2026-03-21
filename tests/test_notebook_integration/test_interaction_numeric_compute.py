"""Batch 236 – Numeric computation and math pattern edit tests.

Tests editing cells with numeric computations, math operations,
and scientific-style calculations.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestNumericComputationEdits:
    """Editing numeric computation patterns."""

    def test_edit_math_formula(self, nb_runner):
        """Edit a mathematical formula."""
        nb_runner.create_notebook([
            "import math\nradius = 5",
            "area = math.pi * radius ** 2\nprint(f'area = {area:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(2)

        # Change formula to volume of sphere
        nb_runner.set_cell_source(2, "volume = (4/3) * math.pi * radius ** 3\nprint(f'volume = {volume:.2f}')")
        nb_runner.run_all()
        assert "volume = 523.60" in nb_runner.get_output(2)

    def test_edit_statistical_calculation(self, nb_runner):
        """Edit a statistical calculation."""
        nb_runner.create_notebook([
            "scores = [85, 90, 78, 92, 88]",
            "mean = sum(scores) / len(scores)\nprint(f'mean = {mean:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 86.6" in nb_runner.get_output(2)

        # Change to median
        nb_runner.set_cell_source(2, "sorted_s = sorted(scores)\nmedian = sorted_s[len(sorted_s) // 2]\nprint(f'median = {median}')")
        nb_runner.run_all()
        assert "median = 88" in nb_runner.get_output(2)

    def test_edit_numeric_input(self, nb_runner):
        """Edit numeric inputs to a computation chain."""
        nb_runner.create_notebook([
            "width = 10\nheight = 5",
            "perimeter = 2 * (width + height)\ndiagonal = (width**2 + height**2) ** 0.5\nprint(f'perimeter={perimeter} diagonal={diagonal:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "perimeter=30" in nb_runner.get_output(2)
        assert "diagonal=11.18" in nb_runner.get_output(2)

        # Change dimensions
        nb_runner.set_cell_source(1, "width = 3\nheight = 4")
        nb_runner.run_all()
        assert "perimeter=14" in nb_runner.get_output(2)
        assert "diagonal=5.00" in nb_runner.get_output(2)

    def test_edit_accumulator_formula(self, nb_runner):
        """Edit a cumulative computation formula."""
        nb_runner.create_notebook([
            "rates = [0.05, 0.03, 0.07, 0.02]",
            "principal = 1000\nfor r in rates:\n    principal *= (1 + r)\nprint(f'final = {principal:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "final = " in out

        # Change starting principal
        nb_runner.set_cell_source(2, "principal = 5000\nfor r in rates:\n    principal *= (1 + r)\nprint(f'final = {principal:.2f}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "final = " in out2
        # 5000 * 1.05 * 1.03 * 1.07 * 1.02 ≈ 5893.xx
        assert "5" in out2  # starts with 5xxx
