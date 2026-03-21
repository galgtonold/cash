"""Batch 194 – Numeric / math computation chain interaction tests.

Tests editing numeric computations including arithmetic chains,
math functions, and statistical calculations.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestArithmeticChainEdits:
    """Editing chains of arithmetic operations."""

    def test_edit_formula(self, nb_runner):
        """Edit a mathematical formula."""
        nb_runner.create_notebook([
            "a = 3\nb = 4  # formula source",
            "result = a**2 + b**2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Change formula
        nb_runner.set_cell_source(
            2, "result = (a + b) ** 2\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 49" in nb_runner.get_output(2)

    def test_edit_chain_operand(self, nb_runner):
        """Edit one operand in a chain."""
        nb_runner.create_notebook([
            "x = 10  # chain operand x",
            "y = 20  # chain operand y",
            "z = 30  # chain operand z",
            "result = x * y + z\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 230" in nb_runner.get_output(4)

        # Change middle operand
        nb_runner.set_cell_source(2, "y = 5  # chain operand y v2")
        nb_runner.run_all()
        assert "result = 80" in nb_runner.get_output(4)


class TestMathFunctionEdits:
    """Editing math function usage."""

    def test_edit_math_function(self, nb_runner):
        """Edit which math function to use."""
        nb_runner.create_notebook([
            "import math",
            "val = 2.0  # math function source",
            "result = math.sqrt(val)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1.4142" in nb_runner.get_output(3)

        # Change to log
        nb_runner.set_cell_source(
            3, "result = math.log(val)\nprint(f'result = {result:.4f}')"
        )
        nb_runner.run_all()
        assert "result = 0.6931" in nb_runner.get_output(3)

    def test_edit_statistical_calc(self, nb_runner):
        """Edit statistical calculations."""
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]  # stats source",
            "mean = sum(data) / len(data)\nprint(f'mean = {mean}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [100, 200, 300]  # stats source v2")
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(2)
