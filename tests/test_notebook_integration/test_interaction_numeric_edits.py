"""Batch 155 – Numeric precision and math interaction tests.

Tests with floating point, integer overflow, precision changes,
and mathematical operations combined with cell edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestFloatingPointEdits:
    """Floating point operations with edits."""

    def test_edit_precision(self, nb_runner):
        """Edit precision of rounding."""
        nb_runner.create_notebook([
            "value = 3.141592653589793",
            "rounded = round(value, 2)\nprint(f'rounded = {rounded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rounded = 3.14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "rounded = round(value, 4)\nprint(f'rounded = {rounded}')"
        )
        nb_runner.run_all()
        assert "rounded = 3.1416" in nb_runner.get_output(2)

    def test_edit_math_operation(self, nb_runner):
        """Edit mathematical operation."""
        nb_runner.create_notebook([
            "import math\nx = 16",
            "result = math.sqrt(x)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = math.log2(x)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(2)

    def test_accumulate_with_precision(self, nb_runner):
        """Accumulation with float precision."""
        nb_runner.create_notebook([
            "values = [0.1] * 10",
            "total = sum(values)\nprint(f'total = {round(total, 1)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "values = [0.1] * 100")
        nb_runner.run_all()
        assert "total = 10.0" in nb_runner.get_output(2)


class TestLargeNumberEdits:
    """Large number operations with edits."""

    def test_edit_exponent(self, nb_runner):
        """Edit exponentiation."""
        nb_runner.create_notebook([
            "base = 2\nexp = 10",
            "result = base ** exp\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1024" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 2\nexp = 20")
        nb_runner.run_all()
        assert "result = 1048576" in nb_runner.get_output(2)

    def test_factorial_edit(self, nb_runner):
        """Edit factorial input."""
        nb_runner.create_notebook([
            "import math\nn = 5",
            "result = math.factorial(n)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "import math\nn = 10")
        nb_runner.run_all()
        assert "result = 3628800" in nb_runner.get_output(2)
