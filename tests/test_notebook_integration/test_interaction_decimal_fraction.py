"""
Batch 321: decimal/fractions precision arithmetic with caching.
Tests Decimal, Fraction operations, and edit propagation for precise math.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecimalFractionOps:
    """Test decimal and fraction operation caching."""

    def test_decimal_precision(self, nb_runner):
        """Decimal arithmetic with precision, verify caching."""
        nb_runner.create_notebook([
            "from decimal import Decimal, getcontext\ngetcontext().prec = 50",
            "a = Decimal('1') / Decimal('7')",
            "result = str(a)[:20]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "0.14285714285714285" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "0.14285714285714285" in out2

    def test_fraction_arithmetic_edit(self, nb_runner):
        """Fraction arithmetic with edit propagation."""
        nb_runner.create_notebook([
            "from fractions import Fraction",
            "a = Fraction(1, 3)\nb = Fraction(1, 6)",
            "result = a + b\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=1/2" in out

        nb_runner.set_cell_source(2, "a = Fraction(2, 3)\nb = Fraction(1, 6)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=5/6" in out2

    def test_decimal_comparison(self, nb_runner):
        """Decimal comparison with floating point equivalence."""
        nb_runner.create_notebook([
            "from decimal import Decimal",
            "x = Decimal('0.1') + Decimal('0.2')",
            "exact = (x == Decimal('0.3'))\nprint(f'exact={exact}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "exact=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "exact=True" in out2
