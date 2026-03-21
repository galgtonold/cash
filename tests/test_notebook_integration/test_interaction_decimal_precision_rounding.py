"""
Interaction test: decimal module precision and rounding.
Tests Decimal arithmetic with custom precision, rounding modes,
quantize operations, and cross-cell financial calculations.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecimalPrecisionRounding:
    """Test Decimal precision and rounding across cells."""

    def test_decimal_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: high precision arithmetic
            "from decimal import Decimal, getcontext, ROUND_HALF_UP\ngetcontext().prec = 50\na = Decimal('1') / Decimal('3')\nb = Decimal('0.1') + Decimal('0.2')\nprint(f'third_start={str(a)[:10]}')\nprint(f'point_three={b}')",
            # Cell 2: quantize for currency
            "price = Decimal('19.995')\ntax = Decimal('0.08875')\ntotal = price * (1 + tax)\nrounded = total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'total_raw={total}')\nprint(f'rounded={rounded}')",
            # Cell 3: comparison
            "check = Decimal('0.1') + Decimal('0.2') == Decimal('0.3')\nfloat_check = (0.1 + 0.2 == 0.3)\nprint(f'decimal_exact={check}')\nprint(f'float_exact={float_check}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "third_start=0.33333333" in out1
        assert "point_three=0.3" in out1
        out2 = nb_runner.get_output(2)
        assert "rounded=" in out2
        out3 = nb_runner.get_output(3)
        assert "decimal_exact=True" in out3
        assert "float_exact=False" in out3

    def test_decimal_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal, ROUND_HALF_UP\nprice = Decimal('100.00')\ndiscount = Decimal('0.15')\nfinal = (price * (1 - discount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'final={final}')",
            "savings = price - final\nprint(f'savings={savings}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=85.00" in nb_runner.get_output(1)
        assert "savings=15.00" in nb_runner.get_output(2)

        # Change discount
        nb_runner.set_cell_source(1, "from decimal import Decimal, ROUND_HALF_UP\nprice = Decimal('100.00')\ndiscount = Decimal('0.20')\nfinal = (price * (1 - discount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'final={final}')")
        nb_runner.run_cells([1, 2])
        assert "final=80.00" in nb_runner.get_output(1)
        assert "savings=20.00" in nb_runner.get_output(2)

    def test_decimal_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal\nvals = [Decimal(str(x)) for x in [10.5, 20.3, 30.7]]\ntotal = sum(vals)\nprint(f'total={total}')",
            "avg = total / len(vals)\nprint(f'avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=61.5" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=61.5" in nb_runner.get_output(1)
