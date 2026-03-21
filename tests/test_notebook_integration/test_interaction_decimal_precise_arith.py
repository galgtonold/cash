"""Batch 415: decimal module for precise arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecimalPreciseArith:
    def test_decimal_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal\na = Decimal('0.1')\nb = Decimal('0.2')",
            "result = a + b\nprint(f'result={result} eq={result == Decimal(\"0.3\")}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=0.3" in nb_runner.get_output(2)
        assert "eq=True" in nb_runner.get_output(2)

    def test_decimal_rounding(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal, ROUND_HALF_UP\nval = Decimal('2.345')",
            "r2 = val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nr1 = val.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)\nprint(f'r2={r2} r1={r1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r2=2.35" in nb_runner.get_output(2)
        assert "r1=2.3" in nb_runner.get_output(2)

    def test_decimal_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal\nprice = Decimal('19.99')\nqty = Decimal('3')",
            "total = price * qty\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=59.97" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from decimal import Decimal\nprice = Decimal('9.99')\nqty = Decimal('7')")
        nb_runner.run_all()
        assert "total=69.93" in nb_runner.get_output(2)
