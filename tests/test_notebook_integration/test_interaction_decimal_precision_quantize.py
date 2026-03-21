"""Batch 490: decimal precision and quantize operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecimalPrecisionQuantize:
    def test_decimal_precision(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal, getcontext",
            "getcontext().prec = 50\na = Decimal('1') / Decimal('3')\nb = Decimal('0.1') + Decimal('0.2')\nprint(f'a={a}')\nprint(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "0.33333333" in out
        assert "b=0.3" in out

    def test_quantize_rounding(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN",
            "price = Decimal('19.995')\nup = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\ndown = price.quantize(Decimal('0.01'), rounding=ROUND_DOWN)\nprint(f'up={up} down={down}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "up=20.00" in out
        assert "down=19.99" in out

    def test_decimal_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from decimal import Decimal",
            "x = Decimal('10.5') * Decimal('3')\nprint(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=31.5" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "x = Decimal('7.25') * Decimal('4')\nprint(f'x={x}')")
        nb_runner.run_all()
        assert "x=29.00" in nb_runner.get_output(2)
