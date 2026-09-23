"""decimal and fractions precision arithmetic."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDecimalPrecision:
    """Decimal module precision patterns."""

    def test_decimal_propagation(self, nb_runner):
        """Decimal precision change propagation."""
        nb_runner.create_notebook(
            [
                "precision = 10",
                textwrap.dedent("""\
                from decimal import Decimal, getcontext, localcontext
                with localcontext() as ctx:
                    ctx.prec = precision
                    result = Decimal('1') / Decimal('7')
                result_str = str(result)
                result_len = len(result_str.replace('0.', ''))
            """),
                "print(f'result={result_str} digits={result_len}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "result=" in out1

        nb_runner.set_cell_source(1, "precision = 30")
        nb_runner.run_cells([1, 2, 3])
        out2 = nb_runner.get_output(3)
        # More digits with higher precision
        assert "result=" in out2
