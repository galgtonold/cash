"""Batch 86 – decimal and fractions precision arithmetic."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDecimalPrecision:
    """Decimal module precision patterns."""

    def test_decimal_basic(self, nb_runner):
        """Basic Decimal arithmetic avoids float issues."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from decimal import Decimal, getcontext
                getcontext().prec = 50
                a = Decimal('0.1') + Decimal('0.2')
                b = Decimal('1') / Decimal('3')
                c = Decimal('2').sqrt()
            """),
            "print(f'a={a}')\nprint(f'b_start={str(b)[:10]}')\nprint(f'c_start={str(c)[:10]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a=0.3" in out
        assert "b_start=0.33333333" in out

    def test_decimal_propagation(self, nb_runner):
        """Decimal precision change propagation."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "result=" in out1

        nb_runner.set_cell_source(1, "precision = 30")
        nb_runner.run_cells([1, 2, 3])
        out2 = nb_runner.get_output(3)
        # More digits with higher precision
        assert "result=" in out2

    def test_decimal_financial(self, nb_runner):
        """Decimal for financial calculations with ROUND_HALF_UP."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from decimal import Decimal, ROUND_HALF_UP
                prices = [Decimal('19.99'), Decimal('5.50'), Decimal('12.75')]
                tax_rate = Decimal('0.0825')
                subtotal = sum(prices)
                tax = (subtotal * tax_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                total = subtotal + tax
            """),
            "print(f'subtotal={subtotal} tax={tax} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "subtotal=38.24" in out
        assert "tax=" in out
        assert "total=" in out


class TestFractions:
    """Fractions module patterns."""

    def test_fraction_arithmetic(self, nb_runner):
        """Fraction arithmetic preserving exact values."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from fractions import Fraction
                a = Fraction(1, 3) + Fraction(1, 6)
                b = Fraction(2, 5) * Fraction(5, 7)
                c = Fraction(3, 4) / Fraction(2, 3)
                results = {'a': str(a), 'b': str(b), 'c': str(c)}
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "1/2" in out   # 1/3 + 1/6 = 1/2
        assert "2/7" in out   # 2/5 * 5/7 = 2/7
        assert "9/8" in out   # 3/4 / 2/3 = 9/8

    def test_fraction_from_float(self, nb_runner):
        """Fraction from float and limit_denominator."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from fractions import Fraction
                import math
                pi_frac = Fraction(math.pi).limit_denominator(1000)
                e_frac = Fraction(math.e).limit_denominator(1000)
            """),
            "print(f'pi={pi_frac} e={e_frac}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "355/113" in out  # classic pi approximation
