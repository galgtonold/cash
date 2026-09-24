"""decimal.Decimal and fractions.Fraction arithmetic across cells."""

import textwrap

import pytest


# decimal/fractions precision arithmetic with caching.
# Tests Decimal, Fraction operations, and edit propagation for precise math.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDecimalFractionOps:
    """Test decimal and fraction operation caching."""

    def test_decimal_precision(self, nb_runner):
        """Decimal arithmetic with precision, verify caching."""
        nb_runner.create_notebook(
            [
                "from decimal import Decimal, getcontext\ngetcontext().prec = 50",
                "a = Decimal('1') / Decimal('7')",
                "result = str(a)[:20]\nprint(f'result={result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from fractions import Fraction",
                "a = Fraction(1, 3)\nb = Fraction(1, 6)",
                "result = a + b\nprint(f'result={result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from decimal import Decimal",
                "x = Decimal('0.1') + Decimal('0.2')",
                "exact = (x == Decimal('0.3'))\nprint(f'exact={exact}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "exact=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "exact=True" in out2


# decimal and fractions precision arithmetic.
@pytest.mark.stress
@pytest.mark.integration
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


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDecimalPreciseArith:
    """decimal module for precise arithmetic."""

    def test_decimal_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal\na = Decimal('0.1')\nb = Decimal('0.2')",
                "result = a + b\nprint(f'result={result} eq={result == Decimal(\"0.3\")}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=0.3" in nb_runner.get_output(2)
        assert "eq=True" in nb_runner.get_output(2)

    def test_decimal_rounding(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal, ROUND_HALF_UP\nval = Decimal('2.345')",
                "r2 = val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nr1 = val.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)\nprint(f'r2={r2} r1={r1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r2=2.35" in nb_runner.get_output(2)
        assert "r1=2.3" in nb_runner.get_output(2)

    def test_decimal_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal\nprice = Decimal('19.99')\nqty = Decimal('3')",
                "total = price * qty\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=59.97" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from decimal import Decimal\nprice = Decimal('9.99')\nqty = Decimal('7')")
        nb_runner.run_all()
        assert "total=69.93" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDecimalPrecisionQuantize:
    """decimal precision and quantize operations."""

    def test_decimal_precision(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal, getcontext",
                "getcontext().prec = 50\na = Decimal('1') / Decimal('3')\nb = Decimal('0.1') + Decimal('0.2')\nprint(f'a={a}')\nprint(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "0.33333333" in out
        assert "b=0.3" in out

    def test_quantize_rounding(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN",
                "price = Decimal('19.995')\nup = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\ndown = price.quantize(Decimal('0.01'), rounding=ROUND_DOWN)\nprint(f'up={up} down={down}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "up=20.00" in out
        assert "down=19.99" in out

    def test_decimal_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal",
                "x = Decimal('10.5') * Decimal('3')\nprint(f'x={x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=31.5" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "x = Decimal('7.25') * Decimal('4')\nprint(f'x={x}')")
        nb_runner.run_all()
        assert "x=29.00" in nb_runner.get_output(2)


# Interaction test: decimal module precision and rounding.
# Tests Decimal arithmetic with custom precision, rounding modes,
# quantize operations, and cross-cell financial calculations.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDecimalPrecisionRounding:
    """Test Decimal precision and rounding across cells."""

    def test_decimal_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: high precision arithmetic
                "from decimal import Decimal, getcontext, ROUND_HALF_UP\ngetcontext().prec = 50\na = Decimal('1') / Decimal('3')\nb = Decimal('0.1') + Decimal('0.2')\nprint(f'third_start={str(a)[:10]}')\nprint(f'point_three={b}')",
                # Cell 2: quantize for currency
                "price = Decimal('19.995')\ntax = Decimal('0.08875')\ntotal = price * (1 + tax)\nrounded = total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'total_raw={total}')\nprint(f'rounded={rounded}')",
                # Cell 3: comparison
                "check = Decimal('0.1') + Decimal('0.2') == Decimal('0.3')\nfloat_check = (0.1 + 0.2 == 0.3)\nprint(f'decimal_exact={check}')\nprint(f'float_exact={float_check}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from decimal import Decimal, ROUND_HALF_UP\nprice = Decimal('100.00')\ndiscount = Decimal('0.15')\nfinal = (price * (1 - discount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'final={final}')",
                "savings = price - final\nprint(f'savings={savings}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final=85.00" in nb_runner.get_output(1)
        assert "savings=15.00" in nb_runner.get_output(2)

        # Change discount
        nb_runner.set_cell_source(
            1,
            "from decimal import Decimal, ROUND_HALF_UP\nprice = Decimal('100.00')\ndiscount = Decimal('0.20')\nfinal = (price * (1 - discount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\nprint(f'final={final}')",
        )
        nb_runner.run_cells([1, 2])
        assert "final=80.00" in nb_runner.get_output(1)
        assert "savings=20.00" in nb_runner.get_output(2)

    def test_decimal_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from decimal import Decimal\nvals = [Decimal(str(x)) for x in [10.5, 20.3, 30.7]]\ntotal = sum(vals)\nprint(f'total={total}')",
                "avg = total / len(vals)\nprint(f'avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=61.5" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=61.5" in nb_runner.get_output(1)


# Interaction test: fractions module arithmetic.
# Tests Fraction creation from various inputs, arithmetic operations,
# limit_denominator, and cross-cell fraction pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFractionsArithmetic:
    """Test fractions.Fraction arithmetic across cells."""

    def test_fractions_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: fraction creation and arithmetic
                "from fractions import Fraction\na = Fraction(1, 3)\nb = Fraction(2, 5)\nsum_ab = a + b\nprod = a * b\nprint(f'a={a}')\nprint(f'b={b}')\nprint(f'sum={sum_ab}')\nprint(f'prod={prod}')",
                # Cell 2: from string and float
                "f_str = Fraction('3/7')\nf_float = Fraction(0.1).limit_denominator(1000)\nprint(f'from_str={f_str}')\nprint(f'from_float={f_float}')",
                # Cell 3: comparison
                "bigger = max(a, b, f_str)\nprint(f'biggest={bigger}')\nprint(f'as_float={float(bigger):.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "a=1/3" in out1
        assert "sum=11/15" in out1
        assert "prod=2/15" in out1
        out2 = nb_runner.get_output(2)
        assert "from_str=3/7" in out2
        assert "from_float=1/10" in out2
        out3 = nb_runner.get_output(3)
        assert "biggest=3/7" in out3

    def test_fractions_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction\nf = Fraction(22, 7)\nprint(f'f={f}')\nprint(f'float={float(f):.6f}')",
                "is_close = abs(float(f) - 3.14159) < 0.01\nprint(f'close_to_pi={is_close}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=22/7" in nb_runner.get_output(1)
        assert "close_to_pi=True" in nb_runner.get_output(2)

        # Better approximation
        nb_runner.set_cell_source(
            1, "from fractions import Fraction\nf = Fraction(355, 113)\nprint(f'f={f}')\nprint(f'float={float(f):.6f}')"
        )
        nb_runner.run_cells([1, 2])
        assert "f=355/113" in nb_runner.get_output(1)
        assert "close_to_pi=True" in nb_runner.get_output(2)

    def test_fractions_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction\nparts = [Fraction(1, n) for n in range(1, 6)]\ntotal = sum(parts)\nprint(f'total={total}')",
                "as_float = float(total)\nprint(f'float={as_float:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1 + 1/2 + 1/3 + 1/4 + 1/5 = 60/60 + 30/60 + 20/60 + 15/60 + 12/60 = 137/60
        assert "total=137/60" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=137/60" in nb_runner.get_output(1)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFractionsRational:
    """fractions module exact rational arithmetic."""

    def test_fraction_arithmetic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction",
                "a = Fraction(1, 3)\nb = Fraction(1, 6)\nsum_ab = a + b\nprod = a * b\nprint(f'sum={sum_ab} prod={prod}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sum=1/2" in out
        assert "prod=1/18" in out

    def test_fraction_from_float(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction",
                "f1 = Fraction('0.1') + Fraction('0.2')\nf2 = Fraction(0.1) + Fraction(0.2)\nprint(f'exact={f1}')\nprint(f'float_based={f2}')\nprint(f'exact_eq_03={f1 == Fraction(3, 10)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "exact=3/10" in out
        assert "exact_eq_03=True" in out

    def test_fraction_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction",
                "f = Fraction(3, 4) + Fraction(1, 4)\nprint(f'f={f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "f = Fraction(2, 3) + Fraction(1, 6)\nprint(f'f={f}')")
        nb_runner.run_all()
        assert "f=5/6" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFractionsRationalArith:
    """fractions module exact rational arithmetic."""

    def test_fraction_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction\na = Fraction(1, 3)\nb = Fraction(1, 6)",
                "result = a + b\nprint(f'result={result} float={float(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1/2" in nb_runner.get_output(2)
        assert "float=0.5" in nb_runner.get_output(2)

    def test_fraction_from_string(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction\nf = Fraction('3.14')",
                "num = f.numerator\nden = f.denominator\nprint(f'num={num} den={den}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "num=157" in nb_runner.get_output(2)
        assert "den=50" in nb_runner.get_output(2)

    def test_fraction_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from fractions import Fraction\nx = Fraction(2, 5)\ny = Fraction(3, 5)",
                "product = x * y\nprint(f'product={product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product=6/25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from fractions import Fraction\nx = Fraction(1, 2)\ny = Fraction(1, 3)")
        nb_runner.run_all()
        assert "product=1/6" in nb_runner.get_output(2)
