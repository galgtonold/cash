"""
Interaction test: fractions module arithmetic.
Tests Fraction creation from various inputs, arithmetic operations,
limit_denominator, and cross-cell fraction pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFractionsArithmetic:
    """Test fractions.Fraction arithmetic across cells."""

    def test_fractions_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: fraction creation and arithmetic
            "from fractions import Fraction\na = Fraction(1, 3)\nb = Fraction(2, 5)\nsum_ab = a + b\nprod = a * b\nprint(f'a={a}')\nprint(f'b={b}')\nprint(f'sum={sum_ab}')\nprint(f'prod={prod}')",
            # Cell 2: from string and float
            "f_str = Fraction('3/7')\nf_float = Fraction(0.1).limit_denominator(1000)\nprint(f'from_str={f_str}')\nprint(f'from_float={f_float}')",
            # Cell 3: comparison
            "bigger = max(a, b, f_str)\nprint(f'biggest={bigger}')\nprint(f'as_float={float(bigger):.4f}')",
        ])
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
        nb_runner.create_notebook([
            "from fractions import Fraction\nf = Fraction(22, 7)\nprint(f'f={f}')\nprint(f'float={float(f):.6f}')",
            "is_close = abs(float(f) - 3.14159) < 0.01\nprint(f'close_to_pi={is_close}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=22/7" in nb_runner.get_output(1)
        assert "close_to_pi=True" in nb_runner.get_output(2)

        # Better approximation
        nb_runner.set_cell_source(1, "from fractions import Fraction\nf = Fraction(355, 113)\nprint(f'f={f}')\nprint(f'float={float(f):.6f}')")
        nb_runner.run_cells([1, 2])
        assert "f=355/113" in nb_runner.get_output(1)
        assert "close_to_pi=True" in nb_runner.get_output(2)

    def test_fractions_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction\nparts = [Fraction(1, n) for n in range(1, 6)]\ntotal = sum(parts)\nprint(f'total={total}')",
            "as_float = float(total)\nprint(f'float={as_float:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1 + 1/2 + 1/3 + 1/4 + 1/5 = 60/60 + 30/60 + 20/60 + 15/60 + 12/60 = 137/60
        assert "total=137/60" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=137/60" in nb_runner.get_output(1)
