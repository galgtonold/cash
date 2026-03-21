"""Batch 512: fractions module exact rational arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFractionsRational:
    def test_fraction_arithmetic(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction",
            "a = Fraction(1, 3)\nb = Fraction(1, 6)\nsum_ab = a + b\nprod = a * b\nprint(f'sum={sum_ab} prod={prod}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sum=1/2" in out
        assert "prod=1/18" in out

    def test_fraction_from_float(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction",
            "f1 = Fraction('0.1') + Fraction('0.2')\nf2 = Fraction(0.1) + Fraction(0.2)\nprint(f'exact={f1}')\nprint(f'float_based={f2}')\nprint(f'exact_eq_03={f1 == Fraction(3, 10)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "exact=3/10" in out
        assert "exact_eq_03=True" in out

    def test_fraction_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction",
            "f = Fraction(3, 4) + Fraction(1, 4)\nprint(f'f={f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "f = Fraction(2, 3) + Fraction(1, 6)\nprint(f'f={f}')")
        nb_runner.run_all()
        assert "f=5/6" in nb_runner.get_output(2)
