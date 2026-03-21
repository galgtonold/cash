"""Batch 416: fractions module exact rational arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFractionsRationalArith:
    def test_fraction_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction\na = Fraction(1, 3)\nb = Fraction(1, 6)",
            "result = a + b\nprint(f'result={result} float={float(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1/2" in nb_runner.get_output(2)
        assert "float=0.5" in nb_runner.get_output(2)

    def test_fraction_from_string(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction\nf = Fraction('3.14')",
            "num = f.numerator\nden = f.denominator\nprint(f'num={num} den={den}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "num=157" in nb_runner.get_output(2)
        assert "den=50" in nb_runner.get_output(2)

    def test_fraction_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from fractions import Fraction\nx = Fraction(2, 5)\ny = Fraction(3, 5)",
            "product = x * y\nprint(f'product={product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product=6/25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from fractions import Fraction\nx = Fraction(1, 2)\ny = Fraction(1, 3)")
        nb_runner.run_all()
        assert "product=1/6" in nb_runner.get_output(2)
