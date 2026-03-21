"""Batch 340: math module functions and numeric transformations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathFunctions:
    def test_math_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nangle = math.pi / 4",
            "sin_val = round(math.sin(angle), 4)\ncos_val = round(math.cos(angle), 4)\nprint(f'sin={sin_val} cos={cos_val}')",
            "hyp = round(math.hypot(3, 4), 1)\nprint(f'hyp={hyp}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sin=0.7071" in nb_runner.get_output(2)
        assert "cos=0.7071" in nb_runner.get_output(2)
        assert "hyp=5.0" in nb_runner.get_output(3)

    def test_math_edit_angle(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nangle = math.pi / 6",
            "val = round(math.sin(angle), 1)\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=0.5" in nb_runner.get_output(2)
        # Edit angle
        nb_runner.set_cell_source(1, "import math\nangle = math.pi / 2")
        nb_runner.run_all()
        assert "val=1.0" in nb_runner.get_output(2)

    def test_math_combinatorics(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nn = 10\nk = 3",
            "comb = math.comb(n, k)\nperm = math.perm(n, k)\nprint(f'comb={comb} perm={perm}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "comb=120 perm=720" in nb_runner.get_output(2)
