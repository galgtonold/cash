"""Batch 403: math module functions and constants."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathModuleFunctions:
    def test_trig_functions(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nangle = math.pi / 4",
            "s = round(math.sin(angle), 4)\nc = round(math.cos(angle), 4)\nt = round(math.tan(angle), 4)\nprint(f's={s} c={c} t={t}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "s=0.7071" in out
        assert "c=0.7071" in out
        assert "t=1.0" in out

    def test_log_functions(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nval = 100",
            "lg = math.log10(val)\nln = round(math.log(val), 4)\nlg2 = round(math.log2(val), 4)\nprint(f'log10={lg} ln={ln} log2={lg2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log10=2.0" in out
        assert "ln=4.6052" in out

    def test_math_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nn = 5",
            "f = math.factorial(n)\nsqrt_val = round(math.sqrt(n), 4)\nprint(f'fact={f} sqrt={sqrt_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fact=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import math\nn = 10")
        nb_runner.run_all()
        assert "fact=3628800" in nb_runner.get_output(2)
