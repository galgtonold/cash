"""
Interaction test: math module special functions.
Tests math.gcd, math.lcm, math.comb, math.perm, math.isclose,
and cross-cell mathematical computation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathSpecialFunctions:
    """Test math special functions across cells."""

    def test_math_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: gcd and lcm
            "import math\ng = math.gcd(48, 18)\nl = math.lcm(4, 6)\nprint(f'gcd={g}')\nprint(f'lcm={l}')",
            # Cell 2: comb and perm
            "c = math.comb(10, 3)\np = math.perm(5, 2)\nprint(f'comb_10_3={c}')\nprint(f'perm_5_2={p}')",
            # Cell 3: isclose and other
            "a = 0.1 + 0.2\nb = 0.3\nclose = math.isclose(a, b, rel_tol=1e-9)\nfact = math.factorial(6)\nprint(f'isclose={close}')\nprint(f'factorial_6={fact}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "gcd=6" in out1
        assert "lcm=12" in out1
        out2 = nb_runner.get_output(2)
        assert "comb_10_3=120" in out2
        assert "perm_5_2=20" in out2
        out3 = nb_runner.get_output(3)
        assert "isclose=True" in out3
        assert "factorial_6=720" in out3

    def test_math_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nn = 5\nfact = math.factorial(n)\nprint(f'factorial={fact}')",
            "is_big = fact > 100\nprint(f'is_big={is_big}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "factorial=120" in nb_runner.get_output(1)
        assert "is_big=True" in nb_runner.get_output(2)

        # Edit n
        nb_runner.set_cell_source(1, "import math\nn = 3\nfact = math.factorial(n)\nprint(f'factorial={fact}')")
        nb_runner.run_cells([1, 2])
        assert "factorial=6" in nb_runner.get_output(1)
        assert "is_big=False" in nb_runner.get_output(2)

    def test_math_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nlog_val = math.log2(1024)\nsqrt_val = math.sqrt(144)\nprint(f'log2_1024={log_val}')\nprint(f'sqrt_144={sqrt_val}')",
            "total = log_val + sqrt_val\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log2_1024=10.0" in nb_runner.get_output(1)
        assert "sqrt_144=12.0" in nb_runner.get_output(1)
        assert "total=22.0" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=22.0" in nb_runner.get_output(2)
