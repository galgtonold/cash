"""
Interaction test: math module advanced functions (log, pow, factorial, comb).
Tests math.log, math.pow, math.factorial, math.comb, math.perm,
and cross-cell mathematical computations.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMathAdvancedFunctions:
    """Test math advanced functions across cells."""

    def test_math_advanced(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: logarithms and powers
            "import math\nlog_e = round(math.log(math.e), 1)\nlog_10 = round(math.log10(1000), 1)\nlog_2 = round(math.log2(256), 1)\npow_val = math.pow(2, 10)\nprint(f'log_e={log_e}')\nprint(f'log_10={log_10}')\nprint(f'log_2={log_2}')\nprint(f'pow={int(pow_val)}')",
            # Cell 2: combinatorics
            "fact_10 = math.factorial(10)\ncomb_10_3 = math.comb(10, 3)\nperm_10_3 = math.perm(10, 3)\nprint(f'fact_10={fact_10}')\nprint(f'comb={comb_10_3}')\nprint(f'perm={perm_10_3}')",
            # Cell 3: combine
            "ratio = perm_10_3 / comb_10_3\nprint(f'ratio={int(ratio)}')\nprint(f'ratio_is_factorial={int(ratio) == math.factorial(3)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "log_e=1.0" in out1
        assert "log_10=3.0" in out1
        assert "log_2=8.0" in out1
        assert "pow=1024" in out1
        out2 = nb_runner.get_output(2)
        assert "fact_10=3628800" in out2
        assert "comb=120" in out2
        assert "perm=720" in out2
        out3 = nb_runner.get_output(3)
        assert "ratio=6" in out3
        assert "ratio_is_factorial=True" in out3

    def test_math_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import math\nn = 5\nfact = math.factorial(n)\nprint(f'fact={fact}')",
            "comb_val = math.comb(n, 2)\nprint(f'comb={comb_val}')",
            "ratio = fact // comb_val\nprint(f'ratio={ratio}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fact=120" in nb_runner.get_output(1)
        assert "comb=10" in nb_runner.get_output(2)
        assert "ratio=12" in nb_runner.get_output(3)

        # Change n
        nb_runner.set_cell_source(1, "import math\nn = 8\nfact = math.factorial(n)\nprint(f'fact={fact}')")
        nb_runner.run_cells([1, 2, 3])
        assert "fact=40320" in nb_runner.get_output(1)
        assert "comb=28" in nb_runner.get_output(2)
        assert "ratio=1440" in nb_runner.get_output(3)

    def test_math_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import math\ngcd_val = math.gcd(48, 18)\nlcm_val = math.lcm(48, 18)\nprint(f'gcd={gcd_val}')\nprint(f'lcm={lcm_val}')",
            "product = gcd_val * lcm_val\noriginal = 48 * 18\nprint(f'identity={product == original}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "gcd=6" in nb_runner.get_output(1)
        assert "lcm=144" in nb_runner.get_output(1)
        assert "identity=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "identity=True" in nb_runner.get_output(2)
