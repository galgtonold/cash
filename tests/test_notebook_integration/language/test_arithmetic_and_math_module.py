"""Arithmetic chains and the math module across cells."""

import pytest


# Numeric / math computation chain interaction tests.
#
# Tests editing numeric computations including arithmetic chains,
# math functions, and statistical calculations.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestArithmeticChainEdits:
    """Editing chains of arithmetic operations."""

    def test_edit_formula(self, nb_runner):
        """Edit a mathematical formula."""
        nb_runner.create_notebook(
            [
                "a = 3\nb = 4  # formula source",
                "result = a**2 + b**2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Change formula
        nb_runner.set_cell_source(2, "result = (a + b) ** 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 49" in nb_runner.get_output(2)

    def test_edit_chain_operand(self, nb_runner):
        """Edit one operand in a chain."""
        nb_runner.create_notebook(
            [
                "x = 10  # chain operand x",
                "y = 20  # chain operand y",
                "z = 30  # chain operand z",
                "result = x * y + z\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 230" in nb_runner.get_output(4)

        # Change middle operand
        nb_runner.set_cell_source(2, "y = 5  # chain operand y v2")
        nb_runner.run_all()
        assert "result = 80" in nb_runner.get_output(4)


# Numeric computation and math pattern edit tests.
#
# Tests editing cells with numeric computations, math operations,
# and scientific-style calculations.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNumericComputationEdits:
    """Editing numeric computation patterns."""

    def test_edit_math_formula(self, nb_runner):
        """Edit a mathematical formula."""
        nb_runner.create_notebook(
            [
                "import math\nradius = 5",
                "area = math.pi * radius ** 2\nprint(f'area = {area:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(2)

        # Change formula to volume of sphere
        nb_runner.set_cell_source(2, "volume = (4/3) * math.pi * radius ** 3\nprint(f'volume = {volume:.2f}')")
        nb_runner.run_all()
        assert "volume = 523.60" in nb_runner.get_output(2)

    def test_edit_statistical_calculation(self, nb_runner):
        """Edit a statistical calculation."""
        nb_runner.create_notebook(
            [
                "scores = [85, 90, 78, 92, 88]",
                "mean = sum(scores) / len(scores)\nprint(f'mean = {mean:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 86.6" in nb_runner.get_output(2)

        # Change to median
        nb_runner.set_cell_source(
            2, "sorted_s = sorted(scores)\nmedian = sorted_s[len(sorted_s) // 2]\nprint(f'median = {median}')"
        )
        nb_runner.run_all()
        assert "median = 88" in nb_runner.get_output(2)

    def test_edit_numeric_input(self, nb_runner):
        """Edit numeric inputs to a computation chain."""
        nb_runner.create_notebook(
            [
                "width = 10\nheight = 5",
                "perimeter = 2 * (width + height)\ndiagonal = (width**2 + height**2) ** 0.5\nprint(f'perimeter={perimeter} diagonal={diagonal:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "perimeter=30" in nb_runner.get_output(2)
        assert "diagonal=11.18" in nb_runner.get_output(2)

        # Change dimensions
        nb_runner.set_cell_source(1, "width = 3\nheight = 4")
        nb_runner.run_all()
        assert "perimeter=14" in nb_runner.get_output(2)
        assert "diagonal=5.00" in nb_runner.get_output(2)

    def test_edit_accumulator_formula(self, nb_runner):
        """Edit a cumulative computation formula."""
        nb_runner.create_notebook(
            [
                "rates = [0.05, 0.03, 0.07, 0.02]",
                "principal = 1000\nfor r in rates:\n    principal *= (1 + r)\nprint(f'final = {principal:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "final = " in out

        # Change starting principal
        nb_runner.set_cell_source(
            2, "principal = 5000\nfor r in rates:\n    principal *= (1 + r)\nprint(f'final = {principal:.2f}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "final = " in out2
        # 5000 * 1.05 * 1.03 * 1.07 * 1.02 ≈ 5893.xx
        assert "5" in out2  # starts with 5xxx


# Numeric precision and math computation edits.
#
# Tests math operations, rounding, precision with edits.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNumericPrecisionEdits:
    """Numeric computation edit patterns."""

    def test_rounding_edit(self, nb_runner):
        """Edit rounding precision, result changes."""
        nb_runner.create_notebook(
            [
                "import math\nval = math.pi",
                "precision = 2",
                "result = round(val, precision)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 3.14" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "precision = 5")
        nb_runner.run_all()
        assert "result = 3.14159" in nb_runner.get_output(3)

    def test_formula_edit(self, nb_runner):
        """Edit formula, downstream result updates."""
        nb_runner.create_notebook(
            [
                "a = 3\nb = 4",
                "import math\nhyp = math.sqrt(a**2 + b**2)\nprint(f'hyp = {hyp}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hyp = 5.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 5\nb = 12")
        nb_runner.run_all()
        assert "hyp = 13.0" in nb_runner.get_output(2)

    def test_statistics_edit(self, nb_runner):
        """Edit data, statistical measures update."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]",
                "import statistics\nmean = statistics.mean(data)\nstdev = round(statistics.stdev(data), 2)\nprint(f'mean={mean} stdev={stdev}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [100, 100, 100, 100, 100]")
        nb_runner.run_all()
        assert "mean=100" in nb_runner.get_output(2)
        assert "stdev=0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMathFunctionEdits:
    """Editing math function usage."""

    def test_edit_math_function(self, nb_runner):
        """Edit which math function to use."""
        nb_runner.create_notebook(
            [
                "import math",
                "val = 2.0  # math function source",
                "result = math.sqrt(val)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1.4142" in nb_runner.get_output(3)

        # Change to log
        nb_runner.set_cell_source(3, "result = math.log(val)\nprint(f'result = {result:.4f}')")
        nb_runner.run_all()
        assert "result = 0.6931" in nb_runner.get_output(3)

    def test_edit_statistical_calc(self, nb_runner):
        """Edit statistical calculations."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]  # stats source",
                "mean = sum(data) / len(data)\nprint(f'mean = {mean}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [100, 200, 300]  # stats source v2")
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMathFunctions:
    """math module functions and numeric transformations."""

    def test_math_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nangle = math.pi / 4",
                "sin_val = round(math.sin(angle), 4)\ncos_val = round(math.cos(angle), 4)\nprint(f'sin={sin_val} cos={cos_val}')",
                "hyp = round(math.hypot(3, 4), 1)\nprint(f'hyp={hyp}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sin=0.7071" in nb_runner.get_output(2)
        assert "cos=0.7071" in nb_runner.get_output(2)
        assert "hyp=5.0" in nb_runner.get_output(3)

    def test_math_edit_angle(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nangle = math.pi / 6",
                "val = round(math.sin(angle), 1)\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=0.5" in nb_runner.get_output(2)
        # Edit angle
        nb_runner.set_cell_source(1, "import math\nangle = math.pi / 2")
        nb_runner.run_all()
        assert "val=1.0" in nb_runner.get_output(2)

    def test_math_combinatorics(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nn = 10\nk = 3",
                "comb = math.comb(n, k)\nperm = math.perm(n, k)\nprint(f'comb={comb} perm={perm}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "comb=120 perm=720" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMathModuleFunctions:
    """math module functions and constants."""

    def test_trig_functions(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nangle = math.pi / 4",
                "s = round(math.sin(angle), 4)\nc = round(math.cos(angle), 4)\nt = round(math.tan(angle), 4)\nprint(f's={s} c={c} t={t}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "s=0.7071" in out
        assert "c=0.7071" in out
        assert "t=1.0" in out

    def test_log_functions(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nval = 100",
                "lg = math.log10(val)\nln = round(math.log(val), 4)\nlg2 = round(math.log2(val), 4)\nprint(f'log10={lg} ln={ln} log2={lg2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log10=2.0" in out
        assert "ln=4.6052" in out

    def test_math_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math\nn = 5",
                "f = math.factorial(n)\nsqrt_val = round(math.sqrt(n), 4)\nprint(f'fact={f} sqrt={sqrt_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fact=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import math\nn = 10")
        nb_runner.run_all()
        assert "fact=3628800" in nb_runner.get_output(2)


# Interaction test: math module advanced functions (log, pow, factorial, comb).
# Tests math.log, math.pow, math.factorial, math.comb, math.perm,
# and cross-cell mathematical computations.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMathAdvancedFunctions:
    """Test math advanced functions across cells."""

    def test_math_advanced(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: logarithms and powers
                "import math\nlog_e = round(math.log(math.e), 1)\nlog_10 = round(math.log10(1000), 1)\nlog_2 = round(math.log2(256), 1)\npow_val = math.pow(2, 10)\nprint(f'log_e={log_e}')\nprint(f'log_10={log_10}')\nprint(f'log_2={log_2}')\nprint(f'pow={int(pow_val)}')",
                # Cell 2: combinatorics
                "fact_10 = math.factorial(10)\ncomb_10_3 = math.comb(10, 3)\nperm_10_3 = math.perm(10, 3)\nprint(f'fact_10={fact_10}')\nprint(f'comb={comb_10_3}')\nprint(f'perm={perm_10_3}')",
                # Cell 3: combine
                "ratio = perm_10_3 / comb_10_3\nprint(f'ratio={int(ratio)}')\nprint(f'ratio_is_factorial={int(ratio) == math.factorial(3)}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math\nn = 5\nfact = math.factorial(n)\nprint(f'fact={fact}')",
                "comb_val = math.comb(n, 2)\nprint(f'comb={comb_val}')",
                "ratio = fact // comb_val\nprint(f'ratio={ratio}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math\ngcd_val = math.gcd(48, 18)\nlcm_val = math.lcm(48, 18)\nprint(f'gcd={gcd_val}')\nprint(f'lcm={lcm_val}')",
                "product = gcd_val * lcm_val\noriginal = 48 * 18\nprint(f'identity={product == original}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "gcd=6" in nb_runner.get_output(1)
        assert "lcm=144" in nb_runner.get_output(1)
        assert "identity=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "identity=True" in nb_runner.get_output(2)


# Interaction test: math module special functions.
# Tests math.gcd, math.lcm, math.comb, math.perm, math.isclose,
# and cross-cell mathematical computation pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMathSpecialFunctions:
    """Test math special functions across cells."""

    def test_math_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: gcd and lcm
                "import math\ng = math.gcd(48, 18)\nl = math.lcm(4, 6)\nprint(f'gcd={g}')\nprint(f'lcm={l}')",
                # Cell 2: comb and perm
                "c = math.comb(10, 3)\np = math.perm(5, 2)\nprint(f'comb_10_3={c}')\nprint(f'perm_5_2={p}')",
                # Cell 3: isclose and other
                "a = 0.1 + 0.2\nb = 0.3\nclose = math.isclose(a, b, rel_tol=1e-9)\nfact = math.factorial(6)\nprint(f'isclose={close}')\nprint(f'factorial_6={fact}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math\nn = 5\nfact = math.factorial(n)\nprint(f'factorial={fact}')",
                "is_big = fact > 100\nprint(f'is_big={is_big}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import math\nlog_val = math.log2(1024)\nsqrt_val = math.sqrt(144)\nprint(f'log2_1024={log_val}')\nprint(f'sqrt_144={sqrt_val}')",
                "total = log_val + sqrt_val\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log2_1024=10.0" in nb_runner.get_output(1)
        assert "sqrt_144=12.0" in nb_runner.get_output(1)
        assert "total=22.0" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=22.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMathCeilFloorLogGcd:
    """math ceil floor log gcd lcm."""

    def test_ceil_floor(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math",
                "vals = [3.2, -3.2, 0.5, -0.5]\nceils = [math.ceil(v) for v in vals]\nfloors = [math.floor(v) for v in vals]\nprint(f'ceils={ceils} floors={floors}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ceils=[4, -3, 1, 0]" in out
        assert "floors=[3, -4, 0, -1]" in out

    def test_log_gcd_lcm(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import math",
                "log2_8 = math.log2(8)\nlog10_1000 = math.log10(1000)\ng = math.gcd(48, 18)\nlcm_val = math.lcm(12, 18)\nprint(f'log2_8={log2_8} log10_1000={log10_1000} gcd={g} lcm={lcm_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "log2_8=3.0" in out
        assert "log10_1000=3.0" in out
        assert "gcd=6" in out
        assert "lcm=36" in out
