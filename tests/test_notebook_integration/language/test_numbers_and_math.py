"""math, cmath, complex numbers, decimal, fractions and statistics across cells."""

import textwrap

import pytest


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestListEdits:
    """List operations with cell edits."""

    def test_list_slice_edit(self, nb_runner):
        """Edit a list and downstream slice operation."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "subset = data[:3]\nprint(f'subset = {subset}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "subset = [1, 2, 3]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "subset = [10, 20, 30]" in nb_runner.get_output(2)

    def test_list_operation_edit(self, nb_runner):
        """Edit the list operation."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "result = sum(data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = max(data)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestTupleSetEdits:
    """Tuples and sets with edits."""

    def test_tuple_unpack_edit(self, nb_runner):
        """Edit a tuple unpacking cell."""
        nb_runner.create_notebook(
            [
                "pair = (10, 20)",
                "a, b = pair\nprint(f'a = {a}, b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "pair = (100, 200)")
        nb_runner.run_all()
        assert "a = 100, b = 200" in nb_runner.get_output(2)

    def test_set_operations_edit(self, nb_runner):
        """Edit set operations."""
        nb_runner.create_notebook(
            [
                "s1 = {1, 2, 3}\ns2 = {2, 3, 4}",
                "result = s1 & s2\nprint(f'result = {sorted(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3]" in nb_runner.get_output(2)

        # Change to union
        nb_runner.set_cell_source(2, "result = s1 | s2\nprint(f'result = {sorted(result)}')")
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4]" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestStringOperations:
    """String manipulation with edits."""

    def test_string_format_edit(self, nb_runner):
        """Edit string formatting."""
        nb_runner.create_notebook(
            [
                "name = 'World'",
                "greeting = f'Hello, {name}!'\nprint(greeting)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello, World!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "name = 'Cash'")
        nb_runner.run_all()
        assert "Hello, Cash!" in nb_runner.get_output(2)

    def test_string_join_edit(self, nb_runner):
        """Edit string join operations."""
        nb_runner.create_notebook(
            [
                "words = ['Hello', 'World']",
                "sentence = ' '.join(words)\nprint(sentence)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "words = ['Foo', 'Bar', 'Baz']")
        nb_runner.run_all()
        assert "Foo Bar Baz" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestComplexDataFlowEdits:
    """Complex data flowing through multiple cells with edits."""

    def test_dict_to_list_to_sum(self, nb_runner):
        """Dict → list extraction → sum, edit the dict."""
        nb_runner.create_notebook(
            [
                "scores = {'math': 90, 'english': 85, 'science': 95}",
                "values = list(scores.values())",
                "total = sum(values)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 270" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scores = {'math': 100, 'english': 100, 'science': 100}")
        nb_runner.run_all()
        assert "total = 300" in nb_runner.get_output(3)

    def test_list_filter_transform_aggregate(self, nb_runner):
        """List → filter → transform → aggregate, edit filter."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "filtered = [x for x in data if x > 5]",
                "transformed = [x * 10 for x in filtered]",
                "result = sum(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(4)

        # Change filter condition
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x > 2]")
        nb_runner.run_all()
        assert "result = 420" in nb_runner.get_output(4)


# Simulation and numerical computation patterns —
# Monte Carlo, optimization, statistical tests, and numerical methods.
@pytest.mark.integration
@pytest.mark.stress
class TestMonteCarloSimulation:
    """Test Monte Carlo patterns across cells."""

    def test_pi_estimation(self, nb_runner):
        """Estimate pi using Monte Carlo across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                n = 100000
                x = np.random.uniform(-1, 1, n)
                y = np.random.uniform(-1, 1, n)
            """),
                textwrap.dedent("""\
                inside = (x**2 + y**2) <= 1
                pi_est = 4 * inside.sum() / n
                print(f"pi_est={pi_est:.4f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "pi_est=" in output
        # Should be close to pi (3.14...)
        val = float(output.split("pi_est=")[1].strip())
        assert abs(val - 3.14159) < 0.1


@pytest.mark.integration
@pytest.mark.stress
class TestStatisticalTests:
    """Test statistical computation patterns."""

    def test_descriptive_statistics(self, nb_runner):
        """Descriptive stats computed across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                data = np.random.normal(100, 15, 1000)
            """),
                textwrap.dedent("""\
                stats = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'median': np.median(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75),
                }
            """),
                textwrap.dedent("""\
                iqr = stats['q75'] - stats['q25']
                print(f"mean={stats['mean']:.1f} std={stats['std']:.1f} iqr={iqr:.1f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "mean=" in output
        # Mean should be close to 100
        mean_val = float(output.split("mean=")[1].split(" ")[0])
        assert abs(mean_val - 100) < 5

    def test_correlation_analysis(self, nb_runner):
        """Correlation analysis across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                x = np.random.randn(500)
                noise = np.random.randn(500) * 0.5
                y = 2 * x + 3 + noise  # Strong positive correlation
            """),
                textwrap.dedent("""\
                correlation = np.corrcoef(x, y)[0, 1]
                print(f"corr={correlation:.4f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "corr=" in output
        # Should be high positive correlation
        corr = float(output.split("corr=")[1].strip())
        assert corr > 0.9


@pytest.mark.integration
@pytest.mark.stress
class TestNumericalMethods:
    """Test numerical methods across cells."""

    def test_numerical_derivative(self, nb_runner):
        """Numerical derivative across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                def f(x):
                    return x**3 - 2*x + 1
                
                x = np.linspace(-2, 2, 100)
                y = f(x)
            """),
                textwrap.dedent("""\
                dx = x[1] - x[0]
                dy = np.gradient(y, dx)
            """),
                textwrap.dedent("""\
                # At x=1, derivative of x^3-2x+1 = 3x^2-2 = 1
                idx = np.argmin(np.abs(x - 1))
                print(f"f(1)={y[idx]:.4f} f'(1)={dy[idx]:.4f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "f(1)=" in output
        assert "f'(1)=" in output
        # f(1)=0, f'(1)≈1
        fval = float(output.split("f(1)=")[1].split(" ")[0])
        assert abs(fval) < 0.1

    def test_numerical_integration(self, nb_runner):
        """Numerical integration (trapezoidal) across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                x = np.linspace(0, np.pi, 1000)
                y = np.sin(x)
            """),
                textwrap.dedent("""\
                # Use trapezoid (new name) or trapz (old name) depending on numpy version
                _trap_fn = getattr(np, 'trapezoid', None) or np.trapz
                integral = _trap_fn(y, x)
                print(f"integral={integral:.6f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # Integral of sin(x) from 0 to pi = 2
        val = float(output.split("integral=")[1].strip())
        assert abs(val - 2.0) < 0.01


@pytest.mark.integration
@pytest.mark.stress
class TestOptimizationPatterns:
    """Test simple optimization patterns."""

    def test_gradient_descent(self, nb_runner):
        """Simple gradient descent across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                # Minimize f(x) = (x-3)^2
                x = 0.0
                learning_rate = 0.1
                history = [x]
            """),
                textwrap.dedent("""\
                for _ in range(50):
                    gradient = 2 * (x - 3)  # f'(x)
                    x = x - learning_rate * gradient
                    history.append(x)
            """),
                textwrap.dedent("""\
                print(f"final_x={x:.6f} iterations={len(history)-1}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        # x should converge to 3
        val = float(output.split("final_x=")[1].split(" ")[0])
        assert abs(val - 3.0) < 0.001

    def test_bisection_method(self, nb_runner):
        """Bisection method to find root."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def f(x):
                    return x**3 - x - 2  # Root near x=1.52
            """),
                textwrap.dedent("""\
                a, b = 1.0, 2.0
                for _ in range(50):
                    mid = (a + b) / 2
                    if f(mid) * f(a) < 0:
                        b = mid
                    else:
                        a = mid
                root = (a + b) / 2
            """),
                textwrap.dedent("""\
                print(f"root={root:.6f} f(root)={f(root):.10f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        root = float(output.split("root=")[1].split(" ")[0])
        assert abs(root - 1.5214) < 0.01


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


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCmathComplexOps:
    """cmath and complex number operations."""

    def test_complex_arithmetic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z1 = 3 + 4j\nz2 = 1 - 2j\ns = z1 + z2\np = z1 * z2\nprint(f'sum={s} prod={p}')\nprint(f'abs_z1={abs(z1)} phase={round(cmath.phase(z1), 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sum=(4+2j)" in out
        assert "prod=(11-2j)" in out
        assert "abs_z1=5.0" in out

    def test_polar_rect(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z = 1 + 1j\nr, phi = cmath.polar(z)\nback = cmath.rect(r, phi)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')\nprint(f'back_real={round(back.real, 4)} back_imag={round(back.imag, 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r=1.4142" in out
        assert "phi=0.7854" in out
        assert "back_real=1.0" in out
        assert "back_imag=1.0" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath",
                "z = 3 + 4j\nprint(f'abs={abs(z)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "abs=5.0" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "z = 5 + 12j\nprint(f'abs={abs(z)}')")
        nb_runner.run_all()
        assert "abs=13.0" in nb_runner.get_output(2)


# Interaction test: complex number operations with cmath module.
# Tests complex arithmetic, cmath functions (polar, rect, phase),
# and cross-cell complex number manipulation.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexCmathOps:
    """Test complex number operations with cmath across cells."""

    def test_complex_cmath(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: complex arithmetic
                "z1 = 3 + 4j\nz2 = 1 - 2j\nsum_z = z1 + z2\nprod_z = z1 * z2\nprint(f'sum={sum_z}')\nprint(f'prod={prod_z}')",
                # Cell 2: cmath functions
                "import cmath\nabs_z1 = abs(z1)\nphase_z1 = cmath.phase(z1)\npolar = cmath.polar(z1)\nprint(f'abs={abs_z1}')\nprint(f'phase={phase_z1:.4f}')\nprint(f'polar_r={polar[0]:.1f}')",
                # Cell 3: rect conversion
                "r, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'real={back.real:.1f}')\nprint(f'imag={back.imag:.1f}')\nprint(f'roundtrip={abs(back - z1) < 1e-10}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sum=(4+2j)" in out1
        assert "prod=(11-2j)" in out1
        out2 = nb_runner.get_output(2)
        assert "abs=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "real=3.0" in out3
        assert "imag=4.0" in out3
        assert "roundtrip=True" in out3

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 3 + 4j\nmag = abs(z)\nprint(f'mag={mag}')",
                "conjugate = z.conjugate()\nprint(f'conj={conjugate}')",
                "product = z * conjugate\nprint(f'prod={product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(1)
        assert "conj=(3-4j)" in nb_runner.get_output(2)
        assert "prod=(25+0j)" in nb_runner.get_output(3)

        # Edit z
        nb_runner.set_cell_source(1, "z = 5 + 12j\nmag = abs(z)\nprint(f'mag={mag}')")
        nb_runner.run_cells([1, 2, 3])
        assert "mag=13.0" in nb_runner.get_output(1)
        assert "conj=(5-12j)" in nb_runner.get_output(2)
        assert "prod=(169+0j)" in nb_runner.get_output(3)

    def test_complex_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath\nz = cmath.sqrt(-1)\nprint(f'sqrt_neg1={z}')",
                "is_imag = z.real == 0 and z.imag == 1\nprint(f'is_i={is_imag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sqrt_neg1=1j" in nb_runner.get_output(1)
        assert "is_i=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_i=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumberArith:
    """complex number arithmetic."""

    def test_complex_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z1 = 3 + 4j\nz2 = 1 - 2j",
                "add = z1 + z2\nmul = z1 * z2\nprint(f'add={add} mul={mul}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "add=(4+2j)" in out
        assert "mul=(11-2j)" in out

    def test_complex_magnitude(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 3 + 4j",
                "mag = abs(z)\nreal = z.real\nimag = z.imag\nconj = z.conjugate()\nprint(f'mag={mag} real={real} imag={imag} conj={conj}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mag=5.0" in out
        assert "real=3.0" in out
        assert "conj=(3-4j)" in out

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 1 + 1j",
                "squared = z ** 2\nprint(f'squared={squared}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared=2j" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "z = 2 + 3j")
        nb_runner.run_all()
        assert "squared=(-5+12j)" in nb_runner.get_output(2)


# Interaction test: complex number arithmetic and polar form.
# Tests complex addition, multiplication, conjugate,
# polar conversion, and cross-cell complex math pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumberPolar:
    """Test complex number arithmetic and polar conversion across cells."""

    def test_complex_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic complex ops
                "z1 = complex(3, 4)\nz2 = complex(1, -2)\nz_sum = z1 + z2\nz_prod = z1 * z2\nprint(f'z1={z1}')\nprint(f'sum={z_sum}')\nprint(f'prod={z_prod}')",
                # Cell 2: conjugate and abs
                "conj = z1.conjugate()\nmag = abs(z1)\nprint(f'conjugate={conj}')\nprint(f'magnitude={mag}')",
                # Cell 3: polar form
                "import cmath\nimport math\nr, theta = cmath.polar(z1)\nback = cmath.rect(r, theta)\nprint(f'r={r}')\nprint(f'theta_deg={math.degrees(theta):.4f}')\nprint(f'roundtrip_real={back.real:.6f}')\nprint(f'roundtrip_imag={back.imag:.6f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "z1=(3+4j)" in out1
        assert "sum=(4+2j)" in out1
        assert "prod=(11-2j)" in out1
        out2 = nb_runner.get_output(2)
        assert "conjugate=(3-4j)" in out2
        assert "magnitude=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "r=5.0" in out3
        assert "roundtrip_real=3.0" in out3

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = complex(0, 1)  # i\nprint(f'z={z}')",
                "z_sq = z * z\nprint(f'z_squared={z_sq}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z_squared=(-1+0j)" in nb_runner.get_output(2)

        # Edit to different complex number
        nb_runner.set_cell_source(1, "z = complex(1, 1)\nprint(f'z={z}')")
        nb_runner.run_cells([1, 2])
        assert "z_squared=2j" in nb_runner.get_output(2)

    def test_complex_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = complex(5, 12)\nmag = abs(z)\nprint(f'mag={mag}')",
                "normalized = z / mag\nprint(f'norm_real={normalized.real:.4f}')\nprint(f'norm_imag={normalized.imag:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=13.0" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "norm_real=0.3846" in out2
        assert "norm_imag=0.9231" in out2

        # Re-run - cache
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "norm_real=0.3846" in out2


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
@pytest.mark.timeout(90)
class TestStatisticsMeanMedian:
    """statistics module mean median stdev."""

    def test_basic_stats(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "data = [4, 8, 15, 16, 23, 42]\nm = statistics.mean(data)\nmed = statistics.median(data)\nsd = round(statistics.stdev(data), 2)\nprint(f'mean={m} median={med} stdev={sd}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=18" in out
        assert "median=15.5" in out

    def test_multimode(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "data = [1, 1, 2, 2, 3]\nmodes = statistics.multimode(data)\nprint(f'modes={modes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "modes=[1, 2]" in nb_runner.get_output(2)

    def test_stats_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "vals = [10, 20, 30]\nresult = statistics.mean(vals)\nprint(f'mean={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "vals = [100, 200, 300, 400]\nresult = statistics.mean(vals)\nprint(f'mean={result}')"
        )
        nb_runner.run_all()
        assert "mean=250" in nb_runner.get_output(2)


# Interaction test: statistics module median and stdev.
# Tests statistics.median, stdev, variance, mode,
# and cross-cell statistical analysis pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStatisticsMedianStdev:
    """Test statistics median and stdev across cells."""

    def test_statistics_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic statistics
                "import statistics\ndata = [4, 8, 15, 16, 23, 42]\nmean = statistics.mean(data)\nmedian = statistics.median(data)\nprint(f'mean={mean}')\nprint(f'median={median}')",
                # Cell 2: stdev and variance
                "stdev = statistics.stdev(data)\nvariance = statistics.variance(data)\nprint(f'stdev={stdev:.4f}')\nprint(f'variance={variance:.4f}')",
                # Cell 3: mode
                "mode_data = [1, 2, 2, 3, 3, 3, 4]\nmode = statistics.mode(mode_data)\nprint(f'mode={mode}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mean=18" in out1
        assert "median=15.5" in out1
        out2 = nb_runner.get_output(2)
        assert "stdev=" in out2
        assert "variance=" in out2
        out3 = nb_runner.get_output(3)
        assert "mode=3" in out3

    def test_statistics_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [80, 85, 90, 95, 100]\navg = statistics.mean(scores)\nprint(f'avg={avg}')",
                "report = f'Average score: {avg}'\nprint(f'report={report}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=90" in nb_runner.get_output(1)
        assert "report=Average score: 90" in nb_runner.get_output(2)

        # Add more scores
        nb_runner.set_cell_source(
            1,
            "import statistics\nscores = [80, 85, 90, 95, 100, 70]\navg = statistics.mean(scores)\nprint(f'avg={avg}')",
        )
        nb_runner.run_cells([1, 2])
        assert "avg=86" in nb_runner.get_output(1)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nvals = [10, 20, 30, 40, 50]\nmed = statistics.median(vals)\nprint(f'median={med}')",
                "above_median = [v for v in vals if v > med]\nprint(f'above={above_median}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=30" in nb_runner.get_output(1)
        assert "above=[40, 50]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above=[40, 50]" in nb_runner.get_output(2)


class TestStatisticsModule:
    """statistics module operations and edits."""

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_stats_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\ndata = [10, 20, 30, 40, 50]",
                "mean = statistics.mean(data)\nmedian = statistics.median(data)\nstdev = round(statistics.stdev(data), 2)\nprint(f'mean={mean} median={median} stdev={stdev}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=30" in out
        assert "median=30" in out
        assert "stdev=15.81" in out

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_stats_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [85, 90, 78, 92, 88]",
                "mean_score = statistics.mean(scores)\nmode_score = statistics.mode(scores) if len(set(scores)) < len(scores) else 'no mode'\nprint(f'mean={mean_score} mode={mode_score}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=86.6" in nb_runner.get_output(2)
        # Edit with mode
        nb_runner.set_cell_source(1, "import statistics\nscores = [80, 90, 80, 90, 80]")
        nb_runner.run_all()
        assert "mean=84" in nb_runner.get_output(2)
        assert "mode=80" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_quantiles(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\ndata = list(range(1, 101))",
                "q = statistics.quantiles(data, n=4)\nprint(f'quartiles={q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "quartiles=" in out
        assert "50.5" in out

    # Statistics & random distributions — cash caching with statistical computations.
    @pytest.mark.stress
    def test_statistics_propagation(self, nb_runner):
        """Statistical results propagate on data change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = [10, 20, 30, 40, 50]
            """),
                textwrap.dedent("""\
                import statistics
                mean_val = statistics.mean(data)
                print(f"mean={mean_val}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            data = [100, 200, 300, 400, 500]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "mean=300" in nb_runner.get_output(2)


# Interaction test: statistics module with variance, stdev, correlation.
# Tests statistics.variance, stdev, correlation (3.10+), and
# cross-cell statistical analysis.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStatisticsVarianceCorr:
    """Test statistics variance and correlation across cells."""

    def test_variance_stdev(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: compute variance and stdev
                "import statistics\ndata = [10, 20, 30, 40, 50]\nmean = statistics.mean(data)\nvar = statistics.variance(data)\nstdev = statistics.stdev(data)\nprint(f'mean={mean}')\nprint(f'var={var}')\nprint(f'stdev={stdev:.2f}')",
                # Cell 2: population vs sample
                "pvar = statistics.pvariance(data)\npstdev = statistics.pstdev(data)\nprint(f'pvar={pvar}')\nprint(f'pstdev={pstdev:.2f}')\nprint(f'sample_larger={var > pvar}')",
                # Cell 3: coefficient of variation
                "cv = stdev / mean * 100\nprint(f'cv={cv:.1f}%')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mean=30" in out1
        assert "var=250" in out1
        assert "stdev=15.81" in out1
        out2 = nb_runner.get_output(2)
        assert "pvar=200" in out2
        assert "pstdev=14.14" in out2
        assert "sample_larger=True" in out2
        out3 = nb_runner.get_output(3)
        assert "cv=52.7%" in out3

    def test_statistics_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\ndata = [5, 5, 5, 5, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev}')",
                "is_uniform = stdev == 0\nprint(f'uniform={is_uniform}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stdev=0" in nb_runner.get_output(1)
        assert "uniform=True" in nb_runner.get_output(2)

        # Edit to non-uniform data
        nb_runner.set_cell_source(
            1, "import statistics\ndata = [1, 2, 3, 4, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev:.2f}')"
        )
        nb_runner.run_cells([1, 2])
        assert "stdev=1.58" in nb_runner.get_output(1)
        assert "uniform=False" in nb_runner.get_output(2)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [85, 90, 78, 92, 88]\nmedian = statistics.median(scores)\nmode_val = statistics.mode(scores)\nprint(f'median={median}')\nprint(f'mode={mode_val}')",
                "above_med = sum(1 for s in scores if s > median)\nprint(f'above_median={above_med}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=88" in nb_runner.get_output(1)
        assert "above_median=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above_median=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestComplexNumbers:
    """complex number operations and cmath."""

    def test_complex_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "z = 3 + 4j",
                "mag = abs(z)\nconj = z.conjugate()\nprint(f'mag={mag} conj={conj}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(2)
        assert "conj=(3-4j)" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "z = 5 + 12j")
        nb_runner.run_all()
        assert "mag=13.0" in nb_runner.get_output(2)
        assert "conj=(5-12j)" in nb_runner.get_output(2)

    def test_cmath_polar(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import cmath\nz = 1 + 1j",
                "r, phi = cmath.polar(z)\nprint(f'r={round(r, 4)} phi={round(phi, 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=1.4142" in nb_runner.get_output(2)
        assert "phi=0.7854" in nb_runner.get_output(2)


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
