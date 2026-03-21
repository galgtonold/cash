"""
Batch 43: Simulation and numerical computation patterns —
Monte Carlo, optimization, statistical tests, and numerical methods.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestMonteCarloSimulation:
    """Test Monte Carlo patterns across cells."""

    def test_pi_estimation(self, nb_runner):
        """Estimate pi using Monte Carlo across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "pi_est=" in output
        # Should be close to pi (3.14...)
        val = float(output.split("pi_est=")[1].strip())
        assert abs(val - 3.14159) < 0.1

    def test_random_walk(self, nb_runner):
        """Random walk simulation across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                steps = np.random.choice([-1, 1], size=1000)
                walk = np.cumsum(steps)
            """),
            textwrap.dedent("""\
                final_pos = walk[-1]
                max_pos = walk.max()
                min_pos = walk.min()
                print(f"final={final_pos} max={max_pos} min={min_pos}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "final=" in output
        assert "max=" in output
        assert "min=" in output


class TestStatisticalTests:
    """Test statistical computation patterns."""

    def test_descriptive_statistics(self, nb_runner):
        """Descriptive stats computed across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "mean=" in output
        # Mean should be close to 100
        mean_val = float(output.split("mean=")[1].split(" ")[0])
        assert abs(mean_val - 100) < 5

    def test_correlation_analysis(self, nb_runner):
        """Correlation analysis across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "corr=" in output
        # Should be high positive correlation
        corr = float(output.split("corr=")[1].strip())
        assert corr > 0.9


class TestNumericalMethods:
    """Test numerical methods across cells."""

    def test_numerical_derivative(self, nb_runner):
        """Numerical derivative across cells."""
        nb_runner.create_notebook([
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
        ])
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
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # Integral of sin(x) from 0 to pi = 2
        val = float(output.split("integral=")[1].strip())
        assert abs(val - 2.0) < 0.01


class TestOptimizationPatterns:
    """Test simple optimization patterns."""

    def test_gradient_descent(self, nb_runner):
        """Simple gradient descent across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        # x should converge to 3
        val = float(output.split("final_x=")[1].split(" ")[0])
        assert abs(val - 3.0) < 0.001

    def test_bisection_method(self, nb_runner):
        """Bisection method to find root."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        root = float(output.split("root=")[1].split(" ")[0])
        assert abs(root - 1.5214) < 0.01
