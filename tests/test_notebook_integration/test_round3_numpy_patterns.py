"""
Batch 20: NumPy integration patterns and numerical computing workflows.

Tests how cash handles numpy arrays, matrix operations, random number
generation, and numerical computation pipelines across cells.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestNumpyBasicOperations:
    """Test basic numpy operations across cells."""

    def test_array_creation_and_ops(self, nb_runner):
        """Create numpy array in one cell, operate in another."""
        nb_runner.create_notebook([
            "import numpy as np",
            "arr = np.array([1, 2, 3, 4, 5])",
            textwrap.dedent("""\
                result = arr.sum()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)

    def test_matrix_operations(self, nb_runner):
        """Matrix multiplication across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                A = np.array([[1, 2], [3, 4]])
                B = np.array([[5, 6], [7, 8]])
            """),
            textwrap.dedent("""\
                C = A @ B
                print(C[0, 0], C[1, 1])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert "19" in nb_runner.get_output(3)
        assert "50" in nb_runner.get_output(3)

    def test_broadcasting(self, nb_runner):
        """Broadcasting operations across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            "matrix = np.ones((3, 4))",
            "row = np.array([1, 2, 3, 4])",
            textwrap.dedent("""\
                result = matrix * row
                print(result.sum())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 3 rows * [1,2,3,4] = 3*(1+2+3+4) = 30
        assert "30" in nb_runner.get_output(4)

    def test_array_slicing(self, nb_runner):
        """Array slicing across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            "data = np.arange(20).reshape(4, 5)",
            textwrap.dedent("""\
                row2 = data[2, :]
                col3 = data[:, 3]
                print(row2.tolist(), col3.tolist())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "[10, 11, 12, 13, 14]" in output
        assert "[3, 8, 13, 18]" in output

    def test_numpy_statistical_functions(self, nb_runner):
        """Statistical functions across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            "data = np.array([10, 20, 30, 40, 50])",
            textwrap.dedent("""\
                stats = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data)
                }
                print(f"mean={stats['mean']:.1f} median={stats['median']:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30.0" in nb_runner.get_output(3)
        assert "median=30.0" in nb_runner.get_output(3)

    def test_numpy_with_pandas(self, nb_runner):
        """Numpy and pandas interop across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import numpy as np
                import pandas as pd
            """),
            textwrap.dedent("""\
                arr = np.random.seed(42)
                values = np.random.randn(100)
            """),
            textwrap.dedent("""\
                df = pd.DataFrame({'value': values})
                stats = df.describe()
                print(f"count={int(stats.loc['count', 'value'])}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=100" in nb_runner.get_output(3)


class TestNumpySeededRandom:
    """Test numpy random with seeds (deterministic)."""

    def test_seeded_random_reproducible(self, nb_runner):
        """Seeded random produces same results."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(123)
                arr = np.random.randint(0, 100, size=5)
                print(arr.tolist())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)

        # Re-run should produce same result (cached)
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert output1 == output2

    def test_linspace_arange(self, nb_runner):
        """linspace and arange across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            "x = np.linspace(0, 1, 5)",
            "y = np.arange(0, 10, 2)",
            textwrap.dedent("""\
                print(x.tolist())
                print(y.tolist())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "0.0" in output
        assert "1.0" in output
        assert "[0, 2, 4, 6, 8]" in output


class TestNumpyChangeInvalidation:
    """Test that numpy array changes propagate correctly."""

    def test_array_change_propagation(self, nb_runner):
        """Changing array definition propagates to downstream."""
        nb_runner.create_notebook([
            "import numpy as np",
            "arr = np.array([1, 2, 3])",
            textwrap.dedent("""\
                total = arr.sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "arr = np.array([10, 20, 30])")
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(3)

    def test_shape_change(self, nb_runner):
        """Changing array shape propagates."""
        nb_runner.create_notebook([
            "import numpy as np",
            "data = np.zeros((3, 3))",
            textwrap.dedent("""\
                print(data.shape)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(3, 3)" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "data = np.zeros((5, 7))")
        nb_runner.run_all()
        assert "(5, 7)" in nb_runner.get_output(3)


class TestNumpyWithRestart:
    """Test numpy array persistence across kernel restarts."""

    def test_array_restore_after_restart(self, nb_runner):
        """Numpy array restored after kernel restart."""
        nb_runner.create_notebook([
            "import numpy as np",
            "arr = np.array([10, 20, 30, 40])",
            textwrap.dedent("""\
                result = arr.mean()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25.0" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25.0" in nb_runner.get_output(3)
