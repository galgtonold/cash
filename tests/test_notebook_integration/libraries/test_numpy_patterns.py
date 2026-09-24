"""numpy arrays and linear algebra across cells."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.stress
class TestNumpyBasicOperations:
    """Test basic numpy operations across cells."""

    def test_numpy_with_pandas(self, nb_runner):
        """Numpy and pandas interop across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=100" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestNumpySeededRandom:
    """Test numpy random with seeds (deterministic)."""

    def test_seeded_random_reproducible(self, nb_runner):
        """Seeded random produces same results."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(123)
                arr = np.random.randint(0, 100, size=5)
                print(arr.tolist())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)

        # Re-run should produce same result (cached)
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert output1 == output2


@pytest.mark.integration
@pytest.mark.stress
class TestNumpyChangeInvalidation:
    """Test that numpy array changes propagate correctly."""

    def test_array_change_propagation(self, nb_runner):
        """Changing array definition propagates to downstream."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "arr = np.array([1, 2, 3])",
                textwrap.dedent("""\
                total = arr.sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "arr = np.array([10, 20, 30])")
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(3)

    def test_shape_change(self, nb_runner):
        """Changing array shape propagates."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "data = np.zeros((3, 3))",
                textwrap.dedent("""\
                print(data.shape)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(3, 3)" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "data = np.zeros((5, 7))")
        nb_runner.run_all()
        assert "(5, 7)" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestNumpyWithRestart:
    """Test numpy array persistence across kernel restarts."""

    def test_array_restore_after_restart(self, nb_runner):
        """Numpy array restored after kernel restart."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "arr = np.array([10, 20, 30, 40])",
                textwrap.dedent("""\
                result = arr.mean()
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25.0" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25.0" in nb_runner.get_output(3)


@pytest.mark.stress
class TestPureMatrixOps:
    """Test pure Python matrix operations."""

    def test_matrix_change_propagates(self, nb_runner):
        """Changing matrix definition propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def dot_product(v1, v2):
                    return sum(a * b for a, b in zip(v1, v2))

                vec_a = [1, 2, 3]
                vec_b = [4, 5, 6]
            """),
                textwrap.dedent("""\
                result = dot_product(vec_a, vec_b)
                print(f"dot={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dot=32" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def dot_product(v1, v2):
                return sum(a * b for a, b in zip(v1, v2))

            vec_a = [10, 20, 30]
            vec_b = [4, 5, 6]
        """),
        )
        nb_runner.run_all()
        assert "dot=320" in nb_runner.get_output(2)
