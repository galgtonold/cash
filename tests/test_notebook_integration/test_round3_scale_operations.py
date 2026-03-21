"""
Batch 42: Large data operations — tests with bigger datasets to stress
the caching/hashing system with non-trivial data sizes.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestLargeListOperations:
    """Test caching with larger list operations."""

    def test_large_list_sort_and_filter(self, nb_runner):
        """Sort and filter a list of 10000 items."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                data = list(np.random.randint(0, 10000, 10000))
            """),
            textwrap.dedent("""\
                sorted_data = sorted(data)
                top_100 = sorted_data[-100:]
            """),
            textwrap.dedent("""\
                print(f"min={sorted_data[0]} max={sorted_data[-1]} top100_min={top_100[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "min=" in output
        assert "max=" in output
        assert "top100_min=" in output

    def test_large_dict_operations(self, nb_runner):
        """Large dict operations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = {f'key_{i}': i * i for i in range(5000)}
            """),
            textwrap.dedent("""\
                filtered = {k: v for k, v in data.items() if v > 20000000}
                print(f"total_keys={len(data)} filtered={len(filtered)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "total_keys=5000" in output
        assert "filtered=" in output


class TestLargeDataFrameOperations:
    """Test with larger DataFrame operations."""

    def test_dataframe_1000_rows(self, nb_runner):
        """DataFrame with 1000 rows processed across cells."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                df = pd.DataFrame({
                    'a': np.random.randn(1000),
                    'b': np.random.randn(1000),
                    'group': np.random.choice(['X', 'Y', 'Z'], 1000)
                })
            """),
            textwrap.dedent("""\
                df = df.assign(c=df['a'] + df['b'])
            """),
            textwrap.dedent("""\
                summary = df.groupby('group')['c'].agg(['mean', 'std', 'count'])
                print(f"groups={len(summary)} total={summary['count'].sum()}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "groups=3" in output
        assert "total=1000" in output

    def test_dataframe_multicolumn_operations(self, nb_runner):
        """DataFrame with many columns."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                cols = {f'col_{i}': np.random.randn(500) for i in range(20)}
                df = pd.DataFrame(cols)
            """),
            textwrap.dedent("""\
                corr = df.corr()
                max_corr = corr.where(corr < 1).max().max()
                print(f"shape={df.shape} max_corr={max_corr:.4f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "shape=(500, 20)" in output
        assert "max_corr=" in output


class TestStringProcessingScale:
    """Test string processing at scale."""

    def test_large_text_processing(self, nb_runner):
        """Process a large text across cells."""
        nb_runner.create_notebook([
            "import re\nfrom collections import Counter",
            textwrap.dedent("""\
                # Generate a large text
                words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'] * 500
                text = ' '.join(words)
            """),
            textwrap.dedent("""\
                tokens = re.findall(r'\\b\\w+\\b', text.lower())
                freq = Counter(tokens)
            """),
            textwrap.dedent("""\
                total = sum(freq.values())
                unique = len(freq)
                print(f"total={total} unique={unique} top={freq.most_common(1)[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "total=4000" in output
        assert "unique=8" in output


class TestNumpyScale:
    """Test numpy operations at moderate scale."""

    def test_matrix_operations(self, nb_runner):
        """Matrix multiplication and decomposition."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                A = np.random.randn(100, 100)
                B = np.random.randn(100, 100)
            """),
            textwrap.dedent("""\
                C = A @ B
                trace = np.trace(C)
                norm = np.linalg.norm(C)
            """),
            textwrap.dedent("""\
                print(f"shape={C.shape} trace={trace:.2f} norm={norm:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "shape=(100, 100)" in output
        assert "trace=" in output
        assert "norm=" in output

    def test_eigenvalue_decomposition(self, nb_runner):
        """Eigenvalue decomposition across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                # Symmetric matrix for real eigenvalues
                A = np.random.randn(50, 50)
                S = (A + A.T) / 2
            """),
            textwrap.dedent("""\
                eigenvalues, eigenvectors = np.linalg.eigh(S)
            """),
            textwrap.dedent("""\
                print(f"n_eigs={len(eigenvalues)} max={eigenvalues[-1]:.2f} min={eigenvalues[0]:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "n_eigs=50" in output
        assert "max=" in output
        assert "min=" in output

    def test_numpy_broadcasting(self, nb_runner):
        """Broadcasting operations across cells."""
        nb_runner.create_notebook([
            "import numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                matrix = np.random.randn(200, 50)
                weights = np.random.randn(50)
            """),
            textwrap.dedent("""\
                weighted = matrix * weights  # Broadcasting
                result = weighted.sum(axis=1)
            """),
            textwrap.dedent("""\
                print(f"shape={result.shape} mean={result.mean():.4f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "shape=(200,)" in output
        assert "mean=" in output
