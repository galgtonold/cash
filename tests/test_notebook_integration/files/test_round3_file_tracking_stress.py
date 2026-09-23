"""
File tracking edge cases, multi-library combos, and stress patterns.

Tests focusing on:
1. File tracking with multiple file formats (CSV, JSON, text)
2. File creation + read in same session
3. Multiple DataFrames from same file
4. Complex pandas operations chain
5. Numpy + pandas combined workflows
6. String processing pipelines
7. Mathematical computation chains
8. Cache invalidation stress tests
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestFileTrackingEdgeCases:
    """Tests for file tracking with various file operations."""

    @pytest.mark.files
    def test_csv_create_read_modify_cycle(self, nb_runner, tmp_path):
        """Create CSV externally, read it in notebook, modify CSV, re-read should detect change."""
        import pandas as pd

        csv_path = str(tmp_path / "data.csv").replace("\\", "/")
        # Create initial CSV externally
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "total = df['a'].sum()",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=3" in output

        # Modify CSV file externally
        import time

        time.sleep(0.1)  # Ensure mtime changes
        pd.DataFrame({"a": [10, 20], "b": [30, 40]}).to_csv(csv_path, index=False)

        # Re-run: should detect file change and recompute
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "total=30" in output2

    @pytest.mark.files
    def test_json_file_dependency(self, nb_runner, tmp_path):
        """JSON file read should be tracked as dependency."""
        json_path = str(tmp_path / "config.json").replace("\\", "/")

        # Create JSON file
        import json

        with open(json_path, "w") as f:
            json.dump({"value": 42, "name": "test"}, f)

        nb_runner.create_notebook(
            [
                "import json",
                f"with open('{json_path}') as f:\n    config = json.load(f)",
                "result = config['value'] * 2",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=84" in output

    @pytest.mark.files
    def test_text_file_line_by_line(self, nb_runner, tmp_path):
        """Read text file line by line."""
        txt_path = str(tmp_path / "data.txt").replace("\\", "/")

        with open(txt_path, "w") as f:
            f.write("line1\nline2\nline3\n")

        nb_runner.create_notebook(
            [
                f"with open('{txt_path}') as f:\n    lines = f.readlines()",
                "count = len(lines)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "count=3" in output

    @pytest.mark.files
    def test_multiple_csv_files(self, nb_runner, tmp_path):
        """Reading multiple CSV files should track all dependencies."""
        import pandas as pd

        csv1 = str(tmp_path / "sales.csv").replace("\\", "/")
        csv2 = str(tmp_path / "costs.csv").replace("\\", "/")

        pd.DataFrame({"amount": [100, 200, 300]}).to_csv(csv1, index=False)
        pd.DataFrame({"cost": [50, 100, 150]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"sales = pd.read_csv('{csv1}')",
                f"costs = pd.read_csv('{csv2}')",
                "profit = sales['amount'].sum() - costs['cost'].sum()",
                "print(f'profit={profit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "profit=300" in output

        # Modify only costs file
        pd.DataFrame({"cost": [10, 20, 30]}).to_csv(csv2, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "profit=540" in output2


class TestNumpyComputationChains:
    """Tests for numpy computation chains."""

    @pytest.mark.libraries
    def test_matrix_operations_chain(self, nb_runner):
        """Chain of matrix operations."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "A = np.array([[1, 2], [3, 4]])",
                "B = A @ A.T",  # matrix multiply with transpose
                "eigenvalues = np.sort(np.linalg.eigvals(B))[::-1]",
                "trace = np.trace(B)",
                "print(f'trace={trace} eigenvalues={eigenvalues}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "trace=" in output

        # Change matrix
        nb_runner.set_cell_source(2, "A = np.array([[2, 0], [0, 3]])")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        assert "trace=13" in output2  # 4+9

    @pytest.mark.libraries
    def test_statistical_computations(self, nb_runner):
        """Statistical operations on numpy arrays."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "np.random.seed(123)\ndata = np.random.randn(1000)",
                "stats = {\n    'mean': round(np.mean(data), 4),\n    'std': round(np.std(data), 4),\n    'median': round(np.median(data), 4),\n    'min': round(np.min(data), 4),\n    'max': round(np.max(data), 4)\n}",
                'print(f\'mean={stats["mean"]} std={stats["std"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "mean=" in output
        assert "std=" in output

        # Re-run: should be cached (deterministic due to seed)
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert output2.strip() == output.strip()


class TestCacheInvalidationStress:
    """Stress tests for cache invalidation correctness."""

    @pytest.mark.stress
    def test_rapid_value_changes(self, nb_runner):
        """Rapidly changing a value and verifying correct propagation each time."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "y = x + 1",
                "print(f'y={y}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(5):
            nb_runner.set_cell_source(1, f"x = {i * 10}")
            nb_runner.run_all()

            output = nb_runner.get_output(3)
            expected = i * 10 + 1
            assert f"y={expected}" in output, f"Iteration {i}: expected y={expected}, got: {output}"

    @pytest.mark.stress
    def test_alternating_code_changes(self, nb_runner):
        """Alternate between two versions of code to stress cache."""
        nb_runner.create_notebook(
            [
                "mode = 'A'",
                "if mode == 'A':\n    result = 'alpha'\nelse:\n    result = 'beta'",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()

        for i in range(4):
            mode = "A" if i % 2 == 0 else "B"
            expected = "alpha" if mode == "A" else "beta"

            nb_runner.set_cell_source(1, f"mode = '{mode}'")
            nb_runner.run_all()

            output = nb_runner.get_output(3)
            assert f"result={expected}" in output, f"Iteration {i}: expected {expected}, got: {output}"

    @pytest.mark.stress
    def test_growing_dependency_chain(self, nb_runner):
        """Build up a longer dependency chain incrementally."""
        # Start with 3 cells
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "print(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "b=2" in output

        # Change root
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "b=11" in output2

    @pytest.mark.stress
    def test_multiple_output_invalidation(self, nb_runner):
        """Cell producing multiple outputs - change input and verify all update."""
        nb_runner.create_notebook(
            [
                "base = 5",
                "x = base * 2\ny = base * 3\nz = base * 4",
                "total = x + y + z",
                "print(f'x={x} y={y} z={z} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "x=10" in output
        assert "y=15" in output
        assert "z=20" in output
        assert "total=45" in output

        # Change base
        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "x=20" in output2
        assert "y=30" in output2
        assert "z=40" in output2
        assert "total=90" in output2


class TestStringProcessingPipelines:
    """Tests for string processing workflows."""

    @pytest.mark.core
    def test_text_cleaning_pipeline(self, nb_runner):
        """Multi-step text cleaning pipeline."""
        nb_runner.create_notebook(
            [
                "raw = '  Hello, WORLD!  This is a TEST.  '",
                "step1 = raw.strip()",
                "step2 = step1.lower()",
                "step3 = step2.replace('!', '').replace('.', '')",
                "words = step3.split()\nword_count = len(words)",
                "print(f'word_count={word_count} first={words[0]} last={words[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "word_count=6" in output
        assert "first=hello" in output
        assert "last=test" in output

        # Change raw text
        nb_runner.set_cell_source(1, "raw = '  Cash is GREAT!  '")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        assert "word_count=3" in output2


class TestMathComputationChains:
    """Tests for mathematical computation chains."""

    @pytest.mark.core
    def test_fibonacci_iterative(self, nb_runner):
        """Iterative Fibonacci computation."""
        nb_runner.create_notebook(
            [
                "n = 10",
                "fib = [0, 1]\nfor i in range(2, n):\n    fib.append(fib[-1] + fib[-2])",
                "result = fib[-1]",
                "print(f'fib({n})={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "fib(10)=34" in output

        # Change n
        nb_runner.set_cell_source(1, "n = 15")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "fib(15)=377" in output2

    @pytest.mark.core
    def test_math_module_computations(self, nb_runner):
        """Using math module for various computations."""
        nb_runner.create_notebook(
            [
                "import math",
                "angle = math.pi / 4",
                "sin_val = round(math.sin(angle), 4)\ncos_val = round(math.cos(angle), 4)\ntan_val = round(math.tan(angle), 4)",
                "print(f'sin={sin_val} cos={cos_val} tan={tan_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "sin=0.7071" in output
        assert "cos=0.7071" in output
        assert "tan=1.0" in output

        # Change angle
        nb_runner.set_cell_source(2, "angle = math.pi / 6")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "sin=0.5" in output2
