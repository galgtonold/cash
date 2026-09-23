"""Data files and cells that change together: edits, rewrites, pathlib, shelve."""

import textwrap
import time

import pytest


# File dependency, kernel restart, and module tests.
#
# Tests:
# - Complex file dependency patterns (multiple files, JSON, Parquet, cross-cell)
# - File creation in one cell, reading in another
# - Disk persistence and restore after kernel restart
# - External Python module changes and hot-reload
# - Module function signature changes
@pytest.mark.files
class TestMultiFileDependencies:
    """Test caching with multiple file dependencies."""

    def test_two_csv_files_merged(self, nb_runner, tmp_path):
        """Read two CSV files and merge them - both should be tracked."""
        import pandas as pd

        csv1 = tmp_path / "users.csv"
        csv2 = tmp_path / "orders.csv"
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")

        pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]}).to_csv(csv1, index=False)
        pd.DataFrame({"user_id": [1, 2], "amount": [100, 200]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
                f"orders = pd.read_csv('{csv2_str}')",
                "merged = users.merge(orders, on='user_id')\nprint(merged.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Alice" in out, f"Expected Alice in output, got: {out}"
        assert "100" in out

        # Modify the second file
        pd.DataFrame({"user_id": [1, 2], "amount": [999, 888]}).to_csv(csv2, index=False)
        time.sleep(0.5)  # Ensure mtime changes

        # Re-run - should detect file change
        nb_runner.run_cells([2, 3])

        out2 = nb_runner.get_output(3)
        assert "999" in out2, f"Expected 999 after file change, got: {out2}"

    def test_json_file_dependency(self, nb_runner, tmp_path):
        """Test that JSON file reads are tracked as dependencies."""
        import json

        config_file = tmp_path / "config.json"
        config_str = str(config_file).replace("\\", "/")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"multiplier": 5, "offset": 10}, f)

        nb_runner.create_notebook(
            [
                f"""import json
with open('{config_str}') as f:
    config = json.load(f)""",
                "result = 100 * config['multiplier'] + config['offset']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 510" in out, f"Got: {out}"

        # Modify config
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"multiplier": 2, "offset": 0}, f)
        time.sleep(0.5)

        nb_runner.run_cells([1, 2])

        out2 = nb_runner.get_output(2)
        assert "result = 200" in out2, f"Expected 200 after config change, got: {out2}"

    def test_file_creation_in_one_cell_read_in_another(self, nb_runner, tmp_path):
        """Create a file in one cell, read in another - should track dependency."""
        fpath = tmp_path / "generated.txt"
        fpath_str = str(fpath).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Write a file
                f"""with open('{fpath_str}', 'w') as f:
    f.write('hello world')""",
                # Cell 2: Read it back
                f"""with open('{fpath_str}') as f:
    content = f.read()
print(f'content = {{content}}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "content = hello world" in out, f"Got: {out}"

    def test_csv_with_pathlib(self, nb_runner, tmp_path):
        """Test that pathlib-based file reads are tracked."""
        import pandas as pd

        data_file = tmp_path / "data.csv"
        data_str = str(data_file).replace("\\", "/")
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(data_file, index=False)

        nb_runner.create_notebook(
            [
                f"""import pandas as pd
from pathlib import Path
df = pd.read_csv(Path('{data_str}'))""",
                "total = df['x'].sum() + df['y'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "total = 21" in out, f"Got: {out}"


@pytest.mark.files
class TestFileDependencyWithUpstreamChanges:
    """Test file deps combined with code changes."""

    def test_file_dep_plus_code_change(self, nb_runner, tmp_path):
        """
        Read a CSV, transform it, modify the transformation code.
        Both file dep and code change should be detected.
        """
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 60" in nb_runner.get_output(2)

        # Change the code to compute mean instead
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "result = 20.0" in out, f"Expected 20.0 after code change, got: {out}"


@pytest.mark.files
@pytest.mark.restore
class TestDiskPersistenceRestore:
    """Test disk persistence and restore after kernel restart."""

    @pytest.mark.timeout(60)
    def test_simple_values_persist_across_restart(self, nb_runner, tmp_path):
        """
        Test that simple values cached to disk can be restored after restart.
        Uses FileBackend to ensure disk persistence.
        """
        cache_dir = tmp_path / "cache"
        cache_dir_str = str(cache_dir).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Setup with file backend
                f"""import time
from cash import Cash
from cash.backends import FileBackend
from cash.notebook.ipython.magics import CashMagics

backend = FileBackend(cache_dir='{cache_dir_str}')
ip = get_ipython()
cash = Cash(backend=backend, register_magic=False)
magics = CashMagics(ip, cash)
ip.register_magics(magics)
%cash_on
time.sleep(0.1)  # Ensure disk writes complete""",
                # Cell 2: Slow computation (persists to disk)
                """# @cash: persist
import time
time.sleep(0.3)
x = 42
print(f'x = {x}')""",
                # Cell 3: Depends on x
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "x = 42" in out2, f"Initial run cell 2: {out2}"
        out3 = nb_runner.get_output(3)
        assert "y = 84" in out3, f"Initial run cell 3: {out3}"


@pytest.mark.files
class TestModuleHotReload:
    """Test module hot-reload detection."""

    def test_external_module_function_change(self, nb_runner, tmp_path):
        """
        Import a module, change it on disk, and verify that cash detects
        the change and recomputes on run_all().
        """
        mod_path = tmp_path / "mymodule.py"
        mod_path.write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import mymodule",
                "result = mymodule.compute(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(2)

        # Modify the module on disk
        mod_path.write_text("def compute(x):\n    return x * 3\n", encoding="utf-8")
        time.sleep(0.5)

        # Re-run all - module reload should detect change
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 30" in out, f"Expected 30 after module change, got: {out}"

    def test_module_class_method_change(self, nb_runner, tmp_path):
        """
        Import a class from a module, change a method, verify detection.
        """
        mod_path = tmp_path / "shapes.py"
        mod_path.write_text(
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        return 3.14 * self.r ** 2\n",
            encoding="utf-8",
        )

        nb_runner.create_notebook(
            [
                "import shapes",
                "c = shapes.Circle(5)\nprint(f'area = {c.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "area = 78.5" in out, f"Got: {out}"

        # Change pi approximation
        mod_path.write_text(
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        import math\n"
            "        return math.pi * self.r ** 2\n",
            encoding="utf-8",
        )
        time.sleep(0.5)

        # Re-run all - module change should be detected
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "area = 78.5398" in out2, f"Expected math.pi result, got: {out2}"

    def test_module_new_function_added(self, nb_runner, tmp_path):
        """
        Add a new function to an existing module and use it.
        """
        mod_path = tmp_path / "utils.py"
        mod_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "import utils",
                "r1 = utils.add(3, 4)\nprint(f'r1 = {r1}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r1 = 7" in nb_runner.get_output(2)

        # Add a multiply function to the module
        mod_path.write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n", encoding="utf-8"
        )
        time.sleep(0.5)

        # Modify cell 2 to also use the new function
        nb_runner.set_cell_source(2, "r1 = utils.add(3, 4)\nr2 = utils.multiply(3, 4)\nprint(f'r1={r1}, r2={r2}')")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "r1=7" in out, f"Got: {out}"
        assert "r2=12" in out, f"Expected r2=12, got: {out}"

    def test_from_import_module_reload(self, nb_runner, tmp_path):
        """
        Test that 'from X import Y' style imports detect module changes.
        When a module is modified, re-running the import should pick up the
        new function definition and downstream cells should use the updated version.

        This tests the F3 bug fix: previously from-import style wouldn't detect
        module changes because (1) the redundant import optimization skipped the
        re-import, (2) tracking for from-imported names wasn't cleared, and
        (3) the cache key for import statements didn't include module source hash.
        """
        mod_path = tmp_path / "mathlib.py"
        mod_path.write_text("def square(x):\n    return x ** 2\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "from mathlib import square",
                "result = square(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 25" in nb_runner.get_output(2)

        # Modify the module to cube instead
        mod_path.write_text("def square(x):\n    return x ** 3\n", encoding="utf-8")
        time.sleep(0.5)

        # Re-run all — cash should detect the module change and re-execute
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 125" in out, f"from-import module reload failed: expected 125 (5**3), got: {out}"


@pytest.mark.files
@pytest.mark.upstream
class TestUpstreamModificationCascades:
    """Test that upstream modifications properly cascade."""

    def test_cascade_through_three_cells(self, nb_runner):
        """Modify cell 1, only run cell 3, verify correct cascade."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "z = 30" in nb_runner.get_output(3)

        # Modify cell 1 and only run cell 3
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 210" in out, f"Expected z=210 after upstream cascade, got: {out}"

    def test_cascade_skips_unrelated_cells(self, nb_runner):
        """
        If cell B doesn't depend on cell A, modifying A shouldn't affect B.
        A: x = 10
        B: y = 99 (independent)
        C: z = x + y
        Modify A -> re-run C. B should not be re-executed.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 99",
                "z = x + y\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "z = 109" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 149" in out, f"Expected z=149, got: {out}"

    def test_cascade_with_function_dependency(self, nb_runner):
        """
        Cell 1: Define function
        Cell 2: Define data
        Cell 3: Apply function to data
        Modify function -> re-run cell 3.
        """
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "data = [1, 2, 3]",
                "result = [transform(x) for x in data]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = [2, 4, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "result = [1, 4, 9]" in out, f"Expected [1,4,9], got: {out}"


@pytest.mark.files
@pytest.mark.core
class TestEdgeCases:
    """Test edge cases and unusual patterns."""

    def test_star_expression_unpacking(self, nb_runner):
        """Test star expression unpacking."""
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3, 4, 5]",
                "first, *rest = items\nprint(f'first={first}, rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "first=1, rest=[2, 3, 4, 5]" in out, f"Got: {out}"


# File tracking edge cases, multi-library combos, and stress patterns.
#
# Tests focusing on:
# 1. File tracking with multiple file formats (CSV, JSON, text)
# 2. File creation + read in same session
# 3. Multiple DataFrames from same file
# 4. Complex pandas operations chain
# 5. Numpy + pandas combined workflows
# 6. String processing pipelines
# 7. Mathematical computation chains
# 8. Cache invalidation stress tests
@pytest.mark.integration
@pytest.mark.timeout(30)
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

        with open(json_path, "w", encoding="utf-8") as f:
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

        with open(txt_path, "w", encoding="utf-8") as f:
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


@pytest.mark.integration
@pytest.mark.timeout(30)
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


# File dependency + cell edit interactions.
#
# Tests the interaction between:
# - Editing cells that read files
# - Changing file contents between runs
# - Editing both the file AND the cell code
# - File changes + kernel restart + cell edits
@pytest.mark.stress
@pytest.mark.files
class TestFileChangeWithCellEdit:
    """Edit the code that reads a file, AND change the file."""

    def test_edit_cell_after_file_change(self, nb_runner, tmp_path):
        """Change file contents, then also edit the code reading it."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file AND edit cell to use mean instead
        csv_path.write_text("val\n100\n200\n300\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "result = 200" in output

    def test_file_change_triggers_recomputation(self, nb_runner, tmp_path):
        """Change file between runs, cell code unchanged → detects stale file."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n1\n2\n3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "total = df['x'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change file only
        csv_path.write_text("x\n10\n20\n30\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "total = 60" in nb_runner.get_output(2)

    def test_file_change_with_upstream_code_edit(self, nb_runner, tmp_path):
        """Upstream cell code changes AND file changes simultaneously."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("a\n5\n10\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "multiplier = 2",
                "result = df['a'].sum() * multiplier\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Change BOTH file AND multiplier
        csv_path.write_text("a\n50\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "multiplier = 10")
        nb_runner.run_cell(3)
        assert "result = 1500" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.files
class TestFileChangeWithRestart:
    """File changes combined with kernel restarts."""

    def test_file_change_after_restart(self, nb_runner, tmp_path):
        """Change file after restart — should detect stale cache."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("val\n1\n2\n3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "s = df['val'].sum()\nprint(f's = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 6" in nb_runner.get_output(2)

        # Restart + change file
        nb_runner.shutdown()
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 60" in nb_runner.get_output(2)

    def test_file_change_and_code_edit_after_restart(self, nb_runner, tmp_path):
        """File changes + code edit + kernel restart — triple stress."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n5\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "result = df['x'].iloc[0] * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        csv_path.write_text("x\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['x'].iloc[0] * 3\nprint(f'result = {result}')")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.files
class TestMultipleFileReads:
    """Multiple cells reading different files."""

    def test_two_files_change_one(self, nb_runner, tmp_path):
        """Two files read in different cells, only one changes."""
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        f1_str = str(f1).replace("\\", "/")
        f2_str = str(f2).replace("\\", "/")
        f1.write_text("a\n10\n", encoding="utf-8")
        f2.write_text("b\n20\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf1 = pd.read_csv('{f1_str}')\ndf2 = pd.read_csv('{f2_str}')",
                "result = df1['a'].iloc[0] + df2['b'].iloc[0]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Only change f1
        f1.write_text("a\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "result = 120" in nb_runner.get_output(2)

    def test_file_deleted_then_recreated(self, nb_runner, tmp_path):
        """Delete a file, then recreate it with new data."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n5\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "val = df['x'].iloc[0]\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 5" in nb_runner.get_output(2)

        # Delete and recreate
        csv_path.unlink()
        csv_path.write_text("x\n999\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)


# File dependency + cell edit advanced interaction tests.
#
# Tests that exercise file reads (CSV, JSON, text) combined with
# cell edits, verifying that cache invalidation works correctly
# when both code and files change.
@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestCSVFileWithEdits:
    """CSV file reads + cell code edits."""

    def test_read_csv_then_edit_processing(self, nb_runner, tmp_path):
        """Read CSV, edit the processing code."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['a'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(2)

        # Edit processing to use column b
        nb_runner.set_cell_source(2, "result = df['b'].sum()\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_edit_csv_then_rerun(self, nb_runner, tmp_path):
        """Read CSV, modify the CSV file, rerun."""
        csv_path = tmp_path / "vals.csv"
        csv_path.write_text("x\n10\n20\n30\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "total = df['x'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify the CSV file
        csv_path.write_text("x\n100\n200\n300\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)

    def test_edit_csv_and_code_together(self, nb_runner, tmp_path):
        """Change both the file and the code."""
        csv_path = tmp_path / "items.csv"
        csv_path.write_text("val\n1\n2\n3\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        # Change both file and code
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20.0" in nb_runner.get_output(2)


@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestJSONFileWithEdits:
    """JSON file reads + cell edits."""

    def test_read_json_edit_processing(self, nb_runner, tmp_path):
        """Read JSON file, edit the processing."""
        json_path = tmp_path / "config.json"
        json_path.write_text('{"scale": 2, "offset": 10}', encoding="utf-8")
        json_str = str(json_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import json\nwith open('{json_str}') as f:\n    config = json.load(f)",
                "result = config['scale'] * 5 + config['offset']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(2, "result = config['scale'] ** 3 + config['offset']\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(2)


@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestTextFileWithEdits:
    """Plain text file reads + cell edits."""

    def test_read_text_file_edit_processing(self, nb_runner, tmp_path):
        """Read text file, edit the processing code."""
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("hello\nworld\nfoo\nbar\n", encoding="utf-8")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    lines = f.read().strip().split('\\n')",
                "count = len(lines)\nprint(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(2, "total_chars = sum(len(l) for l in lines)\nprint(f'chars = {total_chars}')")
        nb_runner.run_all()
        assert "chars = 16" in nb_runner.get_output(2)

    def test_edit_text_file_rerun(self, nb_runner, tmp_path):
        """Modify text file content, rerun."""
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("alpha\nbeta\n", encoding="utf-8")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    content = f.read().strip()",
                "word_count = len(content.split())\nprint(f'words = {word_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 2" in nb_runner.get_output(2)

        # Add more content
        txt_path.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 5" in nb_runner.get_output(2)


# File dependency combined with cell edit tests.
#
# Tests where both file content AND code change, verifying
# that cache correctly invalidates on either trigger.
@pytest.mark.stress
@pytest.mark.files
@pytest.mark.timeout(90)
class TestFilePlusCellEdits:
    """File changes combined with cell edits."""

    def test_edit_code_with_same_file(self, nb_runner, tmp_path):
        """Same file content, different code → recompute."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("hello", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    content = f.read()",
                "result = content.upper()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO" in nb_runner.get_output(2)

        # Edit code (file stays same)
        nb_runner.set_cell_source(2, "result = content.lower()\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = hello" in nb_runner.get_output(2)

    def test_edit_file_with_same_code(self, nb_runner, tmp_path):
        """Same code, different file content → recompute."""
        data_file = tmp_path / "numbers.txt"
        data_file.write_text("1,2,3", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    raw = f.read()",
                "nums = [int(x) for x in raw.split(',')]\ntotal = sum(nums)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Edit file
        data_file.write_text("10,20,30", encoding="utf-8")
        # Restart to ensure file dep is re-checked
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_both_file_and_code(self, nb_runner, tmp_path):
        """Change both file content AND code → recompute."""
        data_file = tmp_path / "config.txt"
        data_file.write_text("scale=2", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    line = f.read()",
                "key, val = line.split('=')\nresult = int(val) * 10\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit file and code
        data_file.write_text("scale=5", encoding="utf-8")
        nb_runner.set_cell_source(
            2,
            "key, val = line.split('=')\nresult = int(val) * 100\nprint(f'result = {result}')",
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.files
@pytest.mark.timeout(90)
class TestCSVFileEdits:
    """CSV file changes with code edits."""

    def test_csv_data_change(self, nb_runner, tmp_path):
        """Change CSV data, verify cache invalidation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        fpath = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import csv",
                f"with open('{fpath}') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)",
                "total_a = sum(int(r['a']) for r in rows)\nprint(f'total_a = {total_a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 4" in nb_runner.get_output(3)

        # Edit CSV
        csv_file.write_text("a,b\n10,20\n30,40\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 40" in nb_runner.get_output(3)


# pathlib and file system operations.
class TestPathlibPatterns:
    """pathlib usage patterns."""

    @pytest.mark.stress
    @pytest.mark.integration
    @pytest.mark.files
    def test_pathlib_operations(self, nb_runner, tmp_path):
        """Path construction, existence checks, iteration."""
        work = tmp_path / "fs_test"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                # Create directory structure
                (base / 'src').mkdir(exist_ok=True)
                (base / 'src' / 'main.py').write_text('print("hello")')
                (base / 'src' / 'utils.py').write_text('x = 1')
                (base / 'data').mkdir(exist_ok=True)
                (base / 'data' / 'input.txt').write_text('data here')
                (base / 'README.md').write_text('# Project')
            """),
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                all_files = sorted([p.name for p in base.rglob('*') if p.is_file()])
                py_files = sorted([p.name for p in base.glob('**/*.py')])
                readme_exists = (base / 'README.md').exists()
            """),
                "print(f'all={all_files}')\nprint(f'py={py_files}')\nprint(f'readme={readme_exists}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "main.py" in out
        assert "utils.py" in out
        assert "input.txt" in out
        assert "readme=True" in out

    @pytest.mark.stress
    @pytest.mark.integration
    @pytest.mark.files
    def test_pathlib_stem_suffix(self, nb_runner, tmp_path):
        """Path parts: stem, suffix, parent."""
        work = tmp_path / "path_parts"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                p = Path('{work_str}') / 'archive' / 'data_2024.csv.gz'
                stem = p.stem
                suffix = p.suffix
                suffixes = p.suffixes
                parent_name = p.parent.name
                parts_count = len(p.parts)
            """),
                "print(f'stem={stem} suffix={suffix} suffixes={suffixes} parent={parent_name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=data_2024.csv" in out
        assert "suffix=.gz" in out
        assert ".csv" in out
        assert "parent=archive" in out

    # Pathlib & IO patterns — cash caching with pathlib, io, tempfile.
    @pytest.mark.stress
    def test_pathlib_basic_operations(self, nb_runner, tmp_path):
        """Pathlib path construction and operations."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                base = Path('{dir_path}')
                sub = base / 'data' / 'output'
                sub.mkdir(parents=True, exist_ok=True)
                exists = sub.exists()
                print(f"exists={{exists}} name={{sub.name}}")
            """),
                textwrap.dedent("""\
                # Create files
                for i in range(3):
                    (sub / f'file_{i}.txt').write_text(f'content_{i}')
                files = sorted(f.name for f in sub.iterdir())
                print(f"files={files}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "exists=True name=output" in nb_runner.get_output(2)
        assert "files=['file_0.txt', 'file_1.txt', 'file_2.txt']" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pathlib_glob(self, nb_runner, tmp_path):
        """Pathlib glob pattern matching."""
        glob_dir = tmp_path / "glob_test"
        glob_dir.mkdir()
        dir_path = str(glob_dir).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                base = Path('{dir_path}')
                # Create mixed files
                (base / 'data.csv').write_text('a,b\\n1,2')
                (base / 'report.csv').write_text('x,y\\n3,4')
                (base / 'notes.txt').write_text('hello')
                (base / 'config.json').write_text('{{}}')
            """),
                textwrap.dedent("""\
                csv_files = sorted(f.name for f in base.glob('*.csv'))
                all_files = sorted(f.name for f in base.glob('*.*'))
                print(f"csv={csv_files}")
                print(f"total={len(all_files)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "csv=['data.csv', 'report.csv']" in nb_runner.get_output(3)
        assert "total=4" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pathlib_read_write(self, nb_runner, tmp_path):
        """Pathlib read_text/write_text across cells."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                p = Path('{dir_path}') / 'data.txt'
                lines = ['line1', 'line2', 'line3']
                p.write_text('\\n'.join(lines))
                size = p.stat().st_size
                print(f"size={{size}}")
            """),
                textwrap.dedent("""\
                content = p.read_text()
                line_count = len(content.strip().split('\\n'))
                print(f"lines={line_count} first={content.split(chr(10))[0]}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=" in nb_runner.get_output(2)
        assert "lines=3 first=line1" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.files
class TestFileSystemOps:
    """File system read/write with caching."""

    def test_file_write_read_chain(self, nb_runner, tmp_path):
        """Write files, read back, process."""
        work = tmp_path / "rw_test"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                for i in range(5):
                    (base / f'log_{{i}}.txt').write_text(f'Entry {{i}}: value={{i*10}}')
            """),
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                contents = []
                for f in sorted(base.glob('log_*.txt')):
                    contents.append(f.read_text())
                total_files = len(contents)
            """),
                "print(f'files={total_files}')\nprint(f'first={contents[0]}')\nprint(f'last={contents[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "files=5" in out
        assert "Entry 0" in out
        assert "Entry 4" in out

    def test_temp_file_processing(self, nb_runner, tmp_path):
        """Process data through temp files."""
        work = tmp_path / "temp_proc"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                import json
                data = {{'items': [1, 2, 3, 4, 5], 'meta': 'test'}}
                path = Path('{work_str}') / 'data.json'
                path.write_text(json.dumps(data))
            """),
                textwrap.dedent(f"""\
                from pathlib import Path
                import json
                path = Path('{work_str}') / 'data.json'
                loaded = json.loads(path.read_text())
                total = sum(loaded['items'])
                meta = loaded['meta']
            """),
                "print(f'total={total} meta={meta}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=15" in out
        assert "meta=test" in out


# Interaction test: shelve module for persistent dict-like storage.
# Tests shelve.open with writeback, cross-cell key access,
# and cache invalidation when shelf contents change.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestShelvePersistentDict:
    """Test shelve persistent storage across cells."""

    def test_shelve_ops(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "test_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                # Cell 1: write to shelf
                f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['name'] = 'Alice'\n    db['scores'] = [90, 85, 95]\n    key_count = len(db)\nprint(f'keys={{key_count}}')",
                # Cell 2: read from shelf
                "with shelve.open(shelf_path) as db:\n    name = db['name']\n    scores = db['scores']\nprint(f'name={name}')\nprint(f'avg={sum(scores)/len(scores):.1f}')",
                # Cell 3: check keys
                "with shelve.open(shelf_path) as db:\n    all_keys = sorted(db.keys())\nprint(f'all_keys={all_keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "keys=2" in out1
        out2 = nb_runner.get_output(2)
        assert "name=Alice" in out2
        assert "avg=90.0" in out2
        out3 = nb_runner.get_output(3)
        assert "name" in out3
        assert "scores" in out3

    def test_shelve_edit(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "edit_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 100\n    stored = db['val']\nprint(f'stored={{stored}}')",
                "doubled = stored * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stored=100" in nb_runner.get_output(1)
        assert "doubled=200" in nb_runner.get_output(2)

        # Edit stored value
        nb_runner.set_cell_source(
            1,
            f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 250\n    stored = db['val']\nprint(f'stored={{stored}}')",
        )
        nb_runner.run_cells([1, 2])
        assert "stored=250" in nb_runner.get_output(1)
        assert "doubled=500" in nb_runner.get_output(2)

    def test_shelve_cache(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "cache_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"import shelve\nwith shelve.open('{shelf_path}') as db:\n    db['x'] = 42\n    x = db['x']\nprint(f'x={{x}}')",
                "is_42 = x == 42\nprint(f'is_42={is_42}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42" in nb_runner.get_output(1)
        assert "is_42=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_42=True" in nb_runner.get_output(2)
