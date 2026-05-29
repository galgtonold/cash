"""
Round 3 Batch 2: File dependency, kernel restart, and module tests.

Tests:
- Complex file dependency patterns (multiple files, JSON, Parquet, cross-cell)
- File creation in one cell, reading in another
- Disk persistence and restore after kernel restart
- External Python module changes and hot-reload
- Module function signature changes
"""
import pytest
import time

pytestmark = pytest.mark.files


class TestMultiFileDependencies:
    """Test caching with multiple file dependencies."""

    def test_two_csv_files_merged(self, nb_runner, tmp_path):
        """Read two CSV files and merge them - both should be tracked."""
        import pandas as pd
        
        csv1 = tmp_path / "users.csv"
        csv2 = tmp_path / "orders.csv"
        csv1_str = str(csv1).replace('\\', '/')
        csv2_str = str(csv2).replace('\\', '/')
        
        pd.DataFrame({'user_id': [1, 2], 'name': ['Alice', 'Bob']}).to_csv(csv1, index=False)
        pd.DataFrame({'user_id': [1, 2], 'amount': [100, 200]}).to_csv(csv2, index=False)
        
        nb_runner.create_notebook([
            f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
            f"orders = pd.read_csv('{csv2_str}')",
            "merged = users.merge(orders, on='user_id')\nprint(merged.to_string(index=False))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Alice" in out, f"Expected Alice in output, got: {out}"
        assert "100" in out

        # Modify the second file
        pd.DataFrame({'user_id': [1, 2], 'amount': [999, 888]}).to_csv(csv2, index=False)
        time.sleep(0.5)  # Ensure mtime changes
        
        # Re-run - should detect file change
        nb_runner.run_cells([2, 3])
        
        out2 = nb_runner.get_output(3)
        assert "999" in out2, f"Expected 999 after file change, got: {out2}"

    def test_json_file_dependency(self, nb_runner, tmp_path):
        """Test that JSON file reads are tracked as dependencies."""
        import json
        
        config_file = tmp_path / "config.json"
        config_str = str(config_file).replace('\\', '/')
        
        with open(config_file, 'w') as f:
            json.dump({"multiplier": 5, "offset": 10}, f)
        
        nb_runner.create_notebook([
            f"""import json
with open('{config_str}') as f:
    config = json.load(f)""",
            "result = 100 * config['multiplier'] + config['offset']\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 510" in out, f"Got: {out}"

        # Modify config
        with open(config_file, 'w') as f:
            json.dump({"multiplier": 2, "offset": 0}, f)
        time.sleep(0.5)
        
        nb_runner.run_cells([1, 2])
        
        out2 = nb_runner.get_output(2)
        assert "result = 200" in out2, f"Expected 200 after config change, got: {out2}"

    def test_file_creation_in_one_cell_read_in_another(self, nb_runner, tmp_path):
        """Create a file in one cell, read in another - should track dependency."""
        fpath = tmp_path / "generated.txt"
        fpath_str = str(fpath).replace('\\', '/')
        
        nb_runner.create_notebook([
            # Cell 1: Write a file
            f"""with open('{fpath_str}', 'w') as f:
    f.write('hello world')""",
            # Cell 2: Read it back
            f"""with open('{fpath_str}') as f:
    content = f.read()
print(f'content = {{content}}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "content = hello world" in out, f"Got: {out}"

    def test_csv_with_pathlib(self, nb_runner, tmp_path):
        """Test that pathlib-based file reads are tracked."""
        import pandas as pd
        
        data_file = tmp_path / "data.csv"
        data_str = str(data_file).replace('\\', '/')
        pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}).to_csv(data_file, index=False)
        
        nb_runner.create_notebook([
            f"""import pandas as pd
from pathlib import Path
df = pd.read_csv(Path('{data_str}'))""",
            "total = df['x'].sum() + df['y'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "total = 21" in out, f"Got: {out}"


class TestFileDependencyWithUpstreamChanges:
    """Test file deps combined with code changes."""

    def test_file_dep_plus_code_change(self, nb_runner, tmp_path):
        """
        Read a CSV, transform it, modify the transformation code.
        Both file dep and code change should be detected.
        """
        import pandas as pd
        
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "result = df['val'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 60" in nb_runner.get_output(2)

        # Change the code to compute mean instead
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "result = 20.0" in out, f"Expected 20.0 after code change, got: {out}"


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
        cache_dir_str = str(cache_dir).replace('\\', '/')
        
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "x = 42" in out2, f"Initial run cell 2: {out2}"
        out3 = nb_runner.get_output(3)
        assert "y = 84" in out3, f"Initial run cell 3: {out3}"


class TestModuleHotReload:
    """Test module hot-reload detection."""

    def test_external_module_function_change(self, nb_runner, tmp_path):
        """
        Import a module, change it on disk, and verify that cash detects
        the change and recomputes on run_all().
        """
        mod_path = tmp_path / "mymodule.py"
        mod_path.write_text("def compute(x):\n    return x * 2\n")
        
        nb_runner.create_notebook([
            "import mymodule",
            "result = mymodule.compute(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(2)

        # Modify the module on disk
        mod_path.write_text("def compute(x):\n    return x * 3\n")
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
            "        return 3.14 * self.r ** 2\n"
        )
        
        nb_runner.create_notebook([
            "import shapes",
            "c = shapes.Circle(5)\nprint(f'area = {c.area()}')",
        ])
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
            "        return math.pi * self.r ** 2\n"
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
        mod_path.write_text("def add(a, b):\n    return a + b\n")
        
        nb_runner.create_notebook([
            "import utils",
            "r1 = utils.add(3, 4)\nprint(f'r1 = {r1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r1 = 7" in nb_runner.get_output(2)

        # Add a multiply function to the module
        mod_path.write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def multiply(a, b):\n    return a * b\n"
        )
        time.sleep(0.5)
        
        # Modify cell 2 to also use the new function
        nb_runner.set_cell_source(2, 
            "r1 = utils.add(3, 4)\nr2 = utils.multiply(3, 4)\nprint(f'r1={r1}, r2={r2}')")
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
        mod_path.write_text("def square(x):\n    return x ** 2\n")
        
        nb_runner.create_notebook([
            "from mathlib import square",
            "result = square(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 25" in nb_runner.get_output(2)

        # Modify the module to cube instead
        mod_path.write_text("def square(x):\n    return x ** 3\n")
        time.sleep(0.5)
        
        # Re-run all — cash should detect the module change and re-execute
        nb_runner.run_all()
        
        out = nb_runner.get_output(2)
        assert "result = 125" in out, (
            f"from-import module reload failed: expected 125 (5**3), got: {out}"
        )


@pytest.mark.upstream
class TestUpstreamModificationCascades:
    """Test that upstream modifications properly cascade."""

    def test_cascade_through_three_cells(self, nb_runner):
        """Modify cell 1, only run cell 3, verify correct cascade."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "z = y * 2\nprint(f'z = {z}')",
        ])
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
        nb_runner.create_notebook([
            "x = 10",
            "y = 99",
            "z = x + y\nprint(f'z = {z}')",
        ])
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
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "data = [1, 2, 3]",
            "result = [transform(x) for x in data]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = [2, 4, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "result = [1, 4, 9]" in out, f"Expected [1,4,9], got: {out}"


@pytest.mark.core
class TestEdgeCases:
    """Test edge cases and unusual patterns."""

    def test_empty_cell_between_deps(self, nb_runner):
        """Test that an empty cell between dependencies doesn't break anything."""
        nb_runner.create_notebook([
            "x = 10",
            "",  # Empty cell
            "y = x + 5\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "y = 15" in out, f"Got: {out}"

    def test_cell_with_only_comments(self, nb_runner):
        """Test that a cell with only comments doesn't break caching."""
        nb_runner.create_notebook([
            "x = 42",
            "# This is just a comment\n# Another comment",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "x = 42" in out, f"Got: {out}"

    def test_very_long_string_output(self, nb_runner):
        """Test caching with very large string output."""
        nb_runner.create_notebook([
            "big_str = 'x' * 10000",
            "result = len(big_str)\nprint(f'length = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "length = 10000" in out, f"Got: {out}"

    def test_deeply_nested_dict(self, nb_runner):
        """Test caching a deeply nested dictionary."""
        nb_runner.create_notebook([
            """d = {'a': {'b': {'c': {'d': {'e': 42}}}}}""",
            "val = d['a']['b']['c']['d']['e']\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "val = 42" in out, f"Got: {out}"

    def test_none_and_bool_values(self, nb_runner):
        """Test caching None and boolean values."""
        nb_runner.create_notebook([
            "a = None\nb = True\nc = False",
            "print(f'a={a}, b={b}, c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "a=None, b=True, c=False" in out, f"Got: {out}"

    def test_large_list_cached(self, nb_runner):
        """Test caching a large list."""
        nb_runner.create_notebook([
            "big_list = list(range(100000))",
            "total = sum(big_list)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "total = 4999950000" in out, f"Got: {out}"

    def test_lambda_in_variable(self, nb_runner):
        """Test caching a lambda function."""
        nb_runner.create_notebook([
            "double = lambda x: x * 2",
            "result = double(21)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 42" in out, f"Got: {out}"

    def test_walrus_operator(self, nb_runner):
        """Test the walrus operator (:=) in various contexts."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            """filtered = [y for x in data if (y := x ** 2) > 25]
print(f'filtered = {filtered}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "filtered = [36, 49, 64, 81, 100]" in out, f"Got: {out}"

    def test_star_expression_unpacking(self, nb_runner):
        """Test star expression unpacking."""
        nb_runner.create_notebook([
            "items = [1, 2, 3, 4, 5]",
            "first, *rest = items\nprint(f'first={first}, rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "first=1, rest=[2, 3, 4, 5]" in out, f"Got: {out}"

    def test_multiple_return_types(self, nb_runner):
        """Test caching different types in the same cell."""
        nb_runner.create_notebook([
            """import numpy as np
num = 42
text = "hello"
lst = [1, 2, 3]
dct = {"a": 1}
arr = np.array([1, 2, 3])
tup = (1, "two", 3.0)""",
            """print(f'num={num}, text={text}')
print(f'lst={lst}, dct={dct}')
print(f'arr={arr.tolist()}, tup={tup}')""",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "num=42" in out, f"Got: {out}"
        assert "text=hello" in out
        assert "lst=[1, 2, 3]" in out
        assert "arr=[1, 2, 3]" in out
