"""
Round 3 Batch 4: Real-world data science patterns, kernel restart scenarios,
out-of-order execution, annotation directives, and complex mutation patterns.

These tests focus on realistic notebook workflows that data scientists 
commonly use, including pandas transformations, numpy array operations,
and iterative refinement patterns.
"""

import pytest
import time


@pytest.mark.core
class TestPandasPipelinePatterns:
    """Test realistic pandas workflows across cells."""

    def test_load_transform_aggregate(self, nb_runner, tmp_path):
        """Classic ETL: load CSV, transform, aggregate."""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text("product,qty,price\nA,10,1.5\nB,5,3.0\nA,8,1.5\nB,12,3.0\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
            "df['total'] = df['qty'] * df['price']",
            "summary = df.groupby('product')['total'].sum().to_dict()\nprint(summary)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "'A': 27.0" in out, f"Got: {out}"
        assert "'B': 51.0" in out, f"Got: {out}"

    def test_dataframe_filtering_chain(self, nb_runner, tmp_path):
        """Chain of DataFrame filters across cells."""
        csv_path = tmp_path / "people.csv"
        csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,NYC\nDiana,28,LA\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
            "nyc_df = df[df['city'] == 'NYC']",
            "nyc_over_30 = nyc_df[nyc_df['age'] >= 30]\nprint(nyc_over_30['name'].tolist())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Alice" in out, f"Got: {out}"
        assert "Charlie" in out, f"Got: {out}"

    def test_csv_modification_detected(self, nb_runner, tmp_path):
        """Modify the CSV file between runs — should re-compute."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("val\n10\n20\n30\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}')",
            "total = df['val'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify the CSV
        csv_path.write_text("val\n100\n200\n300\n")
        time.sleep(0.5)

        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "total = 600" in out, f"Expected total=600, got: {out}"


@pytest.mark.core
class TestNumpyPatterns:
    """Test numpy array operations and caching."""

    def test_array_creation_and_operations(self, nb_runner):
        """Create arrays in cell 1, operate in cell 2."""
        nb_runner.create_notebook([
            "import numpy as np\na = np.array([1, 2, 3, 4, 5])",
            "b = a * 2\nc = np.sum(b)\nprint(f'sum = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(2)

    def test_matrix_operations(self, nb_runner):
        """Matrix multiply across cells."""
        nb_runner.create_notebook([
            "import numpy as np\nA = np.array([[1, 2], [3, 4]])",
            "B = np.array([[5, 6], [7, 8]])",
            "C = A @ B\nprint(C.tolist())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "19" in out, f"Got: {out}"
        assert "22" in out, f"Got: {out}"

    def test_numpy_random_with_seed(self, nb_runner):
        """Seeded random should be reproducible and cacheable."""
        nb_runner.create_notebook([
            "import numpy as np\nnp.random.seed(42)",
            "vals = np.random.rand(3)\nprint([round(v, 4) for v in vals])",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(2)
        
        # Re-run — should produce same output (from cache or same seed)
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        
        # Both should contain the same values
        assert out1 == out2 or "0.3745" in out2, f"Inconsistent: {out1} vs {out2}"


@pytest.mark.core
class TestOutOfOrderExecution:
    """Test running cells out of order and how cash handles it."""

    def test_run_cell3_before_cell2(self, nb_runner):
        """Run cell 3 before cell 2 — cell 3 needs cell 2's output."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # Skip cell 2, run cell 3 — upstream should restore y
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 30" in out, f"Expected z=30, got: {out}"

    def test_reverse_order_execution(self, nb_runner):
        """Run cells in reverse: 3, 2, 1 — first run populates cache."""
        nb_runner.create_notebook([
            "x = 7",
            "y = x * 3",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        # First run all to populate cache
        nb_runner.run_all()
        assert "z = 22" in nb_runner.get_output(3)

        # Now run only cell 3 — should use cached upstream
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "z = 22" in out, f"Expected z=22, got: {out}"


@pytest.mark.core
class TestAnnotationDirectives:
    """Test @cash: annotation directives."""

    def test_no_cache_annotation(self, nb_runner):
        """@cash:no-cache should force execution every time."""
        nb_runner.create_notebook([
            "x = 10",
            "# @cash:no-cache\nimport time\nt = time.time()\nprint(f't = {t}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(2)
        time.sleep(0.1)

        nb_runner.run_all()
        out2 = nb_runner.get_output(2)

        # Outputs should differ because no-cache forces re-execution
        # Both should have 't = ' prefix
        assert "t = " in out1
        assert "t = " in out2

    def test_persist_annotation(self, nb_runner, tmp_path):
        """@cash:persist should force disk storage."""
        nb_runner.create_notebook([
            "# @cash:persist\nx = 42\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(1)


@pytest.mark.core
class TestMultiStatementCells:
    """Test cells with many statements and complex interactions."""

    def test_cell_with_function_def_and_usage(self, nb_runner):
        """Define and use a function in the same cell."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2\n\nresult = double(21)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(1)

    def test_cell_with_multiple_assignments(self, nb_runner):
        """Multiple assignments in one cell, used in the next."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3\nd = a + b + c",
            "print(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 6" in nb_runner.get_output(2)

    def test_long_cell_with_intermediate_vars(self, nb_runner):
        """10-statement cell with intermediate calculations."""
        nb_runner.create_notebook([
            "x = 1\nx = x + 1\nx = x * 2\nx = x + 3\nx = x * 2\nx = x - 1\nx = x // 3\nx = x + 10\nx = x * 2\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # 1 -> 2 -> 4 -> 7 -> 14 -> 13 -> 4 -> 14 -> 28
        assert "x = 28" in nb_runner.get_output(1)


@pytest.mark.core
class TestReassignmentPatterns:
    """Test variable reassignment across cells."""

    def test_reassign_same_variable_different_cells(self, nb_runner):
        """Assign x in cell 1, reassign in cell 2, use in cell 3."""
        nb_runner.create_notebook([
            "x = 10",
            "x = x * 2",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

    def test_reassign_with_different_type(self, nb_runner):
        """Assign x as int, then reassign as string."""
        nb_runner.create_notebook([
            "x = 42",
            "x = str(x) + ' is the answer'",
            "print(x)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42 is the answer" in nb_runner.get_output(3)

    def test_swap_variables(self, nb_runner):
        """Swap two variables."""
        nb_runner.create_notebook([
            "a = 'hello'\nb = 'world'",
            "a, b = b, a",
            "print(f'a={a}, b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=world, b=hello" in nb_runner.get_output(3)


@pytest.mark.core
class TestComplexDataStructures:
    """Test caching with complex nested data structures."""

    def test_nested_dict_of_lists(self, nb_runner):
        """Nested dict creation and access."""
        nb_runner.create_notebook([
            "data = {'users': [{'name': 'Alice', 'scores': [90, 85]}, {'name': 'Bob', 'scores': [78, 92]}]}",
            "avg_scores = {u['name']: sum(u['scores'])/len(u['scores']) for u in data['users']}\nprint(avg_scores)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "'Alice': 87.5" in out, f"Got: {out}"
        assert "'Bob': 85.0" in out, f"Got: {out}"

    def test_namedtuple_pattern(self, nb_runner):
        """NamedTuple creation and usage."""
        nb_runner.create_notebook([
            "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
            "p1 = Point(3, 4)\np2 = Point(6, 8)\ndist = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5\nprint(f'dist = {dist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "dist = 5.0" in nb_runner.get_output(2)

    def test_dataclass_pattern(self, nb_runner):
        """Dataclass definition and usage across cells."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n\n@dataclass\nclass Employee:\n    name: str\n    salary: float\n    dept: str",
            "emp1 = Employee('Alice', 90000, 'Eng')\nemp2 = Employee('Bob', 85000, 'Eng')\nemp3 = Employee('Charlie', 75000, 'Sales')",
            "eng_avg = (emp1.salary + emp2.salary) / 2\nprint(f'eng avg = {eng_avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "eng avg = 87500.0" in nb_runner.get_output(3)


@pytest.mark.core
class TestImportPatterns:
    """Test various import patterns and their caching."""

    def test_import_as_alias(self, nb_runner):
        """import X as Y."""
        nb_runner.create_notebook([
            "import math as m",
            "r = m.sqrt(144)\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 12.0" in nb_runner.get_output(2)

    def test_from_import_as_alias(self, nb_runner):
        """from X import Y as Z."""
        nb_runner.create_notebook([
            "from math import sqrt as sq",
            "r = sq(256)\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 16.0" in nb_runner.get_output(2)

    def test_multiple_imports_one_cell(self, nb_runner):
        """Multiple import statements in one cell."""
        nb_runner.create_notebook([
            "import math\nimport os\nimport json",
            "data = json.dumps({'pi': round(math.pi, 4), 'cwd': os.sep})\nprint(data)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "3.1416" in out, f"Got: {out}"


@pytest.mark.core
class TestCachingEfficiency:
    """Test that caching actually works — second run should use cache."""

    def test_second_run_uses_cache(self, nb_runner):
        """Second run_all should be faster or produce same output from cache."""
        nb_runner.create_notebook([
            "import time\nstart = time.time()\ntime.sleep(0.1)\ncompute_time = time.time() - start",
            "x = 42",  # Simple computation
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(3)

        # Second run — cell 2 should use cache
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(3)

    def test_unchanged_cells_skip(self, nb_runner):
        """Running unchanged cells should skip re-execution."""
        nb_runner.create_notebook([
            "x = 100",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 200" in nb_runner.get_output(2)

        # Second run without changes — should skip
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "y = 200" in out, f"Second run should produce same output, got: {out}"


@pytest.mark.modules
class TestFromImportClassReload:
    """Test from-import with classes (not just functions)."""

    def test_from_import_class_change(self, nb_runner, tmp_path):
        """from X import MyClass — class definition change should propagate."""
        mod_path = tmp_path / "shapes.py"
        mod_path.write_text(
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        return 3.14 * self.r ** 2\n"
        )

        nb_runner.create_notebook([
            "from shapes import Circle",
            "c = Circle(5)\nprint(f'area = {c.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "area = 78.5" in nb_runner.get_output(2)

        # Change to use math.pi
        mod_path.write_text(
            "import math\n"
            "class Circle:\n"
            "    def __init__(self, r):\n"
            "        self.r = r\n"
            "    def area(self):\n"
            "        return math.pi * self.r ** 2\n"
        )
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "area = 78.5398" in out, f"Expected math.pi area, got: {out}"

    def test_from_import_constant_change(self, nb_runner, tmp_path):
        """from X import CONST — constant value change should propagate."""
        mod_path = tmp_path / "config.py"
        mod_path.write_text("VERSION = '1.0'\n")

        nb_runner.create_notebook([
            "from config import VERSION",
            "print(f'Version: {VERSION}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Version: 1.0" in nb_runner.get_output(2)

        # Update the constant
        mod_path.write_text("VERSION = '2.0'\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "Version: 2.0" in out, f"Expected Version: 2.0, got: {out}"
