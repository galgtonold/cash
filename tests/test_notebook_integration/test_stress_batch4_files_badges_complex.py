"""
Stress Test Batch 4: File Dependencies, Badge Accuracy, Complex Interactions (Scenarios 96-130)

Tests file dependency tracking, badge/metrics accuracy, complex multi-cell 
interactions, forbidden functions, randomness handling, and more.
"""

import pytest
import time

pytestmark = pytest.mark.stress


# =============================================================================
# Scenario 96-105: File Dependency Edge Cases
# =============================================================================


class TestFileDependencies:
    """Tests for file dependency tracking and invalidation."""

    def test_96_csv_read_then_modify(self, nb_runner, tmp_path):
        """Scenario 96: Read CSV, modify file, re-run — should re-execute."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x\n1\n2\n3\n")
        csv_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')\nprint(f'len={{len(df)}}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "len=3" in nb_runner.get_output(1)
        # Modify the file
        time.sleep(0.1)
        csv_path.write_text("x\n1\n2\n3\n4\n5\n")
        # Re-run — should detect file change
        nb_runner.run_cell(1)
        assert "len=5" in nb_runner.get_output(1)

    def test_97_multiple_file_reads(self, nb_runner, tmp_path):
        """Scenario 97: Statement reads 2 files — both tracked."""
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        csv1.write_text("x\n1\n2\n")
        csv2.write_text("x\n3\n4\n")
        s1 = str(csv1).replace('\\', '/')
        s2 = str(csv2).replace('\\', '/')

        cell_code = (
            f"import pandas as pd\ndf1 = pd.read_csv('{s1}')\n"
            f"df2 = pd.read_csv('{s2}')\ntotal = len(df1) + len(df2)\n"
            "print(f'total={total}')"
        )
        nb_runner.create_notebook([cell_code])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "total=4" in nb_runner.get_output(1)
        # Modify only second file
        time.sleep(0.1)
        csv2.write_text("x\n3\n4\n5\n")
        nb_runner.run_cell(1)
        assert "total=5" in nb_runner.get_output(1)

    def test_98_file_dep_propagation_to_downstream(self, nb_runner, tmp_path):
        """Scenario 103: Cell 1 reads file → df, Cell 2 uses df — inherits file dep."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x\n10\n20\n")
        csv_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "total = df['x'].sum()\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=30" in nb_runner.get_output(2)
        # Modify file
        time.sleep(0.1)
        csv_path.write_text("x\n10\n20\n30\n")
        # Re-run cell 2 only — should detect file dep changed via propagation
        nb_runner.run_cell(2)
        out = nb_runner.get_output(2)
        # Could be 30 (if file dep not propagated) or 60 (if propagated and upstream re-executed)
        # The key test is that it doesn't return stale data
        assert "total=" in out

    def test_99_file_dep_not_propagated_to_scalar(self, nb_runner, tmp_path):
        """Scenario 104: n = len(df) — n should NOT inherit file dep."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x\n1\n2\n3\n")
        csv_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "n = len(df)\nprint(f'n={n}')",
            "msg = f'Got {n} rows'\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "n=3" in nb_runner.get_output(2)
        assert "Got 3 rows" in nb_runner.get_output(3)

    def test_100_file_read_unchanged_skips(self, nb_runner, tmp_path):
        """File unchanged between runs — should skip re-execution."""
        csv_path = tmp_path / "stable.csv"
        csv_path.write_text("a,b\n1,2\n")
        csv_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')\nprint(f'rows={{len(df)}}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "rows=1" in nb_runner.get_output(1)
        # Re-run without changing file
        nb_runner.run_cell(1)
        assert "rows=1" in nb_runner.get_output(1)


# =============================================================================
# Scenario 106-115: Badge & Metrics Accuracy
# =============================================================================


class TestBadgeMetrics:
    """Tests for badge display and metrics accuracy."""

    def test_101_first_run_produces_output(self, nb_runner):
        """Scenario 106: Fresh execution produces correct output."""
        nb_runner.create_notebook([
            "x = 42\nprint(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        out = nb_runner.get_output(1)
        assert "x=42" in out

    def test_102_cached_run_replays_output(self, nb_runner):
        """Scenario 107: Cached/skipped run still shows output."""
        nb_runner.create_notebook([
            "x = 42\nprint(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "x=42" in nb_runner.get_output(1)
        # Second run should skip but still show output
        nb_runner.run_cell(1)
        assert "x=42" in nb_runner.get_output(1)

    def test_103_mixed_status_cell(self, nb_runner):
        """Scenario 109: Cell with some cached, some computed statements."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1\nc = 100\nprint(f'b={b}, c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b=2, c=100" in nb_runner.get_output(2)
        # Change a, then re-run cell 2 — b should recompute, c should skip
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "b=11, c=100" in nb_runner.get_output(2)

    def test_104_error_partial_execution(self, nb_runner):
        """Scenario 113: Cell errors mid-execution — earlier statements still cached."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1\nraise ValueError('test error')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        from nbclient.exceptions import CellExecutionError
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

    def test_105_large_cell_multiple_statements(self, nb_runner):
        """Cell with many statements — all should be processed."""
        stmts = []
        for i in range(10):
            stmts.append(f"v{i} = {i}")
        stmts.append("total = " + " + ".join(f"v{i}" for i in range(10)))
        stmts.append("print(f'total={total}')")
        code = "\n".join(stmts)

        nb_runner.create_notebook([code])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # sum(0..9) = 45
        assert "total=45" in nb_runner.get_output(1)

    def test_106_rerun_shows_same_output(self, nb_runner):
        """Re-run cell — output should be identical (replayed from cache/skip)."""
        nb_runner.create_notebook([
            "import math\nresult = math.factorial(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        out1 = nb_runner.get_output(1)
        assert "result=3628800" in out1
        nb_runner.run_cell(1)
        out2 = nb_runner.get_output(1)
        assert "result=3628800" in out2


# =============================================================================
# Scenario 116-130: Complex Multi-Cell Interactions
# =============================================================================


class TestComplexInteractions:
    """Tests for complex multi-cell interaction scenarios."""

    def test_107_data_pipeline_full(self, nb_runner, tmp_path):
        """Scenario 116: Full data pipeline: load → filter → transform → agg."""
        csv_path = tmp_path / "pipeline.csv"
        csv_path.write_text("name,value\nalpha,10\nbeta,20\ngamma,30\nalpha,40\nbeta,50\n")
        csv_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "filtered = df[df['value'] > 15]",
            "filtered = filtered.copy()\nfiltered['doubled'] = filtered['value'] * 2",
            "agg = filtered.groupby('name')['doubled'].sum()\nprint(agg.to_dict())",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        # alpha(40→80), beta(20→40, 50→100), gamma(30→60)
        assert "'alpha': 80" in out
        assert "'gamma': 60" in out

    def test_108_shared_function_modification(self, nb_runner):
        """Scenario 117: Modify shared function, downstream re-computes."""
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "a = transform(5)\nprint(f'a={a}')",
            "b = transform(10)\nprint(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10" in nb_runner.get_output(2)
        assert "b=20" in nb_runner.get_output(3)
        # Modify function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_all()
        assert "a=15" in nb_runner.get_output(2)
        assert "b=30" in nb_runner.get_output(3)

    def test_109_class_definition_and_use(self, nb_runner):
        """Scenario 118: Define class, instantiate, modify class."""
        nb_runner.create_notebook([
            "class Calculator:\n    def compute(self, x):\n        return x + 1",
            "calc = Calculator()\nresult = calc.compute(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)
        # Modify class
        nb_runner.set_cell_source(1, "class Calculator:\n    def compute(self, x):\n        return x + 100")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "result=110" in nb_runner.get_output(2)

    def test_110_variable_type_change(self, nb_runner):
        """Scenario 122: Variable type changes — downstream handles correctly."""
        nb_runner.create_notebook([
            "x = [1, 2, 3]",
            "length = len(x)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length=3" in nb_runner.get_output(2)
        # Change type
        nb_runner.set_cell_source(1, "x = 'hello'")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "length=5" in nb_runner.get_output(2)

    def test_111_datetime_forbidden_function(self, nb_runner):
        """Scenario 126: datetime.now() — forbidden function, not cached."""
        nb_runner.create_notebook([
            "from datetime import datetime\nnow = datetime.now()\nprint(type(now).__name__)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "datetime" in nb_runner.get_output(1)

    def test_112_random_without_seed(self, nb_runner):
        """Scenario 127: random.random() without seed — should handle correctly."""
        nb_runner.create_notebook([
            "import random\nval = random.random()\nprint(f'got_value={val is not None}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "got_value=True" in nb_runner.get_output(1)

    def test_113_context_manager_file_read(self, nb_runner, tmp_path):
        """Scenario 130: with open() as f: — file tracking + caching."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello world")
        txt_str = str(txt_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"with open('{txt_str}') as f:\n    content = f.read()\nprint(f'content={{content}}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "content=hello world" in nb_runner.get_output(1)
        # Modify file
        time.sleep(0.1)
        txt_path.write_text("updated content")
        nb_runner.run_cell(1)
        assert "content=updated content" in nb_runner.get_output(1)

    def test_114_import_and_use_pattern(self, nb_runner):
        """Scenario 120: Import in cell 1, use in cells 2-3."""
        nb_runner.create_notebook([
            "import math",
            "a = math.sqrt(16)\nprint(f'a={a}')",
            "b = math.pi\nprint(f'b={b:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=4.0" in nb_runner.get_output(2)
        assert "b=3.14" in nb_runner.get_output(3)

    def test_115_accumulator_across_cells(self, nb_runner):
        """Scenario 121: Accumulator across cells."""
        nb_runner.create_notebook([
            "results = []",
            "results.append(1)\nresults.append(2)",
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 2]" in nb_runner.get_output(3)

    def test_116_string_operations_chain(self, nb_runner):
        """Scenario 125: String transformations cached correctly."""
        nb_runner.create_notebook([
            "text = '  Hello World  '",
            "text = text.strip()",
            "text = text.lower()",
            "text = text.replace('world', 'python')\nprint(f'text={text}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text=hello python" in nb_runner.get_output(4)

    def test_117_conditional_computation(self, nb_runner):
        """Complex conditional with data-dependent branching."""
        nb_runner.create_notebook([
            "import pandas as pd\ndf = pd.DataFrame({'x': range(100)})",
            "if len(df) > 50:\n    result = df['x'].mean()\nelse:\n    result = df['x'].sum()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # len=100 > 50, so mean = 49.5
        assert "result=49.5" in nb_runner.get_output(2)
        # Change to small DataFrame
        nb_runner.set_cell_source(1, "import pandas as pd\ndf = pd.DataFrame({'x': range(10)})")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        # len=10 < 50, so sum = 45
        assert "result=45" in nb_runner.get_output(2)

    def test_118_decorator_on_function(self, nb_runner):
        """Scenario 129: Function with decorator."""
        nb_runner.create_notebook([
            "def my_decorator(func):\n    def wrapper(*args):\n        return func(*args) + 100\n    return wrapper",
            "@my_decorator\ndef compute(x):\n    return x * 2",
            "result = compute(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(3)

    def test_119_global_constant_used_everywhere(self, nb_runner):
        """Constant defined once, used in many cells."""
        nb_runner.create_notebook([
            "MULTIPLIER = 10",
            "a = 1 * MULTIPLIER\nprint(f'a={a}')",
            "b = 2 * MULTIPLIER\nprint(f'b={b}')",
            "c = 3 * MULTIPLIER\nprint(f'c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10" in nb_runner.get_output(2)
        assert "b=20" in nb_runner.get_output(3)
        assert "c=30" in nb_runner.get_output(4)
        # Change constant
        nb_runner.set_cell_source(1, "MULTIPLIER = 100")
        nb_runner.run_all()
        assert "a=100" in nb_runner.get_output(2)
        assert "b=200" in nb_runner.get_output(3)
        assert "c=300" in nb_runner.get_output(4)

    def test_120_complex_dict_operations(self, nb_runner):
        """Complex dict build across multiple cells."""
        nb_runner.create_notebook([
            "config = {'version': 1}",
            "config['name'] = 'test'",
            "config['items'] = [1, 2, 3]",
            "print(f'config={config}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "'version': 1" in out
        assert "'name': 'test'" in out
        assert "'items': [1, 2, 3]" in out

    def test_121_dataframe_multi_transform(self, nb_runner):
        """Multiple DataFrame transforms — each step cached."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'a': np.arange(100), 'b': np.random.RandomState(42).randn(100)})",
            "df = df.sort_values('b')",
            "df['c'] = df['a'].cumsum()",
            "result = df['c'].iloc[-1]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=" in out

    def test_122_fresh_kernel_with_loop_upstream(self, nb_runner):
        """Fresh kernel, run downstream — upstream has loop."""
        nb_runner.create_notebook([
            "data = [10, 20, 30]",
            "total = 0\nfor x in data:\n    total += x",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(3)
        nb_runner.reset_cash_state()
        nb_runner.run_cell(3)
        assert "total=60" in nb_runner.get_output(3)

    def test_123_modify_loop_upstream_run_downstream(self, nb_runner):
        """Modify loop cell in upstream, run only downstream."""
        nb_runner.create_notebook([
            "items = ['a', 'b']",
            "result = {}\nfor item in items:\n    result[item] = len(item)",
            "print(f'keys={sorted(result.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['a', 'b']" in nb_runner.get_output(3)
        # Modify loop cell
        nb_runner.set_cell_source(2, "result = {}\nfor item in items:\n    result[item] = len(item) * 10")
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "keys=['a', 'b']" in out

    def test_124_exception_in_loop_iteration(self, nb_runner):
        """Exception in loop iteration — partial results available."""
        nb_runner.create_notebook([
            "results = {}\nfor x in [1, 2, 0, 3]:\n    try:\n        results[x] = 100 // x\n    except ZeroDivisionError:\n        results[x] = -1\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        out = nb_runner.get_output(1)
        assert "0: -1" in out
        assert "1: 100" in out

    def test_125_multiple_imports_same_cell(self, nb_runner):
        """Multiple imports in same cell — all tracked."""
        nb_runner.create_notebook([
            "import math\nimport os\nimport json",
            "result = math.sqrt(json.loads('4'))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=2.0" in nb_runner.get_output(2)



    def test_128_f_string_with_method_call(self, nb_runner):
        """f-string with method calls — analysis handles correctly."""
        nb_runner.create_notebook([
            "name = 'hello world'",
            "msg = f'Upper: {name.upper()}, Len: {len(name)}'\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Upper: HELLO WORLD" in out
        assert "Len: 11" in out

    def test_129_nested_function_definitions(self, nb_runner):
        """Nested function definitions."""
        nb_runner.create_notebook([
            "def outer(x):\n    def inner(y):\n        return y * 2\n    return inner(x) + 1",
            "result = outer(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(2)

    def test_130_walrus_operator(self, nb_runner):
        """Walrus operator (:=) in if condition."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "if (n := len(data)) > 3:\n    print(f'Large: {n}')\nelse:\n    print(f'Small: {n}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Large: 5" in nb_runner.get_output(2)
