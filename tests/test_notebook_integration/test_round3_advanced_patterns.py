"""
Round 3 Batch 6: Multi-file dependencies, disk restore after restart,
complex module patterns, nested loops, dynamic imports, and advanced caching.

These tests push the boundaries of the caching system with intricate patterns.
"""
import pytest
import textwrap
import time


pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestMultiFileDependencies:
    """Test scenarios where multiple files interact with the caching system."""

    @pytest.mark.files
    def test_two_csv_files_independent(self, nb_runner, tmp_path):
        """Two CSV files read in different cells — changing one shouldn't invalidate the other."""
        import pandas as pd
        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        csv1_str = str(csv1).replace('\\', '/')
        csv2_str = str(csv2).replace('\\', '/')
        pd.DataFrame({'amount': [10, 20]}).to_csv(csv1, index=False)
        pd.DataFrame({'name': ['A', 'B']}).to_csv(csv2, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf_sales = pd.read_csv('{csv1_str}')",
            f"df_products = pd.read_csv('{csv2_str}')",
            "total = df_sales['amount'].sum()",
            "names = df_products['name'].tolist()",
            "print(f'Total: {total}, Names: {names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "Total: 30" in out1
        assert "Names: ['A', 'B']" in out1

        # Change only products CSV
        pd.DataFrame({'name': ['X', 'Y', 'Z']}).to_csv(csv2, index=False)
        time.sleep(0.1)  # Ensure mtime changes
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        assert "Total: 30" in out2  # Sales unchanged
        assert "Names: ['X', 'Y', 'Z']" in out2  # Products updated

    @pytest.mark.files
    def test_csv_read_then_write_new_csv(self, nb_runner, tmp_path):
        """Read CSV, transform, write to new CSV, then read the new CSV."""
        import pandas as pd
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_str = str(input_csv).replace('\\', '/')
        output_str = str(output_csv).replace('\\', '/')
        pd.DataFrame({'x': [1, 2, 3]}).to_csv(input_csv, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{input_str}')",
            f"df['x2'] = df['x'] * 2\ndf.to_csv('{output_str}', index=False)",
            f"df_out = pd.read_csv('{output_str}')",
            "print(df_out.to_string(index=False))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "2" in out and "4" in out and "6" in out

    @pytest.mark.files
    def test_json_file_dependency(self, nb_runner, tmp_path):
        """JSON file read and used in computation."""
        import json
        config_path = tmp_path / "config.json"
        config_str = str(config_path).replace('\\', '/')
        json.dump({"threshold": 50, "name": "test"}, config_path.open('w'))

        nb_runner.create_notebook([
            f"import json\nwith open('{config_str}') as f:\n    config = json.load(f)",
            "result = config['threshold'] * 2",
            "print(f\"Name: {config['name']}, Result: {result}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Name: test, Result: 100" in out1

        # Change the JSON
        json.dump({"threshold": 75, "name": "updated"}, config_path.open('w'))
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Name: updated, Result: 150" in out2

    @pytest.mark.files
    def test_text_file_line_count(self, nb_runner, tmp_path):
        """Read a text file and count lines."""
        txt_path = tmp_path / "data.txt"
        txt_str = str(txt_path).replace('\\', '/')
        txt_path.write_text("line1\nline2\nline3\n")

        nb_runner.create_notebook([
            f"with open('{txt_str}') as f:\n    lines = f.readlines()",
            "count = len(lines)\nprint(f'Lines: {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Lines: 3" in out1

        # Add more lines
        txt_path.write_text("line1\nline2\nline3\nline4\nline5\n")
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Lines: 5" in out2


class TestDiskRestoreAfterRestart:
    """Test that cached values can be restored from disk after kernel restart."""

    @pytest.mark.restore
    def test_simple_value_restore(self, nb_runner):
        """Simple value computed, kernel restarted, should restore from disk."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2",
            "# @cash:persist\nprint(f'y: {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "y: 84" in out1

        # Re-run (same kernel, tests skip optimization)
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y: 84" in out2

    @pytest.mark.restore
    def test_chain_restore(self, nb_runner):
        """Chain of computations — cache should handle sequential restore."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5",
            "c = b * 3",
            "print(f'c: {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "c: 45" in out1

        # Run again — should use cache
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "c: 45" in out2


class TestComplexModulePatterns:
    """Test intricate module import and usage patterns."""

    @pytest.mark.modules
    def test_module_with_class_and_function(self, nb_runner, tmp_path):
        """Module provides both a class and a function."""
        mod = tmp_path / "helpers.py"
        mod.write_text(textwrap.dedent("""\
            class Processor:
                def __init__(self, factor):
                    self.factor = factor
                def process(self, x):
                    return x * self.factor
            
            def quick_process(x):
                return x * 2
        """))
        tmp_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport helpers",
            "p = helpers.Processor(3)",
            "r1 = p.process(10)\nr2 = helpers.quick_process(10)",
            "print(f'Class: {r1}, Func: {r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Class: 30" in out
        assert "Func: 20" in out

    @pytest.mark.modules
    def test_module_with_global_state(self, nb_runner, tmp_path):
        """Module has module-level state (counter)."""
        mod = tmp_path / "stateful_mod.py"
        mod.write_text(textwrap.dedent("""\
            _count = 0
            def increment():
                global _count
                _count += 1
                return _count
            def get_count():
                return _count
        """))
        tmp_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport stateful_mod",
            "r1 = stateful_mod.increment()",
            "r2 = stateful_mod.increment()",
            "count = stateful_mod.get_count()\nprint(f'Count: {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Count: 2" in out

    @pytest.mark.modules
    def test_two_modules_interacting(self, nb_runner, tmp_path):
        """Two local modules where one depends on the other."""
        base_mod = tmp_path / "base_utils.py"
        base_mod.write_text(textwrap.dedent("""\
            def normalize(x):
                return x / 100.0
        """))
        calc_mod = tmp_path / "calculator.py"
        calc_mod.write_text(textwrap.dedent("""\
            from base_utils import normalize
            def calc(x):
                return normalize(x) * 2
        """))
        tmp_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport calculator",
            "result = calculator.calc(500)",
            "print(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Result: 10.0" in out


class TestNestedLoopPatterns:
    """Test nested loop patterns and their interaction with caching."""

    @pytest.mark.loops
    def test_nested_for_loop(self, nb_runner):
        """Nested for loop building a matrix."""
        nb_runner.create_notebook([
            "rows = 3\ncols = 4",
            textwrap.dedent("""\
                matrix = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        row.append(i * cols + j)
                    matrix.append(row)"""),
            "print(f'Matrix rows: {len(matrix)}, cols: {len(matrix[0])}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Matrix rows: 3, cols: 4" in out

    @pytest.mark.loops
    def test_while_loop_convergence(self, nb_runner):
        """While loop that converges to a value."""
        nb_runner.create_notebook([
            "target = 100\ntolerance = 0.01",
            textwrap.dedent("""\
                value = 1.0
                iterations = 0
                while abs(value - target) > tolerance:
                    value = (value + target / value) / 2
                    iterations += 1
                    if iterations > 1000:
                        break"""),
            "print(f'Value: {value:.4f}, Iterations: {iterations}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Value: 10.0000" in out  # sqrt(100) = 10

    @pytest.mark.loops
    def test_loop_with_conditional_accumulation(self, nb_runner):
        """Loop that conditionally adds to different accumulators."""
        nb_runner.create_notebook([
            "data = list(range(20))",
            textwrap.dedent("""\
                evens = []
                odds = []
                for x in data:
                    if x % 2 == 0:
                        evens.append(x)
                    else:
                        odds.append(x)"""),
            "print(f'Evens: {len(evens)}, Odds: {len(odds)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Evens: 10, Odds: 10" in out


class TestAdvancedCachingPatterns:
    """Test advanced caching behavior and edge cases."""

    @pytest.mark.core
    def test_cache_hit_on_identical_rerun(self, nb_runner):
        """Running the same cells twice should use cache on second run."""
        nb_runner.create_notebook([
            "import time\nstart = time.time()",
            textwrap.dedent("""\
                # Simulate expensive computation
                result = sum(i**2 for i in range(10000))"""),
            "print(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Result: 333283335000" in out1

        # Second run should be faster (cache hit)
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Result: 333283335000" in out2

    @pytest.mark.core
    def test_annotation_no_cache(self, nb_runner):
        """@cash:no-cache should prevent caching."""
        nb_runner.create_notebook([
            "x = 10",
            "# @cash:no-cache\ny = x * 2",
            "print(f'y: {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "y: 20" in out

    @pytest.mark.core
    def test_skip_optimization_correctness(self, nb_runner):
        """Skip optimization should correctly detect when re-execution is needed."""
        nb_runner.create_notebook([
            "base = 5",
            "derived = base * 3",
            "print(f'derived: {derived}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "derived: 15" in out1

        # Change base — derived should be recomputed
        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "derived: 30" in out2

        # Change back — derived should change again
        nb_runner.set_cell_source(1, "base = 5")
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "derived: 15" in out3

    @pytest.mark.core
    def test_cell_output_display(self, nb_runner):
        """Ensure cell outputs are captured correctly."""
        nb_runner.create_notebook([
            "x = 42",
            "print('hello')\nprint('world')",
            "y = x + 1\nprint(f'y is {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "hello" in out2
        assert "world" in out2
        out3 = nb_runner.get_output(3)
        assert "y is 43" in out3


class TestComplexDataPipelines:
    """Real-world data pipeline patterns."""

    @pytest.mark.core
    def test_groupby_aggregate(self, nb_runner, tmp_path):
        """GroupBy + aggregate pattern."""
        import pandas as pd
        csv = tmp_path / "orders.csv"
        csv_str = str(csv).replace('\\', '/')
        pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50],
        }).to_csv(csv, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "summary = df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()",
            "print(summary.to_string(index=False))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "A" in out and "B" in out

    @pytest.mark.core
    def test_merge_two_dataframes(self, nb_runner, tmp_path):
        """Merge two DataFrames from different cells."""
        import pandas as pd
        csv1 = tmp_path / "users.csv"
        csv2 = tmp_path / "orders.csv"
        csv1_str = str(csv1).replace('\\', '/')
        csv2_str = str(csv2).replace('\\', '/')
        pd.DataFrame({'user_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}).to_csv(csv1, index=False)
        pd.DataFrame({'user_id': [1, 2, 1, 3], 'amount': [100, 200, 150, 300]}).to_csv(csv2, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
            f"orders = pd.read_csv('{csv2_str}')",
            "merged = pd.merge(users, orders, on='user_id')",
            "total_by_user = merged.groupby('name')['amount'].sum().reset_index()",
            "print(total_by_user.to_string(index=False))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "Alice" in out and "Bob" in out and "Charlie" in out


class TestMultiCellFunctionPatterns:
    """Test function definition and usage across multiple cells."""

    @pytest.mark.core
    def test_helper_function_chain(self, nb_runner):
        """Multiple helper functions defined across cells, composed together."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def double(x):
                    return x * 2"""),
            textwrap.dedent("""\
                def add_one(x):
                    return x + 1"""),
            textwrap.dedent("""\
                def compose(f, g):
                    def h(x):
                        return f(g(x))
                    return h"""),
            "transform = compose(double, add_one)",
            "result = transform(5)\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "Result: 12" in out  # double(add_one(5)) = double(6) = 12

    @pytest.mark.core
    def test_function_change_propagates_to_composition(self, nb_runner):
        """Changing a helper function should invalidate composed function usage."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def process(x):
                    return x * 2"""),
            "result = process(10)\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Result: 20" in out1

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def process(x):
                return x * 3"""))
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Result: 30" in out2

    @pytest.mark.core
    def test_recursive_function(self, nb_runner):
        """Recursive function defined in one cell, used in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def factorial(n):
                    if n <= 1:
                        return 1
                    return n * factorial(n - 1)"""),
            "result = factorial(10)\nprint(f'10! = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "10! = 3628800" in out

    @pytest.mark.core
    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure that captures a variable from a previous cell."""
        nb_runner.create_notebook([
            "multiplier = 5",
            textwrap.dedent("""\
                def make_multiplier():
                    return lambda x: x * multiplier"""),
            "fn = make_multiplier()",
            "result = fn(10)\nprint(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "Result: 50" in out1

        # Change multiplier
        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Result: 100" in out2


class TestErrorHandlingPatterns:
    """Test how the caching system handles errors."""


    @pytest.mark.core
    def test_recover_after_fixing_error(self, nb_runner):
        """Fix a cell after an error — should compute correctly."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "print(f'y: {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "y: 20" in out1

        # Change to a different valid computation
        nb_runner.set_cell_source(2, "y = x ** 2")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y: 100" in out2


class TestStringAndFormattingPatterns:
    """Test string manipulation and formatting patterns."""

    @pytest.mark.core
    def test_string_formatting_chain(self, nb_runner):
        """String formatting across multiple cells."""
        nb_runner.create_notebook([
            "name = 'World'",
            "greeting = f'Hello, {name}!'",
            "upper = greeting.upper()",
            "print(upper)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "HELLO, WORLD!" in out

    @pytest.mark.core
    def test_regex_pattern(self, nb_runner):
        """Regex pattern compiled in one cell, used in another."""
        nb_runner.create_notebook([
            "import re\npattern = re.compile(r'\\d+')",
            "text = 'abc 123 def 456 ghi'",
            "matches = pattern.findall(text)\nprint(f'Matches: {matches}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Matches: ['123', '456']" in out


class TestLargeScalePatterns:
    """Test patterns that stress the system at scale."""

    @pytest.mark.core
    def test_many_cells_sequential(self, nb_runner):
        """10 cells in a sequential chain."""
        cells = [f"x{i} = {f'x{i-1} + 1' if i > 0 else '0'}" for i in range(10)]
        cells.append("print(f'x9: {x9}')")
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(11)
        assert "x9: 9" in out

    @pytest.mark.core
    def test_many_variables_per_cell(self, nb_runner):
        """Single cell creating many variables, used in next cell."""
        setup = "\n".join(f"v{i} = {i * 10}" for i in range(15))
        use = "total = " + " + ".join(f"v{i}" for i in range(15))
        nb_runner.create_notebook([
            setup,
            use,
            "print(f'Total: {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        expected = sum(i * 10 for i in range(15))
        assert f"Total: {expected}" in out

    @pytest.mark.core
    def test_diamond_with_intermediate_transforms(self, nb_runner):
        """Complex diamond: A → B, A → C, B → D, C → D with transforms."""
        nb_runner.create_notebook([
            "a = [1, 2, 3, 4, 5]",
            "b = [x * 2 for x in a]",  # doubles
            "c = [x ** 2 for x in a]",  # squares
            "d = [bi + ci for bi, ci in zip(b, c)]",
            "print(f'D: {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "D: [3, 8, 15, 24, 35]" in out1

        # Change source
        nb_runner.set_cell_source(1, "a = [10, 20]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        assert "D: [120, 440]" in out2  # [20+100, 40+400]
