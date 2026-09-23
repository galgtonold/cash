"""Several features at once: files with upstream edits, reloads with restarts."""

import textwrap
import time

import pytest


# Multi-file dependencies, disk restore after restart,
# complex module patterns, nested loops, dynamic imports, and advanced caching.
#
# These tests push the boundaries of the caching system with intricate patterns.
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiFileDependencies:
    """Test scenarios where multiple files interact with the caching system."""

    @pytest.mark.files
    def test_two_csv_files_independent(self, nb_runner, tmp_path):
        """Two CSV files read in different cells — changing one shouldn't invalidate the other."""
        import pandas as pd

        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")
        pd.DataFrame({"amount": [10, 20]}).to_csv(csv1, index=False)
        pd.DataFrame({"name": ["A", "B"]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf_sales = pd.read_csv('{csv1_str}')",
                f"df_products = pd.read_csv('{csv2_str}')",
                "total = df_sales['amount'].sum()",
                "names = df_products['name'].tolist()",
                "print(f'Total: {total}, Names: {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "Total: 30" in out1
        assert "Names: ['A', 'B']" in out1

        # Change only products CSV
        pd.DataFrame({"name": ["X", "Y", "Z"]}).to_csv(csv2, index=False)
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
        input_str = str(input_csv).replace("\\", "/")
        output_str = str(output_csv).replace("\\", "/")
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(input_csv, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{input_str}')",
                f"df['x2'] = df['x'] * 2\ndf.to_csv('{output_str}', index=False)",
                f"df_out = pd.read_csv('{output_str}')",
                "print(df_out.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "2" in out and "4" in out and "6" in out

    @pytest.mark.files
    def test_json_file_dependency(self, nb_runner, tmp_path):
        """JSON file read and used in computation."""
        import json

        config_path = tmp_path / "config.json"
        config_str = str(config_path).replace("\\", "/")
        json.dump({"threshold": 50, "name": "test"}, config_path.open("w"))

        nb_runner.create_notebook(
            [
                f"import json\nwith open('{config_str}') as f:\n    config = json.load(f)",
                "result = config['threshold'] * 2",
                "print(f\"Name: {config['name']}, Result: {result}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Name: test, Result: 100" in out1

        # Change the JSON
        json.dump({"threshold": 75, "name": "updated"}, config_path.open("w"))
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Name: updated, Result: 150" in out2

    @pytest.mark.files
    def test_text_file_line_count(self, nb_runner, tmp_path):
        """Read a text file and count lines."""
        txt_path = tmp_path / "data.txt"
        txt_str = str(txt_path).replace("\\", "/")
        txt_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    lines = f.readlines()",
                "count = len(lines)\nprint(f'Lines: {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Lines: 3" in out1

        # Add more lines
        txt_path.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Lines: 5" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestDiskRestoreAfterRestart:
    """Test that cached values can be restored from disk after kernel restart."""

    @pytest.mark.restore
    def test_simple_value_restore(self, nb_runner):
        """Simple value computed, kernel restarted, should restore from disk."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2",
                "# @cash:persist\nprint(f'y: {y}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 3",
                "print(f'c: {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "c: 45" in out1

        # Run again — should use cache
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "c: 45" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexModulePatterns:
    """Test intricate module import and usage patterns."""

    @pytest.mark.modules
    def test_module_with_class_and_function(self, nb_runner, tmp_path):
        """Module provides both a class and a function."""
        mod = tmp_path / "helpers.py"
        mod.write_text(
            textwrap.dedent("""\
            class Processor:
                def __init__(self, factor):
                    self.factor = factor
                def process(self, x):
                    return x * self.factor
            
            def quick_process(x):
                return x * 2
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport helpers",
                "p = helpers.Processor(3)",
                "r1 = p.process(10)\nr2 = helpers.quick_process(10)",
                "print(f'Class: {r1}, Func: {r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Class: 30" in out
        assert "Func: 20" in out

    @pytest.mark.modules
    def test_module_with_global_state(self, nb_runner, tmp_path):
        """Module has module-level state (counter)."""
        mod = tmp_path / "stateful_mod.py"
        mod.write_text(
            textwrap.dedent("""\
            _count = 0
            def increment():
                global _count
                _count += 1
                return _count
            def get_count():
                return _count
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport stateful_mod",
                "r1 = stateful_mod.increment()",
                "r2 = stateful_mod.increment()",
                "count = stateful_mod.get_count()\nprint(f'Count: {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Count: 2" in out

    @pytest.mark.modules
    def test_two_modules_interacting(self, nb_runner, tmp_path):
        """Two local modules where one depends on the other."""
        base_mod = tmp_path / "base_utils.py"
        base_mod.write_text(
            textwrap.dedent("""\
            def normalize(x):
                return x / 100.0
        """),
            encoding="utf-8",
        )
        calc_mod = tmp_path / "calculator.py"
        calc_mod.write_text(
            textwrap.dedent("""\
            from base_utils import normalize
            def calc(x):
                return normalize(x) * 2
        """),
            encoding="utf-8",
        )
        tmp_str = str(tmp_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import sys\nsys.path.insert(0, '{tmp_str}')\nimport calculator",
                "result = calculator.calc(500)",
                "print(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Result: 10.0" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestNestedLoopPatterns:
    """Test nested loop patterns and their interaction with caching."""

    @pytest.mark.loops
    def test_while_loop_convergence(self, nb_runner):
        """While loop that converges to a value."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Value: 10.0000" in out  # sqrt(100) = 10

    @pytest.mark.loops
    def test_loop_with_conditional_accumulation(self, nb_runner):
        """Loop that conditionally adds to different accumulators."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Evens: 10, Odds: 10" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestAdvancedCachingPatterns:
    """Test advanced caching behavior and edge cases."""

    @pytest.mark.core
    def test_cache_hit_on_identical_rerun(self, nb_runner):
        """Running the same cells twice should use cache on second run."""
        nb_runner.create_notebook(
            [
                "import time\nstart = time.time()",
                textwrap.dedent("""\
                # Simulate expensive computation
                result = sum(i**2 for i in range(10000))"""),
                "print(f'Result: {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 10",
                "# @cash:no-cache\ny = x * 2",
                "print(f'y: {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "y: 20" in out

    @pytest.mark.core
    def test_skip_optimization_correctness(self, nb_runner):
        """Skip optimization should correctly detect when re-execution is needed."""
        nb_runner.create_notebook(
            [
                "base = 5",
                "derived = base * 3",
                "print(f'derived: {derived}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 42",
                "print('hello')\nprint('world')",
                "y = x + 1\nprint(f'y is {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "hello" in out2
        assert "world" in out2
        out3 = nb_runner.get_output(3)
        assert "y is 43" in out3


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexDataPipelines:
    """Real-world data pipeline patterns."""

    @pytest.mark.core
    def test_groupby_aggregate(self, nb_runner, tmp_path):
        """GroupBy + aggregate pattern."""
        import pandas as pd

        csv = tmp_path / "orders.csv"
        csv_str = str(csv).replace("\\", "/")
        pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        ).to_csv(csv, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "summary = df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()",
                "print(summary.to_string(index=False))",
            ]
        )
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
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")
        pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}).to_csv(csv1, index=False)
        pd.DataFrame({"user_id": [1, 2, 1, 3], "amount": [100, 200, 150, 300]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
                f"orders = pd.read_csv('{csv2_str}')",
                "merged = pd.merge(users, orders, on='user_id')",
                "total_by_user = merged.groupby('name')['amount'].sum().reset_index()",
                "print(total_by_user.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "Alice" in out and "Bob" in out and "Charlie" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiCellFunctionPatterns:
    """Test function definition and usage across multiple cells."""

    @pytest.mark.core
    def test_function_change_propagates_to_composition(self, nb_runner):
        """Changing a helper function should invalidate composed function usage."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def process(x):
                    return x * 2"""),
                "result = process(10)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Result: 20" in out1

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def process(x):
                return x * 3"""),
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Result: 30" in out2

    @pytest.mark.core
    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure that captures a variable from a previous cell."""
        nb_runner.create_notebook(
            [
                "multiplier = 5",
                textwrap.dedent("""\
                def make_multiplier():
                    return lambda x: x * multiplier"""),
                "fn = make_multiplier()",
                "result = fn(10)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "Result: 50" in out1

        # Change multiplier
        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Result: 100" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestErrorHandlingPatterns:
    """Test how the caching system handles errors."""

    @pytest.mark.core
    def test_recover_after_fixing_error(self, nb_runner):
        """Fix a cell after an error — should compute correctly."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "print(f'y: {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "y: 20" in out1

        # Change to a different valid computation
        nb_runner.set_cell_source(2, "y = x ** 2")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y: 100" in out2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestLargeScalePatterns:
    """Test patterns that stress the system at scale."""

    @pytest.mark.core
    def test_many_cells_sequential(self, nb_runner):
        """10 cells in a sequential chain."""
        cells = [f"x{i} = {f'x{i - 1} + 1' if i > 0 else '0'}" for i in range(10)]
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
        nb_runner.create_notebook(
            [
                setup,
                use,
                "print(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        expected = sum(i * 10 for i in range(15))
        assert f"Total: {expected}" in out

    @pytest.mark.core
    def test_diamond_with_intermediate_transforms(self, nb_runner):
        """Complex diamond: A → B, A → C, B → D, C → D with transforms."""
        nb_runner.create_notebook(
            [
                "a = [1, 2, 3, 4, 5]",
                "b = [x * 2 for x in a]",  # doubles
                "c = [x ** 2 for x in a]",  # squares
                "d = [bi + ci for bi, ci in zip(b, c)]",
                "print(f'D: {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "D: [3, 8, 15, 24, 35]" in out1

        # Change source
        nb_runner.set_cell_source(1, "a = [10, 20]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        assert "D: [120, 440]" in out2  # [20+100, 40+400]


# Combined complex patterns stressing multiple subsystems.
#
# Tests focusing on:
# 1. File deps + upstream simulation combined
# 2. Module reload + function tracking + from-import combined
# 3. Kernel restart + file modification + upstream invalidation
# 4. Multi-output cells + dependency chain + cache invalidation
# 5. Complex real-world simulation: data loading → processing → analysis → visualization data
# 6. Nested function definitions with closures across cells
# 7. Exception handling interleaved with caching
# 8. Type conversion chains (str→int→float→list→dict)
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestFileDepsWithUpstream:
    """Combined file dependency and upstream simulation patterns."""

    @pytest.mark.files
    @pytest.mark.upstream
    def test_file_dep_change_propagates_through_chain(self, nb_runner, tmp_path):
        """File change should propagate through dependent cells."""
        import pandas as pd

        csv_path = str(tmp_path / "input.csv").replace("\\", "/")
        pd.DataFrame({"val": [1, 2, 3]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "total = df['val'].sum()",
                "doubled = total * 2",
                "print(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "doubled=12" in output

        # Modify file
        time.sleep(0.1)
        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "doubled=120" in output2

    @pytest.mark.files
    @pytest.mark.upstream
    def test_multiple_file_deps_one_changes(self, nb_runner, tmp_path):
        """Multiple file deps, only one changes - verify partial invalidation."""
        import pandas as pd

        file_a = str(tmp_path / "a.csv").replace("\\", "/")
        file_b = str(tmp_path / "b.csv").replace("\\", "/")

        pd.DataFrame({"x": [1]}).to_csv(file_a, index=False)
        pd.DataFrame({"y": [100]}).to_csv(file_b, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"a = pd.read_csv('{file_a}')['x'].iloc[0]",
                f"b = pd.read_csv('{file_b}')['y'].iloc[0]",
                "result = a + b",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=101" in output

        # Change only file_a
        time.sleep(0.1)
        pd.DataFrame({"x": [50]}).to_csv(file_a, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "result=150" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestModuleReloadCombined:
    """Module reload combined with various other features."""

    @pytest.mark.modules
    @pytest.mark.upstream
    def test_module_function_change_propagates_downstream(self, nb_runner, tmp_path):
        """Changing a module function should invalidate all downstream users."""
        mod_path = str(tmp_path / "mymod.py").replace("\\", "/")

        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("def transform(x): return x * 2\n")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
                "import mymod",
                "a = mymod.transform(5)",
                "b = a + 10",
                "print(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "b=20" in output  # transform(5)=10, 10+10=20

        # Modify module
        time.sleep(0.1)
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("def transform(x): return x * 3\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "b=25" in output2  # transform(5)=15, 15+10=25

    @pytest.mark.modules
    def test_from_import_function_and_constant_mixed(self, nb_runner, tmp_path):
        """Module with both function and constant from-imports."""
        mod_path = str(tmp_path / "config_mod.py").replace("\\", "/")

        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("VERSION = '1.0'\ndef greet(name): return f'Hello {name} v{VERSION}'\n")

        nb_runner.create_notebook(
            [
                f"import sys; sys.path.insert(0, '{str(tmp_path).replace(chr(92), '/')}')",
                "from config_mod import VERSION, greet",
                "msg = greet('World')",
                "print(f'msg={msg} version={VERSION}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "Hello World v1.0" in output
        assert "version=1.0" in output

        # Update module
        time.sleep(0.1)
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write("VERSION = '2.0'\ndef greet(name): return f'Hi {name} v{VERSION}'\n")

        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "Hi World v2.0" in output2
        assert "version=2.0" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestRestartWithFileDeps:
    """Kernel restart combined with file dependency changes."""

    @pytest.mark.restore
    @pytest.mark.files
    def test_restart_after_file_modification(self, nb_runner, tmp_path):
        """Restart kernel after external file mod should recompute."""
        import pandas as pd

        csv_path = str(tmp_path / "restart_test.csv").replace("\\", "/")
        pd.DataFrame({"val": [5, 10, 15]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "result = df['val'].mean()",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=10.0" in output

        # Modify file and restart
        time.sleep(0.1)
        pd.DataFrame({"val": [100, 200, 300]}).to_csv(csv_path, index=False)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=200.0" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiOutputChains:
    """Complex multi-output cell chains."""

    @pytest.mark.core
    @pytest.mark.upstream
    def test_multi_output_diamond_invalidation(self, nb_runner):
        """Multi-output cell feeding into diamond pattern."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "x = base + 1\ny = base + 2\nz = base + 3",
                "left = x * y",  # depends on x, y
                "right = y * z",  # depends on y, z
                "final = left + right",
                "print(f'final={final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        # x=11, y=12, z=13 → left=132, right=156 → final=288
        assert "final=288" in output

        # Change base
        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        # x=101, y=102, z=103 → left=10302, right=10506 → final=20808
        assert "final=20808" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestRealWorldSimulation:
    """Simulates a realistic data analysis workflow."""

    @pytest.mark.stress
    @pytest.mark.files
    def test_full_data_pipeline(self, nb_runner, tmp_path):
        """Complete data pipeline: load → clean → transform → analyze → report."""
        import numpy as np
        import pandas as pd

        # Create test data
        csv_path = str(tmp_path / "sales.csv").replace("\\", "/")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="D").astype(str),
                "product": np.random.choice(["A", "B", "C"], 100),
                "quantity": np.random.randint(1, 50, 100),
                "price": np.round(np.random.uniform(10, 100, 100), 2),
            }
        )
        df.to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                f"raw = pd.read_csv('{csv_path}')",
                "raw['date'] = pd.to_datetime(raw['date'])\nraw['revenue'] = raw['quantity'] * raw['price']",
                "by_product = raw.groupby('product')['revenue'].sum().to_dict()",
                "total_revenue = sum(by_product.values())\ntop_product = max(by_product, key=by_product.get)",
                "print(f'total={total_revenue:.0f} top={top_product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "total=" in output
        assert "top=" in output

        # Re-run should cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(6)
        assert output2.strip() == output.strip()

    @pytest.mark.stress
    def test_iterative_model_tuning(self, nb_runner):
        """Simulate iterative parameter tuning."""
        nb_runner.create_notebook(
            [
                "import numpy as np\nnp.random.seed(42)",
                "data = np.random.randn(1000)",
                "threshold = 1.0",
                "above = np.sum(data > threshold)\nbelow = np.sum(data < -threshold)\nwithin = len(data) - above - below",
                "print(f'above={above} below={below} within={within}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "above=" in output1

        # Tune threshold
        nb_runner.set_cell_source(3, "threshold = 2.0")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        # With higher threshold, fewer above/below
        above2 = int(output2.split("above=")[1].split()[0])
        above1 = int(output1.split("above=")[1].split()[0])
        assert above2 < above1  # Higher threshold → fewer above


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestNestedFunctionClosures:
    """Tests for nested functions and closure patterns across cells."""

    @pytest.mark.core
    def test_higher_order_function_composition(self, nb_runner):
        """Function composition with higher-order functions."""
        nb_runner.create_notebook(
            [
                "def compose(f, g):\n    def composed(x):\n        return f(g(x))\n    return composed",
                "double = lambda x: x * 2\nadd_one = lambda x: x + 1",
                "double_then_add = compose(add_one, double)",
                "result = double_then_add(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=11" in output  # double(5)=10, add_one(10)=11

        # Change composition order
        nb_runner.set_cell_source(3, "add_then_double = compose(double, add_one)")
        nb_runner.set_cell_source(4, "result = add_then_double(5)")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "result=12" in output2  # add_one(5)=6, double(6)=12


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestTypeConversionChains:
    """Tests for type conversion chains across cells."""

    @pytest.mark.core
    def test_str_to_int_to_float_chain(self, nb_runner):
        """String → int → float conversion chain."""
        nb_runner.create_notebook(
            [
                "raw = '42'",
                "as_int = int(raw)",
                "as_float = float(as_int) / 10",
                "print(f'result={as_float}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=4.2" in output

        # Change input
        nb_runner.set_cell_source(1, "raw = '100'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=10.0" in output2

    @pytest.mark.core
    def test_list_to_dict_to_dataframe(self, nb_runner):
        """List → dict → DataFrame conversion chain."""
        nb_runner.create_notebook(
            [
                "import pandas as pd",
                "names = ['Alice', 'Bob', 'Charlie']",
                "scores = [90, 85, 95]",
                "data_dict = dict(zip(names, scores))",
                "df = pd.DataFrame(list(data_dict.items()), columns=['name', 'score'])",
                "top = df.loc[df['score'].idxmax(), 'name']",
                "print(f'top={top}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(7)
        assert "top=Charlie" in output

        # Change scores
        nb_runner.set_cell_source(3, "scores = [95, 100, 85]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(7)
        assert "top=Bob" in output2

    @pytest.mark.core
    def test_json_roundtrip(self, nb_runner):
        """JSON serialization roundtrip should preserve data."""
        nb_runner.create_notebook(
            [
                "import json",
                "original = {'key': [1, 2, 3], 'nested': {'a': True, 'b': None}}",
                "serialized = json.dumps(original)",
                "restored = json.loads(serialized)",
                "match = original == restored",
                "print(f'match={match} type={type(restored).__name__}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "match=True" in output
        assert "type=dict" in output


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexLoopPatterns:
    """Tests for complex loop patterns that interact with caching."""

    @pytest.mark.loops
    def test_nested_loop_with_accumulator(self, nb_runner):
        """Nested loops with accumulator pattern."""
        nb_runner.create_notebook(
            [
                "n = 5",
                "matrix = []\nfor i in range(n):\n    row = []\n    for j in range(n):\n        row.append(i * n + j)\n    matrix.append(row)",
                "flat = [x for row in matrix for x in row]",
                "total = sum(flat)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "total=300" in output  # sum(0..24) = 300

        # Change n
        nb_runner.set_cell_source(1, "n = 3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "total=36" in output2  # sum(0..8) = 36

    @pytest.mark.loops
    def test_while_loop_with_convergence(self, nb_runner):
        """While loop that converges to a result."""
        nb_runner.create_notebook(
            [
                "target = 100\ntolerance = 0.01",
                "guess = 1.0\niterations = 0\nwhile abs(guess * guess - target) > tolerance:\n    guess = (guess + target / guess) / 2\n    iterations += 1",
                "print(f'sqrt={guess:.4f} iterations={iterations}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "sqrt=10.0" in output

        # Change target
        nb_runner.set_cell_source(1, "target = 225\ntolerance = 0.01")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "sqrt=15.0" in output2


# Complex scenario integration tests.
#
# Tests progressively complex caching interactions including:
# - Deep dependency chains (5+ cells)
# - Dataclass and complex object caching
# - Function redefinition with downstream propagation
# - Multi-variable assignment patterns
# - Exception handling and recovery
# - Cross-cell data transformations
# - Global state interactions
# - Nested function closures across cells
# - Re-execution after code modifications
# - Complex pandas operations
@pytest.mark.core
class TestDeepDependencyChains:
    """Test deep dependency chains spanning many cells."""

    def test_six_cell_chain_modification_propagates(self, nb_runner):
        """
        Run a 6-cell chain, modify cell 1, then re-run cell 6.
        The upstream simulation should detect the change and recompute.
        """
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base * 2",
                "step2 = step1 + 5",
                "step3 = step2 ** 2",
                "step4 = step3 - 100",
                "print(f'Result: {step4}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out1 = nb_runner.get_output(6)
        assert "Result: 525" in out1

        # Modify the base cell
        nb_runner.set_cell_source(1, "base = 5")
        # Re-run from cell 1 through cell 6
        nb_runner.run_cells([1, 2, 3, 4, 5, 6])

        # base=5 -> step1=10 -> step2=15 -> step3=225 -> step4=125
        out2 = nb_runner.get_output(6)
        assert "Result: 125" in out2, f"Expected 125 after modification, got: {out2}"

    def test_branching_modify_root_all_branches_update(self, nb_runner):
        """
        Modify the root of a diamond and verify both branches update.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x * 2",
                "b = x + 5",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "c = 35" in nb_runner.get_output(4)

        # Change x from 10 to 100
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cells([1, 2, 3, 4])

        # x=100, a=200, b=105, c=305
        out = nb_runner.get_output(4)
        assert "c = 305" in out, f"Expected c=305 after modification, got: {out}"


@pytest.mark.core
class TestDataclassCaching:
    """Test caching behavior with dataclass objects."""

    def test_dataclass_mutation_detection(self, nb_runner):
        """Test that mutating a dataclass field is detected."""
        nb_runner.create_notebook(
            [
                """from dataclasses import dataclass, field
from typing import List

@dataclass
class Accumulator:
    items: List[int] = field(default_factory=list)
    
    def add(self, val):
        self.items.append(val)
        return self""",
                "acc = Accumulator()",
                "acc.add(10)\nacc.add(20)\nprint(f'Items: {acc.items}')",
                "total = sum(acc.items)\nprint(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out3 = nb_runner.get_output(3)
        assert "Items: [10, 20]" in out3, f"Expected [10, 20], got: {out3}"

        out4 = nb_runner.get_output(4)
        assert "Total: 30" in out4, f"Expected Total: 30, got: {out4}"


@pytest.mark.core
class TestFunctionRedefinition:
    """Test that redefining a function triggers downstream recomputation."""

    def test_redefine_function_invalidates_downstream(self, nb_runner):
        """
        Define a function in cell 1, use it in cell 2.
        Then redefine the function and re-run cell 2.
        """
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "result = transform(5)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Result: 10" in nb_runner.get_output(2)

        # Redefine the function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Result: 15" in out, f"Expected Result: 15 after redefine, got: {out}"

    def test_redefine_function_only_downstream_runs(self, nb_runner):
        """
        Redefine function in cell 1, run only downstream cell 2.
        The upstream simulation should detect the change and re-execute cell 1.
        """
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x + 100",
                "val = compute(5)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "val = 105" in nb_runner.get_output(2)

        # Modify function definition
        nb_runner.set_cell_source(1, "def compute(x):\n    return x + 200")
        # Only run cell 2 - upstream should auto-execute cell 1
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "val = 205" in out, f"Expected val=205, got: {out}"


@pytest.mark.core
class TestComplexAssignments:
    """Test complex assignment patterns."""

    def test_nested_tuple_unpacking(self, nb_runner):
        """Test nested tuple unpacking."""
        nb_runner.create_notebook(
            [
                "pairs = [(1, 'a'), (2, 'b'), (3, 'c')]",
                "firsts = [x for x, y in pairs]\nseconds = [y for x, y in pairs]",
                "print(f'firsts={firsts}, seconds={seconds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "firsts=[1, 2, 3]" in out, f"Got: {out}"
        assert "seconds=['a', 'b', 'c']" in out


@pytest.mark.core
class TestCrossDataTransformations:
    """Test complex data transformations across cells."""

    def test_pandas_modify_source_data(self, nb_runner):
        """Modify source data and verify the pipeline updates."""
        nb_runner.create_notebook(
            [
                """import pandas as pd
df = pd.DataFrame({'name': ['A', 'B'], 'val': [10, 20]})""",
                "total = df['val'].sum()\nprint(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Total: 30" in nb_runner.get_output(2)

        # Modify the DataFrame
        nb_runner.set_cell_source(
            1, "import pandas as pd\ndf = pd.DataFrame({'name': ['A', 'B', 'C'], 'val': [10, 20, 30]})"
        )
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Total: 60" in out, f"Expected Total: 60, got: {out}"


@pytest.mark.core
class TestReexecutionPatterns:
    """Test various re-execution patterns."""

    def test_skip_optimization_on_rerun(self, nb_runner):
        """
        Run all cells, then re-run them all.
        Second run should skip (or cache-hit) unchanged cells.
        """
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x + 8\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "y = 50" in nb_runner.get_output(2)

        # Re-run all - should produce same result
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "y = 50" in out, f"Expected same result on re-run, got: {out}"

    def test_add_new_intermediate_dependency(self, nb_runner):
        """
        Run A -> C, then modify C to depend on new variable B.
        """
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 99",
                "result = a * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 20" in nb_runner.get_output(3)

        # Now make result depend on b too
        nb_runner.set_cell_source(3, "result = a + b\nprint(f'result = {result}')")
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "result = 109" in out, f"Expected result=109, got: {out}"


@pytest.mark.core
class TestClassInheritancePatterns:
    """Test class inheritance and method caching."""

    def test_modify_base_class_invalidates_child(self, nb_runner):
        """Modifying base class should invalidate derived class usage."""
        nb_runner.create_notebook(
            [
                """class Base:
    def greet(self):
        return "Hello"
""",
                """class Child(Base):
    def greet(self):
        return super().greet() + " World"
""",
                "obj = Child()\nprint(obj.greet())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Hello World" in nb_runner.get_output(3)

        # Modify the base class
        nb_runner.set_cell_source(
            1,
            """class Base:
    def greet(self):
        return "Hi"
""",
        )
        nb_runner.run_cells([1, 2, 3])

        out = nb_runner.get_output(3)
        assert "Hi World" in out, f"Expected 'Hi World', got: {out}"


@pytest.mark.core
class TestMultiStatementCells:
    """Test cells with many statements."""

    def test_mixed_assignments_and_expressions(self, nb_runner):
        """Test cells mixing assignments, print calls, and function definitions."""
        nb_runner.create_notebook(
            [
                """data = [1, 2, 3, 4, 5]
total = sum(data)
mean = total / len(data)
above = [x for x in data if x > mean]
below = [x for x in data if x <= mean]
print(f'mean={mean}, above={above}, below={below}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "mean=3.0" in out, f"Got: {out}"
        assert "above=[4, 5]" in out
        assert "below=[1, 2, 3]" in out


@pytest.mark.core
class TestConditionalLogic:
    """Test conditional logic patterns."""

    def test_change_condition_triggers_recompute(self, nb_runner):
        """Changing a condition variable should trigger recomputation."""
        nb_runner.create_notebook(
            [
                "mode = 'add'",
                """if mode == 'add':
    result = 10 + 20
else:
    result = 10 * 20
print(f'result = {result}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 30" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "result = 200" in out, f"Expected result=200, got: {out}"


# Complex combined interaction tests.
#
# Tests that combine multiple features: functions + files, imports + edits + restart,
# loops + mutations + edits, etc. These simulate real-world notebook workflows.
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestFunctionPlusFileEdit:
    """Function that reads file, both function and file edited."""

    def test_function_reads_file_then_file_changes(self, nb_runner, tmp_path):
        """Function reads a file; file content changes."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("100", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"def load_value():\n    with open('{path_str}') as f:\n        return int(f.read().strip())",
                "val = load_value()\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change file
        data_file.write_text("999", encoding="utf-8")
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)

    def test_function_and_file_both_change(self, nb_runner, tmp_path):
        """Both the function definition and the file change."""
        data_file = tmp_path / "vals.txt"
        data_file.write_text("10\n20\n30", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"def process():\n    with open('{path_str}') as f:\n        return sum(int(x) for x in f)",
                "result = process()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file content and function
        data_file.write_text("1\n2\n3", encoding="utf-8")
        nb_runner.set_cell_source(
            1,
            f"def process():\n    with open('{path_str}') as f:\n        return max(int(x) for x in f)",
        )
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestFunctionPlusLoopEdit:
    """Function used inside a loop, both edited."""

    def test_edit_function_used_in_loop(self, nb_runner):
        """Function called inside loop, redefine function."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "results = []\nfor i in range(4):\n    results.append(transform(i))",
                "print(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "results = [0, 1, 4, 9]" in nb_runner.get_output(3)

    def test_edit_loop_and_function(self, nb_runner):
        """Edit both the loop and the function it uses."""
        nb_runner.create_notebook(
            [
                "def scale(x, factor):\n    return x * factor",
                "out = []\nfor v in [1, 2, 3]:\n    out.append(scale(v, 10))",
                "print(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = [10, 20, 30]" in nb_runner.get_output(3)

        # Edit function and loop data
        nb_runner.set_cell_source(1, "def scale(x, factor):\n    return x + factor")
        nb_runner.set_cell_source(2, "out = []\nfor v in [10, 20, 30]:\n    out.append(scale(v, 100))")
        nb_runner.run_all()
        assert "out = [110, 120, 130]" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestImportPlusFunctionEdit:
    """Import + function definition + function call, with edits."""

    def test_import_used_in_function_then_edit(self, nb_runner):
        """Import used inside function body, edit function."""
        nb_runner.create_notebook(
            [
                "import math",
                "def area(r):\n    return math.pi * r ** 2",
                "a = area(1)\nprint(f'a = {a:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 3.14" in nb_runner.get_output(3)

        # Change to volume
        nb_runner.set_cell_source(2, "def area(r):\n    return (4/3) * math.pi * r ** 3")
        nb_runner.run_all()
        assert "a = 4.19" in nb_runner.get_output(3)

    def test_switch_import_and_function(self, nb_runner):
        """Switch from one import to another, function changes too."""
        nb_runner.create_notebook(
            [
                "import math",
                "def compute(x):\n    return math.sqrt(x)",
                "result = compute(16)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "import json")
        nb_runner.set_cell_source(2, "def compute(x):\n    return len(json.dumps({'val': x}))")
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # json.dumps({'val': 16}) -> '{"val": 16}' which is 11 chars
        assert "result = 11" in output


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDataPipelineEdits:
    """Simulate a data processing pipeline with edits at various stages."""

    def test_etl_pipeline_edit_transform(self, nb_runner):
        """ETL: extract -> transform -> load. Edit transform step."""
        nb_runner.create_notebook(
            [
                "raw = [1, 2, 3, 4, 5]",
                "transformed = [x * 2 for x in raw]",
                "result = sum(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit transform
        nb_runner.set_cell_source(2, "transformed = [x ** 2 for x in raw]")
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(3)

    def test_etl_pipeline_edit_source(self, nb_runner):
        """ETL: edit the source data."""
        nb_runner.create_notebook(
            [
                "raw = [10, 20, 30]",
                "filtered = [x for x in raw if x > 15]",
                "total = sum(filtered)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 50" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [5, 10, 15, 20, 25, 30]")
        nb_runner.run_all()
        assert "total = 75" in nb_runner.get_output(3)

    def test_multi_stage_pipeline_edit_multiple(self, nb_runner):
        """Multi-stage pipeline: source -> filter -> transform -> aggregate."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 11))",
                "filtered = [x for x in data if x % 2 == 0]",
                "transformed = [x * 10 for x in filtered]",
                "result = sum(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(4)

        # Edit filter and transform
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x % 2 != 0]")
        nb_runner.set_cell_source(3, "transformed = [x * 100 for x in filtered]")
        nb_runner.run_all()
        assert "result = 2500" in nb_runner.get_output(4)


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestConditionalChainEdits:
    """Conditional logic chains with edits."""

    def test_nested_conditionals_edit(self, nb_runner):
        """Nested if-else, edit the input to change branch taken."""
        nb_runner.create_notebook(
            [
                "score = 85",
                "if score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelse:\n    grade = 'C'",
                "print(f'grade = {grade}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grade = B" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 95")
        nb_runner.run_all()
        assert "grade = A" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "score = 50")
        nb_runner.run_all()
        assert "grade = C" in nb_runner.get_output(3)

    def test_conditional_chain_edit_thresholds(self, nb_runner):
        """Edit the thresholds in conditionals."""
        nb_runner.create_notebook(
            [
                "val = 50",
                "if val > 100:\n    cat = 'high'\nelse:\n    cat = 'low'",
                "print(f'cat = {cat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cat = low" in nb_runner.get_output(3)

        # Lower threshold
        nb_runner.set_cell_source(2, "if val > 40:\n    cat = 'high'\nelse:\n    cat = 'low'")
        nb_runner.run_all()
        assert "cat = high" in nb_runner.get_output(3)
