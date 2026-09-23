"""Values flowing through many cells, edited in the middle."""

import textwrap
import time

import pytest


# Cross-cell patterns, class instances, generators,
# exception recovery, partial re-runs, multi-module cascades.
#
# Tests focus on complex real-world usage patterns that span multiple cells
# and exercise the caching framework's ability to track dependencies across
# execution boundaries.
@pytest.mark.core
class TestClassInstanceCrossCells:
    """Test class instance creation, method calls, and mutation across cells."""

    def test_class_redefinition_invalidates_instances(self, nb_runner):
        """Redefining a class should invalidate cells using instances of it."""
        nb_runner.create_notebook(
            [
                "class Greeter:\n    def greet(self):\n        return 'hello'",
                "g = Greeter()\nprint(g.greet())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "hello" in nb_runner.get_output(2)

        # Change the class
        nb_runner.set_cell_source(1, "class Greeter:\n    def greet(self):\n        return 'hi there'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "hi there" in out, f"Expected 'hi there', got: {out}"

    def test_inheritance_chain_across_cells(self, nb_runner):
        """Base class in cell 1, child in cell 2, usage in cell 3."""
        nb_runner.create_notebook(
            [
                "class Animal:\n    def speak(self):\n        return 'generic sound'",
                "class Dog(Animal):\n    def speak(self):\n        return 'woof'",
                "d = Dog()\nprint(d.speak())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "woof" in nb_runner.get_output(3)

        # Modify base class — doesn't change Dog.speak(), but tests revalidation
        nb_runner.set_cell_source(
            1,
            "class Animal:\n    def speak(self):\n        return 'roar'\n    def name(self):\n        return 'animal'",
        )
        nb_runner.run_all()

        # Dog.speak still returns woof
        assert "woof" in nb_runner.get_output(3)


@pytest.mark.core
class TestGeneratorAndIteratorCaching:
    """Test caching behavior with generators and iterators."""

    def test_list_comprehension_from_range(self, nb_runner):
        """List comprehension result should be cached."""
        nb_runner.create_notebook(
            [
                "n = 5",
                "squares = [x**2 for x in range(n)]\nprint(squares)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run — should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)


@pytest.mark.core
class TestExceptionRecovery:
    """Test that errors in one cell don't break caching in subsequent cells."""

    def test_error_cell_doesnt_break_next_cell(self, nb_runner):
        """An error in cell 2 shouldn't prevent cell 3 from running."""
        from nbclient.exceptions import CellExecutionError

        nb_runner.create_notebook(
            [
                "x = 42",
                "y = 1/0  # ZeroDivisionError",
                "z = x * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib

        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # This will error
        nb_runner.run_cell(3)

        out = nb_runner.get_output(3)
        assert "z = 84" in out, f"Got: {out}"

    def test_fix_error_and_rerun(self, nb_runner):
        """Fix a broken cell and re-run — should work correctly."""
        from nbclient.exceptions import CellExecutionError

        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + undefined_var",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        import contextlib

        with contextlib.suppress(CellExecutionError):
            nb_runner.run_cell(2)  # NameError

        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x + 5")
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "y = 15" in out, f"Got: {out}"


@pytest.mark.core
class TestPartialReExecution:
    """Test re-running only some cells while others use cache."""

    def test_rerun_middle_cell_only(self, nb_runner):
        """Run all, then only re-run cell 2 — cell 3 should still have correct output."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 3",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 31" in nb_runner.get_output(3)

        # Only re-run cell 3 (without running cells 1 and 2 again)
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "z = 31" in out, f"Expected z=31, got: {out}"

    def test_skip_to_last_cell(self, nb_runner):
        """Run only the last cell — upstream system should restore deps."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 2",
                "c = b + 3\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        # Run all to populate cache
        nb_runner.run_all()
        assert "c = 6" in nb_runner.get_output(3)

        # Now only run cell 3 — should restore a and b from cache/upstream
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "c = 6" in out, f"Expected c=6, got: {out}"


@pytest.mark.modules
class TestFromImportCascadeChain:
    """Test from-import with multi-cell cascade chains."""

    def test_from_import_used_in_three_cells(self, nb_runner, tmp_path):
        """
        from X import Y in cell 1, used in cells 2 and 3.
        Module change should invalidate both downstream cells.
        """
        mod_path = tmp_path / "calc.py"
        mod_path.write_text("def double(x):\n    return x * 2\n")

        nb_runner.create_notebook(
            [
                "from calc import double",
                "a = double(5)\nprint(f'a = {a}')",
                "b = double(a)\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 20" in nb_runner.get_output(3)

        # Change double to triple
        mod_path.write_text("def double(x):\n    return x * 3\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        assert "a = 15" in out2, f"Cell 2: expected a=15, got: {out2}"
        assert "b = 45" in out3, f"Cell 3: expected b=45, got: {out3}"

    def test_from_import_multiple_names(self, nb_runner, tmp_path):
        """
        from X import Y, Z — both names should update when module changes.
        """
        mod_path = tmp_path / "ops.py"
        mod_path.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")

        nb_runner.create_notebook(
            [
                "from ops import add, sub",
                "r1 = add(10, 3)\nr2 = sub(10, 3)\nprint(f'add={r1}, sub={r2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "add=13" in out, f"Got: {out}"
        assert "sub=7" in out, f"Got: {out}"

        # Change add to multiply
        mod_path.write_text("def add(a, b):\n    return a * b\n\ndef sub(a, b):\n    return a - b\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        assert "add=30" in out2, f"Expected add=30, got: {out2}"
        assert "sub=7" in out2, f"sub should still be 7, got: {out2}"


@pytest.mark.modules
class TestMultiModuleDependencies:
    """Test scenarios with multiple interdependent modules."""

    def test_two_modules_one_cell(self, nb_runner, tmp_path):
        """Import two different modules and use both."""
        mod_a = tmp_path / "mod_a.py"
        mod_a.write_text("def fa(x):\n    return x + 1\n")
        mod_b = tmp_path / "mod_b.py"
        mod_b.write_text("def fb(x):\n    return x * 2\n")

        nb_runner.create_notebook(
            [
                "import mod_a\nimport mod_b",
                "r = mod_a.fa(mod_b.fb(5))\nprint(f'r = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "r = 11" in nb_runner.get_output(2)

        # Change mod_b
        mod_b.write_text("def fb(x):\n    return x * 3\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "r = 16" in out, f"Expected r=16, got: {out}"

    def test_module_importing_module(self, nb_runner, tmp_path):
        """
        Module A imports Module B. Change B, both should update.
        """
        mod_b = tmp_path / "helper.py"
        mod_b.write_text("FACTOR = 10\n")
        mod_a = tmp_path / "processor.py"
        mod_a.write_text("from helper import FACTOR\ndef process(x):\n    return x * FACTOR\n")

        nb_runner.create_notebook(
            [
                "import processor",
                "result = processor.process(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 50" in nb_runner.get_output(2)

        # Change the helper module's FACTOR
        mod_b.write_text("FACTOR = 100\n")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        # This tests transitive dependency detection
        assert "result = 500" in out, f"Expected result=500 after helper change, got: {out}"


@pytest.mark.files
class TestFileOperationPatterns:
    """Test various file operation patterns and their caching behavior."""

    def test_write_then_read_same_cell(self, nb_runner, tmp_path):
        """Write and read a file in the same cell."""
        nb_runner.create_notebook(
            [
                f"import json\ndata = {{'key': 'value'}}\nwith open(r'{(tmp_path / 'test.json').as_posix()}', 'w') as f:\n    json.dump(data, f)",
                f"import json\nwith open(r'{(tmp_path / 'test.json').as_posix()}') as f:\n    loaded = json.load(f)\nprint(loaded['key'])",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "value" in nb_runner.get_output(2)

    def test_csv_with_different_separators(self, nb_runner, tmp_path):
        """Read CSV with semicolons."""
        csv_path = tmp_path / "semi.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}', sep=';')",
                "print(df.sum().to_string())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "5" in out, f"Expected sum of column a=5, got: {out}"


class TestComplexUpstreamPatterns:
    """Test complex upstream dependency resolution patterns."""

    @pytest.mark.upstream
    def test_diamond_dependency(self, nb_runner):
        """
        Cell 1 → Cell 2 and Cell 3 → Cell 4 (diamond).
        Modify cell 1, run cell 4 — should cascade through both paths.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1",
                "b = x + 2",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 23" in nb_runner.get_output(4)

        # Modify the root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(4)

        out = nb_runner.get_output(4)
        assert "c = 203" in out, f"Expected c=203, got: {out}"

    @pytest.mark.upstream
    def test_long_chain_six_cells(self, nb_runner):
        """Six-cell chain: each transforms the previous."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "x2 = x + 1",
                "x3 = x2 + 1",
                "x4 = x3 + 1",
                "x5 = x4 + 1",
                "x6 = x5 + 1\nprint(f'x6 = {x6}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x6 = 6" in nb_runner.get_output(6)

        # Modify root
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(6)

        out = nb_runner.get_output(6)
        assert "x6 = 105" in out, f"Expected x6=105, got: {out}"

    @pytest.mark.upstream
    def test_independent_branches_no_interference(self, nb_runner):
        """
        Two independent branches should not interfere with each other.
        Cell 1: x = 10
        Cell 2: a = x + 1
        Cell 3: y = 20 (independent)
        Cell 4: b = y + 1 (depends only on y)
        Modifying x should not re-execute cell 4.
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1\nprint(f'a = {a}')",
                "y = 20",
                "b = y + 1\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(4)

        # Modify x — only cell 2 should change, cell 4 stays
        nb_runner.set_cell_source(1, "x = 99")
        nb_runner.run_all()

        out2 = nb_runner.get_output(2)
        out4 = nb_runner.get_output(4)
        assert "a = 100" in out2, f"Expected a=100, got: {out2}"
        assert "b = 21" in out4, f"Expected b=21 unchanged, got: {out4}"

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_diamond_dependency_on_run_all(self, nb_runner):
        """
        Diamond pattern: A → B, A → C, B+C → D.
        Changing A should propagate through both paths to D.
        """
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a * 2",  # b depends on a
                "c = a * 3",  # c depends on a
                "d = b + c",  # d depends on b and c
                "print(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "d=50" in output1  # 20+30

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "d=500" in output2  # 200+300

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_deep_dependency_chain(self, nb_runner):
        """Deep chain: a → b → c → d → e → f → result."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1",
                "result = f + 1",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(8)
        assert "result=7" in output1

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(8)
        assert "result=106" in output2

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_upstream_with_function_call(self, nb_runner):
        """Upstream should track through function definitions and calls."""
        nb_runner.create_notebook(
            [
                "def multiply(x, y): return x * y",
                "a = 5",
                "b = multiply(a, 3)",
                "print(f'b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "b=15" in output1

        # Change function definition
        nb_runner.set_cell_source(1, "def multiply(x, y): return x * y + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "b=16" in output2

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    @pytest.mark.upstream
    def test_upstream_with_conditional_dependency(self, nb_runner):
        """Upstream tracks through conditionals that select different paths."""
        nb_runner.create_notebook(
            [
                "mode = 'add'",
                "x = 10",
                "if mode == 'add':\n    result = x + 100\nelse:\n    result = x * 100",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "result=110" in output1

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=1000" in output2


@pytest.mark.core
class TestConditionalExecution:
    """Test conditional patterns across cells."""

    def test_conditional_variable_assignment(self, nb_runner):
        """Condition in cell 1 affects cell 2's computation."""
        nb_runner.create_notebook(
            [
                "mode = 'double'",
                "x = 10\nresult = x * 2 if mode == 'double' else x * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'triple'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 30" in out, f"Expected 30, got: {out}"


@pytest.mark.core
class TestGlobalStateInteraction:
    """Test caching with global/module-level state modifications."""

    def test_memoization_pattern(self, nb_runner):
        """Test a memoized function pattern."""
        nb_runner.create_notebook(
            [
                "def memoize(f):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = f(*args)\n        return cache[args]\n    return wrapper",
                "@memoize\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
                "r = fib(10)\nprint(f'fib(10) = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "fib(10) = 55" in nb_runner.get_output(3)


@pytest.mark.core
class TestTryCatchPatterns:
    """Test try/except patterns in cached cells."""

    def test_try_except_switch_paths(self, nb_runner):
        """Change input to switch from success to error path."""
        nb_runner.create_notebook(
            [
                "x = '42'",
                "try:\n    val = int(x)\nexcept ValueError:\n    val = -1\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change to trigger error path
        nb_runner.set_cell_source(1, "x = 'abc'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "val = -1" in out, f"Expected val=-1, got: {out}"


# Advanced cross-cell interaction torture tests.
#
# Tests that combine multiple features simultaneously: function definitions
# referencing external variables, class hierarchies with file dependencies,
# decorator + module reload combos, and long multi-cell computation chains.
@pytest.mark.integration
@pytest.mark.stress
class TestCrossCellFunctionState:
    """Test functions that capture state from other cells."""

    def test_function_reads_global_config(self, nb_runner):
        """Function in cell 2 reads config dict from cell 1."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                config = {
                    'multiplier': 3,
                    'offset': 10,
                    'precision': 2
                }
            """),
                textwrap.dedent("""\
                def transform(x):
                    result = x * config['multiplier'] + config['offset']
                    return round(result, config['precision'])
            """),
                textwrap.dedent("""\
                values = [1.0, 2.5, 3.7]
                results = [transform(v) for v in values]
                print(results)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*3+10=13, 2.5*3+10=17.5, 3.7*3+10=21.1
        assert "13" in output
        assert "17.5" in output
        assert "21.1" in output

        # Change config -> function results should change
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            config = {
                'multiplier': 10,
                'offset': 0,
                'precision': 1
            }
        """),
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*10+0=10, 2.5*10+0=25, 3.7*10+0=37
        assert "10" in output
        assert "25" in output
        assert "37" in output

    def test_recursive_function_across_cells(self, nb_runner):
        """Recursive function defined in one cell, called in another."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                memo = {}
                def fib(n):
                    if n in memo:
                        return memo[n]
                    if n < 2:
                        return n
                    result = fib(n-1) + fib(n-2)
                    memo[n] = result
                    return result
            """),
                textwrap.dedent("""\
                result = fib(20)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6765" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestClassWithFileDependency:
    """Test class definitions that interact with file operations."""

    def test_class_reads_config_file(self, nb_runner, tmp_path):
        """Class method reads from a config file."""
        config_file = tmp_path / "app_config.json"
        config_file.write_text('{"version": "1.0", "debug": false}')
        path_str = str(config_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import json
                class AppConfig:
                    def __init__(self):
                        with open('{path_str}') as f:
                            self._data = json.load(f)
                    @property
                    def version(self):
                        return self._data['version']
            """),
                textwrap.dedent("""\
                cfg = AppConfig()
                print(cfg.version)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1.0" in nb_runner.get_output(2)

    def test_data_processor_with_csv(self, nb_runner, tmp_path):
        """Data processing class that reads CSV files."""
        csv_path = tmp_path / "processor_data.csv"
        csv_path.write_text("metric,value\nCPU,75\nMEM,60\nDISK,45\n")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                textwrap.dedent("""\
                class MetricAnalyzer:
                    def __init__(self, path):
                        self.df = pd.read_csv(path)
                    def summary(self):
                        return self.df['value'].describe()
                    def max_metric(self):
                        idx = self.df['value'].idxmax()
                        return self.df.loc[idx, 'metric']
            """),
                textwrap.dedent(f"""\
                analyzer = MetricAnalyzer('{path_str}')
                print(analyzer.max_metric())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "CPU" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestComplexDependencyGraphs:
    """Test complex dependency graph patterns."""

    def test_diamond_with_function_deps(self, nb_runner):
        """Diamond dependency with functions: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook(
            [
                "base_value = 10",
                textwrap.dedent("""\
                def path_b(x):
                    return x ** 2
                b_result = path_b(base_value)
            """),
                textwrap.dedent("""\
                def path_c(x):
                    return x * 3
                c_result = path_c(base_value)
            """),
                textwrap.dedent("""\
                final = b_result + c_result
                print(final)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "130" in nb_runner.get_output(4)  # 100 + 30

        # Change root
        nb_runner.set_cell_source(1, "base_value = 5")
        nb_runner.run_all()
        assert "40" in nb_runner.get_output(4)  # 25 + 15

    def test_wide_fan_out_fan_in(self, nb_runner):
        """Many independent computations merging into one result."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1",
                "b = x + 2",
                "c = x + 3",
                "d = x + 4",
                "e = x + 5",
                "total = a + b + c + d + e",
                "print(total)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 11+12+13+14+15 = 65
        assert "65" in nb_runner.get_output(8)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        # 101+102+103+104+105 = 515
        assert "515" in nb_runner.get_output(8)


@pytest.mark.integration
@pytest.mark.stress
class TestMultipleRestorePhases:
    """Test complex restore-after-restart scenarios."""

    def test_function_and_data_restore(self, nb_runner):
        """Both function definitions and data restore after restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def process(items):
                    return [x.upper() for x in items]
            """),
                textwrap.dedent("""\
                data = ['hello', 'world', 'test']
            """),
                textwrap.dedent("""\
                result = process(data)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "HELLO" in output
        assert "WORLD" in output

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "HELLO" in output
        assert "WORLD" in output

    def test_class_instance_restore(self, nb_runner):
        """Class definition and instance restore after restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Counter:
                    def __init__(self, start=0):
                        self.value = start
                    def increment(self):
                        self.value += 1
                        return self.value
            """),
                textwrap.dedent("""\
                c = Counter(10)
                c.increment()
                c.increment()
                print(c.value)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(2)


# Out-of-order execution, selective cell runs, and re-execution
# patterns that stress the upstream simulation system.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestSelectiveCellExecution:
    """Test running specific cells rather than all."""

    def test_run_only_modified_and_downstream(self, nb_runner):
        """Run notebook, modify a cell, run only that cell + downstream."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=10, b=15, c=30
        assert "30" in nb_runner.get_output(4)

        # Modify cell 1 and re-run cells 1-4
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_cells([1, 2, 3, 4])
        # a=20, b=25, c=50
        assert "50" in nb_runner.get_output(4)

    def test_run_cell_range(self, nb_runner):
        """Run a range of cells, skipping early ones."""
        nb_runner.create_notebook(
            [
                "x = 100",
                "y = 200",
                "z = x + y",
                "print(z)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(4)

        # Change x and run all again
        nb_runner.set_cell_source(1, "x = 500")
        nb_runner.run_all()
        assert "700" in nb_runner.get_output(4)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestRerunSameCell:
    """Test re-running the same cell multiple times."""

    def test_rerun_print_cell(self, nb_runner):
        """Re-run print cell gives fresh output."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "print(sum(data))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)

        # Re-run
        nb_runner.run_cell(2)
        assert "6" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestCellDependencyChain:
    """Test long dependency chains with selective execution."""

    def test_long_chain_head_change(self, nb_runner):
        """Change head of a 6-cell chain, re-run all."""
        nb_runner.create_notebook(
            [
                "base = 1",
                "step1 = base * 2",  # 2
                "step2 = step1 + 3",  # 5
                "step3 = step2 * 4",  # 20
                "step4 = step3 - 5",  # 15
                "print(step4)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        # 10*2=20, +3=23, *4=92, -5=87
        assert "87" in nb_runner.get_output(6)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestParallelBranches:
    """Test parallel independent branches merging downstream."""

    def test_three_branches_one_change(self, nb_runner):
        """Three parallel branches, only one changes."""
        nb_runner.create_notebook(
            [
                "x = 1\ny = 2\nz = 3",
                "ax = x * 10",
                "by = y * 10",
                "cz = z * 10",
                "total = ax + by + cz",
                "print(total)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+20+30=60
        assert "60" in nb_runner.get_output(6)

        # Change only y
        nb_runner.set_cell_source(1, "x = 1\ny = 20\nz = 3")
        nb_runner.run_all()
        # 10+200+30=240
        assert "240" in nb_runner.get_output(6)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestRapidModification:
    """Test rapid modifications and re-runs."""

    def test_multiple_rapid_changes(self, nb_runner):
        """Make several changes and re-run each time."""
        nb_runner.create_notebook(
            [
                "n = 1",
                "result = n * 100",
                "print(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

        for val in [2, 5, 10, 50]:
            nb_runner.set_cell_source(1, f"n = {val}")
            nb_runner.run_all()
            assert str(val * 100) in nb_runner.get_output(3)
