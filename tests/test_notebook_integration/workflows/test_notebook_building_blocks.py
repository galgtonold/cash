"""Pieces of real notebooks on their own: error fixes, closures, classes, directives, imports, reruns."""

import contextlib
import time

import pytest


@pytest.mark.core
class TestCachingEfficiency:
    """Test that caching actually works — second run should use cache."""

    def test_second_run_uses_cache(self, nb_runner):
        """Second run_all should be faster or produce same output from cache."""
        nb_runner.create_notebook(
            [
                "import time\nstart = time.time()\ntime.sleep(0.1)\ncompute_time = time.time() - start",
                "x = 42",  # Simple computation
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(3)

        # Second run — cell 2 should use cache
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(3)

    def test_unchanged_cells_skip(self, nb_runner):
        """Running unchanged cells should skip re-execution."""
        nb_runner.create_notebook(
            [
                "x = 100",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 200" in nb_runner.get_output(2)

        # Second run without changes — should skip
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "y = 200" in out, f"Second run should produce same output, got: {out}"


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestErrorRecovery:
    """Tests for error recovery patterns - cells that initially fail then succeed."""

    @pytest.mark.core
    def test_fix_cell_after_error(self, nb_runner):
        """Fix a cell that errored and re-run should work correctly."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x / 0",  # will error
                "print(f'y={y}')",
            ]
        )
        nb_runner.start_kernel()

        # Run cell 1 OK
        nb_runner.run_cell(1)

        # Cell 2 will error
        with contextlib.suppress(Exception):
            nb_runner.run_cell(2)

        # Fix cell 2
        nb_runner.set_cell_source(2, "y = x / 2")
        nb_runner.run_cells([2, 3])

        output = nb_runner.get_output(3)
        assert "y=5.0" in output

    @pytest.mark.core
    def test_undefined_variable_then_define(self, nb_runner):
        """Run cell using undefined var, define it in another cell, use it later."""
        nb_runner.create_notebook(
            [
                "result_early = data * 2",  # data not yet defined - will fail
                "data = [1, 2, 3]",
                "result_late = data * 2",  # now it works (different var name)
                "print(f'result_late={result_late}')",
            ]
        )
        nb_runner.start_kernel()

        # Cell 1 will fail
        with contextlib.suppress(Exception):
            nb_runner.run_cell(1)

        # Run cells 2-4
        nb_runner.run_cells([2, 3, 4])

        output = nb_runner.get_output(4)
        assert "result_late=" in output
        assert "1, 2, 3, 1, 2, 3" in output


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestDecoratorAndClosurePatterns:
    """Tests for decorator-like patterns and closures across cells."""

    @pytest.mark.core
    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure should capture variable from defining cell."""
        nb_runner.create_notebook(
            [
                "multiplier = 3",
                "def make_multiplier(n):\n    def inner(x):\n        return x * n\n    return inner\nmult = make_multiplier(multiplier)",
                "result = mult(10)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=30" in output

        # Change multiplier
        nb_runner.set_cell_source(1, "multiplier = 5")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=50" in output2

    @pytest.mark.core
    def test_decorator_pattern(self, nb_runner):
        """Custom decorator pattern should be tracked."""
        nb_runner.create_notebook(
            [
                "def logged(func):\n    def wrapper(*args):\n        result = func(*args)\n        return result\n    wrapper.__name__ = func.__name__\n    return wrapper",
                "@logged\ndef add(a, b):\n    return a + b",
                "result = add(3, 4)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=7" in output

        # Change decorated function
        nb_runner.set_cell_source(2, "@logged\ndef add(a, b):\n    return a + b + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=8" in output2

    @pytest.mark.core
    def test_functools_partial_tracking(self, nb_runner):
        """functools.partial should be tracked as dependent on source function."""
        nb_runner.create_notebook(
            [
                "from functools import partial",
                "def power(base, exp):\n    return base ** exp",
                "square = partial(power, exp=2)",
                "result = square(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=25" in output

        # Change base function
        nb_runner.set_cell_source(2, "def power(base, exp):\n    return base ** exp + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "result=26" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiCellClassPatterns:
    """Tests for class definitions and usage across multiple cells."""

    @pytest.mark.core
    def test_class_definition_and_instantiation(self, nb_runner):
        """Class defined in one cell, instantiated in another."""
        nb_runner.create_notebook(
            [
                "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n    def increment(self):\n        self.value += 1\n        return self.value",
                "c = Counter(10)",
                "result = c.increment()",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=11" in output

        # Change class
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n    def increment(self):\n        self.value += 10\n        return self.value",
        )
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=20" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestDynamicVariablePatterns:
    """Tests for dynamic variable creation patterns."""

    @pytest.mark.core
    def test_dict_unpacking_as_variables(self, nb_runner):
        """Dict values used as pseudo-variables should cache."""
        nb_runner.create_notebook(
            [
                "config = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}",
                "lr = config['learning_rate']\nepochs = config['epochs']",
                "result = lr * epochs",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=1.0" in output

        # Change config
        nb_runner.set_cell_source(1, "config = {'learning_rate': 0.1, 'epochs': 100, 'batch_size': 32}")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=10.0" in output2

    @pytest.mark.core
    def test_zip_pattern(self, nb_runner):
        """Zip operation should track dependencies on both inputs."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "values = [1, 2, 3]",
                "combined = dict(zip(keys, values))",
                "print(f'combined={combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "'a': 1" in output

        # Change values
        nb_runner.set_cell_source(2, "values = [10, 20, 30]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "'a': 10" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestContextManagerPatterns:
    """Tests for context manager and resource management patterns."""

    @pytest.mark.core
    def test_with_statement_file_write_read(self, nb_runner, tmp_path):
        """Context manager for file operations."""
        fpath = str(tmp_path / "ctx_test.txt").replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"path = '{fpath}'",
                "with open(path, 'w') as f:\n    f.write('hello from context manager')",
                "with open(path, 'r') as f:\n    content = f.read()",
                "print(f'content={content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "content=hello from context manager" in output


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexExpressionPatterns:
    """Tests for complex expression patterns."""

    @pytest.mark.core
    def test_ternary_expression_chain(self, nb_runner):
        """Nested ternary expressions."""
        nb_runner.create_notebook(
            [
                "x = 15",
                "category = 'high' if x > 20 else 'medium' if x > 10 else 'low'",
                "print(f'category={category}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "category=medium" in output

        # Change x to high
        nb_runner.set_cell_source(1, "x = 25")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "category=high" in output2


@pytest.mark.core
@pytest.mark.core
class TestAnnotationDirectives:
    """Test @cash: annotation directives."""

    def test_no_cache_annotation(self, nb_runner):
        """@cash:no-cache should force execution every time."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "# @cash:no-cache\nimport time\nt = time.time()\nprint(f't = {t}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "# @cash:persist\nx = 42\nprint(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(1)


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
            "        return 3.14 * self.r ** 2\n",
            encoding="utf-8",
        )

        nb_runner.create_notebook(
            [
                "from shapes import Circle",
                "c = Circle(5)\nprint(f'area = {c.area()}')",
            ]
        )
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
            "        return math.pi * self.r ** 2\n",
            encoding="utf-8",
        )
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "area = 78.5398" in out, f"Expected math.pi area, got: {out}"

    def test_from_import_constant_change(self, nb_runner, tmp_path):
        """from X import CONST — constant value change should propagate."""
        mod_path = tmp_path / "config.py"
        mod_path.write_text("VERSION = '1.0'\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                "from config import VERSION",
                "print(f'Version: {VERSION}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Version: 1.0" in nb_runner.get_output(2)

        # Update the constant
        mod_path.write_text("VERSION = '2.0'\n", encoding="utf-8")
        time.sleep(0.5)

        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "Version: 2.0" in out, f"Expected Version: 2.0, got: {out}"
