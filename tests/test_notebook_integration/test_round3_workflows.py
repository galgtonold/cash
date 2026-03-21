"""
Round 3 - Batch 9: Real-world workflows, error recovery, and complex patterns.

Tests focusing on:
1. Real-world data science workflows (sklearn, scipy, datetime)
2. Error recovery - cells that error on first run then succeed
3. Complex decorator patterns and closures
4. Multi-cell class definitions and method usage
5. Dynamic variable creation patterns (globals(), exec())
6. Notebook-realistic iteration patterns
7. Complex import patterns (reimport, conditional import)
8. Context managers and resource management
"""

import pytest
import contextlib



pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestRealWorldDataScience:
    """Tests mimicking real-world data science notebook workflows."""

    @pytest.mark.libraries
    def test_sklearn_like_pipeline(self, nb_runner):
        """Simulate a typical sklearn train/predict pipeline."""
        nb_runner.create_notebook([
            "import numpy as np",
            "np.random.seed(42)\nX = np.random.randn(100, 3)\ny = (X[:, 0] + X[:, 1] * 2 > 0).astype(int)",
            "# Simple manual logistic regression-like scoring\nweights = np.array([1.0, 2.0, 0.0])\nscores = X @ weights\npredictions = (scores > 0).astype(int)",
            "accuracy = np.mean(predictions == y)\nprint(f'accuracy={accuracy:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "accuracy=" in output
        acc_val = float(output.split("=")[1])
        assert acc_val > 0.5  # should be reasonably accurate

        # Re-run: should be cached
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert output2.strip() == output.strip()

    @pytest.mark.libraries
    def test_datetime_computation_workflow(self, nb_runner):
        """Date/time computations typical in analytics notebooks."""
        nb_runner.create_notebook([
            "from datetime import datetime, timedelta",
            "start = datetime(2024, 1, 1)\nend = datetime(2024, 12, 31)",
            "duration = (end - start).days",
            "months = duration // 30",
            "print(f'days={duration} months={months}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "days=365" in output
        assert "months=12" in output

    @pytest.mark.libraries
    def test_json_api_response_processing(self, nb_runner):
        """Processing JSON data like API responses."""
        nb_runner.create_notebook([
            "import json",
            "raw = '{\"users\": [{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]}'",
            "data = json.loads(raw)",
            "names = [u['name'] for u in data['users']]\navg_age = sum(u['age'] for u in data['users']) / len(data['users'])",
            "print(f'names={names} avg_age={avg_age}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "Alice" in output
        assert "Bob" in output
        assert "avg_age=27.5" in output

    @pytest.mark.libraries
    def test_regex_text_processing(self, nb_runner):
        """Regex-based text processing common in NLP notebooks."""
        nb_runner.create_notebook([
            "import re",
            "text = 'Hello World! This is test 123. Contact: user@email.com'",
            "emails = re.findall(r'[\\w.]+@[\\w.]+', text)\nnumbers = re.findall(r'\\d+', text)\nwords = re.findall(r'\\b[A-Z][a-z]+\\b', text)",
            "print(f'emails={emails} numbers={numbers} words={words}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "user@email.com" in output
        assert "123" in output

        # Change text
        nb_runner.set_cell_source(2, "text = 'New text with number 456 and admin@site.org'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "admin@site.org" in output2
        assert "456" in output2


class TestErrorRecovery:
    """Tests for error recovery patterns - cells that initially fail then succeed."""

    @pytest.mark.core
    def test_fix_cell_after_error(self, nb_runner):
        """Fix a cell that errored and re-run should work correctly."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x / 0",  # will error
            "print(f'y={y}')",
        ])
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
        nb_runner.create_notebook([
            "result_early = data * 2",  # data not yet defined - will fail
            "data = [1, 2, 3]",
            "result_late = data * 2",  # now it works (different var name)
            "print(f'result_late={result_late}')",
        ])
        nb_runner.start_kernel()

        # Cell 1 will fail
        with contextlib.suppress(Exception):
            nb_runner.run_cell(1)

        # Run cells 2-4 
        nb_runner.run_cells([2, 3, 4])

        output = nb_runner.get_output(4)
        assert "result_late=" in output
        assert "1, 2, 3, 1, 2, 3" in output

    @pytest.mark.core
    def test_import_error_then_workaround(self, nb_runner):
        """Handle ImportError by providing a fallback."""
        nb_runner.create_notebook([
            "try:\n    import nonexistent_package\n    source = 'package'\nexcept ImportError:\n    source = 'fallback'",
            "print(f'source={source}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "source=fallback" in output


class TestDecoratorAndClosurePatterns:
    """Tests for decorator-like patterns and closures across cells."""

    @pytest.mark.core
    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure should capture variable from defining cell."""
        nb_runner.create_notebook([
            "multiplier = 3",
            "def make_multiplier(n):\n    def inner(x):\n        return x * n\n    return inner\nmult = make_multiplier(multiplier)",
            "result = mult(10)",
            "print(f'result={result}')",
        ])
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
        nb_runner.create_notebook([
            "def logged(func):\n    def wrapper(*args):\n        result = func(*args)\n        return result\n    wrapper.__name__ = func.__name__\n    return wrapper",
            "@logged\ndef add(a, b):\n    return a + b",
            "result = add(3, 4)",
            "print(f'result={result}')",
        ])
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
        nb_runner.create_notebook([
            "from functools import partial",
            "def power(base, exp):\n    return base ** exp",
            "square = partial(power, exp=2)",
            "result = square(5)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=25" in output

        # Change base function
        nb_runner.set_cell_source(2, "def power(base, exp):\n    return base ** exp + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "result=26" in output2


class TestMultiCellClassPatterns:
    """Tests for class definitions and usage across multiple cells."""

    @pytest.mark.core
    def test_class_definition_and_instantiation(self, nb_runner):
        """Class defined in one cell, instantiated in another."""
        nb_runner.create_notebook([
            "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n    def increment(self):\n        self.value += 1\n        return self.value",
            "c = Counter(10)",
            "result = c.increment()",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=11" in output

        # Change class
        nb_runner.set_cell_source(1, "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n    def increment(self):\n        self.value += 10\n        return self.value")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=20" in output2

    @pytest.mark.core
    def test_class_with_classmethod_and_staticmethod(self, nb_runner):
        """Class with classmethods and staticmethods."""
        nb_runner.create_notebook([
            "class MathHelper:\n    factor = 2\n    \n    @classmethod\n    def scale(cls, x):\n        return x * cls.factor\n    \n    @staticmethod\n    def add(a, b):\n        return a + b",
            "scaled = MathHelper.scale(5)\nadded = MathHelper.add(3, 4)",
            "print(f'scaled={scaled} added={added}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "scaled=10" in output
        assert "added=7" in output


class TestDynamicVariablePatterns:
    """Tests for dynamic variable creation patterns."""

    @pytest.mark.core
    def test_dict_unpacking_as_variables(self, nb_runner):
        """Dict values used as pseudo-variables should cache."""
        nb_runner.create_notebook([
            "config = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}",
            "lr = config['learning_rate']\nepochs = config['epochs']",
            "result = lr * epochs",
            "print(f'result={result}')",
        ])
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
    def test_enumerate_pattern(self, nb_runner):
        """Enumerate in loops should cache results."""
        nb_runner.create_notebook([
            "items = ['a', 'b', 'c']",
            "indexed = {i: v for i, v in enumerate(items)}",
            "print(f'indexed={indexed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "0" in output
        assert "'a'" in output

    @pytest.mark.core
    def test_zip_pattern(self, nb_runner):
        """Zip operation should track dependencies on both inputs."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "values = [1, 2, 3]",
            "combined = dict(zip(keys, values))",
            "print(f'combined={combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "'a': 1" in output

        # Change values
        nb_runner.set_cell_source(2, "values = [10, 20, 30]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "'a': 10" in output2


class TestNotebookIterativeWorkflow:
    """Tests for realistic iterative development patterns."""

    @pytest.mark.core
    def test_iterative_refinement(self, nb_runner):
        """Simulate iterative development: define, test, refine."""
        nb_runner.create_notebook([
            "def process(data):\n    return [x * 2 for x in data]",
            "test_data = [1, 2, 3, 4, 5]",
            "result = process(test_data)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "2, 4, 6, 8, 10" in output

        # Refine function
        nb_runner.set_cell_source(1, "def process(data):\n    return [x ** 2 for x in data]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "1, 4, 9, 16, 25" in output2

    @pytest.mark.core
    def test_parameter_sweep_pattern(self, nb_runner):
        """Simulate changing parameters and re-running analysis."""
        nb_runner.create_notebook([
            "threshold = 0.5",
            "data = [0.1, 0.3, 0.5, 0.7, 0.9]",
            "above = [x for x in data if x > threshold]\nbelow = [x for x in data if x <= threshold]",
            "print(f'above={len(above)} below={len(below)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "above=2" in output
        assert "below=3" in output

        # Change threshold
        nb_runner.set_cell_source(1, "threshold = 0.3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "above=3" in output2
        assert "below=2" in output2


class TestConditionalImportPatterns:
    """Tests for conditional and dynamic import patterns."""

    @pytest.mark.modules
    def test_try_import_with_fallback(self, nb_runner):
        """Try importing an optional package with fallback."""
        nb_runner.create_notebook([
            "try:\n    import scipy\n    HAS_SCIPY = True\nexcept ImportError:\n    HAS_SCIPY = False",
            "if HAS_SCIPY:\n    result = 'scipy available'\nelse:\n    result = 'scipy not available'",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=" in output
        assert "scipy" in output

    @pytest.mark.modules
    def test_import_from_pathlib(self, nb_runner):
        """Import Path from pathlib and use it."""
        nb_runner.create_notebook([
            "from pathlib import Path",
            "p = Path('/tmp/test')",
            "parts = p.parts",
            "print(f'parts={parts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "parts=" in output

    @pytest.mark.modules
    def test_collections_import_and_use(self, nb_runner):
        """Use collections module types."""
        nb_runner.create_notebook([
            "from collections import Counter, defaultdict",
            "data = ['a', 'b', 'a', 'c', 'a', 'b']",
            "counts = Counter(data)\ndd = defaultdict(list)\nfor item in data:\n    dd[item].append(1)",
            "print(f'counts={dict(counts)} dd_keys={sorted(dd.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "'a': 3" in output
        assert "'b': 2" in output

    @pytest.mark.modules
    def test_import_star_equivalent_manual(self, nb_runner):
        """Multiple specific imports from same module."""
        nb_runner.create_notebook([
            "from os.path import join, dirname, basename, exists",
            "p = join('/tmp', 'test', 'file.txt')",
            "d = dirname(p)\nb = basename(p)",
            "print(f'd={d} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "b=file.txt" in output


class TestContextManagerPatterns:
    """Tests for context manager and resource management patterns."""

    @pytest.mark.core
    def test_with_statement_file_write_read(self, nb_runner, tmp_path):
        """Context manager for file operations."""
        fpath = str(tmp_path / "ctx_test.txt").replace('\\', '/')

        nb_runner.create_notebook([
            f"path = '{fpath}'",
            "with open(path, 'w') as f:\n    f.write('hello from context manager')",
            "with open(path, 'r') as f:\n    content = f.read()",
            "print(f'content={content}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "content=hello from context manager" in output

    @pytest.mark.core
    def test_custom_context_manager(self, nb_runner):
        """Custom context manager class."""
        nb_runner.create_notebook([
            "class Timer:\n    def __enter__(self):\n        self.started = True\n        return self\n    def __exit__(self, *args):\n        self.started = False",
            "with Timer() as t:\n    result = 42\n    was_started = t.started",
            "print(f'result={result} was_started={was_started}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=42" in output
        assert "was_started=True" in output


class TestComplexExpressionPatterns:
    """Tests for complex expression patterns."""

    @pytest.mark.core
    def test_chained_comparisons(self, nb_runner):
        """Python chained comparisons."""
        nb_runner.create_notebook([
            "x = 5",
            "result = 1 < x < 10\nresult2 = 10 < x < 20",
            "print(f'result={result} result2={result2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=True" in output
        assert "result2=False" in output

    @pytest.mark.core
    def test_ternary_expression_chain(self, nb_runner):
        """Nested ternary expressions."""
        nb_runner.create_notebook([
            "x = 15",
            "category = 'high' if x > 20 else 'medium' if x > 10 else 'low'",
            "print(f'category={category}')",
        ])
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
    def test_nested_list_comprehension(self, nb_runner):
        """Nested list comprehension (matrix flatten)."""
        nb_runner.create_notebook([
            "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            "flat = [x for row in matrix for x in row]",
            "total = sum(flat)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=45" in output

    @pytest.mark.core
    def test_dict_merge_operator(self, nb_runner):
        """Dict merge with | operator (Python 3.9+)."""
        nb_runner.create_notebook([
            "d1 = {'a': 1, 'b': 2}",
            "d2 = {'b': 3, 'c': 4}",
            "merged = d1 | d2",
            "print(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "'a': 1" in output
        assert "'b': 3" in output  # d2 wins
        assert "'c': 4" in output

    @pytest.mark.core
    def test_multiple_assignment_targets(self, nb_runner):
        """Multiple assignment targets in one statement."""
        nb_runner.create_notebook([
            "x = y = z = 42",
            "total = x + y + z",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "total=126" in output
