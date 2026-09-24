"""Editing closures, function factories and nested functions."""

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestClosureEdits:
    """Edit closures and verify behavior updates."""

    def test_edit_closure_variable(self, nb_runner):
        """Closure captures a variable, edit the variable."""
        nb_runner.create_notebook(
            [
                "factor = 3",
                "def scale(x):\n    return x * factor",
                "result = scale(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit the captured variable
        nb_runner.set_cell_source(1, "factor = 7")
        nb_runner.run_all()
        assert "result = 70" in nb_runner.get_output(3)

    def test_function_with_default_arg_edit(self, nb_runner):
        """Function with default arg, edit the default."""
        nb_runner.create_notebook(
            [
                "default_power = 2",
                "def raise_to(x, p=None):\n    if p is None:\n        p = default_power\n    return x ** p",
                "result = raise_to(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "default_power = 3")
        nb_runner.run_all()
        assert "result = 27" in nb_runner.get_output(3)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestClosureFactoryEdits:
    """Closure/factory patterns + cell edits."""

    def test_edit_closure_implementation(self, nb_runner):
        """Edit the closure implementation itself."""
        nb_runner.create_notebook(
            [
                "def make_op(n):\n    def op(x):\n        return x + n\n    return op",
                "op = make_op(3)",
                "result = op(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(3)

        # Change closure to multiply
        nb_runner.set_cell_source(1, "def make_op(n):\n    def op(x):\n        return x * n\n    return op")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)


@pytest.mark.timeout(90)
class TestNestedFunctionClosure:
    """nested function definitions and closures with edits."""

    def test_closure_counter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def make_counter(start=0):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
                "counter = make_counter(10)\nresults = [counter() for _ in range(3)]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[11, 12, 13]" in nb_runner.get_output(2)

    def test_closure_edit_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def multiplier(factor):\n    def multiply(x):\n        return x * factor\n    return multiply",
                "double = multiplier(2)\nresult = double(7)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=14" in nb_runner.get_output(2)
        # Edit factory
        nb_runner.set_cell_source(
            1, "def multiplier(factor):\n    def multiply(x):\n        return x * factor + 1\n    return multiply"
        )
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)

    def test_nested_helper(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def process(items):\n    def clean(s):\n        return s.strip().lower()\n    return [clean(i) for i in items]",
                "data = ['  Hello ', ' WORLD ', '  Python  ']\nresult = process(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['hello', 'world', 'python']" in nb_runner.get_output(2)


# Nested function definition edit patterns.
#
# Tests inner function edits propagating through outer function calls.
@pytest.mark.timeout(90)
class TestNestedFunctionEdits:
    """Nested function definition patterns."""

    def test_inner_function_edit(self, nb_runner):
        """Edit inner function, outer function result changes."""
        nb_runner.create_notebook(
            [
                "def outer(data):\n    def inner(x):\n        return x * 2\n    return [inner(x) for x in data]",
                "result = outer([1, 2, 3, 4])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def outer(data):\n    def inner(x):\n        return x ** 2\n    return [inner(x) for x in data]",
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_nested_with_accumulator(self, nb_runner):
        """Edit inner accumulator function."""
        nb_runner.create_notebook(
            [
                "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x\n        return total\n    return [accumulate(i) for i in items]",
                "running = process([10, 20, 30])\nprint(f'running = {running}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running = [10, 30, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x * 2\n        return total\n    return [accumulate(i) for i in items]",
        )
        nb_runner.run_all()
        assert "running = [20, 60, 120]" in nb_runner.get_output(2)

    def test_factory_with_inner_edit(self, nb_runner):
        """Edit factory function that produces inner functions."""
        nb_runner.create_notebook(
            [
                "def make_processor(op):\n    def process(x):\n        if op == 'double':\n            return x * 2\n        return x + 1\n    return process",
                "proc = make_processor('double')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "proc = make_processor('other')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')"
        )
        nb_runner.run_all()
        assert "results = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)
