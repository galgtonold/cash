"""Batch 107 – Function & class redefinition interaction tests.

Tests that exercise function/class redefinition across cells,
combined with cell edits, reruns, and kernel restarts.
"""

import pytest

pytestmark = [pytest.mark.mutations, pytest.mark.stress, pytest.mark.timeout(30)]


class TestFunctionRedefinition:
    """Redefine a function in one cell, observe downstream effects."""

    def test_redefine_function_downstream_updates(self, nb_runner):
        """Change function body, downstream should recompute."""
        nb_runner.create_notebook([
            "def f(x):\n    return x * 2",
            "result = f(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def f(x):\n    return x * 3")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

    def test_redefine_function_twice(self, nb_runner):
        """Redefine a function two times sequentially."""
        nb_runner.create_notebook([
            "def compute(a, b):\n    return a + b",
            "val = compute(3, 4)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 7" in nb_runner.get_output(2)

        # First edit: multiply
        nb_runner.set_cell_source(1, "def compute(a, b):\n    return a * b")
        nb_runner.run_all()
        assert "val = 12" in nb_runner.get_output(2)

        # Second edit: subtract
        nb_runner.set_cell_source(1, "def compute(a, b):\n    return a - b")
        nb_runner.run_all()
        assert "val = -1" in nb_runner.get_output(2)

    def test_redefine_function_revert(self, nb_runner):
        """Change function, then revert to original. Cache should still work."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "r = double(7)\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_all()
        assert "r = 21" in nb_runner.get_output(2)

        # Revert
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 2")
        nb_runner.run_all()
        assert "r = 14" in nb_runner.get_output(2)

    def test_function_used_in_multiple_cells(self, nb_runner):
        """Function redefined, used in multiple downstream cells."""
        nb_runner.create_notebook([
            "def inc(x):\n    return x + 1",
            "a = inc(10)\nprint(f'a = {a}')",
            "b = inc(20)\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def inc(x):\n    return x + 10")
        nb_runner.run_all()
        assert "a = 20" in nb_runner.get_output(2)
        assert "b = 30" in nb_runner.get_output(3)


    def test_function_redefined_after_restart(self, nb_runner):
        """Restart kernel, redefine function, downstream should recompute."""
        nb_runner.create_notebook([
            "def square(x):\n    return x ** 2",
            "val = square(4)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 16" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "def square(x):\n    return x ** 3")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 64" in nb_runner.get_output(2)


class TestClassRedefinition:
    """Redefine a class, observe downstream effects."""

    def test_redefine_class_method(self, nb_runner):
        """Change a class method, downstream should reflect."""
        nb_runner.create_notebook([
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "c = Calculator()\nresult = c.add(3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Calculator:\n    def add(self, a, b):\n        return a + b + 100",
        )
        nb_runner.run_all()
        assert "result = 107" in nb_runner.get_output(2)

    def test_add_method_to_class(self, nb_runner):
        """Add a new method to an existing class."""
        nb_runner.create_notebook([
            "class MyClass:\n    def foo(self):\n        return 1",
            "obj = MyClass()\nprint(f'foo = {obj.foo()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "foo = 1" in nb_runner.get_output(2)

        # Add bar method and use it
        nb_runner.set_cell_source(
            1,
            "class MyClass:\n    def foo(self):\n        return 1\n    def bar(self):\n        return 2",
        )
        nb_runner.set_cell_source(
            2,
            "obj = MyClass()\nprint(f'foo = {obj.foo()}, bar = {obj.bar()}')",
        )
        nb_runner.run_all()
        assert "foo = 1, bar = 2" in nb_runner.get_output(2)


class TestFunctionCallingFunction:
    """Function that calls another function — redefinition chains."""

    def test_inner_function_redefined(self, nb_runner):
        """Redefine an inner function used by an outer function."""
        nb_runner.create_notebook([
            "def helper(x):\n    return x + 1",
            "def main(x):\n    return helper(x) * 2",
            "result = main(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine helper
        nb_runner.set_cell_source(1, "def helper(x):\n    return x + 10")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

    def test_outer_function_redefined(self, nb_runner):
        """Redefine the outer function while helper stays the same."""
        nb_runner.create_notebook([
            "def helper(x):\n    return x + 1",
            "def main(x):\n    return helper(x) * 2",
            "result = main(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine main
        nb_runner.set_cell_source(2, "def main(x):\n    return helper(x) * 3")
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(3)

    def test_both_functions_redefined(self, nb_runner):
        """Redefine both inner and outer functions simultaneously."""
        nb_runner.create_notebook([
            "def helper(x):\n    return x + 1",
            "def main(x):\n    return helper(x) * 2",
            "result = main(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine both
        nb_runner.set_cell_source(1, "def helper(x):\n    return x * 10")
        nb_runner.set_cell_source(2, "def main(x):\n    return helper(x) + 1")
        nb_runner.run_all()
        assert "result = 51" in nb_runner.get_output(3)

    def test_recursive_function_redefined(self, nb_runner):
        """Redefine a recursive function."""
        nb_runner.create_notebook([
            "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            "val = fib(6)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 8" in nb_runner.get_output(2)

        # Change to factorial
        nb_runner.set_cell_source(
            1,
            "def fib(n):\n    if n <= 1:\n        return 1\n    return n * fib(n-1)",
        )
        nb_runner.run_all()
        assert "val = 720" in nb_runner.get_output(2)

    def test_lambda_redefined(self, nb_runner):
        """Redefine a lambda used downstream."""
        nb_runner.create_notebook([
            "transform = lambda x: x * 2",
            "result = transform(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "transform = lambda x: x ** 2")
        nb_runner.run_all()
        assert "result = 49" in nb_runner.get_output(2)
