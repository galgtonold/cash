"""Batch 168 – Closure and scope interaction tests.

Tests editing closures, nonlocal variables, nested function scopes,
and variable capture patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestClosureEdits:
    """Editing closure definitions and captured variables."""

    def test_edit_closure_body(self, nb_runner):
        """Edit the body of a closure."""
        nb_runner.create_notebook([
            "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
            "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change closure to multiply
        nb_runner.set_cell_source(
            1,
            "def make_adder(n):\n    def adder(x):\n        return x * n\n    return adder",
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)

    def test_edit_captured_value(self, nb_runner):
        """Edit the value passed to create a closure."""
        nb_runner.create_notebook([
            "def make_greeter(name):\n    def greet():\n        return f'Hello, {name}!'\n    return greet",
            "g = make_greeter('Alice')\nresult = g()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Hello, Alice!" in nb_runner.get_output(2)

        # Change captured value
        nb_runner.set_cell_source(
            2,
            "g = make_greeter('Bob')\nresult = g()\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = Hello, Bob!" in nb_runner.get_output(2)

    def test_nested_closure_edit(self, nb_runner):
        """Two levels of closure nesting, edit inner."""
        nb_runner.create_notebook([
            "def outer(a):\n    def middle(b):\n        def inner(c):\n            return a + b + c\n        return inner\n    return middle",
            "fn = outer(1)(2)\nresult = fn(3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        # Change inner to multiply
        nb_runner.set_cell_source(
            1,
            "def outer(a):\n    def middle(b):\n        def inner(c):\n            return a * b * c\n        return inner\n    return middle",
        )
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)  # 1*2*3=6, same value!

        # Use different values to see the difference
        nb_runner.set_cell_source(
            2, "fn = outer(2)(3)\nresult = fn(4)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)


class TestNonlocalEdits:
    """Nonlocal variable patterns."""

    def test_nonlocal_counter(self, nb_runner):
        """Edit a nonlocal counter closure."""
        nb_runner.create_notebook([
            "def make_counter(start=0):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
            "c = make_counter()\nvals = [c() for _ in range(3)]\nprint(f'vals = {vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 2, 3]" in nb_runner.get_output(2)

        # Change start value
        nb_runner.set_cell_source(
            2,
            "c = make_counter(10)\nvals = [c() for _ in range(3)]\nprint(f'vals = {vals}')",
        )
        nb_runner.run_all()
        assert "vals = [11, 12, 13]" in nb_runner.get_output(2)

    def test_nonlocal_accumulator_edit(self, nb_runner):
        """Edit the accumulation logic in a nonlocal pattern."""
        nb_runner.create_notebook([
            "def make_acc():\n    total = 0\n    def add(x):\n        nonlocal total\n        total += x\n        return total\n    return add",
            "acc = make_acc()\nresults = [acc(i) for i in [10, 20, 30]]\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [10, 30, 60]" in nb_runner.get_output(2)

        # Change to multiply accumulator
        nb_runner.set_cell_source(
            1,
            "def make_acc():\n    total = 1\n    def add(x):\n        nonlocal total\n        total *= x\n        return total\n    return add",
        )
        nb_runner.run_all()
        assert "results = [10, 200, 6000]" in nb_runner.get_output(2)


class TestScopeInteraction:
    """Variable scope interactions between cells."""

    def test_same_name_different_scope(self, nb_runner):
        """Same variable name in different scopes."""
        nb_runner.create_notebook([
            "x = 'global_x'  # global scope",
            "def show_x():\n    x = 'local_x'\n    return x",
            "local_val = show_x()\nprint(f'global={x} local={local_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "global=global_x" in nb_runner.get_output(3)
        assert "local=local_x" in nb_runner.get_output(3)

        # Edit global
        nb_runner.set_cell_source(1, "x = 'new_global'  # global scope changed")
        nb_runner.run_all()
        assert "global=new_global" in nb_runner.get_output(3)
        assert "local=local_x" in nb_runner.get_output(3)
