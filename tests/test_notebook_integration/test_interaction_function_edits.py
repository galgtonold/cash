"""Batch 137 – Nested function and closure interaction tests.

Tests where users define functions in one cell and call them in another,
then modify the function definition and verify downstream cells update.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestFunctionDefinitionEdits:
    """Edit function definitions and verify callers update."""

    def test_edit_function_body(self, nb_runner):
        """Change function body, verify callers get new result."""
        nb_runner.create_notebook([
            "def compute(x):\n    return x * 2",
            "result = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit function body
        nb_runner.set_cell_source(1, "def compute(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_add_parameter_to_function(self, nb_runner):
        """Add parameter to function definition."""
        nb_runner.create_notebook([
            "def greet(name):\n    return f'Hello {name}'",
            "msg = greet('World')\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)

        # Add second parameter
        nb_runner.set_cell_source(
            1,
            "def greet(name, greeting='Hi'):\n    return f'{greeting} {name}'",
        )
        nb_runner.set_cell_source(2, "msg = greet('World', 'Hey')\nprint(msg)")
        nb_runner.run_all()
        assert "Hey World" in nb_runner.get_output(2)

    def test_chain_of_functions_edit_middle(self, nb_runner):
        """Chain: f -> g -> h, edit g."""
        nb_runner.create_notebook([
            "def f(x):\n    return x + 1",
            "def g(x):\n    return f(x) * 2",
            "def h(x):\n    return g(x) + 10",
            "result = h(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # h(5) = g(5) + 10 = f(5)*2 + 10 = 6*2+10 = 22
        assert "result = 22" in nb_runner.get_output(4)

        # Edit g
        nb_runner.set_cell_source(2, "def g(x):\n    return f(x) * 3")
        nb_runner.run_all()
        # h(5) = g(5) + 10 = f(5)*3 + 10 = 6*3+10 = 28
        assert "result = 28" in nb_runner.get_output(4)


class TestClosureEdits:
    """Edit closures and verify behavior updates."""

    def test_edit_closure_variable(self, nb_runner):
        """Closure captures a variable, edit the variable."""
        nb_runner.create_notebook([
            "factor = 3",
            "def scale(x):\n    return x * factor",
            "result = scale(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit the captured variable
        nb_runner.set_cell_source(1, "factor = 7")
        nb_runner.run_all()
        assert "result = 70" in nb_runner.get_output(3)

    def test_function_with_default_arg_edit(self, nb_runner):
        """Function with default arg, edit the default."""
        nb_runner.create_notebook([
            "default_power = 2",
            "def raise_to(x, p=None):\n    if p is None:\n        p = default_power\n    return x ** p",
            "result = raise_to(3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "default_power = 3")
        nb_runner.run_all()
        assert "result = 27" in nb_runner.get_output(3)


class TestLambdaEdits:
    """Edit lambda expressions."""

    def test_edit_lambda(self, nb_runner):
        """Change lambda, verify new behavior."""
        nb_runner.create_notebook([
            "transform = lambda x: x + 10",
            "vals = [transform(i) for i in range(5)]",
            "total = sum(vals)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # [10, 11, 12, 13, 14] -> 60
        assert "total = 60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "transform = lambda x: x * 10")
        nb_runner.run_all()
        # [0, 10, 20, 30, 40] -> 100
        assert "total = 100" in nb_runner.get_output(3)

    def test_lambda_to_function(self, nb_runner):
        """Replace lambda with full function def."""
        nb_runner.create_notebook([
            "double = lambda x: x * 2",
            "result = double(15)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def double(x):\n    return x * 2 + 1  # with offset",
        )
        nb_runner.run_all()
        assert "result = 31" in nb_runner.get_output(2)
