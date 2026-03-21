"""Batch 157 – Decorator pattern interaction tests.

Tests where decorators are defined in one cell, applied to functions
in another, and edits to either the decorator or the function
verify correct cache behavior.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDecoratorEdits:
    """Decorator definition and application edits."""

    def test_edit_decorator_logic(self, nb_runner):
        """Edit decorator logic, verify function behavior changes."""
        nb_runner.create_notebook([
            "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
            "@double_result\ndef add(a, b):\n    return a + b",
            "result = add(3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change decorator to triple
        nb_runner.set_cell_source(
            1,
            "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 3\n    return wrapper",
        )
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(3)

    def test_edit_decorated_function(self, nb_runner):
        """Edit the decorated function itself."""
        nb_runner.create_notebook([
            "def negate(fn):\n    def wrapper(*args):\n        return -fn(*args)\n    return wrapper",
            "@negate\ndef compute(x):\n    return x * 2",
            "result = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -10" in nb_runner.get_output(3)

        # Edit the base function
        nb_runner.set_cell_source(
            2, "@negate\ndef compute(x):\n    return x ** 2"
        )
        nb_runner.run_all()
        assert "result = -25" in nb_runner.get_output(3)

    def test_remove_decorator(self, nb_runner):
        """Remove decorator from function."""
        nb_runner.create_notebook([
            "def add_ten(fn):\n    def wrapper(*args):\n        return fn(*args) + 10\n    return wrapper",
            "@add_ten\ndef square(x):\n    return x * x",
            "result = square(3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 19" in nb_runner.get_output(3)

        # Remove decorator
        nb_runner.set_cell_source(2, "def square(x):\n    return x * x")
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)


class TestDecoratorWithArgs:
    """Decorators with arguments."""

    def test_edit_decorator_argument(self, nb_runner):
        """Edit the argument to a decorator factory."""
        nb_runner.create_notebook([
            "def multiply_by(n):\n    def decorator(fn):\n        def wrapper(*args):\n            return fn(*args) * n\n        return wrapper\n    return decorator",
            "@multiply_by(2)\ndef add(a, b):\n    return a + b",
            "result = add(3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change multiplier
        nb_runner.set_cell_source(
            2, "@multiply_by(5)\ndef add(a, b):\n    return a + b"
        )
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(3)
