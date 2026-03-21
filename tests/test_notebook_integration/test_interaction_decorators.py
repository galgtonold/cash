"""Batch 122 – Decorator patterns + cell edit interaction tests.

Tests that exercise decorated functions and how cash handles
changes to decorators, decorated functions, and decorator arguments.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestDecoratorEdits:
    """Basic decorator + edit scenarios."""

    def test_edit_decorated_function(self, nb_runner):
        """Edit a decorated function body."""
        nb_runner.create_notebook([
            "def logged(func):\n    def wrapper(*a, **kw):\n        return func(*a, **kw)\n    return wrapper",
            "@logged\ndef compute(x):\n    return x * 2",
            "result = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "@logged\ndef compute(x):\n    return x ** 2"
        )
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(3)

    def test_edit_decorator(self, nb_runner):
        """Edit the decorator itself."""
        nb_runner.create_notebook([
            "def multiplier(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) * factor\n        return wrapper\n    return decorator",
            "@multiplier(2)\ndef value(x):\n    return x",
            "result = value(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        # Change the multiplier implementation
        nb_runner.set_cell_source(
            1,
            "def multiplier(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) + factor\n        return wrapper\n    return decorator",
        )
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(3)

    def test_change_decorator_argument(self, nb_runner):
        """Change the decorator argument (factor)."""
        nb_runner.create_notebook([
            "def scale(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) * factor\n        return wrapper\n    return decorator",
            "@scale(3)\ndef base(x):\n    return x",
            "result = base(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "@scale(10)\ndef base(x):\n    return x"
        )
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(3)

    def test_remove_decorator(self, nb_runner):
        """Remove a decorator from a function."""
        nb_runner.create_notebook([
            "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
            "@double\ndef calc(x):\n    return x + 1",
            "result = calc(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (5 + 1) * 2 = 12
        assert "result = 12" in nb_runner.get_output(3)

        # Remove decorator
        nb_runner.set_cell_source(
            2, "def calc(x):\n    return x + 1"
        )
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)


class TestStackedDecorators:
    """Stacked decorators + edits."""

    def test_edit_with_stacked_decorators(self, nb_runner):
        """Edit function with two decorators stacked."""
        nb_runner.create_notebook([
            "def add_one(func):\n    def wrapper(*a):\n        return func(*a) + 1\n    return wrapper",
            "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
            "@double\n@add_one\ndef base(x):\n    return x",
            "result = base(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) = 5, add_one → 6, double → 12
        assert "result = 12" in nb_runner.get_output(4)

        # Edit base function
        nb_runner.set_cell_source(
            3, "@double\n@add_one\ndef base(x):\n    return x * 10"
        )
        nb_runner.run_all()
        # base(5) = 50, add_one → 51, double → 102
        assert "result = 102" in nb_runner.get_output(4)

    def test_swap_decorator_order(self, nb_runner):
        """Swap the order of stacked decorators."""
        nb_runner.create_notebook([
            "def add_one(func):\n    def wrapper(*a):\n        return func(*a) + 1\n    return wrapper",
            "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
            "@double\n@add_one\ndef base(x):\n    return x",
            "result = base(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) = 5, add_one → 6, double → 12
        assert "result = 12" in nb_runner.get_output(4)

        # Swap order: now add_one wraps double
        nb_runner.set_cell_source(
            3, "@add_one\n@double\ndef base(x):\n    return x"
        )
        nb_runner.run_all()
        # base(5) = 5, double → 10, add_one → 11
        assert "result = 11" in nb_runner.get_output(4)
