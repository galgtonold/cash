"""Batch 204 – Decorator stacking / chaining interaction tests.

Tests editing stacked decorators and their ordering effects.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDecoratorStackEdits:
    """Editing stacked decorators."""

    def test_edit_decorator_order(self, nb_runner):
        """Edit the order of stacked decorators."""
        nb_runner.create_notebook([
            "def add_prefix(fn):\n    def wrapper(*args, **kwargs):\n        return 'PREFIX_' + fn(*args, **kwargs)\n    return wrapper",
            "def add_suffix(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs) + '_SUFFIX'\n    return wrapper",
            "@add_prefix\n@add_suffix\ndef greet():\n    return 'hello'",
            "result = greet()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # add_suffix first: 'hello_SUFFIX', then add_prefix: 'PREFIX_hello_SUFFIX'
        assert "result = PREFIX_hello_SUFFIX" in nb_runner.get_output(4)

        # Swap decorator order
        nb_runner.set_cell_source(
            3, "@add_suffix\n@add_prefix\ndef greet():\n    return 'hello'"
        )
        nb_runner.run_all()
        # add_prefix first: 'PREFIX_hello', then add_suffix: 'PREFIX_hello_SUFFIX'
        assert "result = PREFIX_hello_SUFFIX" in nb_runner.get_output(4)

    def test_add_decorator(self, nb_runner):
        """Add a new decorator to an existing function."""
        nb_runner.create_notebook([
            "def uppercase(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs).upper()\n    return wrapper",
            "def repeat(fn):\n    def wrapper(*args, **kwargs):\n        r = fn(*args, **kwargs)\n        return r + r\n    return wrapper",
            "@uppercase\ndef say():\n    return 'hi'",
            "result = say()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HI" in nb_runner.get_output(4)

        # Add repeat decorator
        nb_runner.set_cell_source(
            3, "@uppercase\n@repeat\ndef say():\n    return 'hi'"
        )
        nb_runner.run_all()
        # repeat first: 'hihi', then uppercase: 'HIHI'
        assert "result = HIHI" in nb_runner.get_output(4)

    def test_edit_decorator_implementation(self, nb_runner):
        """Edit the implementation of a stacked decorator."""
        nb_runner.create_notebook([
            "def wrap_brackets(fn):\n    def w(*a, **k):\n        return '[' + fn(*a, **k) + ']'\n    return w",
            "def wrap_parens(fn):\n    def w(*a, **k):\n        return '(' + fn(*a, **k) + ')'\n    return w",
            "@wrap_brackets\n@wrap_parens\ndef msg():\n    return 'hi'",
            "result = msg()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [(hi)]" in nb_runner.get_output(4)

        # Change brackets to braces
        nb_runner.set_cell_source(
            1,
            "def wrap_brackets(fn):\n    def w(*a, **k):\n        return '{' + fn(*a, **k) + '}'\n    return w",
        )
        nb_runner.run_all()
        assert "result = {(hi)}" in nb_runner.get_output(4)
