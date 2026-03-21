"""Batch 219 – Partial application interaction tests.

Tests editing cells with functools.partial, closures,
and currying patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestPartialApplyEdits:
    """Editing partial application patterns."""

    def test_edit_partial_arg(self, nb_runner):
        """Edit a partial function's fixed argument."""
        nb_runner.create_notebook([
            "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)",
            "result = square(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Change to cube
        nb_runner.set_cell_source(1, "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=3)")
        nb_runner.run_all()
        assert "result = 125" in nb_runner.get_output(2)

    def test_edit_closure_value(self, nb_runner):
        """Edit a closure factory's captured value."""
        nb_runner.create_notebook([
            "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder\nadd5 = make_adder(5)",
            "result = add5(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change to add10
        nb_runner.set_cell_source(1, "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder\nadd5 = make_adder(10)")
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

    def test_edit_curry_chain(self, nb_runner):
        """Edit curried function application."""
        nb_runner.create_notebook([
            "def mul(a):\n    def inner(b):\n        return a * b\n    return inner\ndouble = mul(2)",
            "result = double(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        # Change multiplier
        nb_runner.set_cell_source(1, "def mul(a):\n    def inner(b):\n        return a * b\n    return inner\ndouble = mul(5)")
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(2)

    def test_edit_partial_with_kwargs(self, nb_runner):
        """Edit partial with keyword arguments."""
        nb_runner.create_notebook([
            "from functools import partial\ndef greet(name, greeting, punct):\n    return f'{greeting} {name}{punct}'\nhello = partial(greet, greeting='Hello', punct='!')",
            "msg = hello('Alice')\nprint(msg)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello Alice!" in nb_runner.get_output(2)

        # Change greeting
        nb_runner.set_cell_source(1, "from functools import partial\ndef greet(name, greeting, punct):\n    return f'{greeting} {name}{punct}'\nhello = partial(greet, greeting='Hi', punct='.')")
        nb_runner.run_all()
        assert "Hi Alice." in nb_runner.get_output(2)
