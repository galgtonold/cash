"""Batch 228 – Closure factory and scope capture interaction tests.

Tests editing cells with closure factories, captured variables,
and function composition patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestClosureFactoryEdits:
    """Editing closure and function factory patterns."""

    def test_edit_closure_captured_argument(self, nb_runner):
        """Edit the argument passed to a closure factory."""
        nb_runner.create_notebook([
            "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
            "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change to add 20
        nb_runner.set_cell_source(2, "add5 = make_adder(20)\nresult = add5(10)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

    def test_edit_lambda_factory(self, nb_runner):
        """Edit a function that returns a lambda."""
        nb_runner.create_notebook([
            "def multiplier(factor):\n    return lambda x: x * factor",
            "double = multiplier(2)\nval = double(7)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 14" in nb_runner.get_output(2)

        # Edit factory to add offset
        nb_runner.set_cell_source(1, "def multiplier(factor):\n    return lambda x: x * factor + 1")
        nb_runner.run_all()
        assert "val = 15" in nb_runner.get_output(2)

    def test_edit_counter_start_value(self, nb_runner):
        """Edit counter closure's starting value."""
        nb_runner.create_notebook([
            "def make_counter(start=0):\n    count = [start]\n    def inc():\n        count[0] += 1\n        return count[0]\n    return inc",
            "counter = make_counter(0)\na = counter()\nb = counter()\nprint(f'a={a} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=2" in nb_runner.get_output(2)

        # Change start value
        nb_runner.set_cell_source(2, "counter = make_counter(10)\na = counter()\nb = counter()\nprint(f'a={a} b={b}')")
        nb_runner.run_all()
        assert "a=11 b=12" in nb_runner.get_output(2)

    def test_edit_function_composition(self, nb_runner):
        """Edit composed functions."""
        nb_runner.create_notebook([
            "def compose(f, g):\n    return lambda x: f(g(x))",
            "inc = lambda x: x + 1\ndbl = lambda x: x * 2\ninc_then_dbl = compose(dbl, inc)\nresult = inc_then_dbl(3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 8" in nb_runner.get_output(2)

        # Change composition order
        nb_runner.set_cell_source(2, "inc = lambda x: x + 1\ndbl = lambda x: x * 2\ndbl_then_inc = compose(inc, dbl)\nresult = dbl_then_inc(3)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)
