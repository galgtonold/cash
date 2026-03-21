"""Batch 242 – Global/nonlocal scope interaction patterns.

Tests editing variables that are used inside functions via closures or global scope.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGlobalNonlocalScope:
    """Scope-related interaction patterns."""

    def test_global_var_used_in_function(self, nb_runner):
        """Edit global variable, function using it re-evaluates."""
        nb_runner.create_notebook([
            "RATE = 0.1",
            "def calc_tax(amount):\n    return amount * RATE",
            "result = calc_tax(1000)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "RATE = 0.25")
        nb_runner.run_all()
        assert "result = 250.0" in nb_runner.get_output(3)

    def test_closure_captured_variable(self, nb_runner):
        """Edit closure factory parameter, downstream reflects."""
        nb_runner.create_notebook([
            "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
            "add5 = make_adder(5)",
            "result = add5(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "add5 = make_adder(100)")
        nb_runner.run_all()
        assert "result = 110" in nb_runner.get_output(3)

    def test_module_level_constant_edit(self, nb_runner):
        """Edit module-level constant used in multiple downstream cells."""
        nb_runner.create_notebook([
            "PI = 3.14",
            "area = PI * 5 * 5\nprint(f'area = {area}')",
            "circumference = 2 * PI * 5\nprint(f'circumference = {circumference}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(2)
        assert "circumference = 31.4" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "PI = 3.14159")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        assert "78.539" in out2
        assert "31.415" in out3

    def test_nested_function_with_nonlocal_pattern(self, nb_runner):
        """Counter pattern with nested function."""
        nb_runner.create_notebook([
            "def make_counter(start):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
            "counter = make_counter(0)\nval1 = counter()\nval2 = counter()\nprint(f'val1 = {val1}, val2 = {val2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val1 = 1, val2 = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "counter = make_counter(10)\nval1 = counter()\nval2 = counter()\nprint(f'val1 = {val1}, val2 = {val2}')")
        nb_runner.run_all()
        assert "val1 = 11, val2 = 12" in nb_runner.get_output(2)
