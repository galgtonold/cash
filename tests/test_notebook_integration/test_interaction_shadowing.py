"""Batch 196 – Variable shadowing and scope interaction tests.

Tests editing code that involves variable shadowing between
function scope, global scope, and comprehension scope.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestVariableShadowingEdits:
    """Editing code with variable shadowing."""

    def test_shadow_with_function_local(self, nb_runner):
        """Edit function that shadows a global variable."""
        nb_runner.create_notebook([
            "x = 'global'  # shadow source",
            "def show_x():\n    x = 'local'\n    return x",
            "result = show_x()\nprint(f'result = {result}')\nprint(f'global_x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result = local" in out
        assert "global_x = global" in out

        # Change function to use global
        nb_runner.set_cell_source(
            2, "def show_x():\n    return x  # use global"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result = global" in out2

    def test_shadow_in_comprehension(self, nb_runner):
        """Edit code that shadows variables in comprehensions."""
        nb_runner.create_notebook([
            "x = 100  # comprehension shadow source",
            "result = [x for x in range(5)]\nprint(f'result = {result}')\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result = [0, 1, 2, 3, 4]" in out
        # In Python 3, comprehension has its own scope
        assert "x = 100" in out

        # Change comprehension range
        nb_runner.set_cell_source(
            2, "result = [x for x in range(3)]\nprint(f'result = {result}')\nprint(f'x = {x}')"
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "result = [0, 1, 2]" in out2
        assert "x = 100" in out2

    def test_edit_global_affects_function(self, nb_runner):
        """Edit a global variable and verify functions see the change."""
        nb_runner.create_notebook([
            "FACTOR = 10  # global factor",
            "def compute(val):\n    return val * FACTOR",
            "result = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

        # Change global
        nb_runner.set_cell_source(1, "FACTOR = 100  # global factor v2")
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(3)

    def test_nested_function_shadowing(self, nb_runner):
        """Edit nested functions that shadow variables."""
        nb_runner.create_notebook([
            "val = 'outer'  # nested shadow source",
            "def outer():\n    val = 'middle'\n    def inner():\n        val = 'inner'\n        return val\n    return f'{val}-{inner()}'",
            "result = outer()\nprint(f'result = {result}')\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result = middle-inner" in out
        assert "val = outer" in out

        # Edit inner to use nonlocal
        nb_runner.set_cell_source(
            2,
            "def outer():\n    val = 'middle'\n    def inner():\n        nonlocal val\n        val = 'changed'\n        return val\n    return f'{inner()}-{val}'",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result = changed-changed" in out2
