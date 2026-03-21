"""Batch 184 – Inter-cell function call interaction tests.

Tests with functions defined in one cell that call functions
from another cell, with edits at various levels.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestInterCellCalls:
    """Functions calling functions from other cells."""

    def test_edit_called_function(self, nb_runner):
        """Edit a function that is called by another function."""
        nb_runner.create_notebook([
            "def helper(x):\n    return x + 1",
            "def main(x):\n    return helper(x) * 2",
            "result = main(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # helper(5)=6, main(5)=12
        assert "result = 12" in nb_runner.get_output(3)

        # Edit helper
        nb_runner.set_cell_source(1, "def helper(x):\n    return x + 100")
        nb_runner.run_all()
        # helper(5)=105, main(5)=210
        assert "result = 210" in nb_runner.get_output(3)

    def test_edit_calling_function(self, nb_runner):
        """Edit a function that calls another function."""
        nb_runner.create_notebook([
            "def square(x):\n    return x ** 2",
            "def process(x):\n    return square(x) + 1",
            "result = process(4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # square(4)=16, process(4)=17
        assert "result = 17" in nb_runner.get_output(3)

        # Edit process
        nb_runner.set_cell_source(
            2, "def process(x):\n    return square(x) * 10"
        )
        nb_runner.run_all()
        assert "result = 160" in nb_runner.get_output(3)

    def test_three_level_call_chain(self, nb_runner):
        """Three functions calling each other across cells."""
        nb_runner.create_notebook([
            "def level1(x):\n    return x + 1",
            "def level2(x):\n    return level1(x) * 2",
            "def level3(x):\n    return level2(x) + 10",
            "result = level3(3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # level1(3)=4, level2(3)=8, level3(3)=18
        assert "result = 18" in nb_runner.get_output(4)

        # Edit bottom of chain
        nb_runner.set_cell_source(1, "def level1(x):\n    return x + 100")
        nb_runner.run_all()
        # level1(3)=103, level2(3)=206, level3(3)=216
        assert "result = 216" in nb_runner.get_output(4)


class TestCallbackEdits:
    """Callback/strategy patterns with edits."""

    def test_edit_callback_function(self, nb_runner):
        """Edit a callback function passed to another function."""
        nb_runner.create_notebook([
            "def apply_fn(fn, data):\n    return [fn(x) for x in data]",
            "def transform(x):\n    return x * 2",
            "result = apply_fn(transform, [1, 2, 3])\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6]" in nb_runner.get_output(3)

        # Edit transform
        nb_runner.set_cell_source(2, "def transform(x):\n    return x ** 3")
        nb_runner.run_all()
        assert "result = [1, 8, 27]" in nb_runner.get_output(3)

    def test_edit_apply_function(self, nb_runner):
        """Edit the higher-order function."""
        nb_runner.create_notebook([
            "def processor(fn, data):\n    return [fn(x) for x in data]",
            "def double(x):\n    return x * 2",
            "result = processor(double, [5, 10])\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [10, 20]" in nb_runner.get_output(3)

        # Change processor to also sum
        nb_runner.set_cell_source(
            1,
            "def processor(fn, data):\n    mapped = [fn(x) for x in data]\n    return sum(mapped)",
        )
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)
