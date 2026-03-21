"""Batch 280 – Built-in function patterns with edits.

Tests map, filter, reduce with function/data edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBuiltinFuncEdits:
    """Map/filter/reduce edit patterns."""

    def test_map_function_edit(self, nb_runner):
        """Edit mapping function, result changes."""
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "data = [1, 2, 3, 4, 5]\nresult = list(map(transform, data))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_filter_predicate_edit(self, nb_runner):
        """Edit filter predicate, filtered set changes."""
        nb_runner.create_notebook([
            "def pred(x):\n    return x % 2 == 0",
            "data = [1, 2, 3, 4, 5, 6]\nresult = list(filter(pred, data))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def pred(x):\n    return x > 3")
        nb_runner.run_all()
        assert "result = [4, 5, 6]" in nb_runner.get_output(2)

    def test_reduce_operator_edit(self, nb_runner):
        """Edit reduce operation."""
        nb_runner.create_notebook([
            "from functools import reduce\ndef op(a, b):\n    return a + b",
            "data = [1, 2, 3, 4]\nresult = reduce(op, data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from functools import reduce\ndef op(a, b):\n    return a * b",
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)
