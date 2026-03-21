"""Batch 183 – Map/filter/reduce pattern interaction tests.

Tests editing functional programming patterns like map, filter,
reduce, and their combinations.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMapEdits:
    """Editing map operations."""

    def test_edit_map_function(self, nb_runner):
        """Edit the function passed to map."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]  # map source data",
            "result = list(map(lambda x: x * 2, data))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change map function
        nb_runner.set_cell_source(
            2,
            "result = list(map(lambda x: x ** 2, data))\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_map_source(self, nb_runner):
        """Edit the source data for map."""
        nb_runner.create_notebook([
            "nums = [10, 20, 30]  # map source",
            "result = list(map(str, nums))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['10', '20', '30']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "nums = [100, 200]  # map source smaller")
        nb_runner.run_all()
        assert "result = ['100', '200']" in nb_runner.get_output(2)


class TestFilterEdits:
    """Editing filter operations."""

    def test_edit_filter_predicate(self, nb_runner):
        """Edit the predicate passed to filter."""
        nb_runner.create_notebook([
            "items = list(range(10))  # filter source",
            "result = list(filter(lambda x: x > 5, items))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [6, 7, 8, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "result = list(filter(lambda x: x % 3 == 0, items))\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = [0, 3, 6, 9]" in nb_runner.get_output(2)

    def test_chain_map_filter(self, nb_runner):
        """Chain map and filter, edit one operation."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6]  # chain source",
            "doubled = map(lambda x: x * 2, data)",
            "result = list(filter(lambda x: x > 6, doubled))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [8, 10, 12]" in nb_runner.get_output(3)

        # Change map to triple
        nb_runner.set_cell_source(2, "doubled = map(lambda x: x * 3, data)")
        nb_runner.run_all()
        assert "result = [9, 12, 15, 18]" in nb_runner.get_output(3)


class TestReduceEdits:
    """Editing reduce operations."""

    def test_edit_reduce_function(self, nb_runner):
        """Edit the function passed to reduce."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "nums = [1, 2, 3, 4]  # reduce source",
            "result = reduce(lambda a, b: a + b, nums)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        # Change to multiply
        nb_runner.set_cell_source(
            3,
            "result = reduce(lambda a, b: a * b, nums)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(3)

    def test_reduce_with_initial(self, nb_runner):
        """Edit reduce initial value."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "vals = [1, 2, 3]  # reduce initial source",
            "result = reduce(lambda a, b: a + b, vals, 0)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        # Change initial value
        nb_runner.set_cell_source(
            3,
            "result = reduce(lambda a, b: a + b, vals, 100)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 106" in nb_runner.get_output(3)
