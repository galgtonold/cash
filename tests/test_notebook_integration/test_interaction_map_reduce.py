"""Batch 213 – Map/reduce/filter interaction tests.

Tests editing cells that use functional programming patterns
(map, filter, reduce) and verifying cache invalidation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMapReduceEdits:
    """Editing map/reduce/filter patterns."""

    def test_edit_map_function(self, nb_runner):
        """Edit the mapping function."""
        nb_runner.create_notebook([
            "nums = [1, 2, 3, 4, 5]",
            "result = list(map(lambda x: x * 2, nums))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change mapping
        nb_runner.set_cell_source(2, "result = list(map(lambda x: x ** 2, nums))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_filter_predicate(self, nb_runner):
        """Edit the filter predicate."""
        nb_runner.create_notebook([
            "nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "evens = list(filter(lambda x: x % 2 == 0, nums))\nprint(f'evens = {evens}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change to filter multiples of 3
        nb_runner.set_cell_source(2, "threes = list(filter(lambda x: x % 3 == 0, nums))\nprint(f'threes = {threes}')")
        nb_runner.run_all()
        assert "threes = [3, 6, 9]" in nb_runner.get_output(2)

    def test_edit_reduce_source(self, nb_runner):
        """Edit data fed into reduce."""
        nb_runner.create_notebook([
            "from functools import reduce\nnums = [1, 2, 3, 4, 5]",
            "product = reduce(lambda a, b: a * b, nums)\nprint(f'product = {product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product = 120" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "from functools import reduce\nnums = [2, 3, 4]")
        nb_runner.run_all()
        assert "product = 24" in nb_runner.get_output(2)

    def test_edit_chained_map_filter(self, nb_runner):
        """Edit source for chained map then filter."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6]",
            "doubled = list(map(lambda x: x * 2, data))\nbig = list(filter(lambda x: x > 6, doubled))\nprint(f'big = {big}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big = [8, 10, 12]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 10, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "20" in out
        assert "40" in out
