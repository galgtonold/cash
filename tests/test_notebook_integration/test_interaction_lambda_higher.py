"""Batch 245 – Complex lambda and higher-order function patterns.

Tests lambdas, map/filter/reduce with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestLambdaHigherOrder:
    """Lambda and higher-order function edit patterns."""

    def test_lambda_in_map(self, nb_runner):
        """Edit lambda used in map."""
        nb_runner.create_notebook([
            "transform = lambda x: x * 2",
            "data = [1, 2, 3, 4, 5]\nresult = list(map(transform, data))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "transform = lambda x: x ** 3")
        nb_runner.run_all()
        assert "result = [1, 8, 27, 64, 125]" in nb_runner.get_output(2)

    def test_filter_lambda_edit(self, nb_runner):
        """Edit filter criteria lambda."""
        nb_runner.create_notebook([
            "is_valid = lambda x: x > 0",
            "nums = [-3, -1, 0, 2, 5, -4, 8]\nvalid = list(filter(is_valid, nums))\nprint(f'valid = {valid}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid = [2, 5, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "is_valid = lambda x: x % 2 == 0")
        nb_runner.run_all()
        assert "valid = [0, 2, -4, 8]" in nb_runner.get_output(2)

    def test_reduce_with_edit(self, nb_runner):
        """Edit reduce function, downstream updates."""
        nb_runner.create_notebook([
            "from functools import reduce\ncombiner = lambda a, b: a + b",
            "data = [1, 2, 3, 4, 5]\nresult = reduce(combiner, data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from functools import reduce\ncombiner = lambda a, b: a * b",
        )
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

    def test_sorted_key_lambda_edit(self, nb_runner):
        """Edit sort key lambda."""
        nb_runner.create_notebook([
            "key_fn = lambda item: item[1]",
            "records = [('a', 3), ('b', 1), ('c', 2)]\nordered = sorted(records, key=key_fn)\nprint(f'ordered = {ordered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ordered = [('b', 1), ('c', 2), ('a', 3)]" in nb_runner.get_output(2)

        # Sort by first element descending
        nb_runner.set_cell_source(1, "key_fn = lambda item: item[0]")
        nb_runner.run_all()
        assert "ordered = [('a', 3), ('b', 1), ('c', 2)]" in nb_runner.get_output(2)
