"""Batch 193 – Sorting and ordering pattern interaction tests.

Tests editing sort keys, reverse flags, custom comparators,
and sorted data propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestSortingEdits:
    """Editing sorting operations."""

    def test_edit_sort_key(self, nb_runner):
        """Edit the key function for sorting."""
        nb_runner.create_notebook([
            "data = [('b', 2), ('a', 3), ('c', 1)]  # sort key source",
            "result = sorted(data, key=lambda x: x[0])\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [('a', 3), ('b', 2), ('c', 1)]" in nb_runner.get_output(2)

        # Sort by second element
        nb_runner.set_cell_source(
            2, "result = sorted(data, key=lambda x: x[1])\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [('c', 1), ('b', 2), ('a', 3)]" in nb_runner.get_output(2)

    def test_edit_sort_reverse(self, nb_runner):
        """Toggle reverse sort."""
        nb_runner.create_notebook([
            "nums = [3, 1, 4, 1, 5, 9]  # sort reverse source",
            "result = sorted(nums)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 1, 3, 4, 5, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = sorted(nums, reverse=True)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [9, 5, 4, 3, 1, 1]" in nb_runner.get_output(2)

    def test_edit_sort_source_data(self, nb_runner):
        """Edit source data, verify sort propagates."""
        nb_runner.create_notebook([
            "words = ['banana', 'apple', 'cherry']  # sort data source",
            "result = sorted(words)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['apple', 'banana', 'cherry']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "words = ['zebra', 'mango', 'fig']  # sort data source v2")
        nb_runner.run_all()
        assert "result = ['fig', 'mango', 'zebra']" in nb_runner.get_output(2)

    def test_sort_then_slice(self, nb_runner):
        """Sort then take a slice, edit the slice."""
        nb_runner.create_notebook([
            "vals = [50, 20, 80, 10, 90, 40]  # sort slice source",
            "sorted_vals = sorted(vals)",
            "top3 = sorted_vals[-3:]\nprint(f'top3 = {top3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3 = [50, 80, 90]" in nb_runner.get_output(3)

        # Change to bottom 2
        nb_runner.set_cell_source(
            3, "top3 = sorted_vals[:2]\nprint(f'top3 = {top3}')"
        )
        nb_runner.run_all()
        assert "top3 = [10, 20]" in nb_runner.get_output(3)
