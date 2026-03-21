"""Batch 229 – Comprehension filter and transform edit tests.

Tests editing filter conditions and transformations in various
comprehension expressions.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestComprehensionFilterEdits:
    """Editing filters and transforms in comprehensions."""

    def test_edit_list_comp_filter(self, nb_runner):
        """Edit the filter condition in a list comprehension."""
        nb_runner.create_notebook([
            "nums = list(range(20))",
            "evens = [x for x in nums if x % 2 == 0]\nprint(f'count = {len(evens)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(2)

        # Change to multiples of 3
        nb_runner.set_cell_source(2, "multiples = [x for x in nums if x % 3 == 0]\nprint(f'count = {len(multiples)}')")
        nb_runner.run_all()
        assert "count = 7" in nb_runner.get_output(2)

    def test_edit_dict_comp_transform(self, nb_runner):
        """Edit a dict comprehension transformation."""
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'python']",
            "lengths = {w: len(w) for w in words}\nprint(f'lengths = {lengths}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'hello': 5" in nb_runner.get_output(2)

        # Change to upper case keys
        nb_runner.set_cell_source(2, "uppers = {w.upper(): len(w) for w in words}\nprint(f'uppers = {uppers}')")
        nb_runner.run_all()
        assert "'HELLO': 5" in nb_runner.get_output(2)

    def test_edit_nested_flat_comprehension(self, nb_runner):
        """Edit a nested comprehension (matrix flattening)."""
        nb_runner.create_notebook([
            "matrix = [[1, 2], [3, 4], [5, 6]]",
            "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)

        # Change matrix
        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40]" in nb_runner.get_output(2)

    def test_edit_sorted_set_comprehension(self, nb_runner):
        """Edit a set comprehension."""
        nb_runner.create_notebook([
            "data = [1, 2, 2, 3, 3, 3]",
            "unique = sorted({x for x in data})\nprint(f'unique = {unique}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique = [1, 2, 3]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [4, 4, 5, 6, 6]")
        nb_runner.run_all()
        assert "unique = [4, 5, 6]" in nb_runner.get_output(2)
