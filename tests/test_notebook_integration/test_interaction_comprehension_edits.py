"""Batch 165 – Comprehension interaction tests.

Tests editing list, dict, set, and generator comprehensions
and verifying cache invalidation and recomputation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestListComprehensionEdits:
    """List comprehension edits."""

    def test_edit_comprehension_expression(self, nb_runner):
        """Edit the expression in a list comprehension."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "result = [x * 2 for x in data]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change expression
        nb_runner.set_cell_source(
            2, "result = [x ** 2 for x in data]\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_comprehension_filter(self, nb_runner):
        """Edit the filter condition in a list comprehension."""
        nb_runner.create_notebook([
            "nums = list(range(10))",
            "evens = [x for x in nums if x % 2 == 0]\nprint(f'evens = {evens}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change filter to odds
        nb_runner.set_cell_source(
            2, "evens = [x for x in nums if x % 2 == 1]\nprint(f'evens = {evens}')"
        )
        nb_runner.run_all()
        assert "evens = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    def test_edit_comprehension_source(self, nb_runner):
        """Edit the source data of a comprehension."""
        nb_runner.create_notebook([
            "src = [10, 20, 30]  # source data",
            "doubled = [x * 2 for x in src]\nprint(f'doubled = {doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled = [20, 40, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "src = [1, 2, 3]  # source data smaller")
        nb_runner.run_all()
        assert "doubled = [2, 4, 6]" in nb_runner.get_output(2)


class TestDictComprehensionEdits:
    """Dict comprehension edits."""

    def test_edit_dict_comprehension_value(self, nb_runner):
        """Edit the value expression in a dict comprehension."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "mapping = {k: len(k) for k in keys}\nprint(f'mapping = {mapping}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "mapping = {k: k.upper() for k in keys}\nprint(f'mapping = {mapping}')"
        )
        nb_runner.run_all()
        assert "'a': 'A'" in nb_runner.get_output(2)

    def test_nested_comprehension_edit(self, nb_runner):
        """Edit a nested comprehension."""
        nb_runner.create_notebook([
            "matrix = [[1, 2], [3, 4]]",
            "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to transform
        nb_runner.set_cell_source(
            2,
            "flat = [x * 10 for row in matrix for x in row]\nprint(f'flat = {flat}')",
        )
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40]" in nb_runner.get_output(2)


class TestSetComprehensionEdits:
    """Set comprehension edits."""

    def test_edit_set_comprehension(self, nb_runner):
        """Edit a set comprehension expression."""
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'hello', 'python']",
            "lengths = {len(w) for w in words}\nprint(f'lengths = {sorted(lengths)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lengths = [5, 6]" in nb_runner.get_output(2)

        # Change to first chars
        nb_runner.set_cell_source(
            2, "lengths = {w[0] for w in words}\nprint(f'lengths = {sorted(lengths)}')"
        )
        nb_runner.run_all()
        assert "'h'" in nb_runner.get_output(2)
        assert "'p'" in nb_runner.get_output(2)
        assert "'w'" in nb_runner.get_output(2)
