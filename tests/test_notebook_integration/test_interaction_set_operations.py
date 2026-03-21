"""Batch 253 – Set operations and edit propagation.

Tests set unions, intersections, differences with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSetOperationEdits:
    """Set operation patterns with edit propagation."""

    def test_set_union_edit(self, nb_runner):
        """Edit one set in union, result updates."""
        nb_runner.create_notebook([
            "a = {1, 2, 3, 4}",
            "b = {3, 4, 5, 6}",
            "union = sorted(a | b)\nprint(f'union = {union}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "union = [1, 2, 3, 4, 5, 6]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "a = {10, 20, 30}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "10" in out
        assert "20" in out
        assert "30" in out

    def test_set_intersection_edit(self, nb_runner):
        """Edit set, intersection changes."""
        nb_runner.create_notebook([
            "primes = {2, 3, 5, 7, 11, 13}",
            "evens = {2, 4, 6, 8, 10, 12}",
            "common = sorted(primes & evens)\nprint(f'common = {common}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common = [2]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "evens = {2, 4, 6, 8, 10, 12, 3, 7}")
        nb_runner.run_all()
        assert "common = [2, 3, 7]" in nb_runner.get_output(3)

    def test_set_difference_edit(self, nb_runner):
        """Edit set, difference changes."""
        nb_runner.create_notebook([
            "all_items = {'a', 'b', 'c', 'd', 'e'}",
            "used = {'a', 'c'}",
            "unused = sorted(all_items - used)\nprint(f'unused = {unused}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unused = ['b', 'd', 'e']" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "used = {'a', 'b', 'c', 'd'}")
        nb_runner.run_all()
        assert "unused = ['e']" in nb_runner.get_output(3)
