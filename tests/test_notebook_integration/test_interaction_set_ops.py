"""Batch 190 – Set operation interaction tests.

Tests editing set operations: union, intersection, difference,
symmetric difference, with propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestSetOperationEdits:
    """Editing set operations."""

    def test_edit_set_union(self, nb_runner):
        """Edit one set in a union."""
        nb_runner.create_notebook([
            "a = {1, 2, 3}  # set A",
            "b = {3, 4, 5}  # set B",
            "result = a | b\nprint(f'result = {sorted(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5]" in nb_runner.get_output(3)

        # Change set B
        nb_runner.set_cell_source(2, "b = {10, 20}  # set B v2")
        nb_runner.run_all()
        assert "result = [1, 2, 3, 10, 20]" in nb_runner.get_output(3)

    def test_edit_set_intersection(self, nb_runner):
        """Edit set intersection."""
        nb_runner.create_notebook([
            "s1 = {1, 2, 3, 4, 5}  # intersection source 1",
            "s2 = {3, 4, 5, 6, 7}  # intersection source 2",
            "common = s1 & s2\nprint(f'common = {sorted(common)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common = [3, 4, 5]" in nb_runner.get_output(3)

        # Change s2
        nb_runner.set_cell_source(2, "s2 = {1, 5, 9}  # intersection source 2 v2")
        nb_runner.run_all()
        assert "common = [1, 5]" in nb_runner.get_output(3)

    def test_edit_set_difference(self, nb_runner):
        """Edit set difference."""
        nb_runner.create_notebook([
            "all_items = {1, 2, 3, 4, 5}  # diff source all",
            "remove = {2, 4}  # diff source remove",
            "remaining = all_items - remove\nprint(f'remaining = {sorted(remaining)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "remaining = [1, 3, 5]" in nb_runner.get_output(3)

        # Change what to remove
        nb_runner.set_cell_source(2, "remove = {1, 3, 5}  # diff source remove v2")
        nb_runner.run_all()
        assert "remaining = [2, 4]" in nb_runner.get_output(3)

    def test_edit_set_chain(self, nb_runner):
        """Edit a chain of set operations."""
        nb_runner.create_notebook([
            "x = {1, 2, 3}  # set chain x",
            "y = {2, 3, 4}  # set chain y",
            "z = {3, 4, 5}  # set chain z",
            "result = (x | y) & z\nprint(f'result = {sorted(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x|y = {1,2,3,4}, (x|y)&z = {3,4}
        assert "result = [3, 4]" in nb_runner.get_output(4)

        # Change z
        nb_runner.set_cell_source(3, "z = {1, 2}  # set chain z v2")
        nb_runner.run_all()
        # x|y = {1,2,3,4}, (x|y)&z = {1,2}
        assert "result = [1, 2]" in nb_runner.get_output(4)
