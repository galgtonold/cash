"""Batch 404: set operations - union, intersection, symmetric_difference."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSetOperations:
    def test_union_intersection(self, nb_runner):
        nb_runner.create_notebook([
            "a = {1, 2, 3, 4}\nb = {3, 4, 5, 6}",
            "union = sorted(a | b)\ninter = sorted(a & b)\ndiff = sorted(a - b)\nprint(f'union={union} inter={inter} diff={diff}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "union=[1, 2, 3, 4, 5, 6]" in out
        assert "inter=[3, 4]" in out
        assert "diff=[1, 2]" in out

    def test_symmetric_difference(self, nb_runner):
        nb_runner.create_notebook([
            "x = {10, 20, 30}\ny = {20, 30, 40}",
            "sym = sorted(x ^ y)\nprint(f'sym_diff={sym}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sym_diff=[10, 40]" in nb_runner.get_output(2)

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook([
            "s1 = {1, 2, 3}\ns2 = {2, 3, 4}",
            "common = sorted(s1 & s2)\nall_items = sorted(s1 | s2)\nprint(f'common={common} all={all_items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=[2, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s1 = {5, 6, 7}\ns2 = {7, 8, 9}")
        nb_runner.run_all()
        assert "common=[7]" in nb_runner.get_output(2)
        assert "all=[5, 6, 7, 8, 9]" in nb_runner.get_output(2)
