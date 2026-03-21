"""Batch 362: set operations (union, intersection, difference, symmetric_difference)."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSetOperationsAdvanced:
    def test_set_operations(self, nb_runner):
        nb_runner.create_notebook([
            "a = {1, 2, 3, 4, 5}\nb = {4, 5, 6, 7, 8}",
            "union = sorted(a | b)\ninter = sorted(a & b)\ndiff = sorted(a - b)\nsym = sorted(a ^ b)\nprint(f'union={union} inter={inter} diff={diff} sym={sym}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "union=[1, 2, 3, 4, 5, 6, 7, 8]" in out
        assert "inter=[4, 5]" in out
        assert "diff=[1, 2, 3]" in out
        assert "sym=[1, 2, 3, 6, 7, 8]" in out

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook([
            "s1 = {'apple', 'banana', 'cherry'}",
            "s2 = {'banana', 'date'}\ncommon = sorted(s1 & s2)\nprint(f'common={common}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=['banana']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "s1 = {'banana', 'date', 'elderberry'}")
        nb_runner.run_all()
        assert "common=['banana', 'date']" in nb_runner.get_output(2)

    def test_frozenset_in_set(self, nb_runner):
        nb_runner.create_notebook([
            "groups = {frozenset({1, 2}), frozenset({3, 4}), frozenset({1, 2})}",
            "count = len(groups)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
