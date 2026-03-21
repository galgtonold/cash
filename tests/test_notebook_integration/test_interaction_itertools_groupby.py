"""Batch 397: itertools.groupby with sorting and key functions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsGroupby:
    def test_groupby_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby\ndata = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]",
            "groups = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}\nprint(f'keys={sorted(groups.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['A', 'B', 'C']" in nb_runner.get_output(2)

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby\nnums = [1, 1, 2, 2, 2, 3, 3]",
            "runs = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'runs={runs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "runs=[(1, 2), (2, 3), (3, 2)]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from itertools import groupby\nnums = [5, 5, 5, 1, 1]")
        nb_runner.run_all()
        assert "runs=[(5, 3), (1, 2)]" in nb_runner.get_output(2)

    def test_groupby_with_sort(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby\nitems = [('b', 2), ('a', 1), ('b', 3), ('a', 4)]",
            "sorted_items = sorted(items, key=lambda x: x[0])\ngrouped = {k: [v for _, v in g] for k, g in groupby(sorted_items, key=lambda x: x[0])}\nprint(f'grouped={dict(sorted(grouped.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': [1, 4]" in nb_runner.get_output(2)
        assert "'b': [2, 3]" in nb_runner.get_output(2)
