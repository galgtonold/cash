"""
Interaction test: itertools.groupby with key function.
Tests groupby with sorted data, key extraction, group aggregation,
and cross-cell grouped data processing.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGroupbyKeyFunction:
    """Test itertools.groupby with key function across cells."""

    def test_groupby_aggregation(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: sort and group
            "from itertools import groupby\ndata = [('A', 10), ('B', 20), ('A', 30), ('B', 40), ('C', 50)]\nsorted_data = sorted(data, key=lambda x: x[0])\ngroups = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}\nprint(f'groups={groups}')",
            # Cell 2: aggregate per group
            "sums = {k: sum(v) for k, v in groups.items()}\navgs = {k: sum(v)/len(v) for k, v in groups.items()}\nprint(f'sums={sums}')\nprint(f'avgs={avgs}')",
            # Cell 3: find best group
            "best = max(sums, key=sums.get)\nprint(f'best={best}')\nprint(f'best_sum={sums[best]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'A': [10, 30]" in out1
        assert "'B': [20, 40]" in out1
        assert "'C': [50]" in out1
        out2 = nb_runner.get_output(2)
        assert "'A': 40" in out2
        assert "'B': 60" in out2
        out3 = nb_runner.get_output(3)
        assert "best=B" in out3
        assert "best_sum=60" in out3

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby\nwords = ['apple', 'avocado', 'banana', 'blueberry', 'cherry']\nby_letter = {k: list(g) for k, g in groupby(sorted(words), key=lambda w: w[0])}\nprint(f'groups={by_letter}')",
            "counts = {k: len(v) for k, v in by_letter.items()}\nprint(f'counts={counts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': ['apple', 'avocado']" in nb_runner.get_output(1)
        assert "'a': 2" in nb_runner.get_output(2)

        # Add more words
        nb_runner.set_cell_source(1, "from itertools import groupby\nwords = ['apple', 'avocado', 'apricot', 'banana', 'blueberry', 'cherry', 'coconut']\nby_letter = {k: list(g) for k, g in groupby(sorted(words), key=lambda w: w[0])}\nprint(f'groups={by_letter}')")
        nb_runner.run_cells([1, 2])
        assert "'a': 3" in nb_runner.get_output(2)
        assert "'c': 2" in nb_runner.get_output(2)

    def test_groupby_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby\nnums = [1, 1, 2, 2, 2, 3, 3]\nruns = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'runs={runs}')",
            "longest_run = max(runs, key=lambda x: x[1])\nprint(f'longest={longest_run}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "runs=[(1, 2), (2, 3), (3, 2)]" in nb_runner.get_output(1)
        assert "longest=(2, 3)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "longest=(2, 3)" in nb_runner.get_output(2)
