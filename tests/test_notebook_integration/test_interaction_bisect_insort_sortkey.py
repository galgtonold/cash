"""Batch 472: bisect insort and sorted key functions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBisectInsortSortedKey:
    def test_bisect_search(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect",
            "grades = [60, 70, 80, 90]\ncutoffs = [60, 70, 80, 90]\nletters = ['F', 'D', 'C', 'B', 'A']\nresults = []\nfor g in [55, 65, 75, 85, 95]:\n    idx = bisect.bisect(cutoffs, g)\n    results.append(letters[idx])\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)

    def test_sorted_key(self, nb_runner):
        nb_runner.create_notebook([
            "items = [('banana', 3), ('apple', 1), ('cherry', 2)]",
            "by_name = sorted(items, key=lambda x: x[0])\nby_count = sorted(items, key=lambda x: x[1])\nprint(f'by_name={[n for n,c in by_name]}')\nprint(f'by_count={[n for n,c in by_count]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_name=['apple', 'banana', 'cherry']" in out
        assert "by_count=['apple', 'cherry', 'banana']" in out

    def test_bisect_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect",
            "data = [1, 3, 5, 7]\nbisect.insort(data, 4)\nprint(f'data={data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[1, 3, 4, 5, 7]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "data = [10, 20, 30]\nbisect.insort(data, 25)\nprint(f'data={data}')")
        nb_runner.run_all()
        assert "data=[10, 20, 25, 30]" in nb_runner.get_output(2)
