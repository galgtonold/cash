"""Batch 474: heapq nlargest nsmallest merge."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHeapqNlargestMerge:
    def test_nlargest_nsmallest(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq",
            "data = [15, 3, 8, 22, 1, 17, 9]\ntop3 = heapq.nlargest(3, data)\nbot3 = heapq.nsmallest(3, data)\nprint(f'top3={top3} bot3={bot3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "top3=[22, 17, 15]" in out
        assert "bot3=[1, 3, 8]" in out

    def test_merge_sorted(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq",
            "a = [1, 4, 7]\nb = [2, 5, 8]\nc = [3, 6, 9]\nmerged = list(heapq.merge(a, b, c))\nprint(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(2)

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq",
            "vals = [10, 20, 30]\nresult = heapq.nlargest(2, vals)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 20]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "vals = [100, 200, 300, 400]\nresult = heapq.nlargest(2, vals)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=[400, 300]" in nb_runner.get_output(2)
