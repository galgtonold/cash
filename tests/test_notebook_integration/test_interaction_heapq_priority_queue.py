"""Batch 421: heapq operations for priority queue patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHeapqPriorityQueue:
    def test_heapq_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\ndata = [5, 1, 8, 3, 2]",
            "heapq.heapify(data)\nsmallest = heapq.heappop(data)\nnext_smallest = heapq.heappop(data)\nprint(f'smallest={smallest} next={next_smallest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=1" in nb_runner.get_output(2)
        assert "next=2" in nb_runner.get_output(2)

    def test_nlargest_nsmallest(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\nnums = [10, 1, 8, 3, 5, 7, 2, 9, 4, 6]",
            "top3 = heapq.nlargest(3, nums)\nbot3 = heapq.nsmallest(3, nums)\nprint(f'top3={top3} bot3={bot3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[10, 9, 8]" in nb_runner.get_output(2)
        assert "bot3=[1, 2, 3]" in nb_runner.get_output(2)

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\nvals = [7, 3, 9, 1]",
            "top2 = heapq.nlargest(2, vals)\nprint(f'top2={top2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top2=[9, 7]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import heapq\nvals = [100, 50, 200, 75]")
        nb_runner.run_all()
        assert "top2=[200, 100]" in nb_runner.get_output(2)
