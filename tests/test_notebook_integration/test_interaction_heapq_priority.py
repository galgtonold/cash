"""Batch 337: heapq priority queue operations and edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHeapqPriority:
    def test_heapq_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\ndata = [5, 1, 8, 3, 2]\nheapq.heapify(data)\nsmallest = heapq.nsmallest(3, data)\nprint(f'smallest={smallest}')",
            "heapq.heappush(data, 0)\ntop = heapq.heappop(data)\nprint(f'top={top}')",
            "merged = list(heapq.merge([1,4,7], [2,5,8], [3,6,9]))\nprint(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=[1, 2, 3]" in nb_runner.get_output(1)
        assert "top=0" in nb_runner.get_output(2)
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(3)

    def test_heapq_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\ndata = [5, 1, 8, 3, 2]",
            "heapq.heapify(data)\ntop3 = heapq.nsmallest(3, data)\nprint(f'top3={top3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[1, 2, 3]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import heapq\ndata = [50, 10, 80, 30, 20]")
        nb_runner.run_all()
        assert "top3=[10, 20, 30]" in nb_runner.get_output(2)

    def test_heapq_nlargest(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\nscores = [88, 92, 75, 100, 63, 95]",
            "top2 = heapq.nlargest(2, scores)\nprint(f'top2={top2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top2=[100, 95]" in nb_runner.get_output(2)
