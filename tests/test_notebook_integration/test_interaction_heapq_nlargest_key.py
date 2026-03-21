"""
Interaction test: heapq nlargest nsmallest with key function.
Tests heapq.nlargest, heapq.nsmallest with key parameter,
heapify, heappush/heappop, and cross-cell priority queue patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHeapqNlargestKey:
    """Test heapq nlargest/nsmallest with key across cells."""

    def test_heapq_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: nlargest/nsmallest
            "import heapq\nscores = [('Alice', 92), ('Bob', 85), ('Charlie', 98), ('Diana', 88), ('Eve', 95)]\ntop2 = heapq.nlargest(2, scores, key=lambda x: x[1])\nbottom2 = heapq.nsmallest(2, scores, key=lambda x: x[1])\nprint(f'top2={top2}')\nprint(f'bottom2={bottom2}')",
            # Cell 2: heap as priority queue
            "heap = []\nfor name, score in scores:\n    heapq.heappush(heap, (-score, name))  # negative for max-heap\nbest_name = heapq.heappop(heap)[1]\nsecond_name = heapq.heappop(heap)[1]\nprint(f'best={best_name}')\nprint(f'second={second_name}')",
            # Cell 3: merge sorted sequences
            "seq1 = [1, 3, 5, 7]\nseq2 = [2, 4, 6, 8]\nmerged = list(heapq.merge(seq1, seq2))\nprint(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Charlie" in out1 and "Eve" in out1
        assert "Bob" in out1
        out2 = nb_runner.get_output(2)
        assert "best=Charlie" in out2
        assert "second=Eve" in out2
        out3 = nb_runner.get_output(3)
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8]" in out3

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\nnums = [5, 1, 8, 3, 9, 2]\ntop3 = heapq.nlargest(3, nums)\nprint(f'top3={top3}')",
            "top_sum = sum(top3)\nprint(f'top_sum={top_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[9, 8, 5]" in nb_runner.get_output(1)
        assert "top_sum=22" in nb_runner.get_output(2)

        # Edit to get top 2 instead
        nb_runner.set_cell_source(1, "import heapq\nnums = [5, 1, 8, 3, 9, 2]\ntop3 = heapq.nlargest(2, nums)\nprint(f'top3={top3}')")
        nb_runner.run_cells([1, 2])
        assert "top3=[9, 8]" in nb_runner.get_output(1)
        assert "top_sum=17" in nb_runner.get_output(2)

    def test_heapq_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import heapq\ndata = [15, 3, 22, 7, 19, 11]\nsmallest = heapq.nsmallest(3, data)\nprint(f'smallest={smallest}')",
            "small_sum = sum(smallest)\nprint(f'small_sum={small_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=[3, 7, 11]" in nb_runner.get_output(1)
        assert "small_sum=21" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "small_sum=21" in nb_runner.get_output(2)
