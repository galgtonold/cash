"""
Batch 285: Bisect/heapq interaction tests.
Tests that editing sorted data or heap structures properly invalidates
downstream lookups and extractions.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestBisectHeapqInteraction:
    """Test bisect/heapq patterns with cache invalidation."""

    def test_bisect_insert_point_edit(self, nb_runner):
        """Editing sorted data should invalidate bisect lookup."""
        nb_runner.create_notebook([
            "import bisect\nsorted_data = [10, 20, 30, 40, 50]",
            "target = 25",
            "pos = bisect.bisect_left(sorted_data, target)",
            "print(f'pos={pos}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "pos=2" in out

        nb_runner.set_cell_source(2, "target = 45")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "pos=4" in out

    def test_heapq_nsmallest_edit(self, nb_runner):
        """Editing data should invalidate heapq nsmallest results."""
        nb_runner.create_notebook([
            "import heapq\nvalues = [5, 1, 8, 3, 9, 2]",
            "n = 3",
            "smallest = heapq.nsmallest(n, values)",
            "result = ','.join(str(x) for x in smallest)",
            "print(f'smallest={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "smallest=1,2,3" in out

        nb_runner.set_cell_source(1, "import heapq\nvalues = [50, 10, 80, 30, 90, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "smallest=10,20,30" in out

    def test_heapq_merge_edit(self, nb_runner):
        """Editing one of the sorted lists to merge should propagate."""
        nb_runner.create_notebook([
            "import heapq\nlist_a = [1, 4, 7]\nlist_b = [2, 5, 8]",
            "merged = list(heapq.merge(list_a, list_b))",
            "result = ','.join(str(x) for x in merged)",
            "print(f'merged={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "merged=1,2,4,5,7,8" in out

        nb_runner.set_cell_source(1, "import heapq\nlist_a = [10, 40, 70]\nlist_b = [2, 5, 8]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "merged=2,5,8,10,40,70" in out
