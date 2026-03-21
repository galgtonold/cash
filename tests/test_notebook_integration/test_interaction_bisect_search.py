"""Batch 380: bisect insert and binary search patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBisectBinarySearch:
    def test_bisect_insort(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\ndata = [10, 20, 30, 40, 50]",
            "pos = bisect.bisect_left(data, 25)\nbisect.insort(data, 25)\nprint(f'pos={pos} data={data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=2" in nb_runner.get_output(2)
        assert "data=[10, 20, 25, 30, 40, 50]" in nb_runner.get_output(2)

    def test_bisect_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\nsorted_list = [1, 3, 5, 7, 9]",
            "idx = bisect.bisect_right(sorted_list, 5)\nprint(f'idx={idx}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=3" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import bisect\nsorted_list = [2, 4, 6, 8, 10]")
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)

    def test_grade_lookup(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\nbreakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'",
            "scores = [55, 65, 75, 85, 95]\nresult = [grades[bisect.bisect(breakpoints, s)] for s in scores]\nprint(f'grades={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grades=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)
