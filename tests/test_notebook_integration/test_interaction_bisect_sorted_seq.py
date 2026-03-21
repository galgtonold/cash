"""Batch 420: bisect module for sorted sequence operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBisectSortedSeq:
    def test_bisect_insort(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\nsorted_list = [10, 20, 30, 40, 50]",
            "pos = bisect.bisect_left(sorted_list, 25)\nbisect.insort(sorted_list, 25)\nprint(f'pos={pos} list={sorted_list}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=2" in nb_runner.get_output(2)
        assert "list=[10, 20, 25, 30, 40, 50]" in nb_runner.get_output(2)

    def test_bisect_grade(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\nbreakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'\nscores = [33, 65, 77, 89, 95]",
            "results = [grades[bisect.bisect(breakpoints, s)] for s in scores]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)

    def test_bisect_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\ndata = [1, 3, 5, 7, 9]",
            "idx = bisect.bisect_left(data, 5)\nprint(f'idx={idx}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import bisect\ndata = [2, 4, 6, 8, 10]")
        nb_runner.set_cell_source(2, "idx = bisect.bisect_left(data, 6)\nprint(f'idx={idx}')")
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)
