"""Batch 480: statistics module mean median stdev."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStatisticsMeanMedian:
    def test_basic_stats(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics",
            "data = [4, 8, 15, 16, 23, 42]\nm = statistics.mean(data)\nmed = statistics.median(data)\nsd = round(statistics.stdev(data), 2)\nprint(f'mean={m} median={med} stdev={sd}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=18" in out
        assert "median=15.5" in out

    def test_multimode(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics",
            "data = [1, 1, 2, 2, 3]\nmodes = statistics.multimode(data)\nprint(f'modes={modes}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "modes=[1, 2]" in nb_runner.get_output(2)

    def test_stats_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics",
            "vals = [10, 20, 30]\nresult = statistics.mean(vals)\nprint(f'mean={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "vals = [100, 200, 300, 400]\nresult = statistics.mean(vals)\nprint(f'mean={result}')")
        nb_runner.run_all()
        assert "mean=250" in nb_runner.get_output(2)
