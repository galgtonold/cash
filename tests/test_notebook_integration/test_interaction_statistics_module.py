"""Batch 367: statistics module operations and edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStatisticsModule:
    def test_stats_basic(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\ndata = [10, 20, 30, 40, 50]",
            "mean = statistics.mean(data)\nmedian = statistics.median(data)\nstdev = round(statistics.stdev(data), 2)\nprint(f'mean={mean} median={median} stdev={stdev}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=30" in out
        assert "median=30" in out
        assert "stdev=15.81" in out

    def test_stats_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\nscores = [85, 90, 78, 92, 88]",
            "mean_score = statistics.mean(scores)\nmode_score = statistics.mode(scores) if len(set(scores)) < len(scores) else 'no mode'\nprint(f'mean={mean_score} mode={mode_score}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=86.6" in nb_runner.get_output(2)
        # Edit with mode
        nb_runner.set_cell_source(1, "import statistics\nscores = [80, 90, 80, 90, 80]")
        nb_runner.run_all()
        assert "mean=84" in nb_runner.get_output(2)
        assert "mode=80" in nb_runner.get_output(2)

    def test_quantiles(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\ndata = list(range(1, 101))",
            "q = statistics.quantiles(data, n=4)\nprint(f'quartiles={q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "quartiles=" in out
        assert "50.5" in out
