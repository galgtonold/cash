"""
Interaction test: statistics module median and stdev.
Tests statistics.median, stdev, variance, mode,
and cross-cell statistical analysis pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStatisticsMedianStdev:
    """Test statistics median and stdev across cells."""

    def test_statistics_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic statistics
            "import statistics\ndata = [4, 8, 15, 16, 23, 42]\nmean = statistics.mean(data)\nmedian = statistics.median(data)\nprint(f'mean={mean}')\nprint(f'median={median}')",
            # Cell 2: stdev and variance
            "stdev = statistics.stdev(data)\nvariance = statistics.variance(data)\nprint(f'stdev={stdev:.4f}')\nprint(f'variance={variance:.4f}')",
            # Cell 3: mode
            "mode_data = [1, 2, 2, 3, 3, 3, 4]\nmode = statistics.mode(mode_data)\nprint(f'mode={mode}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mean=18" in out1
        assert "median=15.5" in out1
        out2 = nb_runner.get_output(2)
        assert "stdev=" in out2
        assert "variance=" in out2
        out3 = nb_runner.get_output(3)
        assert "mode=3" in out3

    def test_statistics_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\nscores = [80, 85, 90, 95, 100]\navg = statistics.mean(scores)\nprint(f'avg={avg}')",
            "report = f'Average score: {avg}'\nprint(f'report={report}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=90" in nb_runner.get_output(1)
        assert "report=Average score: 90" in nb_runner.get_output(2)

        # Add more scores
        nb_runner.set_cell_source(1, "import statistics\nscores = [80, 85, 90, 95, 100, 70]\navg = statistics.mean(scores)\nprint(f'avg={avg}')")
        nb_runner.run_cells([1, 2])
        assert "avg=86" in nb_runner.get_output(1)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\nvals = [10, 20, 30, 40, 50]\nmed = statistics.median(vals)\nprint(f'median={med}')",
            "above_median = [v for v in vals if v > med]\nprint(f'above={above_median}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=30" in nb_runner.get_output(1)
        assert "above=[40, 50]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above=[40, 50]" in nb_runner.get_output(2)
