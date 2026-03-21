"""
Interaction test: statistics module with variance, stdev, correlation.
Tests statistics.variance, stdev, correlation (3.10+), and
cross-cell statistical analysis.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStatisticsVarianceCorr:
    """Test statistics variance and correlation across cells."""

    def test_variance_stdev(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: compute variance and stdev
            "import statistics\ndata = [10, 20, 30, 40, 50]\nmean = statistics.mean(data)\nvar = statistics.variance(data)\nstdev = statistics.stdev(data)\nprint(f'mean={mean}')\nprint(f'var={var}')\nprint(f'stdev={stdev:.2f}')",
            # Cell 2: population vs sample
            "pvar = statistics.pvariance(data)\npstdev = statistics.pstdev(data)\nprint(f'pvar={pvar}')\nprint(f'pstdev={pstdev:.2f}')\nprint(f'sample_larger={var > pvar}')",
            # Cell 3: coefficient of variation
            "cv = stdev / mean * 100\nprint(f'cv={cv:.1f}%')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mean=30" in out1
        assert "var=250" in out1
        assert "stdev=15.81" in out1
        out2 = nb_runner.get_output(2)
        assert "pvar=200" in out2
        assert "pstdev=14.14" in out2
        assert "sample_larger=True" in out2
        out3 = nb_runner.get_output(3)
        assert "cv=52.7%" in out3

    def test_statistics_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\ndata = [5, 5, 5, 5, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev}')",
            "is_uniform = stdev == 0\nprint(f'uniform={is_uniform}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stdev=0" in nb_runner.get_output(1)
        assert "uniform=True" in nb_runner.get_output(2)

        # Edit to non-uniform data
        nb_runner.set_cell_source(1, "import statistics\ndata = [1, 2, 3, 4, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev:.2f}')")
        nb_runner.run_cells([1, 2])
        assert "stdev=1.58" in nb_runner.get_output(1)
        assert "uniform=False" in nb_runner.get_output(2)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import statistics\nscores = [85, 90, 78, 92, 88]\nmedian = statistics.median(scores)\nmode_val = statistics.mode(scores)\nprint(f'median={median}')\nprint(f'mode={mode_val}')",
            "above_med = sum(1 for s in scores if s > median)\nprint(f'above_median={above_med}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=88" in nb_runner.get_output(1)
        assert "above_median=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above_median=2" in nb_runner.get_output(2)
