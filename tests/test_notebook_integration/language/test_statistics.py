"""The statistics module and hand-written statistics across cells."""

import textwrap

import pytest


class TestStatisticsModule:
    """statistics module operations and edits."""

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_stats_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\ndata = [10, 20, 30, 40, 50]",
                "mean = statistics.mean(data)\nmedian = statistics.median(data)\nstdev = round(statistics.stdev(data), 2)\nprint(f'mean={mean} median={median} stdev={stdev}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=30" in out
        assert "median=30" in out
        assert "stdev=15.81" in out

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_stats_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [85, 90, 78, 92, 88]",
                "mean_score = statistics.mean(scores)\nmode_score = statistics.mode(scores) if len(set(scores)) < len(scores) else 'no mode'\nprint(f'mean={mean_score} mode={mode_score}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=86.6" in nb_runner.get_output(2)
        # Edit with mode
        nb_runner.set_cell_source(1, "import statistics\nscores = [80, 90, 80, 90, 80]")
        nb_runner.run_all()
        assert "mean=84" in nb_runner.get_output(2)
        assert "mode=80" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_quantiles(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\ndata = list(range(1, 101))",
                "q = statistics.quantiles(data, n=4)\nprint(f'quartiles={q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "quartiles=" in out
        assert "50.5" in out

    # Statistics & random distributions — cash caching with statistical computations.
    @pytest.mark.stress
    def test_statistics_propagation(self, nb_runner):
        """Statistical results propagate on data change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = [10, 20, 30, 40, 50]
            """),
                textwrap.dedent("""\
                import statistics
                mean_val = statistics.mean(data)
                print(f"mean={mean_val}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            data = [100, 200, 300, 400, 500]
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "mean=300" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStatisticsMeanMedian:
    """statistics module mean median stdev."""

    def test_basic_stats(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "data = [4, 8, 15, 16, 23, 42]\nm = statistics.mean(data)\nmed = statistics.median(data)\nsd = round(statistics.stdev(data), 2)\nprint(f'mean={m} median={med} stdev={sd}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "mean=18" in out
        assert "median=15.5" in out

    def test_multimode(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "data = [1, 1, 2, 2, 3]\nmodes = statistics.multimode(data)\nprint(f'modes={modes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "modes=[1, 2]" in nb_runner.get_output(2)

    def test_stats_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics",
                "vals = [10, 20, 30]\nresult = statistics.mean(vals)\nprint(f'mean={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "vals = [100, 200, 300, 400]\nresult = statistics.mean(vals)\nprint(f'mean={result}')"
        )
        nb_runner.run_all()
        assert "mean=250" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStatisticsMedianStdev:
    """Test statistics median and stdev across cells."""

    def test_statistics_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic statistics
                "import statistics\ndata = [4, 8, 15, 16, 23, 42]\nmean = statistics.mean(data)\nmedian = statistics.median(data)\nprint(f'mean={mean}')\nprint(f'median={median}')",
                # Cell 2: stdev and variance
                "stdev = statistics.stdev(data)\nvariance = statistics.variance(data)\nprint(f'stdev={stdev:.4f}')\nprint(f'variance={variance:.4f}')",
                # Cell 3: mode
                "mode_data = [1, 2, 2, 3, 3, 3, 4]\nmode = statistics.mode(mode_data)\nprint(f'mode={mode}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [80, 85, 90, 95, 100]\navg = statistics.mean(scores)\nprint(f'avg={avg}')",
                "report = f'Average score: {avg}'\nprint(f'report={report}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=90" in nb_runner.get_output(1)
        assert "report=Average score: 90" in nb_runner.get_output(2)

        # Add more scores
        nb_runner.set_cell_source(
            1,
            "import statistics\nscores = [80, 85, 90, 95, 100, 70]\navg = statistics.mean(scores)\nprint(f'avg={avg}')",
        )
        nb_runner.run_cells([1, 2])
        assert "avg=86" in nb_runner.get_output(1)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nvals = [10, 20, 30, 40, 50]\nmed = statistics.median(vals)\nprint(f'median={med}')",
                "above_median = [v for v in vals if v > med]\nprint(f'above={above_median}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=30" in nb_runner.get_output(1)
        assert "above=[40, 50]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above=[40, 50]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStatisticsVarianceCorr:
    """Test statistics variance and correlation across cells."""

    def test_variance_stdev(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: compute variance and stdev
                "import statistics\ndata = [10, 20, 30, 40, 50]\nmean = statistics.mean(data)\nvar = statistics.variance(data)\nstdev = statistics.stdev(data)\nprint(f'mean={mean}')\nprint(f'var={var}')\nprint(f'stdev={stdev:.2f}')",
                # Cell 2: population vs sample
                "pvar = statistics.pvariance(data)\npstdev = statistics.pstdev(data)\nprint(f'pvar={pvar}')\nprint(f'pstdev={pstdev:.2f}')\nprint(f'sample_larger={var > pvar}')",
                # Cell 3: coefficient of variation
                "cv = stdev / mean * 100\nprint(f'cv={cv:.1f}%')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import statistics\ndata = [5, 5, 5, 5, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev}')",
                "is_uniform = stdev == 0\nprint(f'uniform={is_uniform}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stdev=0" in nb_runner.get_output(1)
        assert "uniform=True" in nb_runner.get_output(2)

        # Edit to non-uniform data
        nb_runner.set_cell_source(
            1, "import statistics\ndata = [1, 2, 3, 4, 5]\nstdev = statistics.stdev(data)\nprint(f'stdev={stdev:.2f}')"
        )
        nb_runner.run_cells([1, 2])
        assert "stdev=1.58" in nb_runner.get_output(1)
        assert "uniform=False" in nb_runner.get_output(2)

    def test_statistics_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import statistics\nscores = [85, 90, 78, 92, 88]\nmedian = statistics.median(scores)\nmode_val = statistics.mode(scores)\nprint(f'median={median}')\nprint(f'mode={mode_val}')",
                "above_med = sum(1 for s in scores if s > median)\nprint(f'above_median={above_med}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "median=88" in nb_runner.get_output(1)
        assert "above_median=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "above_median=2" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestStatisticalTests:
    """Test statistical computation patterns."""

    def test_descriptive_statistics(self, nb_runner):
        """Descriptive stats computed across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                data = np.random.normal(100, 15, 1000)
            """),
                textwrap.dedent("""\
                stats = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'median': np.median(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75),
                }
            """),
                textwrap.dedent("""\
                iqr = stats['q75'] - stats['q25']
                print(f"mean={stats['mean']:.1f} std={stats['std']:.1f} iqr={iqr:.1f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "mean=" in output
        # Mean should be close to 100
        mean_val = float(output.split("mean=")[1].split(" ")[0])
        assert abs(mean_val - 100) < 5

    def test_correlation_analysis(self, nb_runner):
        """Correlation analysis across cells."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                x = np.random.randn(500)
                noise = np.random.randn(500) * 0.5
                y = 2 * x + 3 + noise  # Strong positive correlation
            """),
                textwrap.dedent("""\
                correlation = np.corrcoef(x, y)[0, 1]
                print(f"corr={correlation:.4f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "corr=" in output
        # Should be high positive correlation
        corr = float(output.split("corr=")[1].strip())
        assert corr > 0.9
