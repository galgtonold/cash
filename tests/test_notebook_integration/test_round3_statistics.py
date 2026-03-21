"""Batch 76: Statistics & random distributions — cash caching with statistical computations."""
import textwrap
import pytest


@pytest.mark.stress
class TestStatisticsModule:
    """Test statistics module patterns across cells."""

    def test_descriptive_stats(self, nb_runner):
        """Descriptive statistics across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import statistics

                data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
                mean_val = statistics.mean(data)
                median_val = statistics.median(data)
                stdev_val = statistics.stdev(data)
                print(f"mean={mean_val:.4f} median={median_val:.4f} stdev={stdev_val:.4f}")
            """),
            textwrap.dedent("""\
                import statistics
                variance = statistics.variance(data)
                mode_val = statistics.mode(data)
                print(f"var={variance:.4f} mode={mode_val}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mean=" in out1
        assert "median=" in out1
        out2 = nb_runner.get_output(2)
        assert "mode=1.25" in out2

    def test_random_seeded_distributions(self, nb_runner):
        """Seeded random distributions across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import random
                random.seed(42)

                uniform = [round(random.uniform(0, 10), 2) for _ in range(5)]
                gauss = [round(random.gauss(0, 1), 4) for _ in range(5)]
                print(f"uniform={uniform}")
                print(f"gauss={gauss}")
            """),
            textwrap.dedent("""\
                import statistics
                u_mean = round(statistics.mean(uniform), 2)
                g_mean = round(statistics.mean(gauss), 4)
                print(f"u_mean={u_mean} g_mean={g_mean}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "uniform=" in out1
        assert "gauss=" in out1
        assert "u_mean=" in nb_runner.get_output(2)

    def test_statistics_propagation(self, nb_runner):
        """Statistical results propagate on data change."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = [10, 20, 30, 40, 50]
            """),
            textwrap.dedent("""\
                import statistics
                mean_val = statistics.mean(data)
                print(f"mean={mean_val}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            data = [100, 200, 300, 400, 500]
        """))
        nb_runner.run_cells([1, 2])
        assert "mean=300" in nb_runner.get_output(2)


@pytest.mark.stress
class TestNumpyStats:
    """Test numpy statistical patterns."""

    def test_numpy_percentiles(self, nb_runner):
        """Numpy percentile calculations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import numpy as np
                np.random.seed(42)

                data = np.random.normal(100, 15, 1000)
                percentiles = np.percentile(data, [25, 50, 75])
                iqr = percentiles[2] - percentiles[0]
                print(f"p25={percentiles[0]:.1f} p50={percentiles[1]:.1f} p75={percentiles[2]:.1f}")
                print(f"iqr={iqr:.1f}")
            """),
            textwrap.dedent("""\
                import numpy as np
                outliers = data[(data < percentiles[0] - 1.5 * iqr) | (data > percentiles[2] + 1.5 * iqr)]
                print(f"outlier_count={len(outliers)}")
                print(f"data_range={data.min():.1f} to {data.max():.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "p25=" in out1
        assert "iqr=" in out1
        out2 = nb_runner.get_output(2)
        assert "outlier_count=" in out2

    def test_correlation_matrix(self, nb_runner):
        """Correlation matrix computation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import numpy as np
                np.random.seed(42)

                # Generate correlated data
                x = np.random.randn(100)
                y = 2 * x + np.random.randn(100) * 0.5
                z = -x + np.random.randn(100) * 0.3

                data_matrix = np.column_stack([x, y, z])
                corr = np.corrcoef(data_matrix, rowvar=False)
                print(f"shape={corr.shape}")
            """),
            textwrap.dedent("""\
                import numpy as np
                # x-y should be highly positively correlated
                # x-z should be highly negatively correlated
                xy_corr = round(corr[0, 1], 2)
                xz_corr = round(corr[0, 2], 2)
                print(f"xy_corr={xy_corr} xz_corr={xz_corr}")
                print(f"xy_positive={xy_corr > 0.8} xz_negative={xz_corr < -0.8}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "shape=(3, 3)" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "xy_positive=True" in out2
        assert "xz_negative=True" in out2
