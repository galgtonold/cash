"""
Batch 324: counter/statistics patterns with caching.
Tests collections.Counter, statistics module, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterStatistics:
    """Test Counter and statistics operation caching."""

    def test_counter_most_common(self, nb_runner):
        """Counter.most_common with caching."""
        nb_runner.create_notebook([
            "from collections import Counter",
            "words = ['apple', 'banana', 'apple', 'cherry', 'apple', 'banana']",
            "counts = Counter(words)\ntop = counts.most_common(2)",
            "print(f'top={top}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "('apple', 3)" in out
        assert "('banana', 2)" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "('apple', 3)" in out2

    def test_counter_edit_data(self, nb_runner):
        """Edit data, verify counter updates."""
        nb_runner.create_notebook([
            "from collections import Counter",
            "data = [1, 1, 2, 2, 2, 3]",
            "c = Counter(data)\nmost = c.most_common(1)[0]",
            "print(f'most={most}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "most=(2, 3)" in out

        nb_runner.set_cell_source(2, "data = [1, 1, 1, 1, 2, 3]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "most=(1, 4)" in out2

    def test_statistics_measures(self, nb_runner):
        """statistics.mean/median/stdev with caching."""
        nb_runner.create_notebook([
            "import statistics",
            "data = [10, 20, 30, 40, 50]",
            "m = statistics.mean(data)\nmed = statistics.median(data)",
            "print(f'mean={m} median={med}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=30" in out
        assert "median=30" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "mean=30" in out2
