"""Integration tests for fast mode not triggering on cached loop iterations.

Reproduces the issue where running a loop cell a second time would trigger
adaptive fast mode (re-executing all iterations from scratch) instead of
restoring them from cache.
"""
import pytest

pytestmark = pytest.mark.loops


class TestFastModeCacheIntegration:
    """Integration tests for adaptive fast mode with cached iterations."""

    def test_loop_second_run_no_fast_mode(self, nb_runner):
        """Second run of a loop should not trigger FAST MODE when iterations are cached."""
        nb_runner.create_notebook([
            "import time",
            (
                "results = {}\n"
                "for name in ['alpha', 'beta', 'gamma', 'delta']:\n"
                "    # Simulate expensive work\n"
                "    time.sleep(0.05)\n"
                "    results[name] = len(name)\n"
                "print(results)"
            ),
        ])
        nb_runner.start_kernel()

        # First run: computes everything
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert 'alpha' in output1

        # Second run: should use cache, no FAST MODE
        nb_runner.run_all()
        output2 = nb_runner.get_output(2, filter_debug=False)
        assert 'alpha' in output2
        # FAST MODE should NOT appear in the output
        assert 'FAST MODE' not in output2, (
            f"FAST MODE should not trigger on cached iterations. Output: {output2}"
        )

    def test_expensive_loop_cached_correctly(self, nb_runner):
        """Expensive loop iterations should be restored from cache on second run."""
        nb_runner.create_notebook([
            (
                "data = {}\n"
                "for key in ['x', 'y', 'z']:\n"
                "    # Do some work\n"
                "    val = sum(range(100000))\n"
                "    data[key] = val\n"
                "print(f'keys={list(data.keys())}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output1 = nb_runner.get_output(1)
        assert "keys=['x', 'y', 'z']" in output1

        # Second run
        nb_runner.run_all()
        output2 = nb_runner.get_output(1)
        assert "keys=['x', 'y', 'z']" in output2
        assert 'FAST MODE' not in nb_runner.get_output(1, filter_debug=False)

    def test_modified_loop_recomputes_then_caches(self, nb_runner):
        """After modifying a loop, it should recompute, then cache on next run."""
        nb_runner.create_notebook([
            (
                "totals = []\n"
                "for i in [1, 2, 3, 4]:\n"
                "    totals.append(i * 10)\n"
                "print(f'totals={totals}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert 'totals=[10, 20, 30, 40]' in nb_runner.get_output(1)

        # Second run (cached)
        nb_runner.run_all()
        assert 'totals=[10, 20, 30, 40]' in nb_runner.get_output(1)

        # Modify the loop
        nb_runner.set_cell_source(1, (
            "totals = []\n"
            "for i in [1, 2, 3, 4]:\n"
            "    totals.append(i * 100)\n"
            "print(f'totals={totals}')"
        ))
        nb_runner.run_all()
        assert 'totals=[100, 200, 300, 400]' in nb_runner.get_output(1)

        # Third run: modified version should be cached now
        nb_runner.run_all()
        output = nb_runner.get_output(1, filter_debug=False)
        assert 'totals=[100, 200, 300, 400]' in output
        assert 'FAST MODE' not in output

    def test_loop_with_many_iterations_first_run_fast_mode_ok(self, nb_runner):
        """First run of a trivially cheap loop may use fast mode — that's fine."""
        nb_runner.create_notebook([
            (
                "counter = 0\n"
                "for i in range(50):\n"
                "    counter += 1\n"
                "print(f'counter={counter}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run: fast mode may trigger, that's OK for trivial iterations
        nb_runner.run_all()
        assert 'counter=50' in nb_runner.get_output(1)

        # Second run: should still produce correct results
        nb_runner.run_all()
        assert 'counter=50' in nb_runner.get_output(1)

    def test_financial_pattern_loop_cached(self, nb_runner):
        """Pattern similar to financial_analysis_demo: loop over tickers."""
        nb_runner.create_notebook([
            "import time",
            (
                "ticker_stats = {}\n"
                "for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:\n"
                "    time.sleep(0.05)  # Simulate expensive computation\n"
                "    stats = {'mean': len(ticker) * 10, 'std': len(ticker) * 2}\n"
                "    ticker_stats[ticker] = stats\n"
                "    print(f'{ticker}: mean={stats[\"mean\"]}')\n"
                "print(f'Done: {list(ticker_stats.keys())}')"
            ),
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert 'AAPL' in output1
        assert 'Done:' in output1

        # Second run: should be fully cached, no FAST MODE
        nb_runner.run_all()
        output2 = nb_runner.get_output(2, filter_debug=False)
        assert 'AAPL' in output2
        assert 'FAST MODE' not in output2, (
            f"Financial pattern loop should be cached, not fast-moded. Output: {output2}"
        )
