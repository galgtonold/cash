"""
Debug test: reproduce the VolAdj recomputation bug.

The user reports: when changing SMA_58 -> SMA_59 in a cell with both
df['VolAdj_20'] and df['SMA_XX'] self-assignments, the VolAdj statement
gets recomputed instead of being restored from cache.

This test enables debug output to trace exactly what's happening with
lineages, cache keys, and the upstream checker.
"""

import ast

import pytest


@pytest.mark.core
@pytest.mark.upstream
class TestVolAdjRecomputeDebug:
    def test_voladj_recompute_with_debug(self, nb_runner):
        """
        Reproduce the exact issue: change SMA window only, VolAdj should be cached.
        Uses %cash_debug to get full trace of lineage/cache key computation.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                ("import pandas as pd\nimport numpy as np\nimport time"),
                # Cell 2: load ext + cash_on + debug
                ("%load_ext cash\n%cash_on\n%cash_debug on"),
                # Cell 3: Create data
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                    "    'Close': np.random.randn(100).cumsum() + 100\n"
                    "})"
                ),
                # Cell 4: Sort
                "df = df.sort_values(by=['Ticker'])",
                # Cell 5: bare df
                "df",
                # Cell 6: Heavy computation - FIRST version (SMA_58)
                # Includes print() and time.time() like the user's actual notebook
                (
                    "print('Calculating Volatility Adjusted Mean (Statement 1)....')\n"
                    "t0 = time.time()\n"
                    "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform("
                    "lambda x: x.rolling(window=5).apply("
                    "lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
                    "print(f'VolAdj calculated in {time.time() - t0:.2f}s')\n"
                    "print('Calculating Weighted SMA (Statement 2)...')\n"
                    "t0 = time.time()\n"
                    "def custom_weighted_mean(x):\n"
                    "    weights = np.arange(1, len(x) + 1)\n"
                    "    return np.sum(x * weights) / np.sum(weights)\n"
                    "df['SMA_58'] = df.groupby('Ticker')['Close'].transform("
                    "lambda x: x.rolling(window=10).apply("
                    "custom_weighted_mean, raw=True))\n"
                    "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
                    "df"
                ),
            ]
        )
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        first_output = nb_runner.get_raw_output(6)
        print("\n=== FIRST RUN DEBUG OUTPUT (cell 6) ===")
        print(first_output[:3000] if first_output else "(no output)")

        # Now change SMA_58 -> SMA_59 (window 10 -> window 3)
        nb_runner.set_cell_source(
            6,
            "print('Calculating Volatility Adjusted Mean (Statement 1)....')\n"
            "t0 = time.time()\n"
            "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=5).apply("
            "lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
            "print(f'VolAdj calculated in {time.time() - t0:.2f}s')\n"
            "print('Calculating Weighted SMA (Statement 2)...')\n"
            "t0 = time.time()\n"
            "def custom_weighted_mean(x):\n"
            "    weights = np.arange(1, len(x) + 1)\n"
            "    return np.sum(x * weights) / np.sum(weights)\n"
            "df['SMA_59'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=3).apply("
            "custom_weighted_mean, raw=True))\n"
            "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
            "df",
        )
        nb_runner.run_cell(6)
        second_output = nb_runner.get_raw_output(6)
        print("\n=== SECOND RUN DEBUG OUTPUT (cell 6) ===")
        print(second_output[:5000] if second_output else "(no output)")

        # Look for evidence of cache hit for VolAdj in the debug output
        # The debug output should show "[CACHE_HIT_DEBUG]" for the VolAdj statement
        assert second_output is not None, "Cell 6 should have output"

        # Check that VolAdj statement got a cache hit
        has_voladj_cache_hit = "Cache hit for key" in second_output and "VolAdj" in second_output
        has_voladj_restored = "RESTORED" in second_output

        # If neither, the bug is confirmed - let's print the full debug for analysis
        if not has_voladj_cache_hit and not has_voladj_restored:
            print("\n\n=== BUG CONFIRMED: VolAdj not restored from cache ===")
            print("Full debug output above shows the lineage trace")

    @pytest.mark.timeout(60)
    def test_voladj_status_badges(self, nb_runner):
        """After an SMA-only edit, the unchanged VolAdj statement is RESTORED.

        Reads the edited cell's statement statuses from the kernel with
        ``peek`` rather than from a printing cell: under ``%cash_on`` a
        printing cell is itself cached, and a hit would replay the statuses of
        the run that wrote it.
        """
        heavy = (
            "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=5).apply("
            "lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
            "def custom_weighted_mean(x):\n"
            "    weights = np.arange(1, len(x) + 1)\n"
            "    return np.sum(x * weights) / np.sum(weights)\n"
            "df['{col}'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window={window}).apply("
            "custom_weighted_mean, raw=True))\n"
            "df"
        )
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                ("import pandas as pd\nimport numpy as np\nimport time"),
                # Cell 2: load ext + cash_on. No debug output: it would echo the
                # peek below and bury its answer.
                ("%load_ext cash\n%cash_on"),
                # Cell 3: Create data. The dataset must be large enough that the
                # rolling-apply computations clear the 10 ms cost floor;
                # otherwise the VolAdj statement is (correctly) never cached and
                # "RESTORED" can never be observed.
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 10000 + ['GOOGL'] * 10000,\n"
                    "    'Close': np.random.randn(20000).cumsum() + 100\n"
                    "})"
                ),
                # Cell 4: Sort
                "df = df.sort_values(by=['Ticker'])",
                # Cell 5: bare df
                "df",
                # Cell 6: Heavy computation - FIRST version (SMA_58)
                heavy.format(col="SMA_58", window=10),
            ]
        )
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        # Now change SMA_58 -> SMA_59 in cell 6
        nb_runner.set_cell_source(6, heavy.format(col="SMA_59", window=3))
        nb_runner.run_cell(6)

        statuses = nb_runner.peek(
            "[(s['code'][:80], str(s.get('status'))) for s in "
            "get_ipython().magics_manager.magics['line']['cash_status'].__self__"
            "._last_cell_metrics['statements']]"
        )
        print("\n=== SECOND RUN STATUSES ===")
        print(statuses)
        entries = ast.literal_eval(statuses)

        # SMA_59 should be present (proving the updated code ran)
        assert any("SMA_59" in code for code, _ in entries), f"SMA_59 should be in metrics, got: {statuses}"

        voladj = [status for code, status in entries if "VolAdj" in code]
        assert voladj, f"VolAdj should be in metrics, got: {statuses}"
        for status in voladj:
            assert status in ("RESTORED", "SKIPPED"), (
                f"VolAdj statement should be RESTORED or SKIPPED after SMA-only change, but got: {statuses}"
            )
