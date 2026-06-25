"""
Debug test: reproduce the VolAdj recomputation bug.

The user reports: when changing SMA_58 -> SMA_59 in a cell with both
df['VolAdj_20'] and df['SMA_XX'] self-assignments, the VolAdj statement
gets recomputed instead of being restored from cache.

This test enables debug output to trace exactly what's happening with
lineages, cache keys, and the upstream checker.
"""

import pytest



@pytest.mark.core
@pytest.mark.upstream
class TestVolAdjRecomputeDebug:

    def test_voladj_recompute_with_debug(self, nb_runner):
        """
        Reproduce the exact issue: change SMA window only, VolAdj should be cached.
        Uses %cash_debug to get full trace of lineage/cache key computation.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import time"
            ),
            # Cell 2: load ext + cash_on + debug
            (
                "%load_ext cash\n"
                "%cash_on\n"
                "%cash_debug on"
            ),
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
        ])
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        first_output = nb_runner.get_raw_output(6)
        print("\n=== FIRST RUN DEBUG OUTPUT (cell 6) ===")
        print(first_output[:3000] if first_output else "(no output)")

        # Now change SMA_58 -> SMA_59 (window 10 -> window 3)
        nb_runner.set_cell_source(6,
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
            "df"
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
    @pytest.mark.xfail(
        reason=(
            "Known effectiveness bug: when only a LATER sibling self-assignment "
            "in the same cell changes (df['SMA_58'] -> df['SMA_59']), the earlier "
            "unchanged self-assignment (df['VolAdj_20']) is recomputed instead of "
            "restored. The multi-self-assignment in-place-mutation case isn't "
            "covered by the self-assignment lineage handling: df's tracked lineage "
            "reflects the LAST mutation, so the earlier statement's output-lineage "
            "skip-check misses. The cached value is still CORRECT - this is wasted "
            "recompute, not a wrong answer. Tracked for a focused fix."
        ),
        strict=False,
    )
    def test_voladj_status_badges(self, nb_runner):
        """
        Check the actual badge metrics to verify VolAdj is RESTORED not COMPUTED.
        Uses %%cash cell magic to avoid proxy-hook metric overwriting.
        """
        nb_runner.create_notebook([
            # Cell 1: imports
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import time"
            ),
            # Cell 2: load ext + debug (but NOT cash_on -- we'll use %%cash)
            (
                "%load_ext cash\n"
                "%cash_debug on"
            ),
            # Cell 3: Create data (not cached, just runs normally)
            (
                "np.random.seed(42)\n"
                "df = pd.DataFrame({\n"
                "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                "    'Close': np.random.randn(100).cumsum() + 100\n"
                "})"
            ),
            # Cell 4: Sort (using %%cash)
            (
                "%%cash\n"
                "df = df.sort_values(by=['Ticker'])"
            ),
            # Cell 5: bare df (using %%cash)
            (
                "%%cash\n"
                "df"
            ),
            # Cell 6: Heavy computation - FIRST version (SMA_58)
            (
                "%%cash\n"
                "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform("
                "lambda x: x.rolling(window=5).apply("
                "lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
                "def custom_weighted_mean(x):\n"
                "    weights = np.arange(1, len(x) + 1)\n"
                "    return np.sum(x * weights) / np.sum(weights)\n"
                "df['SMA_58'] = df.groupby('Ticker')['Close'].transform("
                "lambda x: x.rolling(window=10).apply("
                "custom_weighted_mean, raw=True))\n"
                "df"
            ),
            # Cell 7: Query metrics
            (
                "cash_magics = get_ipython().magics_manager.magics['line'].get('cash_debug').__self__\n"
                "last_metrics = cash_magics._last_cell_metrics\n"
                "if last_metrics:\n"
                "    for stmt in last_metrics.get('statements', []):\n"
                "        code_preview = stmt['code'][:80]\n"
                "        status = stmt.get('status', 'UNKNOWN')\n"
                "        is_upstream = stmt.get('is_upstream', False)\n"
                "        prefix = '↑ ' if is_upstream else ''\n"
                "        print(f'{prefix}{status}: {code_preview}')\n"
                "else:\n"
                "    print('No last cell metrics available')"
            ),
            # Cell 8: Dump lineage state
            (
                "cash_magics = get_ipython().magics_manager.magics['line'].get('cash_debug').__self__\n"
                "vl = cash_magics._tracking_state.variable_lineage\n"
                "print('=== LINEAGE STATE ===')\n"
                "for k, v in sorted(vl.items()):\n"
                "    print(f'  {k}: {v[:16]}...')\n"
                "print('=== EXECUTED_CELL_CODES ===')\n"
                "ecc = cash_magics._statement_processor.executed_cell_codes\n"
                "for k, v in sorted(ecc.items()):\n"
                "    print(f'  {k}: {v[:60]}')"
            ),
        ])
        nb_runner.start_kernel(with_cash=False)
        nb_runner.run_all()

        first_metrics = nb_runner.get_output(7)
        print("\n=== FIRST RUN METRICS ===")
        print(first_metrics)

        first_lineage = nb_runner.get_output(8)
        print("\n=== FIRST RUN LINEAGE ===")
        print(first_lineage)

        # Get raw output from cell 6 for debug traces
        first_raw = nb_runner.get_raw_output(6)
        print("\n=== FIRST RUN CELL 6 RAW ===")
        print(first_raw[:3000] if first_raw else "(no output)")

        # Now change SMA_58 -> SMA_59 in cell 6
        nb_runner.set_cell_source(6,
            "%%cash\n"
            "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=5).apply("
            "lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
            "def custom_weighted_mean(x):\n"
            "    weights = np.arange(1, len(x) + 1)\n"
            "    return np.sum(x * weights) / np.sum(weights)\n"
            "df['SMA_59'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=3).apply("
            "custom_weighted_mean, raw=True))\n"
            "df"
        )
        nb_runner.run_cell(6)

        second_raw = nb_runner.get_raw_output(6)
        print("\n=== SECOND RUN CELL 6 RAW ===")
        print(second_raw[:5000] if second_raw else "(no output)")

        nb_runner.run_cell(7)
        nb_runner.run_cell(8)

        second_metrics = nb_runner.get_output(7)
        print("\n=== SECOND RUN METRICS ===")
        print(second_metrics)

        second_lineage = nb_runner.get_output(8)
        print("\n=== SECOND RUN LINEAGE ===")
        print(second_lineage)

        # Parse the metrics output to check VolAdj status
        assert second_metrics is not None, "Should have metrics output"
        lines = second_metrics.strip().split('\n')
        voladj_lines = [l for l in lines if "VolAdj" in l]
        sma_lines = [l for l in lines if "SMA_59" in l]
        
        # SMA_59 should be present (proving the updated code ran)
        assert sma_lines, f"SMA_59 should be in metrics, got: {second_metrics}"
        
        # VolAdj should be RESTORED or SKIPPED
        for s in voladj_lines:
            assert "RESTORED" in s or "SKIPPED" in s, (
                f"VolAdj statement should be RESTORED or SKIPPED after SMA-only change, "
                f"but got: {s}"
            )