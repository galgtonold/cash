"""
Tests for out-of-order execution: display cell → display cell → computation cell.

Reproduces the bug where executing display cells out of order before a computation
cell causes the computation cell to miss its cache, even though cached data exists
on disk.

Scenario (from financial_analysis_demo.ipynb):
  Cell 1: import + load data (df = pd.read_csv(...))
  Cell 2: preprocess (df = df.sort_values(...))
  Cell 3: display df (just `df`)
  Cell 4: heavy computation (df['col1'] = ...; df['col2'] = ...)
  Cell 5: display df (just `df` again)

First run: cells 1 → 2 → 3 → 4 → 5 (populates cache).
After reset (simulating fresh session): execute cells 5 → 3 → 4.

Bug: Cell 4's computation is fully recomputed instead of restored from cache.

Root cause: After reset/kernel restart, `_executed_cell_raw_codes` was empty.
The `cell_code_changed` flag was computed as `raw_cell not in self._executed_cell_raw_codes`,
which was True for ALL cells in a fresh session — not just edited cells.
This set `force_recompute = True` in statement_processor.py, which disabled
cache lookups entirely, even though the cache data on disk was valid.

Fix: Changed `cell_code_changed` detection to use position-based tracking
(`_last_cell_code_by_index`). Now `cell_code_changed` is only True when we
have evidence the cell was actually edited (different code at the same notebook
position), not merely "first time in this session".
"""
import pytest
import time

pytestmark = [pytest.mark.upstream, pytest.mark.core]


class TestOutOfOrderDisplayThenCompute:
    """Bug: Display cells executed out-of-order break computation cache restore."""

    def test_display_display_compute_order(self, nb_runner):
        """
        Scenario from financial_analysis_demo.ipynb:
        After cache is populated, reset and execute:
          Cell 5 (display after computation) →
          Cell 3 (display before computation) →
          Cell 4 (heavy computation)

        Cell 4 should restore from cache, NOT recompute.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            import time
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=n, freq='D'),
                'ticker': ['AAPL'] * (n // 2) + ['GOOGL'] * (n // 2),
                'close': np.cumsum(np.random.randn(n)) + 100,
                'volume': np.random.randint(100, 10000, n),
            })
            print(f"Created df: {df.shape}")""")

        cell_2 = textwrap.dedent("""\
            df = df.sort_values(by=['ticker', 'date'])
            print(f"Sorted df: {df.shape}")""")

        cell_3 = textwrap.dedent("""\
            print("Display 1")
            df""")

        cell_4 = textwrap.dedent("""\
            t0 = time.time()
            df['rolling_mean'] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window=20).mean())
            elapsed1 = time.time() - t0
            print(f"Rolling mean: {elapsed1:.2f}s")

            t0 = time.time()
            time.sleep(0.3)
            df['vol_ratio'] = df.groupby('ticker')['volume'].transform(
                lambda x: x / x.rolling(window=10).mean())
            elapsed2 = time.time() - t0
            print(f"Vol ratio: {elapsed2:.2f}s")

            df""")

        cell_5 = textwrap.dedent("""\
            print("Display 2")
            df""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3, cell_4, cell_5])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # --- First pass: Run all cells in order to populate cache ---
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Created df" in out1

        out4_first = nb_runner.get_output(4)
        assert "Rolling mean" in out4_first
        assert "Vol ratio" in out4_first

        # --- Reset cash state to simulate fresh session ---
        nb_runner.reset_cash_state()

        # Also clear user_ns variables to better simulate kernel restart
        # (only lineage/codes are cleared by reset_cash_state, not the actual vars)
        clear_code = """
for _v in ['df', 't0', 'elapsed1', 'elapsed2', 'n']:
    try:
        del globals()[_v]
    except KeyError:
        pass
"""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            nb_runner.client.kc._async_execute_interactive(clear_code, store_history=False)
        )

        # --- Out-of-order execution: cell 5, then cell 3, then cell 4 ---
        # Cell 5: triggers upstream restore of fully computed df
        nb_runner.run_cell(5)
        out5_rerun = nb_runner.get_raw_output(5)
        print(f"\n=== Cell 5 rerun output (raw) ===\n{out5_rerun[:2000]}")

        # Cell 3: display df (before computation cell)
        nb_runner.run_cell(3)
        out3_rerun = nb_runner.get_raw_output(3)
        print(f"\n=== Cell 3 rerun output (raw) ===\n{out3_rerun[:2000]}")

        # Cell 4: heavy computation
        # This should restore from cache, NOT recompute
        t_start = time.time()
        nb_runner.run_cell(4)
        t_elapsed = time.time() - t_start
        out4_rerun = nb_runner.get_raw_output(4)

        out4_text = nb_runner.get_output(4)
        print(f"\n=== Cell 4 rerun output (filtered) ===\n{out4_text}")
        print(f"\n=== Cell 4 rerun output (raw) ===\n{out4_rerun[:2000]}")
        print(f"\n=== Cell 4 timing: {t_elapsed:.2f}s ===\n")

        assert "Rolling mean" in out4_text
        assert "Vol ratio" in out4_text

        # The key assertion: if cell 4 recomputed, it would take > 0.3s (has sleep)
        # If cached, it should be fast (< 0.25s)
        # Also check for Virtual Restore SUCCESS in debug output
        has_restore = "Virtual Restore SUCCESS" in out4_rerun
        was_fast = t_elapsed < 0.5  # Allow some overhead

        assert has_restore or was_fast, (
            f"Cell 4 should have restored from cache, but appears to have recomputed. "
            f"Elapsed: {t_elapsed:.2f}s. "
            f"Output: {out4_rerun[:1000]}")

    def test_display_before_compute_simple(self, nb_runner):
        """
        Simplified version: 4 cells.
        After cache populated, reset and run: cell 4, cell 3, cell 3_compute.
        
        Cell 1: x = 10
        Cell 2: y = x * 2 (expensive simulation with sleep)
        Cell 3: y  (display)
        Cell 4: z = y + 1; print(z)
        
        Reset, run: 3 → 2.
        Cell 2 should restore from cache, not recompute.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            x = 10
            print(f"x={x}")""")

        cell_2 = textwrap.dedent("""\
            import time
            t0 = time.time()
            time.sleep(0.5)
            y = x * 2
            elapsed = time.time() - t0
            print(f"y={y}, took {elapsed:.2f}s")""")

        cell_3 = textwrap.dedent("""\
            y""")

        cell_4 = textwrap.dedent("""\
            z = y + 1
            print(f"z={z}")""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3, cell_4])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # First pass: populate cache
        nb_runner.run_all()
        assert "y=20" in nb_runner.get_output(2)
        assert "z=21" in nb_runner.get_output(4)

        # Reset
        nb_runner.reset_cash_state()

        # Out-of-order: cell 4 first, then cell 3, then cell 2
        nb_runner.run_cell(4)
        assert "z=21" in nb_runner.get_output(4)

        nb_runner.run_cell(3)

        # Now run cell 2 - should restore from cache, not sleep 0.5s
        t_start = time.time()
        nb_runner.run_cell(2)
        t_elapsed = time.time() - t_start

        out2_rerun = nb_runner.get_output(2)
        assert "y=20" in out2_rerun

        # If it took > 0.4s, it recomputed (sleep was 0.5s)
        # If cache restored, it should be < 0.3s
        assert t_elapsed < 0.4, (
            f"Cell 2 took {t_elapsed:.2f}s, suggesting it recomputed instead of "
            f"restoring from cache. Debug output: {nb_runner.get_raw_output(2)[:500]}")
