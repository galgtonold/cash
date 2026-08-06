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

Originally written for a bug where `cell_code_changed` was computed as "this
raw code has not run this session", which is true of EVERY cell in a fresh
session, not just edited ones -- so cache lookups were disabled wholesale even
though the on-disk data was valid. That mechanism is gone: neither
`cell_code_changed` nor `force_recompute` exists in the source anymore.

What still needs guarding is the behaviour: running the cells OUT OF ORDER
after a reset must reconstruct ``df`` to exactly the value a top-to-bottom run
produces. Cell 5 fires first and has to rebuild ``df`` from scratch, including
re-establishing the seeded RNG stream, before cell 4's columns can be right.

Deliberately NOT asserted: that cell 4 restores from cache. Both of its
expensive statements are subscript stores on ``df``
(``df['rolling_mean'] = ...``), which ``analyze_statement`` reports as
``top_level_mutated_vars={'df'}`` -- the in-place-mutation route sets
``skip_cache``, so those statements re-execute BY DESIGN and the sleep inside
cell 4 always runs. The assertion this replaces was ``has_restore or
t_elapsed < 0.5``; ``has_restore`` was False on passing runs too, so the test
was decided purely by whether 0.3s of sleep plus ~0.15s of work fitted under
0.5s. It failed ~2 runs in 4 under parallel load and passed on an idle box,
having never once verified a restore.
"""
import pytest
import time

pytestmark = [pytest.mark.upstream, pytest.mark.core]


def _fingerprint(output: str) -> str:
    """Return cell 4's ``CHK ...`` line, or fail loudly if it is absent.

    Comparing two absent fingerprints would compare equal and quietly assert
    nothing -- the exact way the wall-clock assertion this replaces went
    vacuous. An empty or missing line is a test failure, not a match.
    """
    for line in output.splitlines():
        if line.startswith("CHK "):
            return line.strip()
    raise AssertionError(
        f"cell 4 produced no CHK fingerprint line; got:\n{output[:1000]}")


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

            # Fingerprint of everything cell 1's seeded draw fed into: the
            # oracle for "was df reconstructed to the top-to-bottom value?".
            print(
                f"CHK close={df['close'].sum():.6f} "
                f"vol={df['volume'].sum()} "
                f"rm={df['rolling_mean'].sum():.6f} "
                f"vr={df['vol_ratio'].sum():.6f} "
                f"rows={len(df)}")

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
        chk_first = _fingerprint(out4_first)

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
        nb_runner._run_async(
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

        # Cell 5 ran first with df deleted, so it had to rebuild df -- including
        # re-seeding the RNG stream -- before cells 3 and 4 could see it. If any
        # of that reconstruction is wrong (a draw replayed at the wrong stream
        # position, a stale df restored, the sort skipped), these numbers move.
        chk_rerun = _fingerprint(out4_text)
        assert chk_rerun == chk_first, (
            f"out-of-order execution reconstructed a DIFFERENT df.\n"
            f"  top-to-bottom: {chk_first}\n"
            f"  after 5->3->4: {chk_rerun}")

        # Timing is reported for diagnosis only. It is NOT asserted on: cell 4
        # mutates df in place, so its statements re-execute by design and the
        # 0.3s sleep always runs. See the module docstring.
        print(f"(cell 4 recomputed in {t_elapsed:.2f}s -- expected, not asserted)")

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
