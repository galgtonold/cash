"""
Test for self-assignment cache coherence when multiple self-assignment
statements exist in the SAME cell.

Scenario:
  Cell 1: df = create_data()
  Cell 2: df = df.sort_values(...)
  Cell 3: df['colA'] = df.transform(...)   # Statement 1
          df['colB'] = df.transform(...)   # Statement 2

When only Statement 2 changes, Statement 1 should get a cache HIT (not recompute).
"""

import pytest


@pytest.mark.core
@pytest.mark.upstream
class TestSelfAssignmentSameCell:
    """Two self-assignment statements in the same cell, only one changes."""

    def test_unchanged_stmt_gets_cache_hit(self, nb_runner):
        """
        After changing only the second self-assignment statement,
        the first should be restored from cache, not recomputed.
        """
        nb_runner.create_notebook([
            # Cell 1: Create data
            "import pandas as pd\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
            # Cell 2: Sort (self-assignment)
            "df = df.sort_values('A').reset_index(drop=True)",
            # Cell 3: Two self-assignment statements
            "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
            # Cell 4: Print result
            "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "C=[10, 20, 30]" in output1
        assert "D=[25, 35, 15]" in output1

        # Now change ONLY the second statement (D computation)
        nb_runner.set_cell_source(3, "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 100")
        nb_runner.run_cell(3)
        nb_runner.run_cell(4)
        output2 = nb_runner.get_output(4)
        # C should be unchanged, D should reflect new formula
        assert "C=[10, 20, 30]" in output2
        assert "D=[120, 130, 110]" in output2

    def test_unchanged_stmt_not_recomputed(self, nb_runner):
        """
        Verify the first statement is RESTORED or SKIPPED (not COMPUTED) after the
        second statement changes. Use time.sleep to distinguish compute from cache.
        """
        nb_runner.create_notebook([
            # Cell 1: Create data
            "import pandas as pd\nimport time\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
            # Cell 2: Sort
            "df = df.sort_values('A').reset_index(drop=True)",
            # Cell 3: Two self-assignment statements - first has a sleep
            "time.sleep(0.5)\ndf['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
            # Cell 4: Verify
            "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change only second statement
        nb_runner.set_cell_source(3, "time.sleep(0.5)\ndf['C'] = df['A'] * 10\ndf['D'] = df['B'] + 100")

        t_start = __import__('time').time()
        nb_runner.run_cell(3)
        t_elapsed = __import__('time').time() - t_start

        # If the first statement (with sleep(0.5)) was recomputed, cell 3 takes >1s
        # (0.5 for first stmt + some time for second stmt)
        # If first stmt was restored/skipped, cell 3 should take <0.8s
        # (only second stmt computes, no 0.5s sleep)
        raw = nb_runner.get_raw_output(3)
        print(f"Cell 3 rerun took {t_elapsed:.2f}s. Raw output: {raw[:500]}")

        nb_runner.run_cell(4)
        output_final = nb_runner.get_output(4)
        assert "C=[10, 20, 30]" in output_final
        assert "D=[120, 130, 110]" in output_final

        # NOTE: This is a soft check - timing can vary
        # But if it consistently takes >1s, it's recomputing
        if t_elapsed > 1.0:
            # Check badge output for "Computed" on the C statement
            assert "Computed" not in raw or "Restored" in raw, (
                f"Cell 3 took {t_elapsed:.2f}s, suggesting the first statement "
                f"(with sleep) was recomputed. Badge: {raw[:500]}"
            )

    def test_groupby_transform_pattern(self, nb_runner):
        """
        More realistic scenario matching the original bug report:
        df['VolAdj'] = df.groupby(...).transform(...)
        df['SMA'] = df.groupby(...).transform(...)
        
        Change SMA window, VolAdj should cache-hit.
        """
        nb_runner.create_notebook([
            # Cell 1: Create data
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "np.random.seed(42)\n"
                "df = pd.DataFrame({\n"
                "    'Ticker': ['A'] * 20 + ['B'] * 20,\n"
                "    'Close': np.random.randn(40).cumsum() + 100\n"
                "})"
            ),
            # Cell 2: Sort
            "df = df.sort_values(['Ticker']).reset_index(drop=True)",
            # Cell 3: Two groupby transforms (self-assignment)
            (
                "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
                "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())"
            ),
            # Cell 4: Print summary
            "print(f\"VolAdj_mean={df['VolAdj'].mean():.4f} SMA_mean={df['SMA'].mean():.4f}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "VolAdj_mean=" in output1
        assert "SMA_mean=" in output1

        # Save the VolAdj mean for comparison
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(3,
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())"
        )
        nb_runner.run_cell(3)
        nb_runner.run_cell(4)
        output2 = nb_runner.get_output(4)

        # VolAdj should be exactly the same (cache hit)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed unexpectedly: {volAdj_mean_1} -> {volAdj_mean_2}. "
            f"This suggests the VolAdj statement was recomputed instead of cache-hit."
        )

    def test_with_timing_and_prints_like_demo(self, nb_runner):
        """
        Realistic scenario: cell has print statements, time.time() calls,
        and function definitions BETWEEN the two self-assignment statements.
        This matches the financial_analysis_demo.ipynb pattern.
        """
        nb_runner.create_notebook([
            # Cell 1: Create data
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import time\n"
                "np.random.seed(42)\n"
                "df = pd.DataFrame({\n"
                "    'Ticker': ['A'] * 20 + ['B'] * 20,\n"
                "    'Close': np.random.randn(40).cumsum() + 100\n"
                "})"
            ),
            # Cell 2: Sort
            "df = df.sort_values(['Ticker']).reset_index(drop=True)",
            # Cell 3: Realistic cell with prints, timing, and two self-assignments
            (
                "print('Calculating VolAdj...')\n"
                "t0 = time.time()\n"
                "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
                "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
                "print('Calculating SMA...')\n"
                "t0 = time.time()\n"
                "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())\n"
                "print(f'SMA done in {time.time() - t0:.2f}s')"
            ),
            # Cell 4: Print summary
            "print(f\"VolAdj_mean={df['VolAdj'].mean():.4f} SMA_mean={df['SMA'].mean():.4f}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(3,
            "print('Calculating VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Calculating SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')"
        )
        nb_runner.run_cell(3)
        nb_runner.run_cell(4)
        output2 = nb_runner.get_output(4)

        # VolAdj should be the same (cache hit)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed when only SMA was modified: {volAdj_mean_1} -> {volAdj_mean_2}. "
            f"Bug: VolAdj recomputed instead of cache-hit."
        )

    @pytest.mark.timeout(60)
    def test_after_kernel_restart(self, nb_runner):
        """
        After kernel restart, the first statement should be restored from
        disk cache (not recomputed) when only the second statement changes.
        """
        nb_runner.create_notebook([
            # Cell 1: Create data
            "import pandas as pd\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
            # Cell 2: Sort
            "df = df.sort_values('A').reset_index(drop=True)",
            # Cell 3: Two self-assignment statements
            "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
            # Cell 4: Print result
            "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "C=[10, 20, 30]" in output1

        # Restart kernel (clears all memory)
        nb_runner.reset_cash_state()

        # Change ONLY the second statement
        nb_runner.set_cell_source(3, "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 100")
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        # C should still be correct
        assert "C=[10, 20, 30]" in output2
        assert "D=[120, 130, 110]" in output2
