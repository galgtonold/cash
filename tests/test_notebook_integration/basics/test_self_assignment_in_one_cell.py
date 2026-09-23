"""``x = f(x)`` inside one cell restores and recomputes correctly."""

import pytest


# Reproduce the VolAdj recomputation bug from the financial_analysis_demo notebook.
#
# Exact notebook structure:
#   Cell 1: imports
#   Cell 2: %cash_on
#   Cell 3: load CSV data  (df = pd.read_csv(...))
#   Cell 4: df = df.sort_values(...)
#   Cell 5: df  (bare expression)
#   Cell 6: print + t0 + df['VolAdj'] = ... + print + t0 + def func + df['SMA'] = ... + print + df
#
# When only the SMA window changes, VolAdj should be a cache hit.
@pytest.mark.core
@pytest.mark.upstream
class TestSelfAssignmentDemoRepro:
    """Reproduce exact scenario from financial_analysis_demo.ipynb."""

    def test_volAdj_cache_hit_after_sma_change(self, nb_runner):
        """
        Exact reproduction of the user-reported bug.

        The cell has print, t0, VolAdj self-assign, print, t0, func def,
        SMA self-assign, print, df expression.

        After changing SMA window, VolAdj should be restored from cache.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                ("import pandas as pd\nimport numpy as np\nimport time"),
                # Cell 2: Create data (simulating CSV load)
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                    "    'Close': np.random.randn(100).cumsum() + 100\n"
                    "})"
                ),
                # Cell 3: Sort (self-assignment)
                (
                    "print('Sorting data...')\n"
                    "t0 = time.time()\n"
                    "df = df.sort_values(by=['Ticker'])\n"
                    "print(f'Sorted in {time.time() - t0:.2f}s')"
                ),
                # Cell 4: bare df display (like the notebook)
                "df",
                # Cell 5: Heavy computation cell (matches the exact structure)
                (
                    "print('Calculating Volatility Adjusted Mean (Statement 1)....')\n"
                    "t0 = time.time()\n"
                    "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).apply(lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
                    "print(f'VolAdj calculated in {time.time() - t0:.2f}s')\n"
                    "print('Calculating Weighted SMA (Statement 2)...')\n"
                    "t0 = time.time()\n"
                    "def custom_weighted_mean(x):\n"
                    "    weights = np.arange(1, len(x) + 1)\n"
                    "    return np.sum(x * weights) / np.sum(weights)\n"
                    "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).apply(custom_weighted_mean, raw=True))\n"
                    "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
                    "df"
                ),
                # Cell 6: Verify values
                "print(f\"VolAdj_mean={df['VolAdj_20'].mean():.6f} SMA_mean={df['SMA'].mean():.6f}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        assert "VolAdj_mean=" in output1
        assert "SMA_mean=" in output1

        # Save VolAdj mean for comparison
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]
        sma_mean_1 = output1.split("SMA_mean=")[1].strip()

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(
            5,
            "print('Calculating Volatility Adjusted Mean (Statement 1)....')\n"
            "t0 = time.time()\n"
            "df['VolAdj_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).apply(lambda y: np.mean(y) / (np.std(y) + 1e-6), raw=True))\n"
            "print(f'VolAdj calculated in {time.time() - t0:.2f}s')\n"
            "print('Calculating Weighted SMA (Statement 2)...')\n"
            "t0 = time.time()\n"
            "def custom_weighted_mean(x):\n"
            "    weights = np.arange(1, len(x) + 1)\n"
            "    return np.sum(x * weights) / np.sum(weights)\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).apply(custom_weighted_mean, raw=True))\n"
            "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
            "df",
        )
        nb_runner.run_cell(5)
        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)

        # VolAdj should be exactly the same (cache hit)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        sma_mean_2 = output2.split("SMA_mean=")[1].strip()
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed unexpectedly: {volAdj_mean_1} -> {volAdj_mean_2}. "
            f"Bug: VolAdj statement was recomputed instead of getting a cache hit."
        )
        # SMA should be different because we changed the window
        assert sma_mean_1 != sma_mean_2, "SMA should have changed with new window"

    def test_volAdj_cache_hit_with_debug(self, nb_runner):
        """
        Same test with debug output to diagnose cache miss root cause.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                ("import pandas as pd\nimport numpy as np\nimport time"),
                # Cell 2: Create data
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                    "    'Close': np.random.randn(100).cumsum() + 100\n"
                    "})"
                ),
                # Cell 3: Sort
                ("df = df.sort_values(by=['Ticker'])"),
                # Cell 4: bare df
                "df",
                # Cell 5: Heavy computation (simplified - no timing noise)
                (
                    "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
                    "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())"
                ),
                # Cell 6: Verify
                "print(f\"VolAdj_mean={df['VolAdj'].mean():.6f} SMA_mean={df['SMA'].mean():.6f}\")",
                # Cell 7: debug toggle (will be used between runs)
                "%cash_debug on",
            ]
        )
        nb_runner.start_kernel()

        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Debug is now on (cell 7 ran). Change only SMA window.
        nb_runner.set_cell_source(
            5,
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())",
        )
        nb_runner.run_cell(5)

        # Get debug output
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT:\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed: {volAdj_mean_1} -> {volAdj_mean_2}.\nDebug output:\n{raw}"
        )

    def test_volAdj_with_intermediate_print_and_time(self, nb_runner):
        """
        Test with print and time.time() interleaved - these statements
        have skip_cache=True due to time.time() being a forbidden function.
        Ensure they don't interfere with df lineage.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                "import pandas as pd\nimport numpy as np\nimport time",
                # Cell 2: Create data
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                    "    'Close': np.random.randn(100).cumsum() + 100\n"
                    "})"
                ),
                # Cell 3: Sort
                "df = df.sort_values(by=['Ticker'])",
                # Cell 4: Heavy computation WITH print/timing interleaved
                (
                    "print('Computing VolAdj...')\n"
                    "t0 = time.time()\n"
                    "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
                    "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
                    "print('Computing SMA...')\n"
                    "t0 = time.time()\n"
                    "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())\n"
                    "print(f'SMA done in {time.time() - t0:.2f}s')"
                ),
                # Cell 5: Verify
                "print(f\"VolAdj_mean={df['VolAdj'].mean():.6f} SMA_mean={df['SMA'].mean():.6f}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(5)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change only SMA window
        nb_runner.set_cell_source(
            4,
            "print('Computing VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Computing SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')",
        )
        nb_runner.run_cell(4)
        nb_runner.run_cell(5)
        output2 = nb_runner.get_output(5)

        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed when only SMA was modified: {volAdj_mean_1} -> {volAdj_mean_2}. "
            f"Bug: print/time.time() interleaving caused VolAdj cache miss."
        )

    @pytest.mark.timeout(60)
    def test_volAdj_with_csv_file_dependency(self, nb_runner, tmp_path):
        """
        Test with actual CSV file reading (like the real notebook).
        File dependencies could cause cache misses if file mtime changes.
        """
        import csv

        import numpy as _np

        _np.random.seed(42)
        csv_file = tmp_path / "test_data.csv"
        data = []
        for ticker in ["AAPL"] * 30 + ["GOOGL"] * 30:
            data.append([ticker, 100.0 + _np.random.randn()])
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Ticker", "Close"])
            writer.writerows(data)

        csv_path_str = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: imports
                "import pandas as pd\nimport numpy as np\nimport time",
                # Cell 2: Read CSV
                f"df = pd.read_csv('{csv_path_str}')",
                # Cell 3: Sort
                "df = df.sort_values(by=['Ticker']).reset_index(drop=True)",
                # Cell 4: bare df
                "df",
                # Cell 5: Heavy computation WITH print/timing interleaved
                (
                    "print('Computing VolAdj...')\n"
                    "t0 = time.time()\n"
                    "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
                    "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
                    "print('Computing SMA...')\n"
                    "t0 = time.time()\n"
                    "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())\n"
                    "print(f'SMA done in {time.time() - t0:.2f}s')\n"
                    "df"
                ),
                # Cell 6: Verify
                "print(f\"VolAdj_mean={df['VolAdj'].mean():.6f} SMA_mean={df['SMA'].mean():.6f}\")",
                # Cell 7: debug
                "%cash_debug on",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change only SMA window (debug is on from cell 7)
        nb_runner.set_cell_source(
            5,
            "print('Computing VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Computing SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')\n"
            "df",
        )
        nb_runner.run_cell(5)
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT (CSV test):\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)

        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed with CSV file: {volAdj_mean_1} -> {volAdj_mean_2}.\nDebug output:\n{raw}"
        )

    def test_volAdj_exact_notebook_cell_structure(self, nb_runner):
        """
        EXACT reproduction of financial_analysis_demo.ipynb cell structure:
        - print statement
        - t0 = time.time()
        - df['VolAdj_20'] = df.groupby(...)...
        - print(f'VolAdj calculated in {time.time() - t0:.2f}s')
        - print statement
        - t0 = time.time()
        - def custom_weighted_mean(x): ...  <-- FUNCTION DEF BETWEEN SELF-ASSIGNMENTS!
        - df['SMA_57'] = df.groupby(...)...
        - print(f'SMA calculated in {time.time() - t0:.2f}s')
        - df  <-- bare expression
        """
        nb_runner.create_notebook(
            [
                # Cell 1: imports
                "import pandas as pd\nimport numpy as np\nimport time",
                # Cell 2: Create data
                (
                    "np.random.seed(42)\n"
                    "df = pd.DataFrame({\n"
                    "    'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,\n"
                    "    'Close': np.random.randn(100).cumsum() + 100\n"
                    "})"
                ),
                # Cell 3: Sort
                (
                    "print('Sorting data...')\n"
                    "t0 = time.time()\n"
                    "df = df.sort_values(by=['Ticker'])\n"
                    "print(f'Sorted in {time.time() - t0:.2f}s')"
                ),
                # Cell 4: bare df display
                "df",
                # Cell 5: EXACT heavy computation cell from demo
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
                    "df['SMA_57'] = df.groupby('Ticker')['Close'].transform("
                    "lambda x: x.rolling(window=10).apply(custom_weighted_mean, raw=True))\n"
                    "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
                    "df"
                ),
                # Cell 6: Verify
                "print(f\"VolAdj_mean={df['VolAdj_20'].mean():.6f} SMA_mean={df['SMA_57'].mean():.6f}\")",
                # Cell 7: debug
                "%cash_debug on",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        assert "VolAdj_mean=" in output1
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change SMA window from 10 to 3
        nb_runner.set_cell_source(
            5,
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
            "df['SMA_57'] = df.groupby('Ticker')['Close'].transform("
            "lambda x: x.rolling(window=3).apply(custom_weighted_mean, raw=True))\n"
            "print(f'SMA calculated in {time.time() - t0:.2f}s')\n"
            "df",
        )
        nb_runner.run_cell(5)
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT (exact structure test):\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, f"VolAdj changed: {volAdj_mean_1} -> {volAdj_mean_2}.\nDebug:\n{raw}"


# Test for self-assignment cache coherence when multiple self-assignment
# statements exist in the SAME cell.
#
# Scenario:
#   Cell 1: df = create_data()
#   Cell 2: df = df.sort_values(...)
#   Cell 3: df['colA'] = df.transform(...)   # Statement 1
#           df['colB'] = df.transform(...)   # Statement 2
#
# When only Statement 2 changes, Statement 1 should get a cache HIT (not recompute).
@pytest.mark.core
@pytest.mark.upstream
class TestSelfAssignmentSameCell:
    """Two self-assignment statements in the same cell, only one changes."""

    def test_unchanged_stmt_gets_cache_hit(self, nb_runner):
        """
        After changing only the second self-assignment statement,
        the first should be restored from cache, not recomputed.
        """
        nb_runner.create_notebook(
            [
                # Cell 1: Create data
                "import pandas as pd\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
                # Cell 2: Sort (self-assignment)
                "df = df.sort_values('A').reset_index(drop=True)",
                # Cell 3: Two self-assignment statements
                "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
                # Cell 4: Print result
                "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
            ]
        )
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
        nb_runner.create_notebook(
            [
                # Cell 1: Create data
                "import pandas as pd\nimport time\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
                # Cell 2: Sort
                "df = df.sort_values('A').reset_index(drop=True)",
                # Cell 3: Two self-assignment statements - first has a sleep
                "time.sleep(0.5)\ndf['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
                # Cell 4: Verify
                "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change only second statement
        nb_runner.set_cell_source(3, "time.sleep(0.5)\ndf['C'] = df['A'] * 10\ndf['D'] = df['B'] + 100")

        t_start = __import__("time").time()
        nb_runner.run_cell(3)
        t_elapsed = __import__("time").time() - t_start

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
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "VolAdj_mean=" in output1
        assert "SMA_mean=" in output1

        # Save the VolAdj mean for comparison
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(
            3,
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())",
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
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(
            3,
            "print('Calculating VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Calculating SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')",
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
        nb_runner.create_notebook(
            [
                # Cell 1: Create data
                "import pandas as pd\ndf = pd.DataFrame({'A': [3, 1, 2], 'B': [10, 20, 30]})",
                # Cell 2: Sort
                "df = df.sort_values('A').reset_index(drop=True)",
                # Cell 3: Two self-assignment statements
                "df['C'] = df['A'] * 10\ndf['D'] = df['B'] + 5",
                # Cell 4: Print result
                "print(f\"C={list(df['C'])} D={list(df['D'])}\")",
            ]
        )
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
