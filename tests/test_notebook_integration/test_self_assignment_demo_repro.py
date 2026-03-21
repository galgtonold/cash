"""
Reproduce the VolAdj recomputation bug from the financial_analysis_demo notebook.

Exact notebook structure:
  Cell 1: imports
  Cell 2: %cash_on
  Cell 3: load CSV data  (df = pd.read_csv(...))
  Cell 4: df = df.sort_values(...)
  Cell 5: df  (bare expression)
  Cell 6: print + t0 + df['VolAdj'] = ... + print + t0 + def func + df['SMA'] = ... + print + df

When only the SMA window changes, VolAdj should be a cache hit.
"""

import pytest


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
        nb_runner.create_notebook([
            # Cell 1: imports
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import time"
            ),
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        assert "VolAdj_mean=" in output1
        assert "SMA_mean=" in output1

        # Save VolAdj mean for comparison
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]
        sma_mean_1 = output1.split("SMA_mean=")[1].strip()

        # Change ONLY the SMA window (10 -> 3)
        nb_runner.set_cell_source(5,
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
            "df"
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
        nb_runner.create_notebook([
            # Cell 1: imports
            (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import time"
            ),
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
                "df = df.sort_values(by=['Ticker'])"
            ),
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
        ])
        nb_runner.start_kernel()

        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Debug is now on (cell 7 ran). Change only SMA window.
        nb_runner.set_cell_source(5,
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())"
        )
        nb_runner.run_cell(5)

        # Get debug output
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT:\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed: {volAdj_mean_1} -> {volAdj_mean_2}.\n"
            f"Debug output:\n{raw}"
        )

    def test_volAdj_with_intermediate_print_and_time(self, nb_runner):
        """
        Test with print and time.time() interleaved - these statements
        have skip_cache=True due to time.time() being a forbidden function.
        Ensure they don't interfere with df lineage.
        """
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(5)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change only SMA window
        nb_runner.set_cell_source(4,
            "print('Computing VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Computing SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')"
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
        for ticker in ['AAPL'] * 30 + ['GOOGL'] * 30:
            data.append([ticker, 100.0 + _np.random.randn()])
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Ticker', 'Close'])
            writer.writerows(data)

        csv_path_str = str(csv_file).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change only SMA window (debug is on from cell 7)
        nb_runner.set_cell_source(5,
            "print('Computing VolAdj...')\n"
            "t0 = time.time()\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())\n"
            "print(f'VolAdj done in {time.time() - t0:.2f}s')\n"
            "print('Computing SMA...')\n"
            "t0 = time.time()\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())\n"
            "print(f'SMA done in {time.time() - t0:.2f}s')\n"
            "df"
        )
        nb_runner.run_cell(5)
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT (CSV test):\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)

        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed with CSV file: {volAdj_mean_1} -> {volAdj_mean_2}.\n"
            f"Debug output:\n{raw}"
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
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(6)
        assert "VolAdj_mean=" in output1
        volAdj_mean_1 = output1.split("VolAdj_mean=")[1].split(" ")[0]

        # Change SMA window from 10 to 3
        nb_runner.set_cell_source(5,
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
            "df"
        )
        nb_runner.run_cell(5)
        raw = nb_runner.get_raw_output(5)
        print(f"DEBUG OUTPUT (exact structure test):\n{raw}")

        nb_runner.run_cell(6)
        output2 = nb_runner.get_output(6)
        volAdj_mean_2 = output2.split("VolAdj_mean=")[1].split(" ")[0]
        assert volAdj_mean_1 == volAdj_mean_2, (
            f"VolAdj changed: {volAdj_mean_1} -> {volAdj_mean_2}.\n"
            f"Debug:\n{raw}"
        )
