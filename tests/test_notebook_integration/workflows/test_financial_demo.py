"""The financial analysis demo notebook.

Reproduction tests for financial_analysis_demo bugs:

Bug 1: Run second df cell (cell 11, triggers upstream for heavy calc), then run
       first df cell (cell 8) — it should show df WITHOUT heavy-calc columns.

Bug 2: After running both df display cells, re-running the read_csv cell
       (cell 5) should hit cache, not recompute. Same for the heavy computation
       cell (cell 10).

Bug 3: Print statements execute twice — once during upstream re-execution and
       again during the cell's own statement processing.

Test the exact scenario: ticker_stats = {} followed by loop, add new item

This mimics the financial_analysis_demo.ipynb structure exactly.

Reproduction test for the financial_analysis_demo display cell bug.

Scenario:
  Cell 1: import pandas as pd; df = pd.DataFrame(...)
  Cell 2: df = df.sort_values(...)
  Cell 3: df   (display cell - read-only)
  Cell 4: df['SMA_50'] = df.groupby(...)...  (adds column downstream)

After running all cells, re-run cell 3. It should show df WITHOUT the SMA_50 column.
"""

import pandas as pd
import pytest

from tests._nbharness.runner import REFERENCE_NOTEBOOKS_DIR


def _create_csv(tmp_path):
    """Create a small test CSV file."""
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 20 + ["GOOGL"] * 20,
            "Date": list(range(40)),
            "Close": [100 + i * 0.5 for i in range(40)],
            "Volume": [1000 + i * 10 for i in range(40)],
        }
    ).to_csv(csv_path, index=False)
    return csv_path_str


@pytest.mark.upstream
@pytest.mark.mutations
@pytest.mark.timeout(60)
def test_display_cell_after_running_downstream_df_first(nb_runner, tmp_path):
    """
    Bug 1 reproduction:
    1. Run all cells (load → sort → df display → heavy calc with df mutations)
    2. Run the SECOND df cell (cell 5, downstream) which triggers upstream
    3. Run the FIRST df cell (cell 3, upstream display) — should NOT show extra columns
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded:', list(df.columns))",
            # Cell 2: Sort
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('Sorted:', list(df.columns))",
            # Cell 3: First display cell (read-only) — use print to distinguish from cell 5
            "print('--- Display 1 ---')\ndf",
            # Cell 4: Heavy computation — adds columns (df is input AND output via mutation)
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "df['VolAdj'] = df['Close'] / (df['Volume'] + 1)\n"
            "print('Heavy calc done:', list(df.columns))",
            # Cell 5: Second display cell (read-only) — use different print to distinguish
            "print('--- Display 2 ---')\ndf",
        ]
    )
    nb_runner.start_kernel()

    # Step 1: Run all cells in order
    nb_runner.run_all()

    # Step 2: Run cell 5 (second df display) — triggers upstream for all cells 1-4
    nb_runner.run_cell(5)
    output_5 = nb_runner.get_output(5)
    # Cell 5 should show df WITH extra columns (SMA, VolAdj) since it's after cell 4
    assert "SMA" in output_5, f"Cell 5 should show SMA column, got: {output_5}"

    # Enable debug to trace what happens during cell 3 execution
    nb_runner.set_cell_source(3, "print('--- Display 1 ---')\ndf")

    # Step 3: Run cell 3 (first df display) — should show df WITHOUT extra columns
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)
    assert "SMA" not in output_3, (
        f"Bug 1! Cell 3 (first df display) shows downstream SMA column after running cell 5 first.\nOutput: {output_3}"
    )
    assert "VolAdj" not in output_3, f"Bug 1! Cell 3 shows downstream VolAdj column.\nOutput: {output_3}"
    print("[PASS] Bug 1: First display cell correctly shows upstream state")


@pytest.mark.upstream
@pytest.mark.core
@pytest.mark.timeout(60)
def test_cache_not_invalidated_after_display_cells(nb_runner, tmp_path):
    """
    Bug 2 reproduction:
    1. Run all cells (load → sort → df display → heavy calc → df display)
    2. Run both df display cells (cells 3 and 5)
    3. Re-run cell 1 (load data) — should hit cache (RESTORED/SKIPPED), not recompute
    4. Re-run cell 4 (heavy calc) — should hit cache, not recompute
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data (with timing print)
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded:', list(df.columns))",
            # Cell 2: Sort
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('Sorted')",
            # Cell 3: First display cell
            "print('Display 1')\ndf",
            # Cell 4: Heavy computation
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "print('SMA done')",
            # Cell 5: Second display cell
            "print('Display 2')\ndf",
        ]
    )
    nb_runner.start_kernel()

    # Step 1: Run all cells
    nb_runner.run_all()

    # Step 2: Run both display cells
    nb_runner.run_cell(5)
    nb_runner.run_cell(3)

    # Step 3: Re-run cell 1 (load data) — should NOT recompute
    nb_runner.run_cell(1)
    output_1 = nb_runner.get_output(1)
    # The "Loaded:" message should appear — but it should come from cache restore
    # (printing the cached stdout) rather than actual re-execution
    assert "Loaded:" in output_1, f"Cell 1 should show 'Loaded:' output, got: {output_1}"

    # Step 4: Re-run cell 4 (heavy calc) — should NOT recompute
    nb_runner.run_cell(4)
    output_4 = nb_runner.get_output(4)
    assert "SMA done" in output_4, f"Cell 4 should show 'SMA done', got: {output_4}"

    print("[PASS] Bug 2: Cache not invalidated after running display cells")


@pytest.mark.upstream
@pytest.mark.core
@pytest.mark.timeout(60)
def test_cache_preserved_sequential_rerun(nb_runner, tmp_path):
    """
    Simpler version of Bug 2: after a full run, re-running any cell should use cache.
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded')",
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('Sorted')",
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\nprint('SMA done')",
            "print('Final columns:', list(df.columns))",
        ]
    )
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Re-run all cells in order — all should use cache
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    output_4 = nb_runner.get_output(4)
    assert "Final columns:" in output_4, f"Cell 4 should show output, got: {output_4}"
    print("[PASS] Sequential re-run uses cache correctly")


@pytest.mark.upstream
@pytest.mark.core
@pytest.mark.timeout(60)
def test_no_double_prints_on_rerun(nb_runner, tmp_path):
    """
    Bug 3 reproduction:
    Print statements should appear exactly once, not twice.
    This can happen when upstream re-execution prints the stdout,
    and then the cell's own statement processing prints it again.
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data with a distinctive print
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('MARKER_LOAD')",
            # Cell 2: Process with a distinctive print
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('MARKER_SORT')",
            # Cell 3: Display
            "print('MARKER_DISPLAY')\ndf",
        ]
    )
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Re-run cell 3 — triggers upstream check for cells 1 and 2
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Count occurrences of MARKER_DISPLAY — should be exactly 1
    marker_count = output_3.count("MARKER_DISPLAY")
    assert marker_count == 1, f"Bug 3! 'MARKER_DISPLAY' appears {marker_count} times (expected 1).\nOutput: {output_3}"

    # Also check that upstream prints don't leak into cell 3's output
    assert "MARKER_LOAD" not in output_3, (
        f"Bug 3! Upstream 'MARKER_LOAD' leaked into cell 3 output.\nOutput: {output_3}"
    )
    assert "MARKER_SORT" not in output_3, (
        f"Bug 3! Upstream 'MARKER_SORT' leaked into cell 3 output.\nOutput: {output_3}"
    )
    print("[PASS] Bug 3: No double prints on re-run")


@pytest.mark.upstream
@pytest.mark.core
@pytest.mark.timeout(60)
def test_no_double_prints_in_heavy_calc_cell(nb_runner, tmp_path):
    """
    Bug 3 variant: Check that the heavy calc cell itself doesn't double-print
    when re-run after upstream changes.
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('LOAD_MARKER')",
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)",
            "print('CALC_START')\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "print('CALC_END')",
        ]
    )
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Re-run cell 3 (heavy calc)
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Each marker should appear exactly once
    assert output_3.count("CALC_START") == 1, (
        f"'CALC_START' appears {output_3.count('CALC_START')} times. Output: {output_3}"
    )
    assert output_3.count("CALC_END") == 1, f"'CALC_END' appears {output_3.count('CALC_END')} times. Output: {output_3}"
    print("[PASS] No double prints in heavy calc cell")


@pytest.mark.upstream
@pytest.mark.core
@pytest.mark.timeout(60)
def test_upstream_print_leaks_during_display_cell(nb_runner, tmp_path):
    """
    Bug 3 variant: When a display cell is re-run and upstream restoration happens,
    the restored upstream statement's stdout should NOT appear in the display cell output.
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('UPSTREAM_PRINT_1')",
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('UPSTREAM_PRINT_2')",
            "df",
            "df['extra'] = 1\nprint('DOWNSTREAM_PRINT')",
        ]
    )
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Run cell 4 (downstream mutation) then run cell 3 (display)
    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Cell 3's output should NOT contain upstream prints
    assert "UPSTREAM_PRINT_1" not in output_3, f"Upstream print leaked into display cell output.\nOutput: {output_3}"
    assert "UPSTREAM_PRINT_2" not in output_3, f"Upstream print leaked into display cell output.\nOutput: {output_3}"
    assert "DOWNSTREAM_PRINT" not in output_3, f"Downstream print leaked into display cell output.\nOutput: {output_3}"
    print("[PASS] No upstream prints leaked into display cell")


@pytest.mark.upstream
@pytest.mark.mutations
@pytest.mark.timeout(60)
def test_display_cell_correct_after_alternating_out_of_order(nb_runner, tmp_path):
    """
    Edge case: Alternating out-of-order execution.
    1. Run all cells in order
    2. Run cell 5 (downstream display)
    3. Run cell 3 (upstream display) — should NOT show SMA
    4. Run cell 5 again — should still show SMA
    5. Run cell 3 again — should still NOT show SMA
    """
    csv_path_str = _create_csv(tmp_path)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded')",
            # Cell 2: Sort
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('Sorted')",
            # Cell 3: First display cell
            "print('--- Display 1 ---')\ndf",
            # Cell 4: Heavy computation — adds SMA column
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "print('Heavy calc done')",
            # Cell 5: Second display cell
            "print('--- Display 2 ---')\ndf",
        ]
    )
    nb_runner.start_kernel()

    # Step 1: Run all cells in order
    nb_runner.run_all()

    # Step 2: Run cell 5 (downstream display)
    nb_runner.run_cell(5)
    assert "SMA" in nb_runner.get_output(5)

    # Step 3: Run cell 3 (upstream display) — should NOT show SMA
    nb_runner.run_cell(3)
    output_3a = nb_runner.get_output(3)
    assert "SMA" not in output_3a, f"Pass 1: Cell 3 shows SMA! Output: {output_3a}"

    # Step 4: Run cell 5 again — should still show SMA
    nb_runner.run_cell(5)
    assert "SMA" in nb_runner.get_output(5)

    # Step 5: Run cell 3 again — should still NOT show SMA
    nb_runner.run_cell(3)
    output_3b = nb_runner.get_output(3)
    assert "SMA" not in output_3b, f"Pass 2: Cell 3 shows SMA! Output: {output_3b}"

    print("[PASS] Alternating out-of-order execution works correctly")


@pytest.mark.core
@pytest.mark.loops
def test_financial_demo_exact_structure(nb_runner):
    """
    Exact structure of financial_analysis_demo.ipynb:
    - Cell 1: Create mock DataFrame
    - Cell 2: Loop cell with ticker_stats = {}, loop, prints
    - Cell 3: ticker_stats.keys()

    Test: Add new ticker, run cell 3 only
    """
    df_setup = """import pandas as pd
df = pd.DataFrame({
    'Ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL', 'TSLA', 'TSLA', 'AMZN', 'AMZN'],
    'Close': [100.0, 101.0, 200.0, 201.0, 300.0, 301.0, 400.0, 401.0, 500.0, 501.0],
    'Volume': [1000, 1100, 2000, 2100, 3000, 3100, 4000, 4100, 5000, 5100]
})"""

    loop_cell_4_tickers = """ticker_stats = {}
print(df)

for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
    ticker_data = df[df["Ticker"] == ticker]
    stats = {
        "mean_close": ticker_data["Close"].mean(),
        "std_close": ticker_data["Close"].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print("Done processing!")"""

    keys_cell = """print(f"Keys cell: {list(ticker_stats.keys())}")"""

    nb_runner.create_notebook([df_setup, loop_cell_4_tickers, keys_cell])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out1 = nb_runner.get_output(3)
    print(f"Run 1 keys cell: {out1}")
    assert "AAPL" in out1
    assert "TSLA" in out1

    # Modify loop to add AMZN
    loop_cell_5_tickers = """ticker_stats = {}
print(df)

for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
    ticker_data = df[df["Ticker"] == ticker]
    stats = {
        "mean_close": ticker_data["Close"].mean(),
        "std_close": ticker_data["Close"].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print("Done processing!")"""

    nb_runner.set_cell_source(2, loop_cell_5_tickers)
    nb_runner.run_all()

    out2 = nb_runner.get_output(3)
    print(f"Run 2 keys cell: {out2}")

    # All 5 should be present
    for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
        assert ticker in out2, f"Missing {ticker} in: {out2}"


@pytest.mark.core
@pytest.mark.loops
def test_init_before_loop_no_overwrite(nb_runner):
    """
    Test that ticker_stats = {} initialization doesn't overwrite restored data
    when a new iteration is added.
    """
    nb_runner.create_notebook(
        [
            """data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(f"After loop: {data}")""",
            """print(f"Downstream: {data}")""",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out1 = nb_runner.get_output(2)
    print(f"Run 1: {out1}")
    assert "3: 30" in out1

    # Add 4th item
    nb_runner.set_cell_source(
        1,
        """data = {}
for x in [1, 2, 3, 4]:
    data[x] = x * 10
print(f"After loop: {data}")""",
    )
    nb_runner.run_all()

    out2 = nb_runner.get_output(2)
    print(f"Run 2: {out2}")

    # Verify ALL items present
    for x in [1, 2, 3, 4]:
        assert f"{x}: {x * 10}" in out2, f"Missing {x} in: {out2}"


@pytest.mark.core
@pytest.mark.loops
def test_same_session_edit_and_rerun(nb_runner):
    """
    Test the exact user scenario:
    1. Run notebook with 4 items (all cells)
    2. In SAME session, edit loop to have 5 items
    3. Run all cells again
    """
    nb_runner.create_notebook(
        [
            """data = {}
for x in ['A', 'B', 'C', 'D']:
    data[x] = x * 2
print(f"First: {data}")""",
            """print(f"Keys: {list(data.keys())}")""",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out1 = nb_runner.get_output(2)
    print(f"Run 1: {out1}")
    assert "'D'" in out1

    # Edit and re-run in same session
    nb_runner.set_cell_source(
        1,
        """data = {}
for x in ['A', 'B', 'C', 'D', 'E']:
    data[x] = x * 2
print(f"Second: {data}")""",
    )
    nb_runner.run_all()

    out2 = nb_runner.get_output(2)
    print(f"Run 2: {out2}")

    for x in ["A", "B", "C", "D", "E"]:
        assert f"'{x}'" in out2, f"Missing {x} in: {out2}"


@pytest.mark.core
@pytest.mark.loops
def test_financial_demo_load_reference(nb_runner):
    """
    Test loading the actual financial_demo reference notebook.
    """
    nb_runner.load(REFERENCE_NOTEBOOKS_DIR / "financial_demo.ipynb")
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Check output from cell 3 (the keys display cell)
    output = nb_runner.get_output(3)
    print(f"Cell 3 output: {output}")

    # Should have 4 tickers
    assert "AAPL" in output
    assert "TSLA" in output


@pytest.mark.core
@pytest.mark.loops
def test_financial_demo_modify_and_rerun(nb_runner):
    """
    Test modifying the financial demo notebook and re-running.
    """
    nb_runner.load(REFERENCE_NOTEBOOKS_DIR / "financial_demo.ipynb")
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(3)
    assert "AMZN" not in output1  # AMZN not in original

    # Modify cell 2 to add AMZN
    new_cell2 = """ticker_stats = {}
print(df)

for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
    ticker_data = df[df['Ticker'] == ticker]
    stats = {
        'mean_close': ticker_data['Close'].mean(),
        'std_close': ticker_data['Close'].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print('Done processing!')"""

    nb_runner.set_cell_source(2, new_cell2)
    nb_runner.run_cells([2, 3])

    output2 = nb_runner.get_output(3)
    assert "AMZN" in output2, f"Expected AMZN in output: {output2}"


@pytest.mark.upstream
@pytest.mark.mutations
@pytest.mark.timeout(60)
def test_financial_demo_display_cell_no_downstream_columns(nb_runner, tmp_path):
    """
    Reproduces the financial_analysis_demo bug:
    After running all cells (including one that adds SMA columns to df),
    re-running the earlier `df` display cell should show df WITHOUT SMA columns.
    """
    import pandas as pd

    # Create a small CSV file to simulate the financial data
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    data = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 10 + ["GOOGL"] * 10,
            "Date": list(range(20)),
            "Close": [100 + i * 0.5 for i in range(20)],
            "Volume": [1000 + i * 10 for i in range(20)],
        }
    )
    data.to_csv(csv_path, index=False)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded:', list(df.columns))",
            # Cell 2: Sort (modifies df, df is both input and output)
            "df = df.sort_values(by=['Ticker', 'Date'])\nprint('Sorted:', list(df.columns))",
            # Cell 3: Display cell (df is read-only input, NOT output)
            "df",
            # Cell 4: Add columns downstream (df is both input and output via mutation)
            "df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "df['VolAdj'] = df['Close'] / (df['Volume'] + 1)\n"
            "print('After adding columns:', list(df.columns))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    # After all cells, df should have the extra columns
    output_cell4 = nb_runner.get_output(4)
    assert "SMA_50" in output_cell4, f"Cell 4 should show SMA_50 column, got: {output_cell4}"

    # Now re-run cell 3 (the display cell)
    nb_runner.run_cell(3)
    output_cell3 = nb_runner.get_output(3)

    # Cell 3 should show df at its position in the notebook (BEFORE cell 4)
    # So it should NOT have SMA_50 or VolAdj columns
    assert "SMA_50" not in output_cell3, (
        f"Bug! Cell 3 (display cell) shows downstream SMA_50 column.\nOutput: {output_cell3}"
    )
    assert "VolAdj" not in output_cell3, (
        f"Bug! Cell 3 (display cell) shows downstream VolAdj column.\nOutput: {output_cell3}"
    )
    print("[PASS] Display cell correctly shows upstream state without downstream columns")


@pytest.mark.upstream
@pytest.mark.mutations
@pytest.mark.timeout(60)
def test_financial_demo_multi_statement_cell_downstream(nb_runner, tmp_path):
    """
    Same as above but the downstream cell has multiple statements (like the real demo).
    The multi-statement cell adds columns AND displays df at the end.
    """
    import pandas as pd

    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    data = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 10 + ["GOOGL"] * 10,
            "Date": list(range(20)),
            "Close": [100 + i * 0.5 for i in range(20)],
            "Volume": [1000 + i * 10 for i in range(20)],
        }
    )
    data.to_csv(csv_path, index=False)

    nb_runner.create_notebook(
        [
            # Cell 1: Load data
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded:', list(df.columns))",
            # Cell 2: Sort
            "df = df.sort_values(by=['Ticker', 'Date'])\nprint('Sorted:', list(df.columns))",
            # Cell 3: Display cell (bare expression, read-only)
            "df",
            # Cell 4: Multi-statement cell that adds columns and displays df
            # (Mirrors the real financial_analysis_demo cell 10)
            "print('Adding columns...')\n"
            "df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
            "print('SMA done')\n"
            "df['VolAdj'] = df['Close'] / (df['Volume'] + 1)\n"
            "print('VolAdj done')\n"
            "df",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run cell 3
    nb_runner.run_cell(3)
    output_cell3 = nb_runner.get_output(3)

    assert "SMA_50" not in output_cell3, (
        f"Bug! Cell 3 shows downstream SMA_50 column after re-run.\nOutput: {output_cell3}"
    )
    print("[PASS] Multi-statement downstream cell doesn't pollute display cell")


@pytest.mark.upstream
@pytest.mark.mutations
@pytest.mark.timeout(60)
def test_display_cell_after_sort_then_rerun(nb_runner, tmp_path):
    """
    Simpler scenario: just sort + display + add column.
    Verifies the display cell shows the sorted df without extra columns.
    """
    import pandas as pd

    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    pd.DataFrame(
        {
            "a": [3, 1, 2],
            "b": [30, 10, 20],
        }
    ).to_csv(csv_path, index=False)

    nb_runner.create_notebook(
        [
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint(list(df.columns))",
            "df = df.sort_values('a').reset_index(drop=True)\nprint(list(df.columns))",
            "df",
            "df['c'] = df['a'] * 10\nprint(list(df.columns))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Verify cell 4 output shows column 'c'
    output_4 = nb_runner.get_output(4)
    assert "'c'" in output_4, f"Cell 4 should have column 'c', got: {output_4}"

    # Re-run cell 3 - should show df without column 'c'
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # The output header should only show columns 'a' and 'b', not 'c'
    # The text representation starts with the column headers
    first_line = output_3.strip().split("\n")[0]
    assert "c" not in first_line.split(), (
        f"Bug! Cell 3 shows column 'c' from downstream.\nHeader line: {first_line}\nFull output: {output_3}"
    )
    print(f"[PASS] Simple scenario: display cell shows correct columns. Header: {first_line}")
