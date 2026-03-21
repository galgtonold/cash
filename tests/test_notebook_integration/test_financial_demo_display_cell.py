"""
Reproduction test for the financial_analysis_demo display cell bug.

Scenario:
  Cell 1: import pandas as pd; df = pd.DataFrame(...)
  Cell 2: df = df.sort_values(...)
  Cell 3: df   (display cell - read-only)
  Cell 4: df['SMA_50'] = df.groupby(...)...  (adds column downstream)

After running all cells, re-run cell 3. It should show df WITHOUT the SMA_50 column.
"""
import pytest



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
    csv_path_str = str(csv_path).replace('\\', '/')
    data = pd.DataFrame({
        'Ticker': ['AAPL'] * 10 + ['GOOGL'] * 10,
        'Date': list(range(20)),
        'Close': [100 + i * 0.5 for i in range(20)],
        'Volume': [1000 + i * 10 for i in range(20)],
    })
    data.to_csv(csv_path, index=False)

    nb_runner.create_notebook([
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
    ])
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
        f"Bug! Cell 3 (display cell) shows downstream SMA_50 column.\n"
        f"Output: {output_cell3}"
    )
    assert "VolAdj" not in output_cell3, (
        f"Bug! Cell 3 (display cell) shows downstream VolAdj column.\n"
        f"Output: {output_cell3}"
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
    csv_path_str = str(csv_path).replace('\\', '/')
    data = pd.DataFrame({
        'Ticker': ['AAPL'] * 10 + ['GOOGL'] * 10,
        'Date': list(range(20)),
        'Close': [100 + i * 0.5 for i in range(20)],
        'Volume': [1000 + i * 10 for i in range(20)],
    })
    data.to_csv(csv_path, index=False)

    nb_runner.create_notebook([
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
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run cell 3
    nb_runner.run_cell(3)
    output_cell3 = nb_runner.get_output(3)

    assert "SMA_50" not in output_cell3, (
        f"Bug! Cell 3 shows downstream SMA_50 column after re-run.\n"
        f"Output: {output_cell3}"
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
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({
        'a': [3, 1, 2],
        'b': [30, 10, 20],
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint(list(df.columns))",
        "df = df.sort_values('a').reset_index(drop=True)\nprint(list(df.columns))",
        "df",
        "df['c'] = df['a'] * 10\nprint(list(df.columns))",
    ])
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
    first_line = output_3.strip().split('\n')[0]
    assert 'c' not in first_line.split(), (
        f"Bug! Cell 3 shows column 'c' from downstream.\n"
        f"Header line: {first_line}\n"
        f"Full output: {output_3}"
    )
    print(f"[PASS] Simple scenario: display cell shows correct columns. Header: {first_line}")
