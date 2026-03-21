"""
Reproduction tests for financial_analysis_demo bugs:

Bug 1: Run second df cell (cell 11, triggers upstream for heavy calc), then run
       first df cell (cell 8) — it should show df WITHOUT heavy-calc columns.

Bug 2: After running both df display cells, re-running the read_csv cell
       (cell 5) should hit cache, not recompute. Same for the heavy computation
       cell (cell 10).

Bug 3: Print statements execute twice — once during upstream re-execution and
       again during the cell's own statement processing.
"""
import pytest
import pandas as pd



def _create_csv(tmp_path):
    """Create a small test CSV file."""
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({
        'Ticker': ['AAPL'] * 20 + ['GOOGL'] * 20,
        'Date': list(range(40)),
        'Close': [100 + i * 0.5 for i in range(40)],
        'Volume': [1000 + i * 10 for i in range(40)],
    }).to_csv(csv_path, index=False)
    return csv_path_str


# ─── Bug 1: Display cell after running downstream df cell first ────────────────

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

    nb_runner.create_notebook([
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
    ])
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
        f"Bug 1! Cell 3 (first df display) shows downstream SMA column after "
        f"running cell 5 first.\nOutput: {output_3}"
    )
    assert "VolAdj" not in output_3, (
        f"Bug 1! Cell 3 shows downstream VolAdj column.\nOutput: {output_3}"
    )
    print("[PASS] Bug 1: First display cell correctly shows upstream state")


# ─── Bug 2: Cache invalidation after running display cells ─────────────────────

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

    nb_runner.create_notebook([
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
    ])
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

    nb_runner.create_notebook([
        f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('Loaded')",
        "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('Sorted')",
        "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\nprint('SMA done')",
        "print('Final columns:', list(df.columns))",
    ])
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


# ─── Bug 3: Double print statements ────────────────────────────────────────────

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

    nb_runner.create_notebook([
        # Cell 1: Load data with a distinctive print
        f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('MARKER_LOAD')",
        # Cell 2: Process with a distinctive print
        "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('MARKER_SORT')",
        # Cell 3: Display
        "print('MARKER_DISPLAY')\ndf",
    ])
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Re-run cell 3 — triggers upstream check for cells 1 and 2
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Count occurrences of MARKER_DISPLAY — should be exactly 1
    marker_count = output_3.count('MARKER_DISPLAY')
    assert marker_count == 1, (
        f"Bug 3! 'MARKER_DISPLAY' appears {marker_count} times (expected 1).\n"
        f"Output: {output_3}"
    )

    # Also check that upstream prints don't leak into cell 3's output
    assert "MARKER_LOAD" not in output_3, (
        f"Bug 3! Upstream 'MARKER_LOAD' leaked into cell 3 output.\n"
        f"Output: {output_3}"
    )
    assert "MARKER_SORT" not in output_3, (
        f"Bug 3! Upstream 'MARKER_SORT' leaked into cell 3 output.\n"
        f"Output: {output_3}"
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

    nb_runner.create_notebook([
        f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('LOAD_MARKER')",
        "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)",
        "print('CALC_START')\n"
        "df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())\n"
        "print('CALC_END')",
    ])
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Re-run cell 3 (heavy calc)
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Each marker should appear exactly once
    assert output_3.count('CALC_START') == 1, (
        f"'CALC_START' appears {output_3.count('CALC_START')} times. Output: {output_3}"
    )
    assert output_3.count('CALC_END') == 1, (
        f"'CALC_END' appears {output_3.count('CALC_END')} times. Output: {output_3}"
    )
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

    nb_runner.create_notebook([
        f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('UPSTREAM_PRINT_1')",
        "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)\nprint('UPSTREAM_PRINT_2')",
        "df",
        "df['extra'] = 1\nprint('DOWNSTREAM_PRINT')",
    ])
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Run cell 4 (downstream mutation) then run cell 3 (display)
    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    output_3 = nb_runner.get_output(3)

    # Cell 3's output should NOT contain upstream prints
    assert "UPSTREAM_PRINT_1" not in output_3, (
        f"Upstream print leaked into display cell output.\nOutput: {output_3}"
    )
    assert "UPSTREAM_PRINT_2" not in output_3, (
        f"Upstream print leaked into display cell output.\nOutput: {output_3}"
    )
    assert "DOWNSTREAM_PRINT" not in output_3, (
        f"Downstream print leaked into display cell output.\nOutput: {output_3}"
    )
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

    nb_runner.create_notebook([
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
    ])
    nb_runner.start_kernel()

    # Step 1: Run all cells in order
    nb_runner.run_all()

    # Step 2: Run cell 5 (downstream display)
    nb_runner.run_cell(5)
    assert "SMA" in nb_runner.get_output(5)

    # Step 3: Run cell 3 (upstream display) — should NOT show SMA
    nb_runner.run_cell(3)
    output_3a = nb_runner.get_output(3)
    assert "SMA" not in output_3a, f"Round 1: Cell 3 shows SMA! Output: {output_3a}"

    # Step 4: Run cell 5 again — should still show SMA
    nb_runner.run_cell(5)
    assert "SMA" in nb_runner.get_output(5)

    # Step 5: Run cell 3 again — should still NOT show SMA
    nb_runner.run_cell(3)
    output_3b = nb_runner.get_output(3)
    assert "SMA" not in output_3b, f"Round 2: Cell 3 shows SMA! Output: {output_3b}"

    print("[PASS] Alternating out-of-order execution works correctly")
