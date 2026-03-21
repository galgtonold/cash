"""
Test that display cells show the correct state of a variable, not a
downstream-mutated version.

Bug: When a cell does `df` (bare expression display) and a later cell has
mutated `df` in place (e.g., `df['SMA'] = ...`), re-running the display
cell would show the mutated DataFrame with SMA columns instead of the state
at the display cell's position in the notebook.

Root cause: The upstream checker's "valid extension" logic accepted the
downstream-mutated lineage for required inputs that were NOT also outputs
of the current cell. This meant a read-only display cell (bare `df`) would
accept `df` with extra columns added by a downstream cell, instead of
restoring `df` to its upstream state.

Fix: When a variable is a required input but NOT also an output of the
current cell, reject any "valid downstream extension" and force restoration
to the upstream-simulated state.
"""
import pytest
import pandas as pd

pytestmark = [pytest.mark.upstream, pytest.mark.mutations]


def test_display_cell_shows_upstream_state_not_downstream(nb_runner, tmp_path):
    """
    Test that a bare expression display cell shows the variable state at
    its position in the notebook, not a downstream-mutated version.

    Scenario:
    1. Cell 1: import pandas
    2. Cell 2: df = pd.DataFrame({'A': [1, 2, 3]})
    3. Cell 3: display df (bare expression) — should show only column A
    4. Cell 4: df['B'] = df['A'] * 10  (in-place mutation)
    5. Re-run Cell 3 → should STILL show only column A, not A+B
    """
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    pd.DataFrame({"A": [1, 2, 3], "Close": [100, 200, 300]}).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        # Cell 1: imports
        "import pandas as pd",
        # Cell 2: load data
        f"df = pd.read_csv('{csv_path_str}')\nprint(list(df.columns))",
        # Cell 3: display cell (bare expression + print)
        "print(f'Columns at cell 3: {list(df.columns)}')",
        # Cell 4: mutate df by adding a new column
        "df['SMA'] = df['Close'].rolling(2).mean()\nprint(f'Columns at cell 4: {list(df.columns)}')",
    ])
    nb_runner.start_kernel()

    # Run all cells in order
    nb_runner.run_all()

    out2 = nb_runner.get_output(2)
    assert "A" in out2 and "Close" in out2, f"Cell 2 should show A, Close: {out2}"

    out3 = nb_runner.get_output(3)
    assert "SMA" not in out3, f"Cell 3 should NOT have SMA column on first run: {out3}"

    out4 = nb_runner.get_output(4)
    assert "SMA" in out4, f"Cell 4 should have SMA column: {out4}"

    # Now re-run cell 3 — should restore df to pre-cell-4 state
    nb_runner.run_cell(3)
    out3_rerun = nb_runner.get_output(3)
    assert "SMA" not in out3_rerun, (
        f"Cell 3 re-run should NOT show SMA column (downstream mutation). "
        f"Got: {out3_rerun}"
    )


def test_display_cell_correct_after_multiple_downstream_mutations(nb_runner, tmp_path):
    """
    Test with multiple downstream mutations — the display cell should
    still show the state at its notebook position.
    """
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOGL"],
        "Close": [150.0, 300.0, 2800.0]
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        # Cell 1: imports
        "import pandas as pd\nimport numpy as np",
        # Cell 2: load
        f"df = pd.read_csv('{csv_path_str}')\nprint(f'Loaded columns: {{list(df.columns)}}')",
        # Cell 3: sort (df = df.sort_values, self-assignment)
        "df = df.sort_values('Ticker')\nprint(f'Sorted columns: {{list(df.columns)}}')",
        # Cell 4: display
        "print(f'Display columns: {{list(df.columns)}}')",
        # Cell 5: add column 1
        "df['VolAdj'] = df['Close'] * 1.1\nprint(f'After VolAdj: {{list(df.columns)}}')",
        # Cell 6: add column 2
        "df['SMA'] = df['Close'].rolling(2).mean()\nprint(f'After SMA: {{list(df.columns)}}')",
    ])
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    # Verify cell 5 and 6 added columns
    out5 = nb_runner.get_output(5)
    assert "VolAdj" in out5, f"Cell 5 should add VolAdj: {out5}"
    out6 = nb_runner.get_output(6)
    assert "SMA" in out6, f"Cell 6 should add SMA: {out6}"

    # Re-run cell 4 — should show df state BEFORE cells 5 and 6
    nb_runner.run_cell(4)
    out4_rerun = nb_runner.get_output(4)
    assert "VolAdj" not in out4_rerun, (
        f"Cell 4 re-run should NOT have VolAdj column. Got: {out4_rerun}"
    )
    assert "SMA" not in out4_rerun, (
        f"Cell 4 re-run should NOT have SMA column. Got: {out4_rerun}"
    )


def test_self_assignment_cell_still_works_with_downstream_mutations(nb_runner, tmp_path):
    """
    Ensure that cells where df is both input AND output (e.g., df = df.sort_values())
    still work correctly — the fix should only affect read-only input cells.
    """
    csv_path = tmp_path / "data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    pd.DataFrame({
        "Name": ["Charlie", "Alice", "Bob"],
        "Score": [85, 95, 90]
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        # Cell 1: imports
        "import pandas as pd",
        # Cell 2: load
        f"df = pd.read_csv('{csv_path_str}')\nprint(f'Loaded: {{list(df.columns)}}')",
        # Cell 3: self-assignment (df is both input and output)
        "df = df.sort_values('Name')\nprint('Sorted: ' + str(df['Name'].tolist()))",
        # Cell 4: downstream mutation
        "df['Grade'] = ['A', 'B', 'A']\nprint('With grade: ' + str(list(df.columns)))",
    ])
    nb_runner.start_kernel()

    # Run all cells
    nb_runner.run_all()

    out3 = nb_runner.get_output(3)
    assert "Alice" in out3, f"Cell 3 should show sorted names: {out3}"

    out4 = nb_runner.get_output(4)
    assert "Grade" in out4, f"Cell 4 should have Grade: {out4}"

    # Re-run cell 3 (self-assignment) — this should still work
    # because df is BOTH input AND output
    nb_runner.run_cell(3)
    out3_rerun = nb_runner.get_output(3)
    assert "Alice" in out3_rerun, f"Cell 3 re-run should still show sorted: {out3_rerun}"
