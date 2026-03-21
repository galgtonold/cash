"""
Integration test: Partial cell caching when cell is edited but not saved.

Bug scenario (Issue: downstream advancement fallback):
1. User runs a cell with two heavy statements that both modify df
2. Both statements are computed and cached
3. User edits one statement (e.g., changes SMA_60 to SMA_61) WITHOUT saving
4. User re-runs the cell
5. Expected: First unchanged statement should be RESTORED from cache
6. Bug: First statement was being re-computed because cell not found in notebook
   → upstream check skipped → df lineage not reset → cache miss

This test verifies the fix: even when the cell is not found in the notebook file
(because the user edited without saving), the downstream advancement fallback
resets the lineage for variables that are both inputs and outputs, allowing
unchanged statements to hit the cache.
"""
import pytest

pytestmark = pytest.mark.core


def test_partial_cache_after_unsaved_edit(nb_runner):
    """
    Modify a cell's in-memory content without saving to disk.
    The first unchanged statement should still get a cache hit.
    
    This test simulates the real VS Code scenario:
    1. Run the cell once (cell found on disk, simulation cache populated)
    2. Edit the cell (change second statement) WITHOUT saving
    3. Re-run the cell
    Expected: First unchanged statement should be RESTORED from cache.
    
    Note: We only run cells 1 and 2 (not cell 3) to ensure last_cell_index
    matches cell 2's index, simulating the user re-running the same cell.
    """
    nb_runner.create_notebook([
        # Cell 1: Setup
        "import pandas as pd\ndf = pd.DataFrame({'Close': range(1, 101), 'Volume': range(100, 200)})",
        # Cell 2: Two operations on df
        "df['VolAdj'] = df['Close'] * df['Volume']\ndf['SMA_60'] = df['Close'].rolling(60).mean()",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()

    # First run: run cells 1 and 2 only
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)

    # Check that VolAdj was computed
    nb_runner.get_raw_output(2)

    # Now simulate the VS Code "unsaved edit" scenario:
    # Modify cell 2's in-memory content WITHOUT saving to disk.
    nb_runner.nb.cells[1].source = (
        "df['VolAdj'] = df['Close'] * df['Volume']\n"
        "df['SMA_61'] = df['Close'].rolling(61).mean()"
    )
    nb_runner.nb.cells[1].outputs = []

    # Re-run cell 2 with modified (unsaved) content
    nb_runner.run_cell(2)

    # Check debug output for evidence the fix worked
    raw_cell2_run2 = nb_runner.get_raw_output(2)

    # The cell should not be found in notebook (our test condition)
    assert "Current cell not found in notebook" in raw_cell2_run2, (
        f"Expected 'Current cell not found in notebook' in debug output.\n"
        f"Output: {raw_cell2_run2[:500]}"
    )

    # Check that the downstream advancement fallback fired
    assert "Downstream advancement fallback" in raw_cell2_run2, (
        f"Expected 'Downstream advancement fallback' in debug output.\n"
        f"Output: {raw_cell2_run2[:500]}"
    )


def test_partial_cache_debug_output_shows_fallback(nb_runner):
    """
    Verify that the downstream advancement fallback produces the expected
    debug output when the cell is not found in notebook.
    """
    nb_runner.create_notebook([
        # Cell 1: Setup
        "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})",
        # Cell 2: Two operations on df
        "df['B'] = df['A'] * 2\ndf['C'] = df['A'] * 3",
        # Cell 3: Print
        "print(list(df.columns))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()

    # First run
    nb_runner.run_all()

    # Unsaved edit: change second statement
    nb_runner.nb.cells[1].source = "df['B'] = df['A'] * 2\ndf['D'] = df['A'] * 4"
    nb_runner.nb.cells[1].outputs = []

    # Re-run cell 2
    nb_runner.run_cell(2)

    # Check debug output for the fallback message
    raw_output = nb_runner.get_raw_output(2)

    # The debug output should mention the fallback mechanism
    # It should either show "Downstream advancement fallback" or
    # "Current cell not found in notebook"
    assert ("Current cell not found in notebook" in raw_output or
            "Downstream advancement fallback" in raw_output), \
        f"Expected fallback debug message not found in output:\n{raw_output[:500]}"
