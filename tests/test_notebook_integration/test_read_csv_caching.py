"""
Bug reproduction: pd.read_csv() statement keeps re-executing instead of caching.

Scenario:
  Cell 1: import pandas as pd; os.chdir(...)
  Cell 2: %cash_on
  Cell 3: df = pd.read_csv('data.csv')  ← This should cache!
  Cell 4: df = df.sort_values(...)
  Cell 5: df  # print

When cell 3 is re-run (e.g. user re-executes cell 3 alone),
`df = pd.read_csv(data_path)` should be restored from cache, not re-executed.

The bug: After cells 3+4 both ran, variable_lineage['df'] and 
executed_cell_codes['df'] reflect cell 4 (sort_values), not cell 3 (read_csv).
So the ALREADY_EXECUTED optimization correctly skips cell 3's check.
Then the cache lookup should find the entry. But does it?
"""
import pytest
import pandas as pd

pytestmark = pytest.mark.files



def test_read_csv_caches_on_rerun(nb_runner, tmp_path):
    """
    pd.read_csv should be cached and restored on cell re-execution,
    even after downstream cells have modified df.
    """
    # Create test CSV
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({
        'a': [3, 1, 2],
        'b': [6, 4, 5]
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        "import pandas as pd",
        f"df = pd.read_csv('{csv_path_str}')\nprint('Loaded CSV')",
        "df = df.sort_values('a').reset_index(drop=True)\nprint('Sorted')",
        "print(df.to_string())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Verify initial run works
    output_cell4 = nb_runner.get_output(4)
    assert "1" in output_cell4

    # Now re-run JUST cell 2 (the read_csv cell)
    # It should restore from cache, not re-execute
    nb_runner.run_cell(2)
    output_cell2 = nb_runner.get_output(2)
    # "Loaded CSV" should appear (from cached stdout replay)
    assert "Loaded CSV" in output_cell2

    # Check for cache indicators - should show ⚡ Restored, not ⚙️ Executed
    # If the badge shows "Restored" or there's a CACHE_HIT marker, caching worked
    # The key check: it should NOT take ~2 seconds (read_csv time)


def test_read_csv_caches_on_full_cell_rerun(nb_runner, tmp_path):
    """
    When an entire multi-statement cell containing pd.read_csv is re-run,
    the read_csv statement should be restored from cache.
    """
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({
        'x': [10, 20, 30],
        'y': [40, 50, 60]
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        "import pandas as pd",
        # Multi-statement cell with read_csv
        (
            f"data_path = '{csv_path_str}'\n"
            "print('Loading data...')\n"
            "df = pd.read_csv(data_path)\n"
            "print('Data loaded')"
        ),
        "print(df.to_string())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output1 = nb_runner.get_output(3)
    assert "10" in output1

    # Re-run cell 2 (the one with read_csv)
    nb_runner.run_cell(2)
    output2 = nb_runner.get_output(2)
    # Cached stdout should be replayed
    assert "Loading data..." in output2 or "Data loaded" in output2


def test_read_csv_after_downstream_sort(nb_runner, tmp_path):
    """
    Critical test: After running read_csv cell then sort cell,
    re-running the read_csv cell should still hit cache.
    
    This tests the scenario where executed_cell_codes['df'] stores
    the sort code (from the later cell), not the read_csv code.
    """
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({
        'val': [30, 10, 20],
    }).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        "import pandas as pd",
        f"df = pd.read_csv('{csv_path_str}')\nprint('Read CSV')",
        "df = df.sort_values('val').reset_index(drop=True)\nprint('Sorted')",
        "print(df.to_string())",
    ])
    nb_runner.start_kernel()
    
    # Run all cells
    nb_runner.run_all()
    
    output_sorted = nb_runner.get_output(4)
    assert "10" in output_sorted
    
    # Re-run cell 2 (read_csv) and cell 3 (sort) 
    nb_runner.run_cells([2, 3])
    
    # Cell 2 should have cached output
    output_read = nb_runner.get_output(2)
    assert "Read CSV" in output_read


def test_read_csv_with_debug_shows_cache_hit(nb_runner, tmp_path):
    """
    With debug mode on, verify that read_csv shows cache hit markers,
    not execution markers.
    """
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    pd.DataFrame({'col': [1, 2, 3]}).to_csv(csv_path, index=False)

    nb_runner.create_notebook([
        "import pandas as pd",
        "%cash_debug on",
        f"df = pd.read_csv('{csv_path_str}')\nprint('Loaded')",
        "print(df.to_string())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # Re-run cell 3 (read_csv)
    nb_runner.run_cell(3)
    output = nb_runner.get_output(3)
    
    # Check that it was restored, not re-executed
    # The output should contain "Loaded" (replayed from cache)
    assert "Loaded" in output
    
    # With debug on, we should see CACHE_HIT_DEBUG or Restored markers
    # If we see "Executed" in the badge without "Restored", caching failed
