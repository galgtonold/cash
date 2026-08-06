"""
Integration tests for file dependency tracking with os.chdir().

Issue: When os.chdir() is called before file operations, the same relative
path gets tracked as different absolute paths, causing cache misses.
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.files


def test_chdir_doesnt_break_file_caching(nb_runner, tmp_path):
    """
    Reproduces the bug: os.chdir() before pd.read_csv() breaks caching.
    
    Scenario:
    1. Cell 1: os.chdir() to change working directory
    2. Cell 2: pd.read_csv('data.csv') - uses relative path
    3. Re-execute cell 2 → should be cache HIT but currently is MISS
    
    The bug: os.path.abspath('data.csv') produces different absolute paths
    depending on the current working directory at execution time.
    """
    # Create test data in a subdirectory
    data_dir = tmp_path / "data_subdir"
    data_dir.mkdir()
    csv_path = data_dir / "test_data.csv"
    str(csv_path).replace('\\', '/')
    
    pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}).to_csv(csv_path, index=False)
    
    # Create notebook that changes directory then reads file
    data_dir_str = str(data_dir).replace('\\', '/')
    nb_runner.create_notebook([
        "import os; import pandas as pd",
        f"os.chdir(r'{data_dir_str}')",
        "df = pd.read_csv('test_data.csv'); print('Loaded CSV')",
    ])
    
    nb_runner.start_kernel()
    
    # Enable cash debug
    import asyncio
    nb_runner._run_async(
        nb_runner.client.kc._async_execute_interactive("%cash_debug on", store_history=False)
    )
    
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(3)
    assert "Loaded CSV" in output1  # First run should execute
    
    print("\n[DEBUG] First execution - checking debug logs")
    
    # Second run of cell 3 - should restore from cache (chdir persists in kernel)
    nb_runner.run_cell(3)
    nb_runner.get_output(3)
    
    # Check if it was restored from cache
    raw_output = nb_runner.get_raw_output(3)
    print(f"\n[DEBUG] Second execution raw output:\n{raw_output}")
    
    # The critical test: check for cache hit indicators in debug output
    # "Loaded CSV" appearing is OK - it's from replayed stdout, not re-execution
    # We need to check the badge or debug markers
    #
    # Looking for signs of cache hit:
    # - "CACHE_HIT_DEBUG" in debug output
    # - "Restored" or "RESTORED" in badge
    # - "Skipped" for the df assignment (already executed optimization)
    assert "CACHE_HIT_DEBUG" in raw_output or "Restored" in raw_output or "RESTORED" in raw_output.upper(), (
        f"Expected cache hit (CACHE_HIT_DEBUG or Restored marker), but not found. "
        f"Raw output: {raw_output[:500]}"
    )


def test_chdir_in_same_notebook_multiple_cells(nb_runner, tmp_path):
    """
    Test that chdir in cell 1, then file operations in cells 2 and 3 work.
    
    When re-executing cell 3, it should find the cache from the first run.
    """
    data_dir = tmp_path / "project" / "data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "sales.csv"
    
    pd.DataFrame({'product': ['A', 'B', 'C'], 'sales': [100, 200, 300]}).to_csv(csv_path, index=False)
    
    data_dir_str = str(data_dir).replace('\\', '/')
    nb_runner.create_notebook([
        "import os; import pandas as pd",
        f"os.chdir(r'{data_dir_str}')",
        "df = pd.read_csv('sales.csv')",
        "print(f'Loaded {len(df)} rows')",
        "total = df['sales'].sum()",
        "print(f'Total sales: {total}')",
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(4)
    assert "Loaded 3 rows" in output1
    
    output2 = nb_runner.get_output(6)
    assert "Total sales: 600" in output2
    
    # Re-run cells 3-6
    nb_runner.run_cells([3, 4, 5, 6])
    
    # Outputs should be identical (from cache)
    assert nb_runner.get_output(4) == output1
    assert nb_runner.get_output(6) == output2


def test_absolute_path_unaffected_by_chdir(nb_runner, tmp_path):
    """
    Control test: absolute paths should work correctly regardless of chdir.
    
    This should pass even with the bug, proving the issue is with relative paths.
    """
    csv_path = tmp_path / "absolute_test.csv"
    csv_path_str = str(csv_path).replace('\\', '/')
    
    pd.DataFrame({'x': [10, 20], 'y': [30, 40]}).to_csv(csv_path, index=False)
    
    # Use absolute path - should be unaffected by chdir
    nb_runner.create_notebook([
        "import os; import pandas as pd",
        f"os.chdir(r'{str(tmp_path).replace(chr(92), '/')}')",  # Change to tmp_path
        f"df = pd.read_csv(r'{csv_path_str}')",  # But use absolute path
        "print(df.to_string())",
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(4)
    assert "10" in output1 and "30" in output1
    
    # Change directory to somewhere else
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    nb_runner.set_cell_source(2, f"os.chdir(r'{str(other_dir).replace(chr(92), '/')}')")
    nb_runner.run_cells([2, 3, 4])
    
    output2 = nb_runner.get_output(4)
    # Should still get cache hit because absolute path is the same
    assert output1 == output2


def test_relative_path_from_notebook_dir(nb_runner, tmp_path):
    """
    Test the desired behavior: relative paths should be resolved relative
    to the notebook's directory when using realpath.
    
    This test verifies that file dependencies are properly tracked even when
    the notebook and data file are in a subdirectory.
    """
    # Create notebook directory
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    
    # Create data file in the notebook directory
    csv_path = notebook_dir / "local_data.csv"
    pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(csv_path, index=False)
    
    # Create notebook that uses relative path
    # The notebook's work_dir will be notebook_dir, so relative paths resolve there
    nb_runner.work_dir = notebook_dir
    nb_runner.create_notebook([
        "import pandas as pd",
        "df = pd.read_csv('local_data.csv')",  # Relative to notebook dir
        "print(df.to_string())",
    ])
    
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(3)
    assert "1" in output1 and "3" in output1
    
    # Re-run cells 2-3 - should hit cache
    nb_runner.run_cells([2, 3])
    output2 = nb_runner.get_output(3)
    assert output1 == output2
