"""
Integration test for file dependency invalidation.

This test replicates the exact scenario from financial_analysis_demo.ipynb:
1. Cell 1: df = pd.read_csv(path)
2. Cell 2: df = df.sort_values(...)
3. Cell 3: Display df
4. Modify the CSV file
5. Re-run cells - should detect file change and re-compute
"""
import pytest
import pandas as pd

pytestmark = pytest.mark.files

def test_file_invalidation_full_rerun(nb_runner, tmp_path):
    """
    Test file invalidation with full pipeline re-run.
    
    1. Create notebook: read CSV -> sort -> display
    2. Run all cells (all COMPUTED)
    3. Modify the CSV file
    4. Re-run all cells (should re-compute due to file change)
    """
    # Create a temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    
    # Create initial data
    df_initial = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'Close': [100.0, 200.0, 300.0],
    })
    df_initial.to_csv(csv_path, index=False)

    cell_read = f"""import pandas as pd
df = pd.read_csv('{csv_path_str}')
print(f"Read: {{df['Close'].tolist()}}")"""

    cell_sort = """df = df.sort_values(by='Ticker')
print(f"Sorted: {df['Close'].tolist()}")"""

    cell_display = """print(f"Display: {df['Close'].tolist()}")
df"""

    # Simple test: read, sort, display - then modify file and re-run
    nb_runner.create_notebook([
        cell_read,      # Cell 1: Read
        cell_sort,      # Cell 2: Sort 
        cell_display,   # Cell 3: Display
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # Check first run shows original values
    out_display1 = nb_runner.get_output(3)
    print(f"First display: {out_display1}")
    assert "100.0" in out_display1 or "300.0" in out_display1, f"First display should show original values. Got: {out_display1}"
    
    # Modify the file
    import time
    time.sleep(0.2)  # Ensure mtime changes
    df_modified = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'Close': [999.0, 888.0, 777.0],
    })
    df_modified.to_csv(csv_path, index=False)
    
    # Re-run all cells - should detect file change and re-compute
    nb_runner.run_all()
    
    # Check final display shows NEW values
    out_display2 = nb_runner.get_output(3)
    print(f"Second display (after file modify): {out_display2}")
    assert "999.0" in out_display2 or "888.0" in out_display2 or "777.0" in out_display2, \
        f"Display after file modify should show NEW values. Got: {out_display2}. Cache was NOT invalidated!"


def test_file_modification_detection(nb_runner, tmp_path):
    """
    Test that modifying a file between runs causes cache invalidation.
    """
    # Create initial file
    data_file = tmp_path / "data.txt"
    data_file.write_text("initial content")
    data_file_str = str(data_file).replace("\\", "/")
    
    nb_runner.create_notebook([
        f"""with open('{data_file_str}', 'r') as f:
    content = f.read()
print(f"Content: {{content}}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(1)
    assert "initial content" in out1
    
    # Modify the file
    import time
    time.sleep(0.1)  # Ensure mtime changes
    data_file.write_text("modified content")
    
    # Re-run - should detect file change
    nb_runner.run_all()
    
    out2 = nb_runner.get_output(1)
    assert "modified content" in out2, f"Should detect file change. Got: {out2}"
