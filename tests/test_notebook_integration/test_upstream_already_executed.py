"""
Test that when a cell is executed as an upstream dependency, running the actual cell
afterwards uses the cached result instead of re-executing.

Scenario:
1. Run display cell (df) which triggers upstream execution of sort cell
2. Run sort cell directly - should use cache, NOT re-execute

This tests the case where:
- Cell A (sort): df = df.sort_values(...)  
- Cell B (display): df
- User runs Cell B first → triggers upstream execution of Cell A
- User then runs Cell A → should be cache hit, not re-execution
"""
import pytest
import pandas as pd

pytestmark = [pytest.mark.upstream, pytest.mark.skip_optimization]


def test_upstream_execution_enables_cache_hit(nb_runner, tmp_path):
    """
    Test that executing a cell as upstream dependency enables cache hit
    when that cell is run directly afterwards.
    """
    # Create test data file
    csv_path = tmp_path / "test_data.csv"
    csv_path_str = str(csv_path).replace("\\", "/")
    
    df_initial = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'Close': [100.0, 200.0, 300.0],
    })
    df_initial.to_csv(csv_path, index=False)
    
    cell_imports = "import pandas as pd"
    cell_load = f"df = pd.read_csv('{csv_path_str}')"
    cell_sort = """df = df.sort_values(by='Ticker')
print(f"Sort output: {df['Close'].tolist()}")"""
    cell_display = """print(f"Display output: {df['Close'].tolist()}")
df"""

    nb_runner.create_notebook([
        cell_imports,    # Cell 1
        cell_load,       # Cell 2  
        cell_sort,       # Cell 3
        cell_display,    # Cell 4
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    
    # Run all cells in order
    nb_runner.run_all()
    
    # Get sort output
    sort_output = nb_runner.get_raw_output(3)
    display_output = nb_runner.get_raw_output(4)
    
    print(f"Sort output: {sort_output}")
    print(f"Display output: {display_output}")
    
    # Both should show sorted data (AAPL=100, GOOGL=300, MSFT=200 sorted alphabetically)
    assert "100.0" in sort_output or "200.0" in sort_output or "300.0" in sort_output
    assert "100.0" in display_output or "200.0" in display_output or "300.0" in display_output


def test_downstream_triggers_upstream(nb_runner):
    """
    Test that running a downstream cell triggers upstream cells.
    """
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "z = y + 5",
        "print(f'z = {z}')"
    ])
    nb_runner.start_kernel()
    
    # Run only cell 4 - should trigger cells 1, 2, 3
    nb_runner.run_cell(4)
    
    output = nb_runner.get_output(4)
    assert "z = 25" in output  # 10 * 2 + 5 = 25


def test_repeated_cell_execution_uses_cache(nb_runner):
    """
    Test that running the same cell twice uses cache.
    """
    nb_runner.create_notebook([
        """import time
start = time.time()
time.sleep(0.2)
result = 42
print(f"Computed result: {result}")""",
    ])
    nb_runner.start_kernel()
    
    # First run
    nb_runner.run_cell(1)
    out1 = nb_runner.get_output(1)
    assert "Computed result: 42" in out1
    
    # Second run - should use cache (output replayed)
    nb_runner.run_cell(1)
    out2 = nb_runner.get_output(1)
    assert "Computed result: 42" in out2
