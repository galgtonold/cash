"""
Integration tests for file dependency propagation through upstream simulation cache.

BUG (Finding F1): When a file dependency changes and only a downstream cell is 
re-executed, the upstream simulation cache is reused (no code changed) and 
vars_with_stale_files is never populated. This causes the downstream cell to 
use stale cached data instead of detecting the file change and re-executing 
upstream cells.

Root cause: upstream.py:_simulate_and_find_changes() incremental simulation cache
only checks cell CODE hash for invalidation, not file dependency mtimes.
vars_with_stale_files is not cached/restored from simulation cache entries.

These tests verify that:
1. File changes are detected even when only downstream cells are re-executed
2. Self-assignment cells (df = df.sort_values()) properly propagate file staleness
3. The simulation cache is invalidated when file dependencies change
"""
import pytest
import pandas as pd
import time

pytestmark = pytest.mark.files


def _make_read_cell(csv_path_str):
    """Helper to create a read_csv cell with the path embedded."""
    return (
        "import pandas as pd\n"
        f"df = pd.read_csv('{csv_path_str}')\n"
        "print('read:', df['val'].tolist())"
    )


SORT_CELL = (
    "df = df.sort_values('val', ascending=False)\n"
    "print('sorted:', df['val'].tolist())"
)


SUM_CELL = (
    "total = df['val'].sum()\n"
    "print('total =', total)"
)


class TestFileDepSimulationCache:
    """Tests for file dependency detection through simulation cache."""

    def test_file_change_detected_from_downstream_cell(self, nb_runner, tmp_path):
        """
        Core bug reproduction: changing a file and running only the last cell
        should detect the file change and re-execute upstream cells.
        
        Pipeline: read_csv -> self-assignment sort -> downstream sum
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        
        # Write initial data
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            _make_read_cell(csv_path_str),
            SORT_CELL,
            SUM_CELL,
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        # Verify initial values
        out1 = nb_runner.get_output(3)
        assert "total = 60" in out1, f"Initial sum should be 60. Got: {out1}"
        
        # Modify the file
        time.sleep(0.2)  # Ensure mtime changes
        pd.DataFrame({'val': [100, 200, 300]}).to_csv(csv_path, index=False)
        
        # Run ONLY the downstream cell - should detect file change via upstream sim
        nb_runner.run_cell(3)
        
        out2 = nb_runner.get_output(3)
        assert "total = 600" in out2, (
            f"After file change, running only cell 3 should detect stale file "
            f"and re-execute upstream cells. Expected total=600, got: {out2}"
        )

    def test_file_change_without_self_assignment(self, nb_runner, tmp_path):
        """
        Verify that file change detection works when there's NO self-assignment
        (this is the case that currently works).
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            _make_read_cell(csv_path_str),
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out1 = nb_runner.get_output(2)
        assert "total = 60" in out1, f"Initial sum should be 60. Got: {out1}"
        
        # Modify file
        time.sleep(0.2)
        pd.DataFrame({'val': [100, 200, 300]}).to_csv(csv_path, index=False)
        
        # Run only downstream cell
        nb_runner.run_cell(2)
        
        out2 = nb_runner.get_output(2)
        assert "total = 600" in out2, (
            f"Without self-assignment, file change should still be detected. "
            f"Expected total=600, got: {out2}"
        )

    def test_file_change_with_multiple_self_assignments(self, nb_runner, tmp_path):
        """
        Test with multiple chained self-assignments:
        read -> sort -> filter -> downstream
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        
        pd.DataFrame({'val': [10, 20, 30, 40, 50]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            _make_read_cell(csv_path_str),
            "df = df.sort_values('val')\nprint('sorted:', df['val'].tolist())",
            "df = df[df['val'] > 20]\nprint('filtered:', df['val'].tolist())",
            "result = df['val'].mean()\nprint('mean =', result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        out1 = nb_runner.get_output(4)
        assert "mean = 40.0" in out1, f"Initial mean of [30,40,50] should be 40.0. Got: {out1}"
        
        # Modify file
        time.sleep(0.2)
        pd.DataFrame({'val': [100, 200, 300, 400, 500]}).to_csv(csv_path, index=False)
        
        # Run only the last cell
        nb_runner.run_cell(4)
        
        out2 = nb_runner.get_output(4)
        assert "mean = 300.0" in out2, (
            f"After file change with multiple self-assignments, expected mean=300.0 "
            f"(mean of [100,200,300,400,500] all > 20), got: {out2}"
        )

    def test_file_change_detected_on_second_downstream_run(self, nb_runner, tmp_path):
        """
        Test that after the simulation cache is populated (first downstream run),
        a subsequent file change is still detected on the second downstream run.
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            _make_read_cell(csv_path_str),
            SORT_CELL,
            SUM_CELL,
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        # First downstream run (populates simulation cache)
        nb_runner.run_cell(3)
        out1 = nb_runner.get_output(3)
        assert "total = 60" in out1
        
        # Modify file
        time.sleep(0.2)
        pd.DataFrame({'val': [1000, 2000, 3000]}).to_csv(csv_path, index=False)
        
        # Second downstream run (cache already populated from first downstream run)
        nb_runner.run_cell(3)
        
        out2 = nb_runner.get_output(3)
        assert "total = 6000" in out2, (
            f"Second downstream run after file change should detect staleness. "
            f"Expected total=6000, got: {out2}"
        )

    def test_file_change_run_all_still_works(self, nb_runner, tmp_path):
        """
        Verify that run_all() still correctly detects file changes
        (this should already work, serves as regression test).
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        
        pd.DataFrame({'val': [10, 20, 30]}).to_csv(csv_path, index=False)
        
        nb_runner.create_notebook([
            _make_read_cell(csv_path_str),
            SORT_CELL,
            SUM_CELL,
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        
        assert "total = 60" in nb_runner.get_output(3)
        
        # Modify file and run all
        time.sleep(0.2)
        pd.DataFrame({'val': [100, 200, 300]}).to_csv(csv_path, index=False)
        
        nb_runner.run_all()
        
        out2 = nb_runner.get_output(3)
        assert "total = 600" in out2, f"run_all() after file change should work. Got: {out2}"
