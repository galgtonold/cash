"""
Integration test for file dependency invalidation with display-only cells.

This test reproduces the exact scenario from financial_analysis_demo.ipynb:
1. Cell 1: Load data from CSV with pd.read_csv -> df
2. Cell 2: Process df and write back to df (df = df.sort_values(...))
3. Cell 3: Just display df (no file read, no assignment)

When the CSV file changes, Cell 3 should be invalidated because its input 'df'
was ultimately derived from the file.
"""

import os
import tempfile
import time

import pandas as pd
import pytest

from cash.notebook.cache_status import CacheStatus


@pytest.fixture(autouse=True)
def _pandas_already_imported(mock_shell):
    """An earlier cell ran ``import pandas as pd``."""
    mock_shell.user_ns["pd"] = pd


def test_display_cell_invalidation_on_file_change(statement_processor, mock_shell):
    """
    Test that a display-only cell is invalidated when the source file changes.

    This is the exact scenario from financial_analysis_demo.ipynb:
    1. Cell 1: df = pd.read_csv(path)
    2. Cell 2: df = df.sort_values(...)
    3. Cell 3: df  (just display)

    When the CSV file is modified, Cell 3 should be re-computed, not cached.
    """

    # Create temp CSV file
    temp_path = tempfile.mktemp(suffix=".csv").replace(os.sep, "/")
    df_initial = pd.DataFrame(
        {"Ticker": ["AAPL", "MSFT", "GOOGL"], "Close": [100.0, 200.0, 300.0], "Volume": [1000, 2000, 3000]}
    )
    df_initial.to_csv(temp_path, index=False)

    print(f"\n=== Test file: {temp_path} ===")
    print(f"Initial data:\n{df_initial}")

    try:
        # Cell 1: Read CSV
        code_read = f"df = pd.read_csv('{temp_path}')"

        # Cell 2: Process data (sort)
        code_process = "df = df.sort_values(by='Ticker')"

        # Cell 3: Just display df
        # Given real work on purpose: a statement below the too-cheap floor
        # is no longer stored just because its input came from a file, and an
        # uncached statement would pass the staleness check below trivially. This one is cached, so it must invalidate.
        code_display = "(df, sum(i * i for i in range(400_000)))[0]"

        # === FIRST RUN: Execute all cells ===
        print("\n=== FIRST RUN ===")

        print("\n--- Cell 1: Read CSV ---")
        metrics1 = statement_processor.process_statement(code_read)
        print(f"Status: {metrics1['status']}")
        assert metrics1["status"] == CacheStatus.COMPUTED
        print(f"df loaded: {len(mock_shell.user_ns['df'])} rows")

        print("\n--- Cell 2: Process data ---")
        metrics2 = statement_processor.process_statement(code_process)
        print(f"Status: {metrics2['status']}")
        assert metrics2["status"] == CacheStatus.COMPUTED

        print("\n--- Cell 3: Display df ---")
        metrics3 = statement_processor.process_statement(code_display)
        print(f"Status: {metrics3['status']}")
        assert metrics3["status"] == CacheStatus.COMPUTED

        # Check file dependencies are tracked
        print("\n--- Checking file dependencies ---")
        print(f"executed_file_deps: {statement_processor.tracking_state.executed_file_deps}")
        print(f"variable_sources: {statement_processor.tracking_state.variable_sources}")

        # === SECOND RUN: Re-run display cell (should be cached) ===
        print("\n=== SECOND RUN (no file change) ===")

        print("\n--- Cell 3: Display df (should RESTORE) ---")
        metrics4 = statement_processor.process_statement(code_display)
        print(f"Status: {metrics4['status']}")
        assert metrics4["status"] == CacheStatus.RESTORED, f"Expected RESTORED, got {metrics4['status']}"

        # === MODIFY FILE ===
        print("\n=== MODIFYING CSV FILE ===")
        time.sleep(0.1)  # Small delay

        df_modified = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT", "GOOGL"],
                "Close": [150.0, 250.0, 350.0],  # Changed values
                "Volume": [1500, 2500, 3500],
            }
        )
        df_modified.to_csv(temp_path, index=False)
        print(f"Modified data:\n{df_modified}")
        print(f"New mtime: {os.path.getmtime(temp_path)}")

        # === THIRD RUN: Re-run display cell (should INVALIDATE) ===
        print("\n=== THIRD RUN (after file change) ===")

        print("\n--- Cell 3: Display df (should COMPUTE due to file change) ---")
        metrics5 = statement_processor.process_statement(code_display)
        print(f"Status: {metrics5['status']}")

        # This is the critical assertion!
        assert metrics5["status"] == CacheStatus.COMPUTED, (
            f"Expected COMPUTED but got {metrics5['status']}. "
            "Display cell should be invalidated when source file changes."
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_display_cell_with_intermediate_processing(statement_processor):
    """
    Test with more processing steps between file read and display.

    1. Cell 1: df = pd.read_csv(path)
    2. Cell 2: df['New'] = df['Close'] * 2
    3. Cell 3: summary = df.describe()
    4. Cell 4: df  (display)
    """

    # Create temp CSV file
    temp_path = tempfile.mktemp(suffix=".csv").replace(os.sep, "/")
    df_initial = pd.DataFrame(
        {
            "Close": [100.0, 200.0, 300.0],
        }
    )
    df_initial.to_csv(temp_path, index=False)

    try:
        code_read = f"df = pd.read_csv('{temp_path}')"
        code_transform = "df['New'] = df['Close'] * 2"
        code_summary = "summary = df.describe()"
        # Given real work on purpose: a statement below the too-cheap floor
        # is no longer stored just because its input came from a file, and an
        # uncached statement would pass the staleness check below trivially. This one is cached, so it must invalidate.
        code_display = "(df, sum(i * i for i in range(400_000)))[0]"

        # First run
        print("\n=== FIRST RUN ===")
        statement_processor.process_statement(code_read)
        statement_processor.process_statement(code_transform)
        statement_processor.process_statement(code_summary)
        metrics_display1 = statement_processor.process_statement(code_display)
        assert metrics_display1["status"] == CacheStatus.COMPUTED

        # Second run - should cache
        print("\n=== SECOND RUN (no change) ===")
        metrics_display2 = statement_processor.process_statement(code_display)
        assert metrics_display2["status"] == CacheStatus.RESTORED

        # Modify file
        print("\n=== MODIFYING FILE ===")
        time.sleep(0.1)
        df_modified = pd.DataFrame({"Close": [999.0, 888.0, 777.0]})
        df_modified.to_csv(temp_path, index=False)

        # Third run - should invalidate
        print("\n=== THIRD RUN (after file change) ===")
        metrics_display3 = statement_processor.process_statement(code_display)

        assert metrics_display3["status"] == CacheStatus.COMPUTED, (
            f"Expected COMPUTED but got {metrics_display3['status']}. "
            "Display cell should be invalidated when source file changes."
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
