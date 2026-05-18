from pathlib import Path

from benchmarks._overhead_driver import run_notebook
from benchmarks._overhead_io import CodeCell


def test_run_notebook_cash_off_runs_each_cell_and_times_it():
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 1 + 1\n"),
        CodeCell(index=1, notebook_cell_index=1, source="y = x * 2\n"),
    ]
    timings = run_notebook(cells, cash_enabled=False, cache_dir=None)
    assert len(timings) == 2
    assert timings[0].source_chars == len("x = 1 + 1\n")
    assert timings[0].wall_seconds >= 0
    assert timings[1].wall_seconds >= 0
    # cash_enabled=False -> no statement_metrics captured
    assert timings[0].statement_metrics == []
    assert timings[1].statement_metrics == []


def test_run_notebook_propagates_variable_state_between_cells():
    """Cell 2 reads x from cell 1; if the shell isn't shared, this fails."""
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 42\n"),
        CodeCell(index=1, notebook_cell_index=1, source="assert x == 42\n"),
    ]
    timings = run_notebook(cells, cash_enabled=False, cache_dir=None)
    assert len(timings) == 2  # second cell didn't raise
