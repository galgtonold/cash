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


def test_run_notebook_cash_on_captures_statement_metrics(tmp_path):
    cells = [
        CodeCell(index=0, notebook_cell_index=0, source="x = 1 + 1\n"),
        CodeCell(index=1, notebook_cell_index=1, source="y = x * 2\n"),
    ]
    timings = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    assert len(timings) == 2
    # Each cell has at least one statement metric captured
    assert len(timings[0].statement_metrics) >= 1
    m = timings[0].statement_metrics[0]
    assert m.execution_time >= 0
    assert m.total_time >= m.execution_time  # total includes execution
    assert m.status in {"COMPUTED", "RESTORED", "SKIPPED", "UNKNOWN"}


def test_run_notebook_cash_on_cold_then_warm_status_shifts(tmp_path):
    """Same cells run twice against the same cache dir: second run should
    have at least one RESTORED status (the cache is now populated).

    ``# @cash:persist`` forces the value to disk. Without it the cost model
    declines to persist a trivial statement (compute is cheaper than a
    restore), so no value file is written and the warm run recomputes."""
    cells = [CodeCell(index=0, notebook_cell_index=0, source="# @cash:persist\nz = 7 * 6\n")]
    first = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    second = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    first_statuses = [m.status for m in first[0].statement_metrics]
    second_statuses = [m.status for m in second[0].statement_metrics]
    assert "COMPUTED" in first_statuses
    assert "RESTORED" in second_statuses
