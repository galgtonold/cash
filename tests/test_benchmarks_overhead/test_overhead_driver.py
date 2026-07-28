from pathlib import Path

import pytest

from benchmarks._overhead_driver import new_cash_session, run_notebook
from benchmarks._overhead_io import CodeCell

# Compute that lands between the two cost-model floors: dearer than
# ``min_execution_time_to_cache_seconds`` (0.01s) so it is cached at all, but
# cheaper than ``_SMART_PERSIST_COMPUTE_FLOOR_S`` (0.1s) so the cost model
# keeps it in RAM and never writes it to disk. ~0.04s on the dev machine.
_RAM_ONLY_CELL = "z = sum(i * i for i in range(1_000_000))\n"


def _metric_for_z(timings):
    """The statement metric for the ``z = ...`` assignment."""
    return [m for m in timings[0].statement_metrics if m.code.startswith("z =")]


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


# --- warm-session vs warm-restart -------------------------------------------
#
# These two pin the distinction the benchmark harness draws between its two
# warm modes. A value the cost model keeps in RAM survives re-running a cell in
# a live kernel, but not a kernel restart. Before this split, `--mode warm`
# built a fresh Cash for the measurement pass, so it could only ever observe
# disk-tier restores and scored every RAM-tier hit as a miss.


def _require_ram_only(timings):
    """Assert the fixture really did land RAM-only, so neither test can pass
    for the wrong reason on a machine where it falls outside the floors."""
    metrics = _metric_for_z(timings)
    assert len(metrics) == 1, f"expected one `z =` metric, got {metrics}"
    m = metrics[0]
    if m.storage != ["RAM"]:
        pytest.skip(
            f"fixture did not land RAM-only on this machine "
            f"(exec={m.execution_time:.4f}s storage={m.storage}); the "
            f"RAM/disk distinction under test is not being exercised"
        )
    return m


def test_warm_restart_cannot_restore_a_ram_only_value(tmp_path):
    """Two independent sessions over one cache dir — what a kernel restart does.

    The value never reached disk, and the second session builds its own RAM
    tier, so there is nothing to restore. This is the honest content of the
    old `--mode warm`, and the reason several reference notebooks reported
    zero restores while caching perfectly well.
    """
    cells = [CodeCell(index=0, notebook_cell_index=0, source=_RAM_ONLY_CELL)]
    first = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    _require_ram_only(first)

    second = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path)
    assert [m.status for m in _metric_for_z(second)] == ["COMPUTED"], (
        "a RAM-only value restored across two independent Cash sessions; "
        "either the cost model now persists it, or session isolation broke"
    )


def test_warm_session_restores_a_ram_only_value(tmp_path):
    """The same cells sharing one Cash session — re-running in a live kernel.

    The RAM tier persists, so the value comes back. This is the path the
    harness could not measure at all before `--mode warm-session`.
    """
    cells = [CodeCell(index=0, notebook_cell_index=0, source=_RAM_ONLY_CELL)]
    session = new_cash_session(tmp_path)

    first = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path,
                         session=session)
    _require_ram_only(first)

    second = run_notebook(cells, cash_enabled=True, cache_dir=tmp_path,
                          session=session)
    assert [m.status for m in _metric_for_z(second)] == ["RESTORED"], (
        "reusing the Cash session did not restore a RAM-tier value; "
        "warm-session is measuring the same thing as warm-restart"
    )
