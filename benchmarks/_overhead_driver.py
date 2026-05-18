"""IPython shell driver for the overhead benchmark.

Spins up a fresh ``InteractiveShell`` per call, optionally enables cash,
runs cells via ``shell.run_cell``, and times each one. Captures
per-statement ``ProcessResult`` data when cash is enabled via a monkey-patch
on ``StatementProcessor.process``.
"""
from __future__ import annotations

import time
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

from benchmarks._overhead_io import CodeCell
from benchmarks._overhead_results import CellTiming, StatementMetric


def run_notebook(
    cells: list[CodeCell],
    cash_enabled: bool,
    cache_dir: Path | None,
) -> list[CellTiming]:
    """Run ``cells`` in a fresh in-process IPython shell.

    If ``cash_enabled`` is True, ``cache_dir`` must be supplied and cash is
    enabled with that directory as its backend. Per-statement metrics are
    captured for each cash-processed statement.
    """
    if cash_enabled and cache_dir is None:
        raise ValueError("cache_dir is required when cash_enabled is True")

    shell = InteractiveShell.instance()
    # Defensive: clear user_ns so a re-used singleton doesn't carry state.
    shell.reset(new_session=True)

    statement_sink: list[StatementMetric] = []
    if cash_enabled:
        _enable_cash(shell, Path(cache_dir), statement_sink)

    timings: list[CellTiming] = []
    for cell in cells:
        # Drain the sink so each cell only owns its own statements.
        before = len(statement_sink)
        t0 = time.perf_counter()
        shell.run_cell(cell.source)
        t1 = time.perf_counter()
        cell_metrics = list(statement_sink[before:])
        timings.append(CellTiming(
            index=cell.index,
            notebook_cell_index=cell.notebook_cell_index,
            wall_seconds=t1 - t0,
            source_chars=len(cell.source),
            statement_metrics=cell_metrics,
        ))
    return timings


def _enable_cash(shell, cache_dir: Path, sink: list[StatementMetric]) -> None:
    """Stub for cash setup — implemented in Task 5."""
    raise NotImplementedError("Cash enablement implemented in Task 5")
