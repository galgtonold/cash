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
    """Initialise cash on ``shell`` and install a tee on
    ``StatementProcessor.process_statement`` so each cell's per-statement
    ``ProcessResult`` is appended to ``sink``.

    The tee patches the class method (not the instance) so it fires for every
    StatementProcessor the magics layer creates, including any per-cell
    re-instantiations.
    """
    from cash.core import Cash
    from cash.notebook.magics import CashMagics
    from cash.notebook.statement_processor import StatementProcessor

    # Patch process_statement() at class level so all instances (including those
    # built later inside magics) are observed.  We store the original on the
    # class so re-running this in the same subprocess is idempotent.
    original = getattr(StatementProcessor, "_orig_process_stmt_for_bench", None)
    if original is None:
        original = StatementProcessor.process_statement
        StatementProcessor._orig_process_stmt_for_bench = original  # type: ignore[attr-defined]

    def _teed_process_statement(self, code, *args, **kwargs):
        result = original(self, code, *args, **kwargs)
        try:
            status = result.get("status", "UNKNOWN")
            # CacheStatus is an enum; convert to its string value when needed.
            status_str = status.value if hasattr(status, "value") else str(status)
            sink.append(StatementMetric(
                code=str(result.get("code", code))[:200],
                execution_time=float(result.get("execution_time", 0.0)),
                total_time=float(result.get("total_time", 0.0)),
                status=status_str,
            ))
        except Exception:  # noqa: BLE001 — tee must never break user code
            pass
        return result

    StatementProcessor.process_statement = _teed_process_statement  # type: ignore[method-assign]

    # Create a Cash instance pointed at the bench's cache dir and wire up the
    # magics.  cash_on() doesn't accept a path argument; the dir is configured
    # at Cash construction time.
    cash_instance = Cash(cache_dir=str(cache_dir), register_magic=False)
    magics = CashMagics(shell=shell, cash_instance=cash_instance)
    shell.register_magics(magics)
    # Enable auto-caching (no TTL needed for the benchmark).
    magics.cash_on("")
