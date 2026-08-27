"""IPython shell driver for the overhead benchmark.

Spins up a fresh ``InteractiveShell`` per call, optionally enables cash,
runs cells via ``shell.run_cell``, and times each one. Captures
per-statement ``ProcessResult`` data when cash is enabled via a monkey-patch
on ``StatementProcessor.process``.
"""
from __future__ import annotations

import time
from pathlib import Path

# Pre-import cash BEFORE creating any InteractiveShell so its
# ``_auto_load_in_ipython`` side effect fires once now (no shell active →
# no-op) instead of when the notebook's first ``import cash`` runs.
# Without this, the auto-load would call ``_get_global_cash().register_magic()``
# during cell 0 and silently override the bench's own CashMagics
# registration in cold-mode runs, *and* enable cash in "off" mode runs
# (because the notebook itself imports cash). The bench was measuring
# something other than what its mode flag claimed.
import cash  # noqa: F401 — side-effect import: neutralise IPython auto-load

from IPython.core.interactiveshell import InteractiveShell

from benchmarks._overhead_io import CodeCell
from benchmarks._overhead_results import CellTiming, StatementMetric


def new_cash_session(cache_dir: Path | str):
    """Build a ``Cash`` instance callers can hand to successive
    :func:`run_notebook` calls to model **one long-lived kernel**.

    The point is the RAM tier. Each ``Cash`` owns its own in-memory backend, so
    constructing a new one per run (the default) throws away every value the
    cost model declined to write to disk — which, for a notebook of
    sub-100ms statements, is most of them. Passing one instance to several runs
    keeps that tier alive, which is what re-running a cell in a live Jupyter
    kernel actually does.
    """
    from cash.core import Cash
    return Cash(cache_dir=str(cache_dir), register_magic=False)


def run_notebook(
    cells: list[CodeCell],
    cash_enabled: bool,
    cache_dir: Path | None,
    session=None,
) -> list[CellTiming]:
    """Run ``cells`` in a fresh in-process IPython shell.

    If ``cash_enabled`` is True, ``cache_dir`` must be supplied and cash is
    enabled with that directory as its backend. Per-statement metrics are
    captured for each cash-processed statement.

    ``session`` is an optional ``Cash`` from :func:`new_cash_session`. Supply it
    to reuse one cache instance — and therefore one RAM tier — across calls;
    omit it (the default) for a fresh instance per call, which models a kernel
    restart and can only ever restore what reached disk.
    """
    if cash_enabled and cache_dir is None:
        raise ValueError("cache_dir is required when cash_enabled is True")
    if session is not None and not cash_enabled:
        raise ValueError("session is meaningless when cash_enabled is False")

    shell = InteractiveShell.instance()
    # Defensive: clear user_ns so a re-used singleton doesn't carry state.
    shell.reset(new_session=True)

    # Drop the global Cash singleton + clear in-memory tracking state
    # so the next `%load_ext cash` (or our `_enable_cash` below) gets a
    # fresh instance. Without this, cash's ``executed_cell_codes`` and
    # ``variable_lineage`` survive the shell.reset and silently
    # short-circuit cells in later repeats -- e.g. 10_us_flights cell 3
    # (a pd.read_csv) drops from 2.5s to 40ms in repeats 1+ even with
    # CASH_CACHE_DIR wiped, because cash remembers the cell was
    # "already executed" in repeat 0. ``reset_session`` is the public
    # API that does exactly what we need here.
    cash.reset_session()

    if session is not None:
        # reset_session() nulls the singleton and, under an active IPython,
        # immediately builds a replacement. Put ours back, because notebooks
        # that bootstrap cash themselves (`%load_ext cash` in cell 0 — most of
        # the reference suite) resolve through the singleton and would
        # otherwise bind to a fresh instance with an empty RAM tier, silently
        # measuring warm-restart while claiming warm-session.
        #
        # Only the backend rides along: `executed_cell_codes` /
        # `variable_lineage` live in the notebook layer, not on Cash, so
        # reset_session() still clears them. warm-session therefore differs
        # from warm-restart in exactly one variable — whether the RAM tier
        # survives — rather than also inheriting "this cell already ran"
        # short-circuits, which would flatter the numbers.
        cash._global_cash = session

    statement_sink: list[StatementMetric] = []
    if cash_enabled:
        _enable_cash(shell, Path(cache_dir), statement_sink, session=session)

    timings: list[CellTiming] = []
    for cell in cells:
        # Drain the sink so each cell only owns its own statements.
        before = len(statement_sink)
        t0 = time.perf_counter()
        exec_result = shell.run_cell(cell.source)
        t1 = time.perf_counter()
        cell_metrics = list(statement_sink[before:])
        # run_cell swallows exceptions into the result object. A cell that
        # dies still returns a timing, so an unrecorded error reads as a fast
        # cell rather than a broken run.
        exc = getattr(exec_result, "error_in_exec", None) or getattr(
            exec_result, "error_before_exec", None)
        timings.append(CellTiming(
            index=cell.index,
            notebook_cell_index=cell.notebook_cell_index,
            wall_seconds=t1 - t0,
            source_chars=len(cell.source),
            statement_metrics=cell_metrics,
            error=f"{type(exc).__name__}: {exc}" if exc is not None else None,
        ))

    if cash_enabled and not statement_sink:
        # Cash was asked for and processed nothing. The cells still ran, and
        # the wall times still look plausible — which is exactly the danger:
        # a run like this is a cash-off measurement wearing a cash-on label,
        # and averaging it silently understates every mode it appears in.
        # Fail loudly instead; a sweep that skips one notebook is recoverable,
        # a sweep with an invisible hole in it is not.
        raise RuntimeError(
            f"cash was enabled but not one of {len(cells)} cells produced a "
            f"statement metric. The cell transform was not active, so this "
            f"run measured uncached execution. Refusing to report it as a "
            f"cash run."
        )
    return timings


def _enable_cash(shell, cache_dir: Path, sink: list[StatementMetric],
                 session=None) -> None:
    """Initialise cash on ``shell`` and install a tee on
    ``StatementProcessor.process_statement`` so each cell's per-statement
    ``ProcessResult`` is appended to ``sink``.

    The tee patches the class method (not the instance) so it fires for every
    StatementProcessor the magics layer creates, including any per-cell
    re-instantiations.
    """
    from cash.core import Cash
    from cash.notebook.ipython.magics import CashMagics
    from cash.notebook.statement import StatementProcessor

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
                cost_model_size_bytes=result.get("cost_model_size_bytes"),
                cost_model_restore_seconds=result.get("cost_model_restore_seconds"),
                cost_model_type_name=result.get("cost_model_type_name"),
                cost_model_family=result.get("cost_model_family"),
                uncacheable_reasons=[
                    str(r) for r in (result.get("uncacheable_reasons") or [])
                ],
                skipped_reason=result.get("skipped_reason"),
                miss_reason=result.get("miss_reason"),
                storage=[str(s) for s in (result.get("storage") or [])],
            ))
        except Exception:  # noqa: BLE001 — tee must never break user code
            pass
        return result

    StatementProcessor.process_statement = _teed_process_statement  # type: ignore[method-assign]

    # Create a Cash instance pointed at the bench's cache dir and wire up the
    # magics.  cash_on() doesn't accept a path argument; the dir is configured
    # at Cash construction time. A caller-supplied ``session`` is reused as-is,
    # keeping its RAM tier (and everything the cost model left there) alive
    # across runs.
    cash_instance = session if session is not None else Cash(
        cache_dir=str(cache_dir), register_magic=False)
    magics = CashMagics(shell=shell, cash_instance=cash_instance)
    shell.register_magics(magics)
    # Enable auto-caching (no TTL needed for the benchmark).
    magics.cash_on("")
