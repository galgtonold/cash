"""Cost-model validation CLI orchestrator.

Runs four passes per repeat over bench_cost_model_validation.ipynb:

  1. cold + instrumented  — default policy; captures cost_model_* fields
  2. warm-after-instrumented — rerun against cache from pass 1; captures t_warm_policy
  3. cold + force-cache — fresh cold cache, policy gates disabled; every cell cached
  4. warm-after-force-cache — rerun against pass-3 cache; captures t_restore_actual

Repeat 0 is discarded as warmup (matches the existing bench convention).

Results are written under --results-dir:
  decisions.jsonl       — one CellDecision per (cell, repeat)
  restore_times.jsonl   — one CellRestore per (cell, repeat)
  residuals.csv         — joined table from join_residuals()
  confusion_matrix.json — dict from score_confusion_matrix()
  report.md             — markdown report rendered via render_report()

Force-cache mechanism
---------------------
Option (b) from the plan: a local ``run_notebook_force_cache()`` shim that
mirrors ``_overhead_driver.run_notebook`` but, after constructing the Cash
instance, sets two config knobs on the live instance:

    cash_instance.config.min_cache_savings_pct = -1.0
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0

``min_cache_savings_pct = -1.0`` makes the ratio gate always pass because the
threshold becomes ``(1 - (-1.0)) * t_compute = 2.0 * t_compute``, which the
estimated restore cost virtually always satisfies.

``min_execution_time_to_cache_seconds = 0.0`` disables the "too cheap to
cache" floor.

Known limitations
-----------------
- Cells annotated with ``# @cash:no-cache`` will not be cached regardless.
- Cells that trigger mutation/side-effect detection in the statement processor
  may still be skipped; the config knobs do not bypass safety gates.
- For those cells, restore_times.jsonl will contain t_restore_actual=None and
  the confusion matrix will have lower n.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

# Make the repo root importable when invoked as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cash  # noqa: F401 — neutralise IPython auto-load side-effect

from benchmarks._cost_model_eval import (
    CellDecision,
    CellRestore,
    classify_oracle,
    join_residuals,
    render_report,
    score_confusion_matrix,
)
from benchmarks._overhead_driver import _enable_cash, run_notebook
from benchmarks._overhead_io import CodeCell, load_code_cells
from benchmarks._overhead_results import CellTiming, StatementMetric


# ---------------------------------------------------------------------------
# Force-cache shim (option b — keeps _overhead_driver.py untouched)
# ---------------------------------------------------------------------------

def run_notebook_force_cache(
    cells: list[CodeCell],
    cache_dir: Path,
) -> list[CellTiming]:
    """Run ``cells`` with cash enabled and all policy gates disabled.

    Identical to ``run_notebook(cells, cash_enabled=True, cache_dir=…)``
    except that two config fields on the live Cash instance are overridden
    immediately after construction:

        config.min_cache_savings_pct = -1.0      (ratio gate always passes)
        config.min_execution_time_to_cache_seconds = 0.0  (floor disabled)

    This maximises the number of cells that get a cache entry written, giving
    ground-truth t_restore_actual observations for the confusion matrix.
    """
    from IPython.core.interactiveshell import InteractiveShell
    from cash.core import Cash
    from cash.notebook.magics import CashMagics
    from cash.notebook.statement import StatementProcessor

    shell = InteractiveShell.instance()
    shell.reset(new_session=True)
    cash.reset_session()

    statement_sink: list[StatementMetric] = []

    # Re-use the class-level tee installed by _enable_cash if already present,
    # or install it now.  We still need our own sink wired up.
    original = getattr(StatementProcessor, "_orig_process_stmt_for_bench", None)
    if original is None:
        original = StatementProcessor.process_statement
        StatementProcessor._orig_process_stmt_for_bench = original  # type: ignore[attr-defined]

    def _teed(self, code, *args, **kwargs):
        result = original(self, code, *args, **kwargs)
        try:
            status = result.get("status", "UNKNOWN")
            status_str = status.value if hasattr(status, "value") else str(status)
            statement_sink.append(StatementMetric(
                code=str(result.get("code", code))[:200],
                execution_time=float(result.get("execution_time", 0.0)),
                total_time=float(result.get("total_time", 0.0)),
                status=status_str,
                cost_model_size_bytes=result.get("cost_model_size_bytes"),
                cost_model_restore_seconds=result.get("cost_model_restore_seconds"),
                cost_model_type_name=result.get("cost_model_type_name"),
                cost_model_family=result.get("cost_model_family"),
            ))
        except Exception:  # noqa: BLE001
            pass
        return result

    StatementProcessor.process_statement = _teed  # type: ignore[method-assign]

    cash_instance = Cash(cache_dir=str(cache_dir), register_magic=False)
    # Disable both policy gates so every cacheable cell writes an entry.
    cash_instance.config.min_cache_savings_pct = -1.0
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0

    magics = CashMagics(shell=shell, cash_instance=cash_instance)
    shell.register_magics(magics)
    magics.cash_on("")

    timings: list[CellTiming] = []
    for cell in cells:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_RE = re.compile(r"#\s*expected:\s*(\S+)")


def _extract_expected(source: str) -> str:
    """Return the expected label from a ``# expected: <label>`` comment."""
    m = _EXPECTED_RE.search(source)
    return m.group(1) if m else "unknown"


def _decisions_from_timings(
    timings: list[CellTiming],
    repeat: int,
    expected_labels: dict[int, str],
) -> list[CellDecision]:
    """Build CellDecision records from a cold+instrumented run."""
    decisions: list[CellDecision] = []
    for ct in timings:
        cell_id = ct.notebook_cell_index
        t_compute = ct.wall_seconds

        # Find the statement metric with the largest cost_model_size_bytes —
        # that's the statement whose output dominates the cache decision.
        # Picking the first hit instead would attribute cells like cell 4
        # (which produces both a small `vocab` list and a big csr_matrix)
        # to the wrong family.
        cost_stmt = None
        best_size = -1
        for sm in ct.statement_metrics:
            sz = sm.cost_model_size_bytes
            if sz is None:
                continue
            if sz > best_size:
                best_size = sz
                cost_stmt = sm

        if cost_stmt is None:
            # No cost_model_* fields at all → hit the 10ms floor
            policy_decision: str = "skip-floor"
            policy_reason: str | None = "below_floor"
            est_obj_size_bytes: int | None = None
            est_restore_seconds: float | None = None
            type_name: str | None = None
            family: str | None = None
        elif cost_stmt.status in ("SKIPPED",):
            policy_decision = "skip-cost"
            policy_reason = "cost_model_skip"
            est_obj_size_bytes = cost_stmt.cost_model_size_bytes
            est_restore_seconds = cost_stmt.cost_model_restore_seconds
            type_name = cost_stmt.cost_model_type_name
            family = cost_stmt.cost_model_family
        elif cost_stmt.status in ("COMPUTED", "RESTORED"):
            policy_decision = "cache"
            policy_reason = None
            est_obj_size_bytes = cost_stmt.cost_model_size_bytes
            est_restore_seconds = cost_stmt.cost_model_restore_seconds
            type_name = cost_stmt.cost_model_type_name
            family = cost_stmt.cost_model_family
        else:
            policy_decision = "skip-other"
            policy_reason = cost_stmt.status
            est_obj_size_bytes = cost_stmt.cost_model_size_bytes
            est_restore_seconds = cost_stmt.cost_model_restore_seconds
            type_name = cost_stmt.cost_model_type_name
            family = cost_stmt.cost_model_family

        decisions.append(CellDecision(
            cell_id=cell_id,
            repeat=repeat,
            t_compute=t_compute,
            est_obj_size_bytes=est_obj_size_bytes,
            est_restore_seconds=est_restore_seconds,
            type_name=type_name,
            family=family,
            policy_decision=policy_decision,  # type: ignore[arg-type]
            policy_reason=policy_reason,
            expected_label=expected_labels.get(cell_id, "unknown"),
        ))
    return decisions


def _restores_from_timings(
    timings_force_cold: list[CellTiming],
    timings_force_warm: list[CellTiming],
    repeat: int,
) -> list[CellRestore]:
    """Build CellRestore records by pairing the force-cache cold and warm runs."""
    warm_by_idx = {ct.notebook_cell_index: ct for ct in timings_force_warm}

    restores: list[CellRestore] = []
    for ct_cold in timings_force_cold:
        cell_id = ct_cold.notebook_cell_index
        ct_warm = warm_by_idx.get(cell_id)

        # t_restore_actual: find the RESTORED statement in the warm run.
        t_restore_actual: float | None = None
        if ct_warm is not None:
            restored_times = [
                sm.total_time
                for sm in ct_warm.statement_metrics
                if sm.status == "RESTORED"
            ]
            if restored_times:
                t_restore_actual = max(restored_times)

        # t_warm_policy: total cell wall time from the warm-after-policy run.
        # We'll fill this in separately; for now leave as None.
        restores.append(CellRestore(
            cell_id=cell_id,
            repeat=repeat,
            t_restore_actual=t_restore_actual,
            t_warm_policy=None,
        ))
    return restores


def _fill_warm_policy_times(
    restores: list[CellRestore],
    timings_warm_policy: list[CellTiming],
) -> None:
    """Mutate restores to fill in t_warm_policy from the warm-after-instrumented run."""
    warm_map = {ct.notebook_cell_index: ct.wall_seconds for ct in timings_warm_policy}
    for r in restores:
        r.t_warm_policy = warm_map.get(r.cell_id)


def _oracle_wall(
    residuals: list[dict],
    epsilon_write: float,
    min_savings_pct: float,
) -> float:
    """Sum of cell times under the oracle policy."""
    total = 0.0
    for row in residuals:
        oracle = classify_oracle(
            row["t_compute"], row["t_restore_actual"], epsilon_write, min_savings_pct
        )
        if oracle == "cache":
            total += row["t_restore_actual"] * (1.0 + epsilon_write)
        else:
            total += row["t_compute"]
    return total


# ---------------------------------------------------------------------------
# Per-repeat runner
# ---------------------------------------------------------------------------

def _run_repeat(
    cells: list[CodeCell],
    cache_root: Path,
    repeat: int,
    expected_labels: dict[int, str],
    epsilon_write: float,
    min_savings_pct: float,
) -> tuple[list[CellDecision], list[CellRestore], dict[str, float]]:
    """Run all four passes for one repeat and return (decisions, restores, wall)."""

    # --- Pass 1: cold + instrumented -------------------------------------------
    cache_dir_policy = cache_root / f"repeat_{repeat}_policy"
    if cache_dir_policy.exists():
        shutil.rmtree(cache_dir_policy)
    cache_dir_policy.mkdir(parents=True)

    print(f"  [repeat {repeat}] pass 1/4: cold+instrumented …", flush=True)
    timings_cold = run_notebook(cells, cash_enabled=True, cache_dir=cache_dir_policy)
    decisions = _decisions_from_timings(timings_cold, repeat, expected_labels)
    wall_cold = sum(ct.wall_seconds for ct in timings_cold)

    # --- Pass 2: warm-after-instrumented ----------------------------------------
    print(f"  [repeat {repeat}] pass 2/4: warm-after-instrumented …", flush=True)
    timings_warm_policy = run_notebook(cells, cash_enabled=True, cache_dir=cache_dir_policy)
    wall_warm_policy = sum(ct.wall_seconds for ct in timings_warm_policy)

    # --- Pass 3: cold + force-cache ---------------------------------------------
    cache_dir_force = cache_root / f"repeat_{repeat}_force"
    if cache_dir_force.exists():
        shutil.rmtree(cache_dir_force)
    cache_dir_force.mkdir(parents=True)

    print(f"  [repeat {repeat}] pass 3/4: cold+force-cache …", flush=True)
    timings_force_cold = run_notebook_force_cache(cells, cache_dir_force)

    # --- Pass 4: warm-after-force-cache -----------------------------------------
    # We re-use the same cache_dir so disk entries from pass 3 are visible.
    # cash.reset_session() inside run_notebook_force_cache clears in-memory
    # state but leaves disk entries intact.
    print(f"  [repeat {repeat}] pass 4/4: warm-after-force-cache …", flush=True)
    timings_force_warm = run_notebook_force_cache(cells, cache_dir_force)

    # Build restores and fill warm-policy times.
    restores = _restores_from_timings(timings_force_cold, timings_force_warm, repeat)
    _fill_warm_policy_times(restores, timings_warm_policy)

    # Compute oracle wall-clock for this repeat using joined residuals.
    repeat_residuals = join_residuals(decisions, restores, epsilon_write, min_savings_pct)
    wall_oracle = _oracle_wall(repeat_residuals, epsilon_write, min_savings_pct)

    wall = {
        "cold": wall_cold,
        "warm_policy": wall_warm_policy,
        "warm_oracle": wall_oracle,
    }
    return decisions, restores, wall


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        lines.append(json.dumps(rec.__dict__))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-model validation CLI: runs 4 passes per repeat and emits eval artifacts.",
    )
    parser.add_argument(
        "--notebook",
        default="benchmarks/bench_cost_model_validation.ipynb",
        help="Path to the validation notebook (default: benchmarks/bench_cost_model_validation.ipynb)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats (repeat 0 is warmup and discarded, default: 3)",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results/cost_model_validation",
        help="Directory to write artifacts (default: benchmarks/results/cost_model_validation)",
    )
    parser.add_argument(
        "--epsilon-write",
        type=float,
        default=0.10,
        help="Write-overhead factor epsilon for oracle classification (default: 0.10)",
    )
    parser.add_argument(
        "--min-savings-pct",
        type=float,
        default=0.20,
        help="Minimum savings fraction for oracle (default: 0.20)",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    results_dir = Path(args.results_dir)
    cache_root = results_dir / "_cache"
    epsilon_write: float = args.epsilon_write
    min_savings_pct: float = args.min_savings_pct

    cells = load_code_cells(notebook_path)
    expected_labels: dict[int, str] = {
        cell.notebook_cell_index: _extract_expected(cell.source)
        for cell in cells
    }

    print(f"Loaded {len(cells)} code cells from {notebook_path}")
    print(f"Repeats: {args.repeats}  (repeat 0 = warmup, discarded)")
    print(f"Results -> {results_dir}\n")

    all_decisions: list[CellDecision] = []
    all_restores: list[CellRestore] = []
    wall_accum: dict[str, list[float]] = {"cold": [], "warm_policy": [], "warm_oracle": []}

    for rep in range(args.repeats):
        print(f"--- Repeat {rep} {'(warmup — will be discarded)' if rep == 0 else ''} ---")
        decisions, restores, wall = _run_repeat(
            cells, cache_root, rep,
            expected_labels, epsilon_write, min_savings_pct,
        )
        if rep == 0:
            print("  Warmup repeat discarded.\n")
            continue
        all_decisions.extend(decisions)
        all_restores.extend(restores)
        for k in wall_accum:
            wall_accum[k].append(wall[k])
        print()

    if not all_decisions:
        print("WARNING: all repeats were warmup (--repeats=1). "
              "Using warmup data for the report.")
        # Re-run repeat 0 and use its data for reporting.
        print("--- Repeat 0 (forced single-repeat mode) ---")
        decisions, restores, wall = _run_repeat(
            cells, cache_root, 0,
            expected_labels, epsilon_write, min_savings_pct,
        )
        all_decisions.extend(decisions)
        all_restores.extend(restores)
        for k in wall_accum:
            wall_accum[k].append(wall[k])

    # Aggregate wall-clock (mean across kept repeats).
    mean_wall = {k: (sum(v) / len(v) if v else 0.0) for k, v in wall_accum.items()}

    # Join + score.
    residuals = join_residuals(all_decisions, all_restores, epsilon_write, min_savings_pct)
    cm = score_confusion_matrix(residuals)
    n_repeats_kept = len(wall_accum["cold"])

    # Render report.
    report_text = render_report(
        cm=cm,
        residuals=residuals,
        wall=mean_wall,
        n_cells=len(cells),
        n_repeats=max(n_repeats_kept, 1),
    )

    # Write artifacts.
    results_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(results_dir / "decisions.jsonl", all_decisions)
    _write_jsonl(results_dir / "restore_times.jsonl", all_restores)
    _write_csv(results_dir / "residuals.csv", residuals)
    (results_dir / "confusion_matrix.json").write_text(
        json.dumps(cm, indent=2), encoding="utf-8"
    )
    (results_dir / "report.md").write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)
    print(f"\nArtifacts written to {results_dir}/")
    for name in ("decisions.jsonl", "restore_times.jsonl", "residuals.csv",
                 "confusion_matrix.json", "report.md"):
        p = results_dir / name
        size = p.stat().st_size if p.exists() else 0
        print(f"  {name}: {size} bytes")


if __name__ == "__main__":
    main()
