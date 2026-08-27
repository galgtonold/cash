"""Notebook *edit* benchmark: what does changing one cell cost?

The overhead benchmark (``bench_notebook_overhead.py``) runs a notebook
unchanged in off / cold / warm-session / warm-restart. That measures the
best case. This one measures the working case: populate a cache, change a
cell, run again, and count the compute cash redoes that the edit could not
possibly have invalidated.

Every scenario is a matched pair of full runs -- prime, then measure --
against its own private cache directory. Nothing is shared between
scenarios, so one scenario cannot warm or poison another. It costs twice
the runs and removes an entire class of doubt about the numbers.

Usage:
    python benchmarks/bench_notebook_edit.py <notebook> [--results-dir DIR]
        [--max-sites N] [--session {restart,live}] [--quiet]

Read the numbers as:

    restorable   statements that restore when NOTHING is edited (the
                 ceiling -- cash cannot save more than this)
    wasted       of those, how many recomputed anyway after an edit that
                 changed nothing they depend on

``wasted`` should be zero. Every non-zero row is an open effectiveness
bug with a reproduction attached.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import shutil
import sys
import time
from pathlib import Path

# Make the benchmarks package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._edit_scenarios import (  # noqa: E402
    EditScenario,
    ScenarioResult,
    attribute_waste,
    build_cells,
    plan_scenarios,
)
from benchmarks._overhead_driver import new_cash_session, run_notebook  # noqa: E402
from benchmarks._overhead_io import CodeCell, load_code_cells  # noqa: E402


def _metrics_by_cell(timings) -> dict[int, list]:
    return {t.index: list(t.statement_metrics) for t in timings}


def _fresh(cache_dir: Path) -> Path:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _run_pair(
    cells: list[CodeCell],
    scenario: EditScenario,
    cache_dir: Path,
    session_mode: str,
) -> dict[int, list]:
    """Prime the cache with the scenario's "before" cells, then measure its
    "after" cells against it. Returns the measured run's statement metrics.

    ``session_mode`` decides what survives between the two halves:

        restart   a new ``Cash`` for the measured run -- only what the cost
                  model wrote to DISK can come back. This is the sharper
                  test and the default.
        live      one ``Cash`` across both -- the RAM tier survives, which
                  is what re-running a cell in a live kernel actually gets.
    """
    _fresh(cache_dir)
    session = new_cash_session(cache_dir)

    run_notebook(
        build_cells(cells, scenario, edited=False),
        cash_enabled=True, cache_dir=cache_dir, session=session,
    )
    measured = run_notebook(
        build_cells(cells, scenario, edited=True),
        cash_enabled=True, cache_dir=cache_dir,
        session=session if session_mode == "live" else None,
    )
    return _metrics_by_cell(measured)


def _control_pair(
    cells: list[CodeCell], cache_dir: Path, session_mode: str,
) -> dict[int, list]:
    """The no-edit baseline: the same prime-then-measure protocol with no
    edit applied. Anything that restores here is what cash *can* save on
    this notebook; anything that does not is uncacheable or below the cost
    model's threshold, and recomputing it later is correct, not waste.
    """
    noop = EditScenario(kind="comment", site=-1, label="control")
    return _run_pair(cells, noop, cache_dir, session_mode)


def _restorable_by_cell(control: dict[int, list]) -> tuple[dict, dict]:
    seconds: dict[int, float] = {}
    counts: dict[int, int] = {}
    for cell_index, metrics in control.items():
        for m in metrics:
            if m.status == "RESTORED":
                seconds[cell_index] = seconds.get(cell_index, 0.0) + (
                    m.execution_time or 0.0)
                counts[cell_index] = counts.get(cell_index, 0) + 1
    return seconds, counts


def run_notebook_edit_benchmark(
    notebook_path: Path,
    work_dir: Path,
    max_sites: int = 3,
    session_mode: str = "restart",
    log=print,
) -> dict:
    """Run the full edit benchmark for one notebook and return a result dict."""
    cells = load_code_cells(notebook_path)
    t0 = time.perf_counter()

    log("  control run (no edit)...")
    control = _control_pair(cells, work_dir / "control", session_mode)
    seconds_by_cell, counts_by_cell = _restorable_by_cell(control)
    total_restorable = sum(counts_by_cell.values())
    log(f"    {total_restorable} statement(s) restore with no edit, "
        f"{sum(seconds_by_cell.values()):.2f}s of compute")

    scenarios = plan_scenarios(cells, seconds_by_cell, counts_by_cell,
                               max_sites=max_sites)
    if not scenarios:
        log("    nothing restorable below any cell -- no scenarios to run")

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        log(f"  {scenario.label}...")
        edited = _run_pair(cells, scenario,
                           work_dir / scenario.label.replace("@", "_"),
                           session_mode)
        result = attribute_waste(scenario, control, edited)
        results.append(result)
        if scenario.kind == "linked":
            verdict = ("OK" if result.control_sink_recomputed
                       else "BROKEN -- harness cannot see a real dependency")
            log(f"    positive control: {verdict}")
        else:
            log(f"    wasted {result.wasted_count}/{result.restorable_count} "
                f"statements, {result.wasted_seconds:.2f}s")

    return {
        "notebook": str(notebook_path),
        "session_mode": session_mode,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - t0,
        "restorable_count": total_restorable,
        "restorable_seconds": sum(seconds_by_cell.values()),
        "scenarios": [dataclasses.asdict(r) for r in results],
    }


def _print_table(report: dict) -> None:
    name = Path(report["notebook"]).name
    print()
    print(f"{name}  ({report['session_mode']}, "
          f"{report['restorable_count']} restorable statements)")
    print(f"  {'scenario':28s} {'wasted':>14s} {'of restorable':>14s}")
    for s in report["scenarios"]:
        if s["kind"] == "linked":
            verdict = "OK" if s["control_sink_recomputed"] else "BROKEN"
            print(f"  {s['label']:28s} {'positive control':>14s} {verdict:>14s}")
            continue
        share = (f"{s['wasted_count']}/{s['restorable_count']}"
                 if s["restorable_count"] else "-")
        print(f"  {s['label']:28s} {s['wasted_seconds']:11.2f}s  {share:>14s}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Notebook edit benchmark")
    p.add_argument("notebook", type=Path)
    p.add_argument("--results-dir", type=Path,
                   default=Path("benchmarks/results_edit"))
    p.add_argument("--max-sites", type=int, default=3,
                   help="how many distinct cells to edit (default: 3)")
    p.add_argument("--session", dest="session_mode", default="restart",
                   choices=["restart", "live"],
                   help="'restart' measures the disk tier only (default); "
                        "'live' keeps the RAM tier across the edit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = results_dir / "_caches" / args.notebook.stem
    log = (lambda *a, **k: None) if args.quiet else print

    log(f"{args.notebook}")
    report = run_notebook_edit_benchmark(
        args.notebook, work_dir,
        max_sites=args.max_sites, session_mode=args.session_mode, log=log)

    out = results_dir / f"{args.notebook.stem}-edit-{args.session_mode}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not args.quiet:
        _print_table(report)
        print(f"\nwrote {out}")
    shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
