"""Diff two labelled runs of the size-estimator reference suite.

Reads ``benchmarks/results/size_estimator_eval/before/`` and
``benchmarks/results/size_estimator_eval/after/`` produced by
``eval_size_estimator.py``. Emits ``report.md`` with five sections:
per-notebook wall-time deltas, cold-mode regression watch, warm-mode
improvement table, cell-level decision-flip table, aggregate summary.

Usage:
    python benchmarks/diff_size_estimator.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "benchmarks" / "results" / "size_estimator_eval"


def _load_runs(label_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """Group run-JSONs by (notebook_stem, mode). Returns {(stem, mode): [run_dict, ...]}."""
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not label_dir.exists():
        return out
    for path in sorted(label_dir.glob("*-*-*.json")):
        # filename pattern: <stem>-<mode>-<repeat>.json
        # stems can contain dashes, so split from the right
        name = path.stem
        try:
            stem_mode, repeat_str = name.rsplit("-", 1)
            int(repeat_str)  # validate
            stem, mode = stem_mode.rsplit("-", 1)
        except (ValueError, IndexError):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        out[(stem, mode)].append(data)
    return out


def _median_wall_ms(runs: list[dict]) -> float | None:
    """Median total wall time (ms) across non-warmup repeats."""
    non_warmup = [r for r in runs if r.get("repeat", 0) > 0]
    if not non_warmup:
        return None
    return statistics.median(r["total_wall_seconds"] * 1000.0 for r in non_warmup)


def _cell_decisions(runs: list[dict]) -> dict[int, str]:
    """Per-cell policy decision from the most recent non-warmup run."""
    non_warmup = [r for r in runs if r.get("repeat", 0) > 0]
    if not non_warmup:
        return {}
    r = non_warmup[-1]
    out: dict[int, str] = {}
    for cell in r["cells"]:
        cell_idx = cell["notebook_cell_index"]
        with_pred = [
            sm for sm in cell["statement_metrics"]
            if sm.get("cost_model_family") is not None
        ]
        if not with_pred:
            out[cell_idx] = "skip-floor"
            continue
        # Pick the largest-output statement
        winner = max(with_pred, key=lambda sm: sm.get("cost_model_size_bytes") or 0)
        status = winner.get("status", "")
        if status == "SKIPPED":
            out[cell_idx] = "skip-cost"
        elif status in ("COMPUTED", "RESTORED"):
            out[cell_idx] = "cache"
        else:
            out[cell_idx] = "skip-other"
    return out


def _cell_family(runs: list[dict], cell_idx: int) -> str | None:
    non_warmup = [r for r in runs if r.get("repeat", 0) > 0]
    if not non_warmup:
        return None
    r = non_warmup[-1]
    for cell in r["cells"]:
        if cell["notebook_cell_index"] != cell_idx:
            continue
        with_pred = [
            sm for sm in cell["statement_metrics"]
            if sm.get("cost_model_family") is not None
        ]
        if not with_pred:
            return None
        winner = max(with_pred, key=lambda sm: sm.get("cost_model_size_bytes") or 0)
        return winner.get("cost_model_family")
    return None


def render_report(before: dict, after: dict) -> str:
    keys = sorted(set(before.keys()) | set(after.keys()))

    lines: list[str] = []
    lines.append("# Size-estimator Before/After Report\n")

    # Section 1: per-notebook wall-time
    lines.append("## Per-notebook wall-time (median of non-warmup repeats)\n")
    lines.append("| notebook | mode | before ms | after ms | delta ms | delta % |")
    lines.append("|---|---|---:|---:|---:|---:|")
    cold_deltas_pct: list[float] = []
    warm_deltas_pct: list[float] = []
    cold_total_before = 0.0
    cold_total_after = 0.0
    warm_total_before = 0.0
    warm_total_after = 0.0
    for stem, mode in keys:
        b = _median_wall_ms(before.get((stem, mode), []))
        a = _median_wall_ms(after.get((stem, mode), []))
        if b is None or a is None:
            lines.append(f"| {stem} | {mode} | "
                         f"{b if b is not None else '-'} | "
                         f"{a if a is not None else '-'} | - | - |")
            continue
        delta = a - b
        pct = (delta / b * 100.0) if b > 0 else 0.0
        lines.append(f"| {stem} | {mode} | {b:.0f} | {a:.0f} | {delta:+.0f} | {pct:+.1f}% |")
        if mode == "cold":
            cold_deltas_pct.append(pct)
            cold_total_before += b
            cold_total_after += a
        elif mode == "warm":
            warm_deltas_pct.append(pct)
            warm_total_before += b
            warm_total_after += a
    lines.append("")

    # Section 2: cold regression watch
    lines.append("## Cold-mode regression watch (gate: any > +5% is a red flag)\n")
    cold_regressions = [
        (stem, pct) for (stem, mode), pct in zip(
            ((s, m) for s, m in keys if m == "cold"), cold_deltas_pct
        ) if pct > 5.0
    ]
    if cold_regressions:
        lines.append("| notebook | cold delta % |")
        lines.append("|---|---:|")
        for stem, pct in cold_regressions:
            lines.append(f"| {stem} | **+{pct:.1f}%** |")
    else:
        lines.append("No notebook exceeded +5% cold-mode regression.")
    cold_median = statistics.median(cold_deltas_pct) if cold_deltas_pct else 0.0
    lines.append(f"\n_Median cold delta across the suite: **{cold_median:+.1f}%**_\n")

    # Section 3: warm improvement table
    lines.append("## Warm-mode improvement table (gate: <= -10% on at least one)\n")
    warm_improvements = [
        (stem, pct) for (stem, mode), pct in zip(
            ((s, m) for s, m in keys if m == "warm"), warm_deltas_pct
        ) if pct < -10.0
    ]
    if warm_improvements:
        lines.append("| notebook | warm delta % |")
        lines.append("|---|---:|")
        for stem, pct in warm_improvements:
            lines.append(f"| {stem} | **{pct:+.1f}%** |")
    else:
        lines.append("No notebook improved by >=10% in warm mode. This is the warm gate; investigate before declaring the change successful.")
    lines.append("")

    # Section 4: cell-level decision flips
    lines.append("## Cell-level decision flips (per notebook)\n")
    any_flips = False
    for (stem, mode) in keys:
        if mode != "warm":
            continue
        b_decisions = _cell_decisions(before.get((stem, mode), []))
        a_decisions = _cell_decisions(after.get((stem, mode), []))
        flips = []
        for cell_idx in sorted(set(b_decisions) | set(a_decisions)):
            b_d = b_decisions.get(cell_idx, "-")
            a_d = a_decisions.get(cell_idx, "-")
            if b_d != a_d:
                family = _cell_family(after.get((stem, mode), []), cell_idx) or "-"
                flips.append((cell_idx, b_d, a_d, family))
        if flips:
            any_flips = True
            lines.append(f"\n### {stem}\n")
            lines.append("| cell | before | after | family (after) |")
            lines.append("|---:|---|---|---|")
            for cell_idx, b_d, a_d, family in flips:
                lines.append(f"| {cell_idx} | {b_d} | {a_d} | {family} |")
    if not any_flips:
        lines.append("No cells flipped policy decision between before and after.")
    lines.append("")

    # Section 5: aggregate summary
    lines.append("## Aggregate summary\n")
    if cold_total_before > 0:
        cold_net = (cold_total_after - cold_total_before) / cold_total_before * 100.0
        lines.append(f"- **Cold total**: {cold_total_before:.0f} ms -> {cold_total_after:.0f} ms "
                     f"(**{cold_net:+.1f}%** net)")
    if warm_total_before > 0:
        warm_net = (warm_total_after - warm_total_before) / warm_total_before * 100.0
        lines.append(f"- **Warm total**: {warm_total_before:.0f} ms -> {warm_total_after:.0f} ms "
                     f"(**{warm_net:+.1f}%** net)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diff before/after size-estimator runs")
    p.add_argument("--root", type=Path, default=ROOT,
                   help="Parent of the before/ and after/ subdirs.")
    p.add_argument("--out", type=Path,
                   help="Where to write report.md. Defaults to <root>/report.md.")
    args = p.parse_args(argv)

    before = _load_runs(args.root / "before")
    after = _load_runs(args.root / "after")
    if not before:
        print(f"!! No 'before' runs found under {args.root / 'before'}")
        return 2
    if not after:
        print(f"!! No 'after' runs found under {args.root / 'after'}")
        return 2

    report = render_report(before, after)
    out_path = args.out or (args.root / "report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
