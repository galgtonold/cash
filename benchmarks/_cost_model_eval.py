"""Pure-function helpers for the cost-model validation eval.

Splits joining + scoring + report-rendering from the CLI orchestrator
so the logic is unit-testable without a live IPython shell.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Literal


PolicyDecision = Literal["cache", "skip-floor", "skip-cost", "skip-other"]
OracleDecision = Literal["cache", "skip"]


@dataclass
class CellDecision:
    """One per (cell, repeat) from the instrumented cold run."""
    cell_id: int
    repeat: int
    t_compute: float
    est_obj_size_bytes: int | None
    est_restore_seconds: float | None
    type_name: str | None
    family: str | None
    policy_decision: PolicyDecision
    policy_reason: str | None
    expected_label: str  # from the cell's "# expected:" comment


@dataclass
class CellRestore:
    """One per (cell, repeat) from the force-cache warm run."""
    cell_id: int
    repeat: int
    t_restore_actual: float | None
    t_warm_policy: float | None  # may differ from t_restore_actual


def classify_oracle(
    t_compute: float,
    t_restore_actual: float | None,
    epsilon_write: float,
    min_savings_pct: float,
) -> OracleDecision:
    """Ground-truth decision: would caching actually save time?

    The model: effective_restore = t_restore_actual × (1 + ε); cache iff
    effective_restore < (1 - min_savings_pct) × t_compute.
    """
    if t_restore_actual is None or t_compute <= 0:
        return "skip"
    effective = t_restore_actual * (1.0 + epsilon_write)
    threshold = (1.0 - min_savings_pct) * t_compute
    return "cache" if effective < threshold else "skip"


def join_residuals(
    decisions: list[CellDecision],
    restores: list[CellRestore],
    epsilon_write: float = 0.10,
    min_savings_pct: float = 0.20,
) -> list[dict[str, Any]]:
    """Join decisions and restores by (cell_id, repeat). Compute residuals
    and oracle decisions. Cells with no restore observation are dropped."""
    restore_idx = {(r.cell_id, r.repeat): r for r in restores}
    rows: list[dict[str, Any]] = []
    for d in decisions:
        r = restore_idx.get((d.cell_id, d.repeat))
        if r is None or r.t_restore_actual is None:
            continue
        pred = d.est_restore_seconds if d.est_restore_seconds is not None else float("nan")
        actual = r.t_restore_actual
        oracle = classify_oracle(d.t_compute, actual, epsilon_write, min_savings_pct)
        abs_err = abs(actual - pred) if pred == pred else float("nan")
        rel_err = (abs_err / actual) if (pred == pred and actual > 0) else float("nan")
        rows.append({
            "cell_id": d.cell_id,
            "repeat": d.repeat,
            "family": d.family or "unknown",
            "size_mb": (d.est_obj_size_bytes or 0) / 1e6,
            "t_compute": d.t_compute,
            "t_restore_predicted": pred,
            "t_restore_actual": actual,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "policy_decision": d.policy_decision,
            "oracle_decision": oracle,
            "expected_label": d.expected_label,
        })
    return rows


def score_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, float]:
    """TP/TN/FP/FN counts. Policy 'cache' = predicted positive."""
    tp = sum(1 for r in rows
             if r["policy_decision"] == "cache" and r["oracle_decision"] == "cache")
    fn = sum(1 for r in rows
             if r["policy_decision"].startswith("skip") and r["oracle_decision"] == "cache")
    tn = sum(1 for r in rows
             if r["policy_decision"].startswith("skip") and r["oracle_decision"] == "skip")
    fp = sum(1 for r in rows
             if r["policy_decision"] == "cache" and r["oracle_decision"] == "skip")
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "accuracy": acc}


def _family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median + p95 of rel_err per family."""
    by_family: dict[str, list[float]] = {}
    for r in rows:
        if r["rel_err"] == r["rel_err"]:  # not NaN
            by_family.setdefault(r["family"], []).append(r["rel_err"])
    out = []
    for family, errs in sorted(by_family.items()):
        errs_sorted = sorted(errs)
        n = len(errs_sorted)
        median = statistics.median(errs_sorted)
        p95 = errs_sorted[int(0.95 * (n - 1))] if n > 0 else float("nan")
        out.append({
            "family": family, "n": n,
            "median_rel_err": median, "p95_rel_err": p95,
        })
    return out


def render_report(
    cm: dict[str, float],
    residuals: list[dict[str, Any]],
    wall: dict[str, float],
    n_cells: int,
    n_repeats: int,
) -> str:
    """Render the markdown report.

    ``wall`` keys: ``cold``, ``warm_policy``, ``warm_oracle`` (seconds).
    """
    lines: list[str] = []
    lines.append("# Cost-Model Validation Report\n")
    lines.append(f"**Cells:** {n_cells} | **Repeats:** {n_repeats} | "
                 f"**Accuracy:** {cm['accuracy']*100:.1f}%\n")

    lines.append("## Confusion matrix\n")
    lines.append("| | oracle=cache | oracle=skip |")
    lines.append("|---|---|---|")
    lines.append(f"| policy=cache | TP={cm['TP']} | FP={cm['FP']} |")
    lines.append(f"| policy=skip  | FN={cm['FN']} | TN={cm['TN']} |\n")

    lines.append("## Top 5 worst absolute errors\n")
    worst = sorted(
        residuals,
        key=lambda r: r["abs_err"] if r["abs_err"] == r["abs_err"] else -1,
        reverse=True,
    )[:5]
    lines.append("| cell | family | size MB | pred (s) | actual (s) | abs err (s) | rel err |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in worst:
        lines.append(
            f"| {r['cell_id']} | {r['family']} | {r['size_mb']:.1f} | "
            f"{r['t_restore_predicted']:.3f} | {r['t_restore_actual']:.3f} | "
            f"{r['abs_err']:.3f} | {r['rel_err']*100:.0f}% |"
        )
    lines.append("")

    lines.append("## Per-family residual summary\n")
    lines.append("| family | n | median rel err | p95 rel err |")
    lines.append("|---|---|---|---|")
    for fs in _family_summary(residuals):
        lines.append(f"| {fs['family']} | {fs['n']} | "
                     f"{fs['median_rel_err']*100:.0f}% | "
                     f"{fs['p95_rel_err']*100:.0f}% |")
    lines.append("")

    lines.append("## Counterfactual wall-clock\n")
    headroom = ((wall["warm_policy"] - wall["warm_oracle"]) / wall["warm_policy"]
                if wall["warm_policy"] > 0 else 0.0)
    lines.append(f"- Cold (no cache): **{wall['cold']:.1f}s**")
    lines.append(f"- Warm under current policy: **{wall['warm_policy']:.1f}s**")
    lines.append(f"- Warm under oracle policy: **{wall['warm_oracle']:.1f}s**")
    lines.append(f"- Headroom: **{headroom*100:.1f}%** of the warm wall could be reclaimed "
                 f"by a perfect cache-or-skip oracle.\n")

    lines.append("## Findings\n")
    worst_family = max(_family_summary(residuals),
                       key=lambda f: f["median_rel_err"], default=None)
    if worst_family is not None:
        lines.append(
            f"- The cost model's worst-fit family in this run is "
            f"**`{worst_family['family']}`** "
            f"(median rel err {worst_family['median_rel_err']*100:.0f}%, "
            f"p95 {worst_family['p95_rel_err']*100:.0f}%, n={worst_family['n']})."
        )
    lines.append(
        f"- Policy made **{int(cm['FP'])} false-positive caches** "
        f"(cached when oracle says skip) and "
        f"**{int(cm['FN'])} false-negative skips** "
        f"(skipped when oracle says cache)."
    )
    if headroom < 0.05:
        lines.append("- Headroom is under 5% — the policy is essentially as good as the oracle on this workload.")
    elif headroom < 0.20:
        lines.append("- Headroom is meaningful but not large; targeted family remapping or refit could close most of it.")
    else:
        lines.append("- Headroom is large; the policy is leaving substantial wall-clock on the table.")
    lines.append("")

    return "\n".join(lines)
