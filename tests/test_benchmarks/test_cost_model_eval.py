"""Unit tests for the cost-model eval helpers."""
from __future__ import annotations

import pytest

from benchmarks._cost_model_eval import (
    CellDecision,
    CellRestore,
    classify_oracle,
    join_residuals,
    score_confusion_matrix,
    render_report,
)


def test_classify_oracle_caches_when_restore_much_cheaper():
    # compute = 1.0s, actual restore = 0.05s, write overhead ε=0.10
    # → caching saves 1.0 - 0.05*1.10 = 0.945s; way > 20% threshold
    assert classify_oracle(
        t_compute=1.0, t_restore_actual=0.05,
        epsilon_write=0.10, min_savings_pct=0.20,
    ) == "cache"


def test_classify_oracle_skips_when_restore_dominates():
    # compute = 0.5s, restore = 1.0s → never worth caching
    assert classify_oracle(
        t_compute=0.5, t_restore_actual=1.0,
        epsilon_write=0.10, min_savings_pct=0.20,
    ) == "skip"


def test_classify_oracle_skips_when_savings_below_threshold():
    # compute = 1.0s, restore = 0.85s, ε=0.10 → effective restore 0.935s
    # savings = 0.065s = 6.5% of compute, below 20% threshold
    assert classify_oracle(
        t_compute=1.0, t_restore_actual=0.85,
        epsilon_write=0.10, min_savings_pct=0.20,
    ) == "skip"


def test_join_residuals_pairs_decision_and_restore():
    decisions = [
        CellDecision(cell_id=2, repeat=1, t_compute=2.0,
                     est_obj_size_bytes=50_000_000,
                     est_restore_seconds=0.08,
                     type_name="DataFrame", family="dataframe_numeric",
                     policy_decision="cache",
                     policy_reason=None,
                     expected_label="cache"),
    ]
    restores = [
        CellRestore(cell_id=2, repeat=1, t_restore_actual=0.35,
                    t_warm_policy=0.35),
    ]
    residuals = join_residuals(decisions, restores)
    assert len(residuals) == 1
    r = residuals[0]
    assert r["cell_id"] == 2
    assert r["family"] == "dataframe_numeric"
    assert r["t_restore_predicted"] == 0.08
    assert r["t_restore_actual"] == 0.35
    assert abs(r["abs_err"] - 0.27) < 1e-9
    assert abs(r["rel_err"] - (0.27 / 0.35)) < 1e-9


def test_score_confusion_matrix_counts_each_quadrant():
    rows = [
        {"policy_decision": "cache", "oracle_decision": "cache"},   # TP
        {"policy_decision": "cache", "oracle_decision": "cache"},   # TP
        {"policy_decision": "skip-cost", "oracle_decision": "skip"},  # TN
        {"policy_decision": "skip-floor", "oracle_decision": "skip"},  # TN
        {"policy_decision": "cache", "oracle_decision": "skip"},    # FP
        {"policy_decision": "skip-cost", "oracle_decision": "cache"},  # FN
    ]
    cm = score_confusion_matrix(rows)
    assert cm["TP"] == 2
    assert cm["TN"] == 2
    assert cm["FP"] == 1
    assert cm["FN"] == 1
    assert cm["accuracy"] == pytest.approx(4 / 6)


def test_render_report_contains_required_sections():
    cm = {"TP": 5, "TN": 2, "FP": 1, "FN": 0, "accuracy": 7 / 8}
    residuals = [
        {"cell_id": 2, "repeat": 1, "family": "dataframe_numeric",
         "size_mb": 50.0, "t_compute": 2.0,
         "t_restore_predicted": 0.08, "t_restore_actual": 0.35,
         "abs_err": 0.27, "rel_err": 0.77,
         "policy_decision": "cache", "oracle_decision": "cache",
         "expected_label": "cache"},
    ]
    wall = {"cold": 60.0, "warm_policy": 8.0, "warm_oracle": 6.5}
    report = render_report(cm, residuals, wall, n_cells=12, n_repeats=3)
    assert "# Cost-Model Validation Report" in report
    assert "## Confusion matrix" in report
    assert "## Top 5 worst absolute errors" in report
    assert "## Per-family residual summary" in report
    assert "## Counterfactual wall-clock" in report
    assert "## Findings" in report
    assert "5" in report  # TP count appears somewhere
