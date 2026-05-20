"""Hand-authored metrics_list fixtures for docs/badges.md badge examples.

Each entry maps a scenario name to a metrics_list that conforms to the
input contract of cash.notebook.badge_renderer.view_builder
.build_interactive_badge. Keys used are documented in
cash.notebook.badge_renderer.view_builder._statement_row_from_metric.

Timing values and cache_key prefixes are plausible-looking but
hand-chosen — the rendered HTML structure is what matters for docs.
"""
from __future__ import annotations

from typing import Any

Metric = dict[str, Any]
MetricsList = list[Metric]


FIXTURES: dict[str, MetricsList] = {
    # §3 status reference: a single RESTORED row.
    "status_restored": [
        {
            "status": "RESTORED",
            "code": "df = pd.read_csv('sales.csv')",
            "total_time": 0.018,
            "saved_time": 2.847,
            "evaluated_vars": ["df"],
            "restored_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "source": "RAM",
            "cache_key": "stmt:7a3f0b1c9e2d4567",
            "is_upstream": False,
        },
    ],
    "status_computed": [
        {
            "status": "COMPUTED",
            "code": "summary = df.describe()",
            "total_time": 0.214,
            "evaluated_vars": ["summary"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:c1e8a72f4b9d0156",
            "is_upstream": False,
        },
    ],
    "status_skipped": [
        {
            "status": "SKIPPED",
            "code": "result = expensive_call(x)",
            "total_time": 0.0,
            "evaluated_vars": ["result"],
            "skipped_reason": "downstream value not requested",
            "is_upstream": False,
        },
    ],
    "status_mixed": [
        {
            "status": "RESTORED",
            "code": "df = pd.read_csv('sales.csv')",
            "total_time": 0.018,
            "saved_time": 2.847,
            "evaluated_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:7a3f0b1c9e2d4567",
            "is_upstream": False,
        },
        {
            "status": "COMPUTED",
            "code": "summary = df.describe()",
            "total_time": 0.214,
            "evaluated_vars": ["summary"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:c1e8a72f4b9d0156",
            "is_upstream": False,
        },
    ],
    "status_function_changed": [
        {
            "status": "FUNCTION_CHANGED",
            "code": "scores = score_rows(df)",
            "total_time": 0.041,
            "changed_functions": ["score_rows"],
            "is_upstream": False,
        },
    ],
    "status_module_reloaded": [
        {
            "status": "MODULE_RELOADED",
            "code": "from features import build_features",
            "total_time": 0.012,
            "changed_modules": {"features": "..."},
            "is_upstream": False,
        },
    ],
    "status_warning": [
        {
            "status": "WARNING",
            "code": "x = np.random.rand(1000)",
            "total_time": 0.003,
            "evaluated_vars": ["x"],
            "uncacheable_reasons": ["unseeded random call: numpy.random.rand"],
            "is_upstream": False,
        },
    ],
    "status_error": [
        {
            "status": "ERROR",
            "code": "result = will_raise()",
            "total_time": 0.005,
            "is_upstream": False,
        },
    ],
    # §4.a — cold start: first time seeing this code, no prior lineage.
    "miss_first_time": [
        {
            "status": "COMPUTED",
            "code": "model = train(features, labels)",
            "total_time": 4.812,
            "evaluated_vars": ["model"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:3f2a8e019c7b4d12",
            "miss_reason": "first time seeing this code",
            "is_upstream": False,
        },
    ],
    # §4.b — input lineage changed: an upstream variable was recomputed.
    "miss_input_lineage": [
        {
            "status": "RESTORED",
            "code": "features = encode(df)",
            "total_time": 0.024,
            "saved_time": 0.612,
            "evaluated_vars": ["features"],
            "storage": ["RAM"],
            "cache_key": "stmt:88a1c2d3e4f50617",
            "is_upstream": True,
        },
        {
            "status": "COMPUTED",
            "code": "model = train(features, labels)",
            "total_time": 4.812,
            "evaluated_vars": ["model"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:9d4e2f10a1b6c7d8",
            "miss_reason": "input lineage changed (one of: features)",
            "is_upstream": False,
        },
    ],
    # §4.c — file changed: tracked CSV mtime/hash differs from last run.
    "miss_file_changed": [
        {
            "status": "COMPUTED",
            "code": "df = pd.read_csv('sales.csv')",
            "total_time": 2.913,
            "evaluated_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:4b8c1e3a9f2d5067",
            "miss_reason": "file changed: sales.csv",
            "is_upstream": False,
        },
    ],
    # §4.d — function source changed: a helper this cell calls was edited.
    "miss_function_source_changed": [
        {
            "status": "FUNCTION_CHANGED",
            "code": "scores = score_rows(df)",
            "total_time": 0.041,
            "changed_functions": ["score_rows"],
            "is_upstream": True,
        },
        {
            "status": "COMPUTED",
            "code": "scores = score_rows(df)",
            "total_time": 1.732,
            "evaluated_vars": ["scores"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:5d6e7f80a1b2c3d4",
            "miss_reason": "function source changed: score_rows",
            "is_upstream": False,
        },
    ],
    # §4.e — module reloaded: a tracked local import was edited.
    "miss_module_reloaded": [
        {
            "status": "MODULE_RELOADED",
            "code": "from features import build_features",
            "total_time": 0.012,
            "changed_modules": {"features": "..."},
            "is_upstream": True,
        },
        {
            "status": "COMPUTED",
            "code": "feats = build_features(df)",
            "total_time": 0.876,
            "evaluated_vars": ["feats"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:6a7b8c9d0e1f2a3b",
            "miss_reason": "module reloaded: features",
            "is_upstream": False,
        },
    ],
}
