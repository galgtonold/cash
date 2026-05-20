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
}
