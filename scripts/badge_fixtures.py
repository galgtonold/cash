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
}
