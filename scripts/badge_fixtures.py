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
    # §5.a — side effect detected (file write, print to stdout, plt.show,
    # requests.get, db cursor).
    "not_cached_side_effect": [
        {
            "status": "COMPUTED",
            "code": "df.to_csv('out.csv')",
            "total_time": 0.094,
            "uncacheable_reasons": ["side effect: file write to out.csv"],
            "is_upstream": False,
        },
    ],
    # §5.b — REMOVED (CAS-114). This fixture hand-wrote an
    # ``uncacheable_reasons`` entry ("unseeded random call: numpy.random.rand")
    # that the runtime never emits, so it rendered a badge state that cannot
    # occur.  Unseeded randomness is *cacheable by design*: ``decide_cacheability``
    # has no randomness reason-source. What CAS-114 shipped is a
    # ``CashRandomnessWarning``, which is a Python warning — not a badge row and
    # not an uncacheable reason.  Making the fixture "real" would have required
    # inventing the very cacheability rule the design rejects.
    # §5.c — cost model says caching this would be slower than recomputing.
    "not_cached_too_cheap": [
        {
            "status": "COMPUTED",
            "code": "total = sum(values)",
            "total_time": 0.012,
            "evaluated_vars": ["total"],
            "skipped_reason": "below cost-model threshold (use @cash:persist to force)",
            "is_upstream": False,
        },
    ],
    # §5.d — explicit opt-out via the @cash:no-cache annotation.
    "not_cached_explicit": [
        {
            "status": "COMPUTED",
            "code": "x = compute_with_side_effect()",
            "total_time": 0.241,
            "evaluated_vars": ["x"],
            "skipped_reason": "# @cash:no-cache",
            "is_upstream": False,
        },
    ],
    # §5.e — file accessed through an API cash doesn't intercept.
    "not_cached_untracked_io": [
        {
            "status": "COMPUTED",
            "code": "data = my_custom_loader('data.bin')",
            "total_time": 0.812,
            "evaluated_vars": ["data"],
            "uncacheable_reasons": ["untracked file read: data.bin"],
            "is_upstream": False,
        },
    ],
    # §5.f — function called without @pure; Cash can't verify it has no side
    # effects, so it plays it safe and skips caching.
    "not_cached_purity": [
        {
            "status": "COMPUTED",
            "code": "result = featurize(my_df)",
            "total_time": 0.094,
            "evaluated_vars": ["result"],
            "uncacheable_reasons": ["function purity unknown"],
            "is_upstream": False,
        },
    ],
    # §2 anatomy — one rich badge that exercises every visible region:
    # header with mixed counts, upstream section, current section,
    # decorator-call group, overhead breakdown.
    "anatomy_hero": [
        # Upstream restored row.
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
            "is_upstream": True,
        },
        # Current cell — restored intermediate.
        {
            "status": "RESTORED",
            "code": "features = encode(df)",
            "total_time": 0.024,
            "saved_time": 0.612,
            "evaluated_vars": ["features"],
            "restored_vars": ["features"],
            "storage": ["RAM"],
            "source": "RAM",
            "cache_key": "stmt:88a1c2d3e4f50617",
            "is_upstream": False,
        },
        # Current cell — computed because miss_reason.
        {
            "status": "COMPUTED",
            "code": "preds = decorated_predict(features)",
            "total_time": 0.341,
            "evaluated_vars": ["preds"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:c1e8a72f4b9d0156",
            "miss_reason": "input lineage changed (one of: features)",
            "decorator_calls": [
                {"func_name": "predict_one", "cache_hit": True, "execution_time": 0.002},
                {"func_name": "predict_one", "cache_hit": True, "execution_time": 0.002},
                {"func_name": "predict_one", "cache_hit": False, "execution_time": 0.087},
            ],
            "is_upstream": False,
        },
    ],
    # Quickstart prose — first run of `df = pd.read_csv('large_dataset.csv')`.
    # Code matches docs/getting-started/quickstart.md so the embedded badge
    # is honest about what the reader just typed.
    "quickstart_first_run": [
        {
            "status": "COMPUTED",
            "code": "df = pd.read_csv('large_dataset.csv')",
            "total_time": 2.851,
            "evaluated_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:5b1d8e2af0c34719",
            "is_upstream": False,
        },
    ],
    # Quickstart prose — second run of the same statement, restored from cache.
    "quickstart_second_run": [
        {
            "status": "RESTORED",
            "code": "df = pd.read_csv('large_dataset.csv')",
            "total_time": 0.019,
            "saved_time": 2.832,
            "evaluated_vars": ["df"],
            "restored_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "source": "RAM",
            "cache_key": "stmt:5b1d8e2af0c34719",
            "is_upstream": False,
        },
    ],
    # data_science_workflows tutorial — Cell 2 recomputes because customers.csv changed.
    "miss_file_changed_customers": [
        {
            "status": "COMPUTED",
            "code": "customers = pd.read_csv('customers.csv')",
            "total_time": 1.847,
            "evaluated_vars": ["customers"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:2c9d4e1a7b3f8056",
            "miss_reason": "file changed: customers.csv",
            "is_upstream": False,
        },
    ],
    # data_science_workflows tutorial — Cell 4 after changing how='left' to
    # how='inner'. The expensive groupby statement is restored from cache;
    # only the merge statement (whose code changed) recomputes.
    "workflows_mixed": [
        {
            "status": "RESTORED",
            "code": "tx_features = transactions.groupby('customer_id').agg(...)",
            "total_time": 0.031,
            "saved_time": 3.214,
            "evaluated_vars": ["tx_features"],
            "restored_vars": ["tx_features"],
            "storage": ["RAM", "DISK"],
            "source": "RAM",
            "cache_key": "stmt:1a2b3c4d5e6f7081",
            "is_upstream": False,
        },
        {
            "status": "COMPUTED",
            "code": "df = customers.merge(tx_features, on='customer_id', how='inner')",
            "total_time": 0.183,
            "evaluated_vars": ["df"],
            "storage": ["RAM", "DISK"],
            "cache_key": "stmt:8f9e0d1c2b3a4956",
            "miss_reason": "code changed",
            "is_upstream": False,
        },
    ],
}
