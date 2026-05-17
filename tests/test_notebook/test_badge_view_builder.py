"""Tests for the view-builder that translates ProcessResult metrics into BadgeView nodes."""

from __future__ import annotations

import pytest

from cash.notebook.badge_renderer.view import (
    BadgeStatus,
    ControlGroup,
    DecoratorCallGroup,
    ForLoopGroup,
    OverheadBreakdown,
    Section,
    SectionKind,
    SkippedBucket,
    StatementRow,
)
from cash.notebook.badge_renderer.view_builder import (
    build_bug_report_url,
    build_interactive_badge,
    build_status_badge,
    map_status,
)
from cash.notebook.cache_status import CacheStatus


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (CacheStatus.RESTORED, BadgeStatus.RESTORED),
    (CacheStatus.COMPUTED, BadgeStatus.COMPUTED),
    (CacheStatus.SKIPPED, BadgeStatus.SKIPPED),
    (CacheStatus.ERROR, BadgeStatus.ERROR),
    ("RESTORED", BadgeStatus.RESTORED),
    ("computed", BadgeStatus.COMPUTED),
    ("FUNCTION_CHANGED", BadgeStatus.FUNCTION_CHANGED),
    ("MODULE_RELOADED", BadgeStatus.MODULE_RELOADED),
    ("WARNING", BadgeStatus.WARNING),
    ("garbage", BadgeStatus.WARNING),  # total function
    (None, BadgeStatus.WARNING),
])
def test_map_status_is_total(raw, expected) -> None:
    assert map_status(raw) is expected


# ---------------------------------------------------------------------------
# Empty + minimal
# ---------------------------------------------------------------------------

def test_empty_metrics_produces_only_current_section() -> None:
    badge = build_interactive_badge([])
    assert badge.header.status is BadgeStatus.COMPUTED
    assert tuple(s.kind for s in badge.sections) == (SectionKind.CURRENT,)
    assert badge.sections[0].items == ()
    # Footer is always emitted; URL has the empty-metrics placeholder.
    assert badge.footer is not None
    assert "no%20metrics" in badge.footer.url


def test_single_restored_metric_no_upstream() -> None:
    metrics = [{
        "code": "x = 1",
        "status": str(CacheStatus.RESTORED),
        "total_time": 0.005,
        "saved_time": 0.3,
        "source": "RAM",
        "storage": ["RAM", "DISK"],
        "restored_vars": ["x"],
    }]
    badge = build_interactive_badge(metrics)
    assert badge.header.status is BadgeStatus.RESTORED
    assert badge.header.restored_count == 1
    assert pytest.approx(badge.header.total_saved_s) == 0.3

    current = badge.sections[0]
    assert current.kind is SectionKind.CURRENT
    row = current.items[0]
    assert isinstance(row, StatementRow)
    assert row.status is BadgeStatus.RESTORED
    assert row.source == "RAM"
    assert row.storage_tiers == ("RAM", "DISK")
    # Restored rows separate restored_vars from output_vars (both fields exist).
    assert row.restored_vars == ("x",)
    assert row.is_upstream is False
    # time_s is now actual elapsed restore time; saved_time_s carries the saving.
    assert row.time_s == pytest.approx(0.005)
    assert row.saved_time_s == pytest.approx(0.3)


def test_mixed_status_summary() -> None:
    metrics = [
        {"code": "x=1", "status": str(CacheStatus.RESTORED), "saved_time": 0.5, "total_time": 0.001},
        {"code": "y=x+1", "status": str(CacheStatus.COMPUTED), "total_time": 0.2},
    ]
    badge = build_interactive_badge(metrics)
    assert badge.header.status is BadgeStatus.MIXED


# ---------------------------------------------------------------------------
# Upstream partitioning
# ---------------------------------------------------------------------------

def test_upstream_metrics_go_to_upstream_section() -> None:
    metrics = [
        {"code": "a=1", "status": str(CacheStatus.RESTORED), "is_upstream": True,
         "saved_time": 0.1, "total_time": 0.0},
        {"code": "b=a+1", "status": str(CacheStatus.COMPUTED), "is_upstream": False, "total_time": 0.05},
    ]
    badge = build_interactive_badge(metrics)
    kinds = tuple(s.kind for s in badge.sections)
    assert SectionKind.UPSTREAM in kinds
    assert SectionKind.CURRENT in kinds

    upstream = next(s for s in badge.sections if s.kind is SectionKind.UPSTREAM)
    upstream_row = upstream.items[0]
    assert isinstance(upstream_row, StatementRow)
    assert upstream_row.is_upstream is True
    assert upstream_row.status is BadgeStatus.RESTORED


def test_upstream_skipped_metrics_become_skipped_bucket() -> None:
    metrics = [
        {"code": "step1", "status": str(CacheStatus.SKIPPED), "is_upstream": True, "saved_time": 0.4},
        {"code": "step2", "status": str(CacheStatus.SKIPPED), "is_upstream": True, "saved_time": 0.1},
        {"code": "current", "status": str(CacheStatus.COMPUTED), "total_time": 0.2},
    ]
    badge = build_interactive_badge(metrics)
    upstream = next(s for s in badge.sections if s.kind is SectionKind.UPSTREAM)
    bucket = next(i for i in upstream.items if isinstance(i, SkippedBucket))
    assert len(bucket.items) == 2
    assert pytest.approx(bucket.total_saved_time_s) == 0.5


# ---------------------------------------------------------------------------
# Loop grouping
# ---------------------------------------------------------------------------

def test_for_loop_iterations_collapse_into_for_loop_group() -> None:
    metrics = [
        {"code": "# __iteration_context__:abc\ny = x * 2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.01,
         "loop_vars": {"x": 1}},
        {"code": "# __iteration_context__:abc\ny = x * 2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.02,
         "loop_vars": {"x": 2}},
    ]
    badge = build_interactive_badge(metrics)
    current = next(s for s in badge.sections if s.kind is SectionKind.CURRENT)
    group = current.items[0]
    assert isinstance(group, ForLoopGroup)
    assert group.loop_var_names == ("x",)
    assert len(group.stmts) == 1
    assert len(group.stmts[0].iterations) == 2
    assert group.stmts[0].iterations[0].loop_bindings == (("x", 1),)


def test_control_group_metrics_collapse_into_control_group() -> None:
    metrics = [
        {"code": "z = 1", "status": str(CacheStatus.COMPUTED),
         "control_context": "ctx1", "branch_label": "if",
         "body_statements": ["if x:", "    z = 1"], "total_time": 0.01},
        {"code": "z += 1", "status": str(CacheStatus.COMPUTED),
         "control_context": "ctx1", "total_time": 0.005},
    ]
    badge = build_interactive_badge(metrics)
    current = next(s for s in badge.sections if s.kind is SectionKind.CURRENT)
    group = current.items[0]
    assert isinstance(group, ControlGroup)
    assert len(group.rows) == 2


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def test_decorator_calls_become_their_own_section_grouped_by_func() -> None:
    metrics = [{
        "code": "f()", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
        "decorator_calls": [
            {"func_name": "f", "cache_hit": True, "execution_time": 0.001},
            {"func_name": "f", "cache_hit": False, "execution_time": 0.05},
            {"func_name": "g", "cache_hit": True, "execution_time": 0.002},
        ],
    }]
    badge = build_interactive_badge(metrics)
    dec_section = next(s for s in badge.sections if s.kind is SectionKind.DECORATORS)
    func_names = {item.func_name for item in dec_section.items if isinstance(item, DecoratorCallGroup)}
    assert func_names == {"f", "g"}


def test_decorator_group_is_condensed_when_more_than_three_calls() -> None:
    calls = [{"func_name": "f", "cache_hit": True, "execution_time": 0.001} for _ in range(4)]
    metrics = [{"code": "loop", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
                "decorator_calls": calls}]
    badge = build_interactive_badge(metrics)
    dec_section = next(s for s in badge.sections if s.kind is SectionKind.DECORATORS)
    group = dec_section.items[0]
    assert isinstance(group, DecoratorCallGroup) and group.condensed is True


# ---------------------------------------------------------------------------
# Overhead
# ---------------------------------------------------------------------------

def test_overhead_section_only_when_nontrivial() -> None:
    # No timing breakdown → no section
    badge = build_interactive_badge([], timing_breakdown=None, cell_total_time=None)
    assert all(s.kind is not SectionKind.OVERHEAD for s in badge.sections)

    # Overhead below threshold → no section
    badge = build_interactive_badge(
        [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 1.0}],
        timing_breakdown={"badge_init": 0.0001},
        cell_total_time=1.0001,
    )
    assert all(s.kind is not SectionKind.OVERHEAD for s in badge.sections)

    # Real overhead → section appears
    badge = build_interactive_badge(
        [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 1.0}],
        timing_breakdown={"badge_init": 0.05, "upstream_check": 0.02},
        cell_total_time=1.1,
    )
    overhead_section = next(s for s in badge.sections if s.kind is SectionKind.OVERHEAD)
    breakdown = overhead_section.items[0]
    assert isinstance(breakdown, OverheadBreakdown)
    labels = {e.label for e in breakdown.entries}
    assert "🏷️ Badge init" in labels
    assert "↻ Upstream check" in labels


# ---------------------------------------------------------------------------
# Bug report
# ---------------------------------------------------------------------------

def test_bug_report_footer_is_always_emitted() -> None:
    """Footer is part of the badge UX; context is optional metadata, not a gate."""
    metrics = [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 0.1}]
    badge = build_interactive_badge(metrics)
    assert badge.footer is not None
    assert badge.footer.url.startswith("https://github.com/galgtonold/cash/issues/new")
    with_ctx = build_interactive_badge(metrics, bug_report_context={"version": "1.0"})
    assert "1.0" in with_ctx.footer.url


def test_bug_report_url_under_size_limit() -> None:
    # Huge notebook source must still produce a valid (short-enough) URL.
    huge_cells = ["x = 1\n" * 1000 for _ in range(20)]
    metrics = [{"code": "f()", "status": "COMPUTED", "total_time": 0.1}]
    url = build_bug_report_url(metrics, {
        "version": "9.9", "python_version": "3.13", "backend": "file",
        "notebook_source": huge_cells,
    })
    assert len(url) <= 7800 + len("https://github.com/galgtonold/cash/issues/new?title=...&body=")


# ---------------------------------------------------------------------------
# StatusBadge
# ---------------------------------------------------------------------------

def test_status_badge_round_trips_through_view_builder() -> None:
    sb = build_status_badge(
        status=CacheStatus.RESTORED, execution_time=0.01, time_saved=0.5,
        source="RAM", storage=["RAM", "DISK"],
    )
    assert sb.status is BadgeStatus.RESTORED
    assert sb.storage_tiers == ("RAM", "DISK")
