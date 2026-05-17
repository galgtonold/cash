"""Construction smoke tests for the BadgeView IR.

Tests in this file verify only the *shape* of the node types: that they
construct, that they are immutable, and that a fully-populated tree is
hashable. Behavioural tests (view-builder, renderers) live elsewhere.
"""

from __future__ import annotations

import dataclasses

import pytest

from cash.notebook.badge_renderer.view import (
    BadgeHeader,
    BadgeStatus,
    BugReportLink,
    ControlBody,
    ControlGroup,
    ControlGroupSingle,
    DecoratorCall,
    DecoratorCallGroup,
    ForLoopGroup,
    InteractiveBadge,
    IterationRow,
    LoopStatement,
    OverheadBreakdown,
    OverheadEntry,
    Section,
    SectionKind,
    SkippedBucket,
    StatementRow,
    StatusBadge,
)


def test_badge_status_enum_is_a_string_enum() -> None:
    # Renderers compare status values to literals; keep them serialisable.
    assert BadgeStatus.RESTORED == "restored"
    assert BadgeStatus.MIXED == "mixed"
    assert str(BadgeStatus.COMPUTED) == "BadgeStatus.COMPUTED"  # enum repr


def test_status_enum_covers_all_legacy_string_statuses() -> None:
    """The IR must be able to represent every status the legacy code emits."""
    required = {
        "RESTORED", "COMPUTED", "SKIPPED", "MIXED", "ERROR",
        "FUNCTION_CHANGED", "MODULE_RELOADED", "WARNING",
    }
    assert required.issubset({s.name for s in BadgeStatus})


def test_statement_row_constructs_with_only_required_props() -> None:
    row = StatementRow(status=BadgeStatus.RESTORED, code="x = 1", time_s=0.02)
    assert row.status is BadgeStatus.RESTORED
    assert row.storage_tiers == ()
    assert row.uncacheable_reasons == ()
    assert row.is_upstream is False


def test_frozen_nodes_reject_mutation() -> None:
    row = StatementRow(status=BadgeStatus.COMPUTED, code="y = 2", time_s=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.code = "y = 3"  # type: ignore[misc]


def test_fully_populated_tree_is_hashable() -> None:
    """A complete BadgeView must round-trip through ``set``/``dict``."""
    iteration = IterationRow(
        status=BadgeStatus.COMPUTED,
        code="y = x * 2",
        time_s=0.01,
        storage_tiers=("RAM",),
        loop_bindings=(("x", 1),),
    )
    loop_stmt = LoopStatement(base_code="y = x * 2", iterations=(iteration,))
    for_group = ForLoopGroup(loop_var_names=("x",), stmts=(loop_stmt,))
    ctrl_row = StatementRow(status=BadgeStatus.RESTORED, code="z = 0", time_s=0.0)
    ctrl_group = ControlGroup(branch_label="if", header="if x > 0", rows=(ctrl_row,))
    ctrl_single = ControlGroupSingle(row=ctrl_row)
    ctrl_body = ControlBody(body_stmts=("a = 1", "b = 2"))
    skipped = SkippedBucket(
        items=(StatementRow(status=BadgeStatus.SKIPPED, code="skip", time_s=0.0),),
        total_saved_time_s=0.3,
    )
    dec_call = DecoratorCall(func_name="f", status=BadgeStatus.RESTORED, time_s=0.001)
    dec_group = DecoratorCallGroup(func_name="f", calls=(dec_call,), condensed=False)
    overhead = OverheadBreakdown(
        entries=(OverheadEntry(label="Lineage simulation", time_s=0.005),),
        total_s=0.005,
    )

    upstream_section = Section(
        kind=SectionKind.UPSTREAM, header="UPSTREAM HISTORY",
        items=(for_group, skipped),
    )
    current_section = Section(
        kind=SectionKind.CURRENT, header="",
        items=(ctrl_group, ctrl_single, ctrl_body),
    )
    decorators_section = Section(
        kind=SectionKind.DECORATORS, header="DECORATOR CACHE",
        items=(dec_group,),
    )
    overhead_section = Section(
        kind=SectionKind.OVERHEAD, header="",
        items=(overhead,),
    )

    badge = InteractiveBadge(
        header=BadgeHeader(
            status=BadgeStatus.MIXED, restored_count=1, computed_count=1,
            total_saved_s=0.3, total_exec_s=0.01,
        ),
        sections=(upstream_section, current_section, decorators_section, overhead_section),
        footer=BugReportLink(url="https://github.com/x/y/issues/new?body=..."),
    )

    assert hash(badge) == hash(badge)  # determinism
    assert {badge} == {badge}


def test_status_badge_is_a_distinct_root() -> None:
    """``StatusBadge`` is the compact non-interactive root, not nested under InteractiveBadge."""
    sb = StatusBadge(
        status=BadgeStatus.RESTORED,
        execution_time_s=0.01, time_saved_s=0.5,
        source="RAM", storage_tiers=("RAM", "DISK"),
    )
    assert sb.storage_tiers == ("RAM", "DISK")
