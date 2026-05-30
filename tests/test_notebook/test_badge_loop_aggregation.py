"""Aggregation contract for nested loop trees.

A ``ForLoopGroup`` can carry inner loops/controls in its ``.nested`` edge
(set when the runtime records an enclosing-loop chain). The renderer's
aggregation passes — total time, saved time, status roll-up, iteration
count — must all descend through the *same* child-edges, or the head-row
summary disagrees with the body it summarises.

These tests pin that every aggregation includes a populated ``.nested``.
Before the ``view.iter_iterations`` seam, ``_item_total_time``,
``_item_saved_time`` and ``_walk_statuses`` walked ``ForLoopGroup.stmts``
only and silently dropped nested savings, while ``_collect_iterations``
and the body renderer descended into ``.nested`` — a drift bug.
"""
from __future__ import annotations

import pytest

from cash.notebook.badge_renderer.renderers.html import (
    _collect_iterations,
    _item_saved_time,
    _item_total_time,
    _walk_statuses,
)
from cash.notebook.badge_renderer.view import (
    BadgeStatus,
    ForLoopGroup,
    IterationRow,
    LoopStatement,
)


def _loop(code: str, *, time_s: float, saved_s: float, status: BadgeStatus) -> LoopStatement:
    return LoopStatement(
        base_code=code,
        iterations=(
            IterationRow(status=status, code=code, time_s=time_s, saved_time_s=saved_s),
        ),
    )


def _outer_with_nested() -> ForLoopGroup:
    """An outer loop whose body has one direct statement and one nested
    inner loop, each contributing a distinct, non-overlapping saving."""
    inner = ForLoopGroup(
        loop_var_names=("j",),
        stmts=(_loop("y = f(j)", time_s=0.01, saved_s=0.30, status=BadgeStatus.RESTORED),),
    )
    return ForLoopGroup(
        loop_var_names=("i",),
        stmts=(_loop("x = g(i)", time_s=0.02, saved_s=0.50, status=BadgeStatus.COMPUTED),),
        nested=(inner,),
    )


def test_total_time_includes_nested_loop_time() -> None:
    outer = _outer_with_nested()
    # outer stmt 0.02 + nested inner 0.01
    assert _item_total_time(outer) == pytest.approx(0.03)


def test_saved_time_includes_nested_loop_savings() -> None:
    outer = _outer_with_nested()
    # outer stmt 0.50 + nested inner 0.30
    assert _item_saved_time(outer) == pytest.approx(0.80)


def test_status_rollup_includes_nested_loop_statuses() -> None:
    outer = _outer_with_nested()
    statuses = _walk_statuses((outer,))
    # The outer's COMPUTED stmt and the nested inner's RESTORED stmt.
    assert BadgeStatus.COMPUTED in statuses
    assert BadgeStatus.RESTORED in statuses
    assert len(statuses) == 2


def test_collect_iterations_already_descends_into_nested() -> None:
    """Characterises the one fold that was already correct, so the seam
    extraction can't regress it."""
    outer = _outer_with_nested()
    iters = _collect_iterations((outer,))
    assert len(iters) == 2
