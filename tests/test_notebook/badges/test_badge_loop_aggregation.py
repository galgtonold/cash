"""Aggregation contract for nested loop trees.

A ``ForLoopGroup`` can carry inner loops/controls in its ``.nested`` edge
(set when the runtime records an enclosing-loop chain). Every aggregate a
renderer shows -- total time, saved time, status roll-up, leaf count --
comes from the node's ``rollup``, which sums over the same child edges
(``view.children``), so a head row can never disagree with the body it
summarises.
"""

from __future__ import annotations

import pytest

from cash.notebook.badge_renderer.view import (
    BadgeStatus,
    ControlGroup,
    ForLoopGroup,
    IterationRow,
    LoopStatement,
    SkippedBucket,
    StatementRow,
    iter_leaves,
)


def _loop(code: str, *, time_s: float, saved_s: float, status: BadgeStatus) -> LoopStatement:
    return LoopStatement(
        base_code=code,
        iterations=(IterationRow(status=status, code=code, time_s=time_s, saved_time_s=saved_s),),
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
    assert outer.rollup.time_s == pytest.approx(0.03)


def test_saved_time_includes_nested_loop_savings() -> None:
    outer = _outer_with_nested()
    # outer stmt 0.50 + nested inner 0.30
    assert outer.rollup.saved_s == pytest.approx(0.80)


def test_status_rollup_includes_nested_loop_statuses() -> None:
    outer = _outer_with_nested()
    # The outer's COMPUTED stmt and the nested inner's RESTORED stmt.
    assert (outer.rollup.leaves, outer.rollup.cached, outer.rollup.computed) == (2, 1, 1)
    assert outer.rollup.mixed


def test_leaf_walk_and_rollup_see_the_same_rows() -> None:
    outer = _outer_with_nested()
    leaves = list(iter_leaves(outer))
    assert len(leaves) == outer.rollup.leaves == 2


def test_a_statement_in_a_control_nested_in_a_loop_is_counted() -> None:
    """The loop's time used to count only iterations, so a plain statement in
    an ``if`` inside the loop was in the control's total but not the loop's."""
    row = StatementRow(status=BadgeStatus.COMPUTED, code="z = h()", time_s=0.4)
    loop = ForLoopGroup(
        loop_var_names=("i",),
        stmts=(_loop("x = g(i)", time_s=0.1, saved_s=0.0, status=BadgeStatus.COMPUTED),),
        nested=(ControlGroup(branch_label="if c", header="if c:", rows=(row,)),),
    )
    assert loop.rollup.time_s == pytest.approx(0.5)


def test_a_skipped_bucket_counts_what_was_saved_not_as_time_spent() -> None:
    """Steps that were not re-run cost no time; they only saved it."""
    skipped = StatementRow(status=BadgeStatus.SKIPPED, code="a = 1", time_s=0.0, saved_time_s=2.0)
    bucket = SkippedBucket(items=(skipped,), total_saved_time_s=2.0)
    assert bucket.rollup.time_s == 0.0
    assert bucket.rollup.saved_s == pytest.approx(2.0)
    assert bucket.rollup.kind == "cached"


def test_the_loop_tip_counts_trips_not_statement_runs() -> None:
    """``for mg in [200, 400, 600]:`` with three body
    statements read "Iterations 9" -- the statement-iterations summed. The
    loop ran three times."""
    from cash.notebook.badge_renderer.renderers.html import _for_loop_group_html, _RenderPass

    def stmt(code, statuses):
        return LoopStatement(
            base_code=code,
            iterations=tuple(IterationRow(status=s, code=code, time_s=0.01, saved_time_s=0.1) for s in statuses),
        )

    R, C = BadgeStatus.RESTORED, BadgeStatus.COMPUTED
    loop = ForLoopGroup(
        loop_var_names=("mg",),
        stmts=(stmt("sel = pick(mg)", (R, R, C)), stmt("n = count(sel)", (R, C, C)), stmt("rows.append(n)", (R, R, C))),
        loop_header="for mg in [200, 400, 600]:",
    )
    html = _for_loop_group_html(loop, _RenderPass(max_time=1.0))
    assert "<dt>Iterations</dt><dd>3</dd>" in html, html[:400]
    assert "<dt>Iterations</dt><dd>9</dd>" not in html
