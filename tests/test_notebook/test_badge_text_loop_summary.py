"""The text badge summarises a long loop per statement, not per pass.

Round 25's r25s3: 126+ lines for 63 machines -- a row and a sub-call row per
iteration, the not-cached rows each carrying the same 150-character reason --
and "~400 lines per cell" with all 200 machines. The tester wanted
``scores[mid] = ... 63 iterations: 31 cached, 32 not cached``. A short loop
keeps its rows.
"""

from cash.notebook.badge_renderer.renderers.text import _item_lines
from cash.notebook.badge_renderer.view import (
    BadgeStatus,
    DecoratorCall,
    ForLoopGroup,
    IterationRow,
    LoopStatement,
    SubUnitGroup,
)

CODE = "scores[mid] = fit_detector(feats[mid])"
REASON = "Too cheap to cache after the first 32 iterations of a growing object, see docs"


def _loop(n_restored, n_computed):
    iterations = tuple(
        [IterationRow(status=BadgeStatus.RESTORED, code=CODE, time_s=0.01, saved_time_s=0.8) for _ in range(n_restored)]
        + [
            IterationRow(status=BadgeStatus.COMPUTED, code=CODE, time_s=0.9, skipped_reason=REASON)
            for _ in range(n_computed)
        ]
    )
    calls = tuple(
        [DecoratorCall(func_name="fit_detector", status=BadgeStatus.RESTORED, time_s=0.8)] * n_restored
        + [DecoratorCall(func_name="fit_detector", status=BadgeStatus.COMPUTED, time_s=0.9)] * n_computed
    )
    sub = (
        SubUnitGroup(
            call_source="fit_detector(feats[mid])", occurrence_index=0, calls=calls, condensed=True, key_prefix="call:"
        ),
    )
    stmt = LoopStatement(base_code=CODE, iterations=iterations, sub_units=sub)
    return ForLoopGroup(loop_var_names=("mid",), stmts=(stmt,), loop_header="for mid in feats:", body=(stmt,))


def test_a_long_loop_is_one_line_per_statement():
    lines = _item_lines(_loop(31, 32), is_upstream=False)
    assert len(lines) <= 3, "\n".join(lines)
    text = "\n".join(lines)
    assert "63" in text and "31 cached" in text and "32" in text, text
    assert "fit_detector(feats[mid]): 31/63 hit" in text, text
    assert text.count(REASON) <= 1, text


def test_a_short_loop_keeps_its_rows():
    lines = _item_lines(_loop(2, 2), is_upstream=False)
    assert sum(1 for line in lines if CODE in line) == 4, "\n".join(lines)


def test_upstream_steps_not_re_run_are_one_line():
    """Round 25 (r25s1): 18 ``^SKIPPED: import os, sys`` style rows in a report
    cell's text badge, one per step the repair did not need. The HTML badge
    folds them into a count; so does the text badge."""
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge
    from cash.notebook.statement.processor import CacheStatus

    metrics = [
        {"code": f"step{i} = {i}", "status": str(CacheStatus.SKIPPED), "is_upstream": True, "saved_time": 0.1}
        for i in range(16)
    ]
    metrics.append({"code": "x = 1", "status": str(CacheStatus.COMPUTED), "total_time": 0.01})
    text = render_text(build_interactive_badge(metrics))
    assert "SKIPPED:" not in text, text
    assert "^16 upstream steps not re-run" in text, text


def _chart_cell(slow_s=0.01):
    from cash.notebook.statement.processor import CacheStatus

    steps = [
        "fig, ax = plt.subplots()",
        "ax.plot(xs, ys)",
        "ax.axhline(0)",
        "ax.legend()",
        "ax.set_title('t')",
        "fig.savefig('a.png')",
        "plt.close(fig)",
    ]
    metrics = [
        {
            "code": code,
            "status": str(CacheStatus.COMPUTED),
            "total_time": 0.004,
            "uncacheable_reasons": ["In-place mutation on: ax (receiver lineage bumped; statement re-executes)"],
        }
        for code in steps
    ]
    metrics.insert(
        3,
        {
            "code": "big = fit(ax)",
            "status": str(CacheStatus.COMPUTED),
            "total_time": slow_s,
            "uncacheable_reasons": ["In-place mutation on: ax"],
        },
    )
    return metrics


def test_quick_steps_that_are_never_cached_fold_into_one_line():
    """Round 25 (r25s1, r25s2): 10 of 12 badge lines of a chart cell were
    ``In-place mutation on: ax (...)`` for steps that re-run in milliseconds."""
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    text = render_text(build_interactive_badge(_chart_cell(slow_s=2.5)))
    lines = text.splitlines()
    assert len(lines) <= 5, text
    assert "re-ran 3 quick steps" in text and "re-ran 4 quick steps" in text, text
    assert "ax.axhline" in text and "fig.savefig" in text, text
    assert "big = fit(ax)" in text and "In-place mutation on: ax" in text, "a slow one keeps its row"


def test_two_quick_steps_keep_their_rows():
    from cash.notebook.badge_renderer.renderers.text import render_text
    from cash.notebook.badge_renderer.view_builder import build_interactive_badge

    text = render_text(build_interactive_badge(_chart_cell()[:2]))
    assert "quick steps" not in text and "ax.plot(xs, ys)" in text, text
