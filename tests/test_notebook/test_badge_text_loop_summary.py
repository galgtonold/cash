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
        [IterationRow(status=BadgeStatus.RESTORED, code=CODE, time_s=0.01, saved_time_s=0.8)
         for _ in range(n_restored)]
        + [IterationRow(status=BadgeStatus.COMPUTED, code=CODE, time_s=0.9, skipped_reason=REASON)
           for _ in range(n_computed)])
    calls = tuple([DecoratorCall(func_name="fit_detector", status=BadgeStatus.RESTORED, time_s=0.8)] * n_restored
                  + [DecoratorCall(func_name="fit_detector", status=BadgeStatus.COMPUTED, time_s=0.9)] * n_computed)
    sub = (SubUnitGroup(call_source="fit_detector(feats[mid])", occurrence_index=0, calls=calls,
                        condensed=True, key_prefix="call:"),)
    stmt = LoopStatement(base_code=CODE, iterations=iterations, sub_units=sub)
    return ForLoopGroup(loop_var_names=("mid",), stmts=(stmt,), loop_header="for mid in feats:",
                        body=(stmt,))


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
