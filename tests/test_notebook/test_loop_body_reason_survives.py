"""A loop-body row must keep the reason the runtime worked out for it (#61).

``IterationRow`` had no ``miss_reason`` / ``skipped_reason`` fields, so every
attribution the runtime computed was dropped on the floor for anything inside a
loop. The row rendered with no explanation at all -- and loops are where the
expensive work lives, which makes it the worst place for the badge to go quiet.

This is the multiplier on the miss-attribution work: ``input changed: x`` is
computed for a loop-body statement exactly as it is anywhere else, and before
this it could never be displayed.

Tested at the view/renderer layer rather than through a kernel because that is
where the bug was -- the metric carried the reason the whole time. A test that
drove a notebook would be testing the runtime, which was never broken.
"""
from __future__ import annotations

from cash.notebook.badge_renderer.renderers.text import _iteration_pseudo_row
from cash.notebook.badge_renderer.view import BadgeStatus, IterationRow
from cash.notebook.badge_renderer.view_builder import _iteration_row


def _metric(**over):
    m = {
        "status": "COMPUTED",
        "code": "total = total + expensive(row)",
        "total_time": 0.5,
        "saved_time": 0.0,
        "loop_vars": {"row": 3},
    }
    m.update(over)
    return m


def test_the_builder_carries_a_miss_reason_onto_the_iteration():
    row = _iteration_row(_metric(miss_reason="input changed: df"))
    assert row.miss_reason == "input changed: df"


def test_the_builder_carries_a_skipped_reason_onto_the_iteration():
    row = _iteration_row(_metric(skipped_reason="unstable key"))
    assert row.skipped_reason == "unstable key"


def test_a_reasonless_iteration_stays_reasonless():
    """Control arm: the fields must not invent a reason where there was none.

    Without this, a builder that defaulted to an empty string rather than None
    would satisfy the tests above while putting a blank attribution on every
    loop row.
    """
    row = _iteration_row(_metric())
    assert row.miss_reason is None
    assert row.skipped_reason is None


def test_the_text_renderer_bridge_does_not_drop_the_reason():
    """The pseudo-row conversion is where the reason was actually lost.

    ``_iteration_pseudo_row`` rebuilds an ``IterationRow`` as a
    ``StatementRow`` so the row-rendering helpers can be reused, and every one
    of those helpers reads the reason off the StatementRow -- so a field the
    bridge does not copy is a field that cannot render, no matter what the
    builder put on the iteration.
    """
    it = IterationRow(
        status=BadgeStatus.COMPUTED,
        code="total = total + expensive(row)",
        time_s=0.5,
        miss_reason="input changed: df",
        skipped_reason="unstable key",
    )
    pseudo = _iteration_pseudo_row(it)

    assert pseudo.miss_reason == "input changed: df"
    assert pseudo.skipped_reason == "unstable key"


def test_the_reason_reaches_the_rendered_text_row():
    """End of the chain: it is visible, not merely carried.

    The three tests above each check one hand-off. This checks the thing a
    user would actually see, which is the only claim that matters and the one
    a field-by-field test cannot make.
    """
    from cash.notebook.badge_renderer.renderers.text import _row_line

    it = IterationRow(
        status=BadgeStatus.COMPUTED,
        code="total = total + expensive(row)",
        time_s=0.5,
        skipped_reason="unstable key",
    )
    line = _row_line(_iteration_pseudo_row(it), is_upstream=False)
    assert "unstable key" in line, line
