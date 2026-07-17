"""The CAS-172 guard must announce itself quietly (CAS-182).

The guard's mechanism is confirmed working and is NOT under test here: a tester
verified it fires during edits and drops to zero firings across 12 reps once the
notebook stabilises. This file pins only its VOICE.

It explained itself in a ~46-word paragraph per statement -- 7 times on a single
cell, ~380 words of prose to say one thing seven times. The badge is a
glanceable UI, and noise is how a user learns to ignore the thing that will
later tell them something important.

The bar: one short line per statement, the full explanation once per cell.
"""
from __future__ import annotations

import pytest

from cash.notebook.badge_renderer._reasons import (
    GUARD_SHORT,
    is_guard_reason,
    shorten_skipped_reason,
)
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.statement.miss_guard import GUARD_SKIP_REASON

# The distinctive opening of the guard's paragraph. Used ONLY to detect the
# prose leaking inline; the production code matches on the imported constant.
_PARAGRAPH_MARKER = "Perpetual cache miss"


def _guarded_metrics(n):
    return [
        {'status': 'COMPUTED', 'code': f'df_{i} = transform(df_{i})',
         'execution_time': 0.12, 'skipped_reason': GUARD_SKIP_REASON}
        for i in range(n)
    ]


def _render(n):
    return render_text(build_interactive_badge(_guarded_metrics(n)))


def test_the_paragraph_is_not_repeated_per_statement():
    """The reported bug: 7 statements produced 7 copies of the paragraph."""
    out = _render(7)

    assert out.count(_PARAGRAPH_MARKER) == 0, (
        "the guard's 46-word paragraph is still being printed inline"
    )


def test_each_guarded_row_still_says_it_was_not_cached_and_why():
    """Quieter must not mean silent - the row keeps its verdict and reason."""
    out = _render(3)
    rows = [ln for ln in out.splitlines() if 'df_' in ln]

    assert len(rows) == 3
    for row in rows:
        assert 'NOT CACHED' in row
        assert GUARD_SHORT in row
        # One line per statement, not a paragraph.
        assert len(row) < 100, f"row is still a paragraph: {row!r}"


def test_the_full_explanation_appears_once_per_cell():
    out = _render(7)

    assert out.count('stopped caching') == 1
    assert '7 statements stopped caching' in out
    # The two facts the per-row line cannot carry.
    assert 'still run' in out
    assert 're-probes' in out


def test_singular_wording_for_a_single_statement():
    out = _render(1)
    assert '1 statement stopped caching' in out


def test_no_summary_when_the_guard_did_not_fire():
    """A quiet guard must be silent, not merely brief."""
    out = render_text(build_interactive_badge([
        {'status': 'COMPUTED', 'code': 'x = f()', 'execution_time': 0.5},
    ]))

    assert 'stopped caching' not in out
    assert GUARD_SHORT not in out


def test_volume_is_bounded_as_statements_grow():
    """The whole point: N guarded statements must not cost N paragraphs."""
    one = _render(1)
    seven = _render(7)

    # Before the fix a 7-statement cell cost ~2300 chars / ~380 words.
    assert len(seven) < 800, f"still {len(seven)} chars for 7 statements"
    # Each extra statement costs a short line, not a paragraph.
    growth = (len(seven) - len(one)) / 6
    assert growth < 70, f"each guarded statement adds {growth:.0f} chars"


def test_other_skip_reasons_are_passed_through_untouched():
    """This is a volume fix for ONE message, not a general truncator.

    A reason the shortener has never seen must survive intact rather than be
    silently clipped.
    """
    other = "Object too large to cache (512MB > 100MB limit)"

    assert shorten_skipped_reason(other) == other
    assert not is_guard_reason(other)

    out = render_text(build_interactive_badge([
        {'status': 'COMPUTED', 'code': 'big = load()',
         'execution_time': 1.0, 'skipped_reason': other},
    ]))
    assert other in out
    assert 'stopped caching' not in out


@pytest.mark.parametrize("reason", [None, ""])
def test_absent_reason_is_not_mistaken_for_the_guard(reason):
    assert not is_guard_reason(reason)
    assert shorten_skipped_reason(reason) == reason


def test_html_keeps_the_full_explanation_in_its_drawer():
    """The detail must remain reachable, not be deleted.

    The HTML badge already collapses the reason into a per-row drawer, which is
    exactly where a long explanation belongs -- so it is deliberately left
    carrying the full text.
    """
    html = render_html(build_interactive_badge(_guarded_metrics(2)))

    assert _PARAGRAPH_MARKER in html
