"""One word per state, at every level, in both renderers (CAS-272).

The badge grew two vocabularies. The cell header said ``CACHED`` / ``EXECUTED``;
the statement rows under it said ``RESTORED`` / ``COMPUTED`` for the very same
states, because the HTML row pill rendered the raw ``BadgeStatus`` value and the
text renderer hard-coded a parallel set of words. Nothing caught it -- both
halves were self-consistent -- and four documentation pages ended up describing
whichever label their author happened to have looked at.

The fix routes every displayed word through ``theme.label_of``. These tests
assert the *relationship* (header word == row word) rather than only the
literal strings, so renaming a label deliberately stays a one-line change while
letting the two levels drift apart still fails.

Also covered: ``uncacheable_count``. A statement that ran and was NOT stored
will run again on every future run -- a standing, usually fixable property --
but it was tallied only inside ``computed_count``, so a permanently uncacheable
cell was indistinguishable from one the user had just edited.
"""
from __future__ import annotations

import pytest

from cash.notebook.badge_renderer import theme
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view import BadgeStatus
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def _render_both(metrics: list[dict]) -> tuple[str, str]:
    badge = build_interactive_badge(metrics)
    return render_text(badge), render_html(badge)


# ---------------------------------------------------------------------------
# The two levels agree
# ---------------------------------------------------------------------------

def test_all_cached_cell_uses_one_word_at_both_levels() -> None:
    text, html = _render_both(
        [{"code": "x = load()", "status": str(CacheStatus.RESTORED),
          "total_time": 0.01, "saved_time": 0.5}]
    )
    word = theme.label_of(BadgeStatus.RESTORED.value)
    # Header plus the single row = two occurrences, not one of each vocabulary.
    assert text.count(word) == 2, text
    assert word in html
    for internal in ("RESTORED", "restored"):
        assert internal not in text, f"internal vocabulary {internal!r} in text badge"


def test_executed_cell_uses_one_word_at_both_levels() -> None:
    text, html = _render_both(
        [{"code": "x = work()", "status": str(CacheStatus.COMPUTED), "total_time": 0.3}]
    )
    word = theme.label_of(BadgeStatus.COMPUTED.value)
    assert text.count(word) == 2, text
    assert word in html
    assert "COMPUTED" not in text


@pytest.mark.parametrize("status", list(BadgeStatus))
def test_no_status_displays_its_raw_enum_value(status: BadgeStatus) -> None:
    """``label_of`` must answer for every member, never fall through to the
    enum's snake_case value. The HTML row pill used to print that value
    directly, so ``module_reloaded`` reached the user as MODULE_RELOADED."""
    label = theme.label_of(status.value)
    assert label == label.upper()
    assert "_" not in label


# ---------------------------------------------------------------------------
# NOT CACHED — the row that will run again every time
# ---------------------------------------------------------------------------

_UNCACHEABLE = {"code": "print(df)", "status": str(CacheStatus.COMPUTED),
                "total_time": 0.2, "uncacheable_reasons": ["has a side effect"]}


def test_uncacheable_row_says_so_in_both_renderers() -> None:
    text, html = _render_both([_UNCACHEABLE])
    assert theme.LABEL_UNCACHEABLE in text
    assert theme.LABEL_UNCACHEABLE in html
    assert "has a side effect" in text, "the reason is the actionable part"


def test_uncacheable_row_is_counted_in_the_header() -> None:
    badge = build_interactive_badge([_UNCACHEABLE])
    assert badge.header.uncacheable_count == 1
    # A subset of computed, not a fourth bucket: the row really did run.
    assert badge.header.computed_count == 1


def test_header_names_the_uncacheable_work() -> None:
    text, html = _render_both([_UNCACHEABLE])
    assert "1 not cached" in text, text
    assert "not cached" in html


def test_an_ordinary_executed_cell_claims_nothing_uncacheable() -> None:
    """The control. Without it the assertions above pass on a renderer that
    prints "not cached" unconditionally."""
    metrics = [{"code": "x = work()", "status": str(CacheStatus.COMPUTED),
                "total_time": 0.3}]
    badge = build_interactive_badge(metrics)
    text, html = render_text(badge), render_html(badge)
    assert badge.header.uncacheable_count == 0
    assert "not cached" not in text.lower()
    assert theme.LABEL_UNCACHEABLE not in html


def test_a_fully_cached_cell_claims_nothing_uncacheable() -> None:
    badge = build_interactive_badge(
        [{"code": "x = load()", "status": str(CacheStatus.RESTORED),
          "total_time": 0.01, "saved_time": 0.5}]
    )
    assert badge.header.uncacheable_count == 0
    assert "not cached" not in render_text(badge).lower()


def test_the_count_matches_the_rows_below_it() -> None:
    """Header and rows are computed by different code paths; a mismatch would
    have the summary contradict what the reader can count for themselves."""
    metrics = [
        _UNCACHEABLE,
        {"code": "cheap = 1 + 1", "status": str(CacheStatus.COMPUTED),
         "total_time": 0.0, "skipped_reason": "below the cost floor"},
        {"code": "y = work()", "status": str(CacheStatus.COMPUTED), "total_time": 0.4},
        {"code": "z = load()", "status": str(CacheStatus.RESTORED),
         "total_time": 0.01, "saved_time": 0.9},
    ]
    badge = build_interactive_badge(metrics)
    text = render_text(badge)
    assert badge.header.uncacheable_count == 2
    assert text.count(theme.LABEL_UNCACHEABLE) == 2
    assert "2 not cached" in text


def test_text_badge_stays_encodable_by_a_legacy_console() -> None:
    """``%cash_badge print`` is read by a different process than wrote it."""
    text, _ = _render_both([_UNCACHEABLE])
    text.encode("cp1252")
