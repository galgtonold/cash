"""The loop and branch marker lines are written, found and removed in one place."""

from __future__ import annotations

from cash.control_markers import has_marker, iteration_digest, mark_control, mark_iteration, strip_markers

BODY = "total += x"


def test_a_marked_statement_strips_back_to_its_source():
    for marked in (mark_iteration(BODY, "ab12"), mark_control(BODY, "cd34")):
        assert has_marker(marked)
        assert strip_markers(marked) == BODY


def test_both_kinds_are_stripped_even_nested():
    """A loop inside a branch carries both; several places once removed only
    the iteration marker, so a branch statement never matched its cell."""
    nested = mark_iteration(mark_control(BODY, "cd34"), "ab12")
    assert strip_markers(nested) == BODY


def test_a_marker_without_a_following_newline_is_stripped():
    assert strip_markers("x = 1\n# control_context: cd34") == "x = 1"


def test_iteration_digest():
    assert iteration_digest(mark_iteration(BODY, "ab12")) == "ab12"
    assert iteration_digest(mark_control(mark_iteration(BODY, "ab12"), "cd34")) == "ab12"
    assert iteration_digest(mark_control(BODY, "cd34")) is None


def test_plain_code_is_left_alone():
    """The control: nothing to strip, nothing found."""
    code = "x = '# not a marker'\ny = x"
    assert not has_marker(code)
    assert strip_markers(code) is code
    assert iteration_digest(code) is None
