"""The share check of a statement's outputs is asked twice when holders
alone refuse it: a second time with anchors. The checks of each output's
value on its own (alias, identity, consumable, context variable) see the
same values and namespace both times, so the first answer is kept and the
second ask does not walk the values again."""

from __future__ import annotations

import contextvars

from cash.notebook.statement.output_refusals import unrestorable_output_reason


def test_values_that_pass_on_their_own_are_marked_as_passed() -> None:
    passed: list[bool] = []
    unrestorable_output_reason({"x"}, {"x": [1, 2, 3]}, {}, values_passed=passed)
    assert passed == [True]


def test_a_value_refused_on_its_own_is_not_marked_as_passed() -> None:
    passed: list[bool] = []
    var = contextvars.ContextVar("v")
    reason = unrestorable_output_reason({"x"}, {"x": var}, {}, values_passed=passed)
    assert "context variable" in (reason or "")
    assert passed == []


def test_values_marked_as_passed_are_not_checked_again() -> None:
    var = contextvars.ContextVar("v")
    reason = unrestorable_output_reason({"x"}, {"x": var}, {}, values_passed=[True])
    assert "context variable" not in (reason or "")


def test_without_the_mark_the_values_are_checked_as_before() -> None:
    var = contextvars.ContextVar("v")
    reason = unrestorable_output_reason({"x"}, {"x": var}, {})
    assert "context variable" in (reason or "")
