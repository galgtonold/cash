"""A memo answer made only of numbers, strings and None is remembered by its
pickle between statements, not by its full value digest: a change in place
that keeps the value equal but not the same (an int made a float, 0.0 made
-0.0) is still a change, and a statement that only read the answer is still
one that only read it."""

from __future__ import annotations

import functools

import pytest

from cash.notebook import helper_memos


def _memo():
    @functools.cache
    def seg(n):
        return [n, 0.0, "a", None]

    ns = {"seg": seg}
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    seg(1)  # the statement that fills it, and leaves its answer remembered
    assert helper_memos.used(before, ns=ns)[0] == helper_memos.FILLED
    return seg, ns


@pytest.mark.parametrize(("index", "value"), [(0, 1.0), (0, True), (1, -0.0), (2, b"a"), (3, 0)])
def test_an_equal_value_of_another_form_written_in_place_is_a_change(index, value):
    seg, ns = _memo()
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    seg(1)[index] = value

    verdict = helper_memos.used(before, ns=ns)

    assert isinstance(verdict, tuple), verdict
    assert "changed an answer the helper memo seg holds in place" in verdict[1], verdict[1]


def test_a_statement_that_only_reads_the_answer_is_not_a_change():
    """Positive control: the same read with nothing written."""
    seg, ns = _memo()
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    assert seg(1)[0] == 1

    verdict = helper_memos.used(before, ns=ns)

    assert isinstance(verdict, list), verdict
    assert [record["where"][-1] for record in verdict] == ["seg"]
