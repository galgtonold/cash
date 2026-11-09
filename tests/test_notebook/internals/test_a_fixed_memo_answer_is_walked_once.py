"""A memo answer nothing can change in place is told so once, not before
every statement.

E16 (TSP): ``subsegments(N)``, an ``lru_cache`` helper, holds a tuple of
100,000 index pairs. Before each statement cash asks which answers a run
could change in place (`helper_memos.watched`), and walked that tuple to its
last pair each time to learn it is made of ints: about 1 s of a first
run's overhead over a dozen cells. A tuple of numbers stays one, so the
answer, kept with the memo's other answers it has seen, is not walked again.
"""

from __future__ import annotations

import functools

from cash.notebook import helper_memos


def _memo():
    @functools.cache
    def subsegments(n):
        return tuple((i, i + length) for length in range(2, n) for i in range(n - length + 1))

    @functools.cache
    def seg(n):
        return [n, n + 1]

    return subsegments, seg, {"subsegments": subsegments, "seg": seg}


def _walks(monkeypatch, answer):
    walks = []
    real = helper_memos._fixed

    def counted(value, *args, **kwargs):
        if value is answer:
            walks.append(1)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(helper_memos, "_fixed", counted)
    return walks


def test_a_tuple_of_numbers_is_walked_once_over_many_statements(monkeypatch):
    subsegments, _, ns = _memo()
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    answer = subsegments(60)
    helper_memos.used(before, ns=ns)
    walks = _walks(monkeypatch, answer)
    for _ in range(5):
        helper_memos.new_statement()
        before = helper_memos.watched(ns)
        assert subsegments(60) is answer
        verdict = helper_memos.used(before, ns=ns)
        assert isinstance(verdict, list), verdict
    assert len(walks) <= 1, f"walked {len(walks)} times over 5 statements"


def test_a_list_answer_changed_in_place_is_still_seen(monkeypatch):
    """Positive control: an answer that can change is still compared."""
    _, seg, ns = _memo()
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    seg(1)
    helper_memos.used(before, ns=ns)
    for _ in range(2):
        helper_memos.new_statement()
        before = helper_memos.watched(ns)
        assert seg(1) == [1, 2]
        assert isinstance(helper_memos.used(before, ns=ns), list)
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    seg(1).append(3)
    verdict = helper_memos.used(before, ns=ns)
    assert isinstance(verdict, tuple), verdict
    assert "changed an answer the helper memo seg holds in place" in verdict[1], verdict[1]
