"""CACHE-HIT-COPIES-HELD: a function whose hits keep copying its result,
because the caller still holds the results it got before, is told once
that ``frozen=True`` would hand every hit the same object.

A hit of a big plain result hands back the copy the last hit handed out
when nothing outside holds it. A library that keeps every result it is
given (PyThaiNLP keeps each word trie by its id) leaves cash a held copy on
every hit, so each hit copies it again. The hint names the function, says
the results are correct either way, and names the fix and what it costs if
the promise is broken. Nothing is said when callers drop their results, or
when the function is frozen already.
"""

from __future__ import annotations

import random
import warnings

import pytest

from cash import held_copies_notice, value_watch

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a hit hands back or copies its last copy only where the C extension's watchers follow it",
)

CODE = "CACHE-HIT-COPIES-HELD"


def trie(n=3000, seed=5):
    rnd = random.Random(seed)
    root: dict = {}
    for _ in range(n):
        node = root
        for ch in "".join(rnd.choice("abcdefgh") for _ in range(rnd.randint(3, 9))):
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


@pytest.fixture
def any_copy_time(monkeypatch):
    """The copies of a small trie count, however quick they are."""
    monkeypatch.setattr(held_copies_notice, "MIN_SECONDS", 0.0)


def hints(seen):
    return [str(w.message) for w in seen if getattr(w.message, "code", None) == CODE]


def logged(func):
    return [w for w in func.cache_info()["warnings"] if w["code"] == CODE]


def run(func, calls, keep):
    held = []
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for _ in range(calls):
            result = func()
            if keep:
                held.append(result)
            del result
    return hints(seen)


def test_hits_copying_results_the_caller_holds_are_told_once(cash_instance, any_copy_time):
    @cash_instance.cache(assume_safe=True)
    def word_trie():
        return trie()

    found = run(word_trie, 20, keep=True)
    assert len(found) == 1, found
    shown = found[0]
    assert "word_trie" in shown
    assert "other code still holds the results" in shown
    assert "@cash.cache(frozen=True)" in shown
    [entry] = logged(word_trie)
    message = entry["message"]
    assert "correct either way" in message
    assert "@cash.cache(frozen=True)" in message
    assert "KEY-FROZEN-MUTATED" in message, "the fix says what happens if the result is changed after all"


def test_a_few_held_hits_are_not_a_pattern(cash_instance, any_copy_time):
    @cash_instance.cache(assume_safe=True)
    def word_trie():
        return trie()

    assert run(word_trie, held_copies_notice.MIN_COPIES, keep=True) == []


def test_copies_that_cost_little_are_not_worth_saying(cash_instance, monkeypatch):
    monkeypatch.setattr(held_copies_notice, "MIN_SECONDS", 3600.0)

    @cash_instance.cache(assume_safe=True)
    def word_trie():
        return trie()

    assert run(word_trie, 20, keep=True) == []


def test_callers_that_drop_their_results_are_not_told(cash_instance, any_copy_time):
    @cash_instance.cache(assume_safe=True)
    def word_trie():
        return trie()

    assert run(word_trie, 20, keep=False) == []


def test_a_frozen_function_is_not_told(cash_instance, any_copy_time):
    @cash_instance.cache(frozen=True, assume_safe=True)
    def word_trie():
        return trie()

    assert run(word_trie, 20, keep=True) == []
    assert logged(word_trie) == []
