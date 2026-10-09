"""`DigestHandoff` hands a digest from one check to the next only where no
code of the user's can run between the two.

Before a call: only when the statement is that one call of a bare name on
bare names and constants, and only to the first call through the cache. After
it: only when its result is bound to one name and no other call started
since. A statement started inside the call (a nested run) hands nothing on.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.call_effects import DigestHandoff, _call_shape

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "source, before, after",
    [
        ("y = f(x)", True, True),
        ("y = f(x, 3, k=z, s='a')", True, True),
        ("f(x)", True, False),
        ("a, b = f(x)", False, False),
        ("y: int = f(x)", False, False),
        ("y.attr = f(x)", False, False),
        ("y[0] = f(x)", False, False),
        ("y = obj.f(x)", False, False),
        ("y = f(x.attr)", False, False),
        ("y = f(x[0])", False, False),
        ("y = f(g(x))", False, False),
        ("y = f(*xs)", False, False),
        ("y = f(**kw)", False, False),
        ("y = f(x) + 0", False, False),
        ("y = z = f(x)", False, False),
        ("y = f(x); z = 1", False, False),
    ],
)
def test_the_shapes_with_nothing_between(source, before, after):
    assert _call_shape(ast.parse(source)) == (before, after)


def _watched(source="y = f(x)"):
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse(source))
    return handoff


def _counting(handoff, monkeypatch):
    reads = []
    real = handoff.fingerprints.digest
    monkeypatch.setattr(handoff.fingerprints, "digest", lambda v: reads.append(v) or real(v))
    return reads


def test_the_first_call_takes_the_statement_digest(monkeypatch):
    x = np.arange(10.0)
    handoff = _watched()
    reads = _counting(handoff, monkeypatch)
    before = handoff.digest(x)
    ticket = handoff.call_started()
    assert handoff.digest_before(ticket)(x) == before
    assert len(reads) == 1


def test_a_second_call_hashes_for_itself(monkeypatch):
    x = np.arange(10.0)
    handoff = _watched()
    handoff.digest(x)
    handoff.call_started()
    reads = _counting(handoff, monkeypatch)
    second = handoff.call_started()
    handoff.digest_before(second)(x)
    assert len(reads) == 1


def test_another_shape_hands_nothing_on(monkeypatch):
    x = np.arange(10.0)
    handoff = _watched("y = f(x) + 0")
    handoff.digest(x)
    reads = _counting(handoff, monkeypatch)
    handoff.digest_before(handoff.call_started())(x)
    assert len(reads) == 1


def test_another_object_under_the_name_is_hashed(monkeypatch):
    handoff = _watched()
    handoff.digest(np.arange(10.0))
    reads = _counting(handoff, monkeypatch)
    rebound = np.arange(10.0)
    handoff.digest_before(handoff.call_started())(rebound)
    assert reads == [rebound]


def test_the_after_digest_reaches_the_statement_check(monkeypatch):
    x = np.arange(10.0)
    handoff = _watched()
    handoff.digest(x)
    ticket = handoff.call_started()
    handoff.note_after(ticket, (x,), ("d",))
    reads = _counting(handoff, monkeypatch)
    assert handoff.digest_after(x) == "d"
    assert reads == []


@pytest.mark.parametrize("source", ["f(x)", "a, b = f(x)"])
def test_the_after_digest_is_not_handed_on_when_more_runs_after(monkeypatch, source):
    x = np.arange(10.0)
    handoff = _watched(source)
    handoff.note_after(handoff.call_started(), (x,), ("d",))
    assert handoff.digest_after(x) != "d"


def test_a_call_started_after_hands_nothing_on():
    x = np.arange(10.0)
    handoff = _watched()
    handoff.note_after(handoff.call_started(), (x,), ("d",))
    handoff.call_started()
    assert handoff.digest_after(x) != "d"


def test_a_nested_statement_ends_the_handoff():
    x = np.arange(10.0)
    handoff = _watched()
    handoff.digest(x)
    ticket = handoff.call_started()
    handoff.begin_statement()  # a statement run inside the call
    handoff.watch(ast.parse("y = f(x)"))
    handoff.note_after(ticket, (x,), ("d",))
    assert handoff.digest_after(x) != "d"
    assert handoff.digest_before(ticket)(x) != "d"
