"""After a statement changed an object in place, what else holds the object
is read once.

E8 (sklearn ensembles): after ``voting_clf.voting = 'soft'`` the lists and
models sharing objects with it were each pickled twice, by the check that
their holders kept their content (`_unchanged_holders`, or the anchors of
`_anchorable`) and again for their session hashes
(`replay_derivation_bumps`), with only cash's code between. The read
window closed before the first of them, as the checks that count an
object's weak references must not see one of cash's; it now stays open
with no weak reference, keeping what it read by the name the notebook binds
(`value_hash.no_weak_reads`). The store's pickling and sizing hooks may
change a value: what was read before them is read again after
(`value_hash.forget_reads`).
"""

from __future__ import annotations

import pytest

from cash import value_hash, value_watch
from tests._cell_driver import run_cash_cell

MAKE = "import argparse\nm = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], n=0)\nmodels = [m]"


@pytest.fixture
def unwatched(monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)


def _reads_of(monkeypatch, mock_shell, name):
    reads = []
    real = value_hash._compute_hash

    def counted(v):
        if v is mock_shell.user_ns.get(name):
            reads.append(1)
        return real(v)

    monkeypatch.setattr(value_hash, "_compute_hash", counted)
    return reads


def _after(monkeypatch, cash_magics, reads):
    """How many times *reads* grew after each statement ran."""
    after = []
    kind = type(cash_magics._statement_processor)
    real_post = kind._post_execute

    def post(self, *args):
        n = len(reads)
        real_post(self, *args)
        after.append(len(reads) - n)

    monkeypatch.setattr(kind, "_post_execute", post)
    return after


def test_the_list_holding_a_changed_object_is_read_once(cash_magics, mock_shell, monkeypatch, unwatched):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, MAKE)
    reads = _reads_of(monkeypatch, mock_shell, "models")
    after = _after(monkeypatch, cash_magics, reads)
    run_cash_cell(cash_magics, "m.n += 1")
    assert after and after[-1] >= 1, "the holder is checked after the statement"
    assert after == [1], f"models was read {after} times after the statement"
    state = cash_magics.tracking_state
    assert state.current_session_hashes["models"] == value_hash.compute_hash(mock_shell.user_ns["models"])


def test_a_value_the_stores_hooks_change_is_read_again(monkeypatch):
    """What the window read before the store's hooks ran is read again."""
    namespace = {}
    monkeypatch.setattr(value_watch, "namespaces", lambda: (namespace,))
    monkeypatch.setattr(value_watch, "WATCHING", False)
    rows = namespace["rows"] = [[1, 2.0]]
    with value_hash.one_read():
        value_hash.no_weak_reads()
        before = value_hash.compute_hash(rows)
        assert value_hash.compute_hash(rows) == before
        rows.append([3, 4.0])  # what a hook of the user's could do
        value_hash.forget_reads()
        assert value_hash.compute_hash(rows) != before


def test_no_weak_reference_is_taken_once_weak_reads_are_off(monkeypatch):
    import argparse
    import weakref

    namespace = {}
    monkeypatch.setattr(value_watch, "namespaces", lambda: (namespace,))
    monkeypatch.setattr(value_watch, "WATCHING", False)

    named = namespace["named"] = argparse.Namespace(rows=[[1, 2.0]])
    unnamed = argparse.Namespace(rows=[[3, 4.0]])
    with value_hash.one_read():
        value_hash.compute_hash(named)
        assert weakref.getweakrefcount(named) == 1
        value_hash.no_weak_reads()
        assert weakref.getweakrefcount(named) == 0, "the weak reference taken before is let go"
        value_hash.compute_hash(unnamed)
        assert weakref.getweakrefcount(unnamed) == 0
        reads = []
        real = value_hash._compute_hash
        monkeypatch.setattr(value_hash, "_compute_hash", lambda v: reads.append(v) or real(v))
        value_hash.compute_hash(named)
        assert reads == [], "kept on by its name"
        value_hash.compute_hash(unnamed)
        assert reads == [unnamed], "no name holds it: read again"
