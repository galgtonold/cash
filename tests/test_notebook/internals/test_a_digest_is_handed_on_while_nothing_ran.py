"""A digest is handed from one read window to the next while nothing ran.

E8 (sklearn ensembles): after a restart, each statement replayed above the
cell read the models it changes before it ran, though the statement before
had just read them after it ran, with only cash's code between. Within
`value_hash.handed_on` a digest one window took stands in the next, until
something that may change a value (the user's code, a restore, the store's
pickling hooks) calls `_settled.unsettle`.
"""

from __future__ import annotations

import sys
import weakref

import pytest

from cash import value_hash


class Model:
    """A value only pickling reads."""

    def __init__(self, n):
        self.rows = [[i, i * 2.0] for i in range(n)]


@pytest.fixture
def reads(monkeypatch):
    seen = []
    real = value_hash._compute_hash
    monkeypatch.setattr(value_hash, "_compute_hash", lambda v: seen.append(id(v)) or real(v))
    return seen


@pytest.fixture
def notebook(monkeypatch):
    """A namespace the notebook binds its variables in."""
    namespace = {}
    monkeypatch.setattr(value_hash.value_watch, "namespaces", lambda: (namespace,))
    return namespace


def test_a_digest_stands_in_the_next_window_until_unsettled(reads, notebook):
    from cash._settled import unsettle

    model = notebook["model"] = Model(50)
    models = notebook["models"] = [model]
    with value_hash.handed_on():
        with value_hash.one_read():
            first = value_hash.compute_hash(model), value_hash.compute_hash(models)
        with value_hash.one_read():
            again = value_hash.compute_hash(model), value_hash.compute_hash(models)
        assert again == first
        assert reads == [id(model), id(models)], "nothing ran between: each read once"
        unsettle()
        model.rows.append([0, 0.0])
        with value_hash.one_read():
            changed = value_hash.compute_hash(model), value_hash.compute_hash(models)
    assert changed[0] != first[0] and changed[1] != first[1]
    assert reads == [id(model), id(models)] * 2, "unsettled: each read again"


def test_a_name_bound_to_another_object_is_read_again(reads, notebook):
    notebook["models"] = [Model(5)]
    with value_hash.handed_on():
        with value_hash.one_read():
            value_hash.compute_hash(notebook["models"])
        notebook["models"] = other = [Model(6)]
        with value_hash.one_read():
            value_hash.compute_hash(other)
    assert reads[-1] == id(other)


def test_nothing_is_handed_on_outside_it(reads, notebook):
    model = notebook["model"] = Model(20)
    for _ in range(2):
        with value_hash.one_read():
            value_hash.compute_hash(model)
    assert reads == [id(model)] * 2


def test_not_handed_on_reads_again_inside_and_after(reads, notebook):
    model = notebook["model"] = Model(20)
    with value_hash.handed_on():
        with value_hash.one_read():
            value_hash.compute_hash(model)
        with value_hash.not_handed_on():
            with value_hash.one_read():
                value_hash.compute_hash(model)
        with value_hash.one_read():
            value_hash.compute_hash(model)
    assert reads == [id(model)] * 3


def test_no_weak_reference_of_cash_outlives_a_window(reads, notebook, monkeypatch):
    """The checks that count an object's weak references (`identity_holders`,
    `in_place`) must not see one of cash's: an unnamed object is kept by weak
    reference only while its window is open, and a window without weak
    references takes none. (The watchers, which keep their own, are off.)"""
    monkeypatch.setattr(value_hash.value_watch, "WATCHING", False)
    unnamed = Model(10)
    with value_hash.handed_on():
        with value_hash.one_read():
            value_hash.compute_hash(unnamed)
            assert weakref.getweakrefcount(unnamed) == 1
        assert weakref.getweakrefcount(unnamed) == 0
        with value_hash.one_read(weak=False):
            value_hash.compute_hash(unnamed)
            value_hash.compute_hash(unnamed)
            assert weakref.getweakrefcount(unnamed) == 0
    assert reads == [id(unnamed)] * 3, "kept by nothing: read each time"


def test_a_name_anchor_holds_no_reference(notebook):
    value = notebook["value"] = [Model(3)]
    before = sys.getrefcount(value)
    with value_hash.handed_on():
        with value_hash.one_read(weak=False):
            value_hash.compute_hash(value)
        assert sys.getrefcount(value) == before
