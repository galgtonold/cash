"""The digest a statement took of a holder after it ran is the next
statement's digest before it runs, when nothing ran between (`DigestCarry`).

E8 (sklearn ensembles): a cell of small statements over one fitted
``VotingClassifier`` (``voting_clf.voting = 'soft'``,
``voting_clf.set_params(...)``) digested the lists and models sharing its
objects before and after every statement, the digest after one being the
very digest the next took before it. Carried, each holder is digested once
a statement. Not carried into the first statement of a cell, nor while
another thread runs; a change made between the two, or by the statement
itself, or through another name, still moves the holder on.
"""

from __future__ import annotations

import threading

import pytest

from cash import value_watch
from cash.notebook.statement import processor as processor_module
from tests._cell_driver import run_cash_cell

np = pytest.importorskip("numpy")

MAKE = (
    "import types\n"
    "import numpy as np\n"
    "parts = [np.arange(200_000.0) + i for i in range(3)]\n"
    "alias = parts\n"
    "model = types.SimpleNamespace(parts=parts, flag=0)\n"
    "other = types.SimpleNamespace(parts=parts, flag=0)"
)
FOUR = "model.flag = 1\nmodel.flag = 2\nmodel.flag = 3\nmodel.flag = 4"


@pytest.fixture
def unwatched(monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)


@pytest.fixture
def seen(monkeypatch, cash_magics, mock_shell, unwatched):
    """Each statement's ``(code, digests before, unchanged, changed)``, and
    the digests taken (``"digests"``)."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, MAKE)
    log: dict = {"rows": [], "digests": 0}
    kind = type(cash_magics._statement_processor)
    real_compare = kind._compare_partner_digests

    def compare(self, run):
        real_compare(self, run)
        log["rows"].append((run.code, dict(run.holder_digests), set(run.holders_unchanged), set(run.holders_changed)))

    real_fingerprint = processor_module._bounded_fingerprint

    def fingerprint(*args, **kwargs):
        log["digests"] += 1
        return real_fingerprint(*args, **kwargs)

    monkeypatch.setattr(kind, "_compare_partner_digests", compare)
    monkeypatch.setattr(processor_module, "_bounded_fingerprint", fingerprint)
    return log


def _carry():
    """`DigestCarry`, imported where it is used."""
    from cash.notebook.statement.digest_carry import DigestCarry

    return DigestCarry


def _uncarried(monkeypatch):
    monkeypatch.setattr(_carry(), "before", lambda self, name, value, lineage: None)


def _outcomes(log):
    return [(code, set(before), unchanged, changed) for code, before, unchanged, changed in log["rows"]]


HOLDERS = {"parts", "alias", "other"}


def test_each_holder_is_digested_once_a_statement(cash_magics, seen):
    run_cash_cell(cash_magics, FOUR)
    # One object (parts and alias hold the same list), other its own:
    # both digested before the first statement, then after each one.
    assert seen["digests"] == 2 + 4 * 2
    assert _outcomes(seen) == [(f"model.flag = {i}", HOLDERS, HOLDERS, set()) for i in range(1, 5)]


def test_uncarried_it_is_digested_twice_with_the_same_outcome(cash_magics, seen, monkeypatch):
    """The positive control: the same cell, each digest before taken anew."""
    _uncarried(monkeypatch)
    run_cash_cell(cash_magics, FOUR)
    assert seen["digests"] == 4 * 2 * 2
    assert _outcomes(seen) == [(f"model.flag = {i}", HOLDERS, HOLDERS, set()) for i in range(1, 5)]


def test_not_into_the_first_statement_of_a_cell(cash_magics, seen):
    for i in range(1, 4):
        run_cash_cell(cash_magics, f"model.flag = {i}")
    assert seen["digests"] == 3 * 2 * 2


def test_a_change_the_statement_makes_moves_the_holders_on(cash_magics, seen):
    run_cash_cell(cash_magics, "model.flag = 1\nmodel.parts[0][0] = -5.0\nmodel.flag = 2")
    (_, _, _, first), (code, _, unchanged, changed), (_, _, _, last) = seen["rows"]
    assert code == "model.parts[0][0] = -5.0"
    assert {"parts", "alias", "other"} <= changed
    assert not unchanged & {"parts", "alias", "other"}
    assert not first and not last
    assert seen["digests"] == 2 + 3 * 2


def test_a_change_through_another_name_moves_the_holders_on(cash_magics, seen):
    """``other`` holds the same list: changed through it, ``parts`` and
    ``alias``, whose digests before came from the statement above, read as
    changed."""
    run_cash_cell(cash_magics, "model.flag = 1\nother.parts[1][0] = -7.0")
    (_, _, _, _), (code, before, unchanged, changed) = seen["rows"]
    assert code == "other.parts[1][0] = -7.0"
    assert {"parts", "alias"} <= set(before)
    assert {"parts", "alias", "model"} <= changed
    assert not unchanged & {"parts", "alias", "model"}


def test_a_change_between_two_statements_is_never_taken_as_none(cash_magics, seen, monkeypatch, mock_shell):
    """A change no check sees coming (a finalizer) made after the digest
    was kept: the holder reads as changed by the next statement, never as
    unchanged."""
    real_keep = _carry().keep
    state = {"done": False}

    def keep(self, *args, **kwargs):
        real_keep(self, *args, **kwargs)
        if not state["done"]:
            state["done"] = True
            mock_shell.user_ns["parts"][2][0] = -9.0

    monkeypatch.setattr(_carry(), "keep", keep)
    run_cash_cell(cash_magics, "model.flag = 1\nmodel.flag = 2")
    (_, _, first_unchanged, _), (_, _, unchanged, changed) = seen["rows"]
    assert {"parts", "alias", "other"} <= first_unchanged
    assert {"parts", "alias", "other"} <= changed
    assert not unchanged & {"parts", "alias", "other"}


@pytest.mark.parametrize("daemon", [False, True])
def test_a_write_by_another_thread_between_statements_is_digested_anew(
    cash_magics, seen, monkeypatch, mock_shell, daemon
):
    """While a thread runs, nothing is carried: the second statement digests
    the holders as the thread left them."""
    go, done = threading.Event(), threading.Event()

    def write():
        go.wait(30)
        mock_shell.user_ns["parts"][2][0] = -11.0
        done.set()

    worker = threading.Thread(target=write, daemon=daemon)
    real_keep = _carry().keep

    def keep(self, *args, **kwargs):
        real_keep(self, *args, **kwargs)
        if not go.is_set():
            go.set()
            assert done.wait(30)

    monkeypatch.setattr(_carry(), "keep", keep)
    worker.start()
    try:
        run_cash_cell(cash_magics, "model.flag = 1\nmodel.flag = 2")
    finally:
        go.set()
        worker.join(30)
    (_, _, _, _), (_, before, unchanged, changed) = seen["rows"]
    assert seen["digests"] == 2 * 2 * 2
    assert {"parts", "alias", "other"} <= unchanged
    assert not changed
    assert before["parts"] == processor_module._fingerprint(mock_shell.user_ns["parts"])


def test_nothing_is_held_past_the_cell(cash_magics, seen):
    """The last statement of a cell keeps nothing: the values would be held
    until the next cell, which cannot take them up."""
    run_cash_cell(cash_magics, "model.flag = 1\nmodel.flag = 2")
    assert seen["digests"] == 2 + 2 * 2
    assert cash_magics._statement_processor._digest_carry.held() == [{}]
