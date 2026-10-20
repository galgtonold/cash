"""After a statement ran, its checks read each value once.

E8 (sklearn ensembles): a 500-tree model was read up to 5 times after each
statement that used it -- the in-place check of its receiver, the observed
mutation, the rewatch, its lineage and its session hash -- each a full
pickle, with only cash's code running between them. They are all
`compute_hash` of the same object, so one read gives every one its bytes
(`value_hash.one_read`). Where another thread is alive it could change
the value between two of them: each reads for itself, as before.
"""

from __future__ import annotations

import threading

import pytest

from cash import value_hash, value_watch
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


class Big:
    """A value only pickling can read: no watcher, no content hasher."""

    def __init__(self, n):
        self.rows = [[i, i * 2.0] for i in range(n)]


def test_inside_the_window_an_object_is_read_once(monkeypatch):
    reads = []
    real = value_hash._compute_hash
    monkeypatch.setattr(value_hash, "_compute_hash", lambda v: reads.append(type(v)) or real(v))
    value = Big(50)
    pair = [1, 2]
    with value_hash.one_read():
        first = value_hash.compute_hash(value)
        assert value_hash.compute_hash(value) == first
        value_hash.compute_hash(pair)
        value_hash.compute_hash(pair)
    assert reads.count(Big) == 1
    assert reads.count(list) == 2, "a list is read each time: no weak reference to it"
    assert value_hash.compute_hash(value) == first
    assert reads.count(Big) == 2, "outside the window it is read again"


def test_a_value_changed_after_the_window_is_read_as_changed():
    value = Big(20)
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
    value.rows.append([0, 0.0])
    assert value_hash.compute_hash(value) != before


def test_the_window_is_this_threads_only():
    value = Big(20)
    seen = []
    with value_hash.one_read():
        value_hash.compute_hash(value)
        value.rows.append([1, 1.0])  # what another thread would see changed

        def other():
            seen.append(value_hash.compute_hash(value))

        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen == [value_hash.compute_hash(value)]


SETUP = (
    "import argparse, time\n"
    "def fit(model):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    model.fitted += 1\n"
    "    return model\n"
)
#: A model only pickling reads (a notebook class's instance takes the
#: identity tier), with the watchers off: each check reads it.
MAKE = "m = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], fitted=0)"


def _reads_of(monkeypatch, mock_shell, name):
    reads = []
    real = value_hash._compute_hash

    def counted(v):
        if v is mock_shell.user_ns.get(name):
            reads.append(1)
        return real(v)

    monkeypatch.setattr(value_hash, "_compute_hash", counted)
    return reads


@pytest.fixture
def unwatched(monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)


def test_a_model_changed_in_place_is_read_once_after_its_statement(cash_magics, mock_shell, monkeypatch, unwatched):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, MAKE)
    windows = []
    real_window = value_hash.one_read
    monkeypatch.setattr(value_hash, "one_read", lambda: windows.append(1) or real_window())
    reads = _reads_of(monkeypatch, mock_shell, "m")
    after = []
    real_post = type(cash_magics._statement_processor)._post_execute_reading_once

    def post(self, *args):
        n = len(reads)
        real_post(self, *args)
        after.append(len(reads) - n)

    monkeypatch.setattr(type(cash_magics._statement_processor), "_post_execute_reading_once", post)
    run_cash_cell(cash_magics, "fit(m)")
    assert windows, "no other thread: the window opens"
    assert after == [1], f"read {after} times after the statement"
    state = cash_magics.tracking_state
    assert state.current_session_hashes["m"] == value_hash.compute_hash(mock_shell.user_ns["m"])


def test_a_thread_changing_the_model_between_two_checks_is_seen(cash_magics, mock_shell, monkeypatch, unwatched):
    """Another thread changes the model after the in-place check read it
    and before its session hash is taken: no window, and the session hash
    is the model as it is now."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "import threading\ngo = threading.Event()\ndone = threading.Event()")
    run_cash_cell(
        cash_magics,
        "def change_it():\n    go.wait(30)\n    m.rows.append([-1, -1.0])\n    done.set()\n",
    )
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "worker = threading.Thread(target=change_it, daemon=True)\nworker.start()")
    windows = []
    real_window = value_hash.one_read
    monkeypatch.setattr(value_hash, "one_read", lambda: windows.append(1) or real_window())
    routing = cash_magics._statement_processor._mutation_routing
    real_observe = routing.observe

    def observe(run):
        real_observe(run)  # reads m
        mock_shell.user_ns["go"].set()
        assert mock_shell.user_ns["done"].wait(30)

    monkeypatch.setattr(routing, "observe", observe)
    try:
        run_cash_cell(cash_magics, "fit(m)")
    finally:
        mock_shell.user_ns["go"].set()
        mock_shell.user_ns["worker"].join(30)
    assert not windows, "another thread is alive: no window"
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)
