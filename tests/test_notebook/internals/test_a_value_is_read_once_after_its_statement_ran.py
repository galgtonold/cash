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

import gc
import sys
import threading
import time
import weakref

import pytest

from cash import value_hash, value_watch
from cash.decorator import code_trace
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


class Big:
    """A value only pickling can read: no watcher, no content hasher."""

    def __init__(self, n):
        self.rows = [[i, i * 2.0] for i in range(n)]


@pytest.mark.usefixtures("user_code_watched")
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


@pytest.mark.usefixtures("user_code_watched")
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


#: Code of the user's: a module of theirs, which a test writes and imports
#: (the suite's own code is the host's, `trace_recorder.HOST_MODULES`).
USER_MODULE = """
def change(value, row):
    value.rows.append(row)

class Sneaky:
    "A value whose own __reduce__ changes another one as it is read."

    def __init__(self, other):
        self.other = other

    def __reduce__(self):
        if self.other is not None:
            self.other.rows.append([7, 7.0])
        return (Sneaky, (None,))
"""


@pytest.fixture
def user(tmp_path, monkeypatch):
    name = f"_user_code_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(USER_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = __import__(name)
    yield mod
    sys.modules.pop(name, None)


def test_a_finalizer_changing_the_model_between_two_checks_is_seen(
    cash_magics, mock_shell, monkeypatch, unwatched, user_code_watched
):
    """The collector runs a ``__del__`` of the user's after the in-place
    check read the model and before its session hash is taken: it changed
    the model, so the window ends and the session hash is the model as it
    is now."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(
        cash_magics,
        "fired = []\nclass Bomb:\n    def __del__(self):\n        m.rows.append([-1, -1.0])\n        fired.append(1)\n",
    )
    gc.disable()
    try:
        run_cash_cell(cash_magics, "b = Bomb()\nb.me = b\ndel b")
        mock_shell.user_ns["fired"].clear()
        windows = []
        real_window = value_hash.one_read
        monkeypatch.setattr(value_hash, "one_read", lambda: windows.append(1) or real_window())
        routing = cash_magics._statement_processor._mutation_routing
        real_observe = routing.observe

        def observe(run):
            real_observe(run)  # reads m
            gc.collect()

        monkeypatch.setattr(routing, "observe", observe)
        run_cash_cell(cash_magics, "fit(m)")
    finally:
        gc.enable()
    assert windows, "no other thread: the window opens"
    assert mock_shell.user_ns["fired"] == [1], "the finalizer ran inside the window"
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)


def test_a_value_read_while_code_of_the_users_ran_is_not_kept(user, user_code_watched):
    value = Big(20)
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
        value_hash.compute_hash(user.Sneaky(value))
        assert value_hash.compute_hash(value) != before, "the window ended: read again"


def test_a_callback_of_the_users_ends_the_window(user, user_code_watched):
    value = Big(20)
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
        user.change(value, [3, 3.0])
        assert value_hash.compute_hash(value) != before


def test_a_thread_of_the_users_ends_the_window(user, user_code_watched):
    value = Big(20)
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
        t = threading.Thread(target=user.change, args=(value, [5, 5.0]))
        t.start()
        t.join()
        assert value_hash.compute_hash(value) != before


def test_the_window_does_not_keep_a_model_alive():
    value = Big(200)
    gone = weakref.ref(value)
    with value_hash.one_read():
        value_hash.compute_hash(value)
        del value
        assert gone() is None, "the window holds its values weakly"


def test_a_digest_handed_to_the_call_is_not_one_taken_before_code_of_the_users_ran(user, user_code_watched):
    """``y = f(x)``: the statement's fingerprint of ``x`` before it ran is
    handed to the call's hash before the call, unless code of the user's ran
    in between (a finalizer the collector ran)."""
    import ast

    from cash.notebook.call_effects import DigestHandoff

    value = Big(20)
    for user_code_runs in (False, True):
        handoff = DigestHandoff()
        handoff.begin_statement()
        handoff.watch(ast.parse("y = f(x)"), {"f": len, "x": value})
        handoff.digest(value)
        if user_code_runs:
            user.change(value, [9, 9.0])
        ticket = handoff.call_started()
        assert handoff.digest_before(ticket)(value) == value_hash.compute_hash(value)


def test_a_pickle_kept_for_the_store_is_dropped_once_code_of_the_users_ran(user, user_code_watched):
    from cash import stored_pickle

    value = Big(20)
    with stored_pickle.keeping(lambda v: v is value):
        stored_pickle.object_digest(value)
        assert stored_pickle.kept(value) is not None
        user.change(value, [4, 4.0])
        assert stored_pickle.kept(value) is None, "the value changed since its pickle was kept"


def _settle(pool_alive, want, deadline=20.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if pool_alive() == want:
            return True
        time.sleep(0.05)
    return pool_alive() == want


def test_a_pool_counts_as_alive_only_while_it_has_work():
    """A process pool's manager and its queue's feeder thread stay alive
    after its work is done (joblib's after an ``n_jobs`` fit): idle, they
    change nothing; with work handed to them, they may."""
    from concurrent.futures import ProcessPoolExecutor

    from cash.tracking.trace_recorder import other_thread_alive

    ns = {}
    assert not other_thread_alive(ns)
    with ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(abs, -1).result(timeout=60) == 1
        assert _settle(lambda: other_thread_alive(ns), False), "an idle pool's threads change nothing"
        busy = pool.submit(time.sleep, 3)
        assert other_thread_alive(ns), "a pool with work counts as alive"
        busy.result(timeout=60)
        assert _settle(lambda: other_thread_alive(ns), False)


@pytest.fixture
def nothing_watches(monkeypatch):
    """As on Python 3.10/3.11, or with the ``sys.monitoring`` tool id taken:
    nothing would see code of the user's run between two checks."""
    monkeypatch.setattr(code_trace, "watching_user_code", lambda: False)


def test_where_nothing_watches_each_check_reads_for_itself(nothing_watches, monkeypatch):
    reads = []
    real = value_hash._compute_hash
    monkeypatch.setattr(value_hash, "_compute_hash", lambda v: reads.append(type(v)) or real(v))
    value = Big(20)
    with value_hash.one_read():
        assert not value_hash.reading_once()
        value_hash.compute_hash(value)
        value_hash.compute_hash(value)
    assert reads.count(Big) == 2


def test_where_nothing_watches_no_digest_is_handed_to_the_call(nothing_watches, user):
    import ast

    from cash.notebook.call_effects import DigestHandoff

    value = Big(20)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = f(x)"), {"f": len, "x": value})
    before = handoff.digest(value)
    user.change(value, [9, 9.0])  # unseen: no watch would end
    ticket = handoff.call_started()
    after = handoff.digest_before(ticket)(value)
    assert after != before
    assert after == value_hash.compute_hash(value)


def test_where_nothing_watches_no_pickle_is_kept_for_the_store(nothing_watches):
    from cash import stored_pickle

    value = Big(20)
    with stored_pickle.keeping(lambda v: v is value):
        stored_pickle.object_digest(value)
        assert stored_pickle.kept(value) is None
