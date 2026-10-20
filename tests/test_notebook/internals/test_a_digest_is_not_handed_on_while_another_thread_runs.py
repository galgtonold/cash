"""A digest one check took is handed to the next only when no other thread
could have changed the value in between.

An output's session hash took the digest its in-place check read just
before (`DigestHandoff.take_settled`): a thread of the user's that changed
the model between the two left the session hash naming the model as it was
before. Any live thread but this one, cash's own and the kernel's counts,
daemon or not: a pool's worker runs the user's code under a library's name.
"""

from __future__ import annotations

import ast
import threading

import pytest

from cash import value_hash, value_watch
from cash.notebook.call_effects import DigestHandoff
from cash.tracking import trace_recorder
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SETUP = (
    "import argparse, threading, time\n"
    "def fit(model):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    model.fitted += 1\n"
    "    return model\n"
    "go = threading.Event()\n"
    "done = threading.Event()\n"
    "def change_it():\n"
    "    go.wait(30)\n"
    "    m.rows.append([-1, -1.0])\n"
    "    done.set()\n"
)
#: Read by pickling only (a notebook class's instance takes the identity
#: tier), with the watchers off: each check reads it.
MAKE = "m = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], fitted=0)"


@pytest.fixture
def unwatched(monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)


@pytest.mark.parametrize("daemon", [False, True])
def test_a_thread_changing_the_model_between_two_checks_is_seen(
    cash_magics, mock_shell, monkeypatch, unwatched, daemon
):
    """It changes the model after the in-place check read it and before the
    session hash is taken: the session hash is the model as it is now."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, f"worker = threading.Thread(target=change_it, daemon={daemon})\nworker.start()")
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
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)


def _pool_worker_like(stop):
    """A thread whose target is a library's function, running the user's
    code: what `runs_user_code` cannot see."""
    return threading.Thread(target=threading.Event.wait, args=(stop, 30), daemon=True)


def test_any_other_live_thread_counts():
    ns: dict = {}
    assert not trace_recorder.other_thread_alive(ns)
    stop = threading.Event()
    t = _pool_worker_like(stop)
    t.start()
    try:
        assert trace_recorder.other_thread_alive(ns)
    finally:
        stop.set()
        t.join(30)


def test_a_call_takes_the_statements_digest_only_when_alone(monkeypatch):
    x = np.arange(10.0)
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = round(f(x), 2)"))
    handoff.digest(x)
    reads = []
    real = handoff.fingerprints.digest
    monkeypatch.setattr(handoff.fingerprints, "digest", lambda v: reads.append(v) or real(v))
    monkeypatch.setattr("cash.notebook.call_effects.other_thread_alive", lambda ns: True)
    ticket = handoff.call_started()
    handoff.digest_before(ticket, "f(x)")(x)
    assert len(reads) == 1, "another thread is alive: read again"


@pytest.mark.parametrize(
    ("named", "counts"),
    [
        ("zmq.utils.garbage:GarbageCollectorThread", False),
        ("ipykernel.heartbeat:Heartbeat", False),
        ("zmq.utils.garbage:Other", True),
        ("IPython.lib.backgroundjobs:BackgroundJobFunc", True),
        ("tqdm._monitor:TMonitor", True),
    ],
)
def test_a_thread_started_later_counts_unless_it_is_one_of_the_kernels_own(named, counts):
    """pyzmq starts its garbage thread after cash was imported (when a
    message is first sent without a copy). Only the kernel's own threads,
    by exact class, are passed over: another class of those packages can run
    the user's code (IPython's background jobs)."""
    module, _, qualname = named.partition(":")
    stop = threading.Event()
    def run(self):
        stop.wait(30)

    run.__module__ = module  # the package's code, as the real class's is
    cls = type(qualname.rpartition(".")[2], (threading.Thread,), {"run": run})
    cls.__module__, cls.__qualname__ = module, qualname
    t = cls(daemon=True)
    t.start()
    try:
        assert trace_recorder.other_thread_alive({}) is counts
    finally:
        stop.set()
        t.join(30)
