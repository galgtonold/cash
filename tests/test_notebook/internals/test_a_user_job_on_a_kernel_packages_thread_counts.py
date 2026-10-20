"""A user's job on a thread of a kernel package's counts as another thread.

Two of the kernel's packages run the user's code on threads the user
starts: IPython's background jobs (``jobs.new(f)``: `BackgroundJobFunc` is a
Thread subclass with its own ``run`` in IPython, calling the user's
function) and a tornado IOLoop the user runs on a thread
(``threading.Thread(target=loop.start)``, as bokeh or a small web app in a
notebook does), every callback added to it the user's code. Only the
kernel's own threads, named exactly, are passed over
(`trace_recorder._KERNEL_THREADS`). The last test is the model scenario
with the change made by the tornado loop's callback: the session hash
names the model as it is.
"""

from __future__ import annotations

import threading
import time

import pytest

from cash import value_hash, value_watch
from cash.tracking import trace_recorder
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

tornado_ioloop = pytest.importorskip("tornado.ioloop")


def test_an_ipython_background_job_counts():
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.lib.backgroundjobs import BackgroundJobManager

    InteractiveShell.instance()
    go = threading.Event()
    x = [1]

    def job():  # the user's
        go.wait(30)
        x.append(2)

    jobs = BackgroundJobManager()
    jobs.new(job)
    try:
        time.sleep(0.1)
        assert trace_recorder.other_thread_alive({"x": x, "job": job}), "the user's job runs on IPython's thread"
    finally:
        go.set()
        for t in list(jobs.running):
            t.join(30)


def _loop_on_a_thread():
    loop = tornado_ioloop.IOLoop(make_current=False)
    t = threading.Thread(target=loop.start, daemon=True)
    t.start()
    return loop, t


def test_a_tornado_loop_the_user_runs_on_a_thread_counts():
    loop, t = _loop_on_a_thread()
    try:
        time.sleep(0.05)
        assert trace_recorder.other_thread_alive({"loop": loop}), "the loop runs the user's callbacks"
    finally:
        loop.add_callback(loop.stop)
        t.join(30)


SETUP = (
    "import argparse, time\n"
    "def fit(model):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    model.fitted += 1\n"
    "    return model\n"
    "m = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], fitted=0)\n"
)


def test_a_tornado_callback_changing_the_model_between_two_checks_is_seen(cash_magics, mock_shell, monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    loop, t = _loop_on_a_thread()
    done = threading.Event()

    def change_it():  # the user's callback, run on the loop's thread
        mock_shell.user_ns["m"].rows.append([-1, -1.0])
        done.set()

    routing = cash_magics._statement_processor._mutation_routing
    real_observe = routing.observe

    def observe(run):
        real_observe(run)  # reads m
        loop.add_callback(change_it)
        assert done.wait(30)

    monkeypatch.setattr(routing, "observe", observe)
    try:
        run_cash_cell(cash_magics, "fit(m)")
    finally:
        loop.add_callback(loop.stop)
        t.join(30)
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)
