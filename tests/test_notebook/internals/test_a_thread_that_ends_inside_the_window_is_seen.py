"""Review of claude/speed-model-once 14026f75 (base-gap fix): failing tests.

Put this file in tests/test_notebook/internals/ (it relies on that folder's
autouse fixture taking threads an earlier test left as the host's).

1. Aliveness is sampled when a digest is HANDED ON, not over the window it
   stands for. `DigestHandoff.digest_before`, `offer_to_decorator` and
   `_digest_after` ask `other_thread_alive` at the end of the window; a
   thread alive when the statement's fingerprint read the value, which
   changes it and ends before the call starts, is gone by then, and the
   call's key takes the digest of the value as it was. The call's result
   is then stored under the old value's key: a later call with the old
   value is served the result computed from the changed one.
   (`_post_execute` samples at the start of its window, which is right.)
   End to end I could not turn this into a wrong result: with a notebook
   call (`y = total(x)`) and a `@cash.cache` call the store was lost
   instead (the call read as changing x), and with a changed result the
   next statement recomputed. So: a broken contract the later checks
   catch today, worth closing (take the alive answer when the digest is
   TAKEN and again when it is handed on; any yes means no handoff).

2. `THREADS_AT_IMPORT` holds thread idents, and the OS reuses them: on
   glibc a new thread commonly gets the ident of one that ended. A worker
   started after a thread that was alive at `import cash` ended is taken
   for the kernel's, and digests are handed on while it runs.

3. A thread the user started before `import cash` (a ThreadPoolExecutor
   worker: its target is the library's `_worker`, which `runs_user_code`
   cannot see) is trusted for the whole session.

2 and 3 reach the branch's own fix: the third test is its model scenario
with the change made by such a worker, and the session hash is stale.
Suggestion: keep the import-time threads as objects (weak refs), not
idents, and trust only those running no target at all or a known kernel
or IPython class; any other thread alive at import counts.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import threading

import pytest

from cash import value_hash, value_watch
from cash.notebook.call_effects import DigestHandoff
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")


def _changer(x, go):
    """A thread of a library's (no user code by `runs_user_code`) that
    changes *x* once *go* is set, then ends."""

    def run():
        go.wait(30)
        x[0] = 99.0

    return threading.Thread(target=run, daemon=True)


def test_a_thread_that_changes_the_value_and_ends_before_the_call_is_seen():
    x = np.arange(10.0)
    ns = {"x": x}
    go = threading.Event()
    t = _changer(x, go)
    t.start()
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = f(x)"), ns)
    handoff.digest(x)  # the statement's fingerprint, while the thread runs
    go.set()
    t.join(30)  # it changed x and ended; no code of the main thread ran
    ticket = handoff.call_started()
    got = handoff.digest_before(ticket, "f(x)")(x)
    assert x[0] == 99.0
    assert got == value_hash.compute_hash(x), "the call's key took the digest of x before the thread changed it"


def test_a_thread_that_changes_the_result_and_ends_before_the_statements_check_is_seen():
    y = np.arange(10.0)
    go = threading.Event()
    t = _changer(y, go)
    t.start()
    handoff = DigestHandoff()
    handoff.begin_statement()
    handoff.watch(ast.parse("y = f(x)"), {"y": y})
    ticket = handoff.call_started()
    handoff.note_after(ticket, (y,), (value_hash.compute_hash(y),))  # the call hashed its result
    go.set()
    t.join(30)
    assert y[0] == 99.0
    assert handoff.digest_after(y) == value_hash.compute_hash(y), "the statement took the call's digest of y"


MODEL_SETUP = (
    "import argparse, threading, time\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
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
    "m = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], fitted=0)\n"
)


def test_a_pool_worker_alive_at_import_changing_the_model_between_two_checks_is_seen(
    cash_magics, mock_shell, monkeypatch
):
    """The branch's own scenario (`test_a_thread_changing_the_model_between_
    two_checks_is_seen`) with the change made by a ThreadPoolExecutor worker
    whose ident is in `THREADS_AT_IMPORT`: the pool was started before
    `import cash`, or the worker got the ident of a thread that was alive
    then (tests 3 and 4 below show both happen). The session hash names the
    model as it was before the change."""
    import cash

    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, MODEL_SETUP)
    run_cash_cell(cash_magics, "pool = ThreadPoolExecutor(1)\njob = pool.submit(change_it)")
    (worker,) = mock_shell.user_ns["pool"]._threads
    monkeypatch.setattr(cash, "THREADS_AT_IMPORT", cash.THREADS_AT_IMPORT | {worker})
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
        mock_shell.user_ns["job"].result(30)
        mock_shell.user_ns["pool"].shutdown()
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)


_IDENT_REUSE = textwrap.dedent(
    """
    import threading
    import numpy as np
    stop = threading.Event()
    early = threading.Thread(target=stop.wait, daemon=True)
    early.start()                     # alive at `import cash` (a kernel thread that ends later)
    import cash
    from cash.tracking import trace_recorder
    stop.set(); early.join()
    x = np.arange(10.0)
    go = threading.Event()
    def change():
        go.wait(30)
        x[0] = 99.0
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)       # started after import: its worker runs the library's `_worker`
    job = pool.submit(change)
    (later,) = pool._threads
    print(int(later.ident == early.ident), int(trace_recorder.other_thread_alive({"x": x, "pool": pool})))
    go.set(); job.result(); pool.shutdown()
    """
)


def test_a_thread_reusing_the_ident_of_one_alive_at_import_counts():
    out = subprocess.run([sys.executable, "-c", _IDENT_REUSE], capture_output=True, text=True, timeout=120, check=True)
    reused, alive = map(int, out.stdout.split())
    if not reused:
        pytest.skip("the OS gave the later thread a new ident this time")
    assert alive == 1, "a live thread was taken for the kernel's because it got a dead thread's ident"


_STARTED_BEFORE_IMPORT = textwrap.dedent(
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    x = np.arange(10.0)
    go = threading.Event()
    def change():
        go.wait(30)
        x[0] = 99.0
    pool = ThreadPoolExecutor(1)
    job = pool.submit(change)        # cell 1: a training job in the background
    import cash                      # cell 2: import cash; %cash_on
    from cash.tracking import trace_recorder
    print(int(trace_recorder.other_thread_alive({"x": x, "pool": pool, "job": job})))
    go.set(); job.result(); pool.shutdown()
    """
)


def test_a_pool_the_user_started_before_import_counts():
    out = subprocess.run(
        [sys.executable, "-c", _STARTED_BEFORE_IMPORT], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "1", "the user's pool worker, alive at import, is taken for the kernel's"
