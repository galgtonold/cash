"""A cell that ends with a thread of the user's still running, one it
started, is named once (``NOTEBOOK-THREAD-ALIVE``).

What that thread changes after the cell finished is outside cash's view.
Quiet for a cell that joins its thread, for a cell that starts none, for a
thread running library code (a pool's idle workers), and on a re-run.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

CODE = "NOTEBOOK-THREAD-ALIVE"

SETUP = "import threading, concurrent.futures\nstop = threading.Event()\ndef work():\n    stop.wait(10)"
LEFT_RUNNING = "t = threading.Thread(target=work, name='filler')\nt.start()"
JOINED = "stop.set()\nt2 = threading.Thread(target=work, name='joined')\nt2.start()\nt2.join()"
NO_THREAD = "x = 1 + 1"
POOL = "pool = concurrent.futures.ThreadPoolExecutor(2)\nn = pool.submit(lambda: 3).result()"
CELLS = [SETUP, LEFT_RUNNING, JOINED, NO_THREAD, POOL]


def _run(magics, *cells: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=CELLS)
    return [w for w in caught if getattr(w.message, "code", None) == CODE]


def _stop(magics) -> None:
    ns = magics.shell.user_ns
    ns["stop"].set()
    for name in ("t", "t2"):
        if name in ns:
            ns[name].join(5)
    if "pool" in ns:
        ns["pool"].shutdown()


def test_a_cell_that_leaves_its_thread_running_is_named(cash_magics):
    try:
        found = _run(cash_magics, SETUP, LEFT_RUNNING)
        assert len(found) == 1, found
        text = str(found[0].message)
        assert "cell 2" in text
        assert "'filler'" in text and "work" in text, "names the thread and what it runs"
        assert "t.join()" in text
    finally:
        _stop(cash_magics)


def test_it_is_named_once_per_cell(cash_magics):
    try:
        assert len(_run(cash_magics, SETUP, LEFT_RUNNING)) == 1, "control: the first run is named"
        assert not _run(cash_magics, LEFT_RUNNING), "named again on a re-run"
    finally:
        _stop(cash_magics)


def test_a_joined_thread_no_thread_and_a_pool_s_workers_are_quiet(cash_magics):
    try:
        assert not _run(cash_magics, SETUP, JOINED, NO_THREAD, POOL)
        assert cash_magics.shell.user_ns["n"] == 3
        assert any(t.is_alive() for t in cash_magics.shell.user_ns["pool"]._threads), (
            "control: the pool's workers are still alive"
        )
    finally:
        _stop(cash_magics)
