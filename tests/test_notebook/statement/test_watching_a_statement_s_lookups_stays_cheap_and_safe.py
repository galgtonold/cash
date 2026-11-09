"""Watching what a statement finds by a lookup costs little, and fails safe.

``seen = logging.getLogger('app').level`` is watched as it runs
(`lookups_seen`). As first written, every statement read the state of every
logger the process had ever been seen holding, before and after it ran: 200
loggers made ``q = 1 + 1`` take 83 ms. Every call a statement made paid for
a Python callback, a loop of a million calls 0.11 s more. A monitoring tool
id freed by another tool made every later cell raise. Where nothing can be
watched (before Python 3.12), nothing said so. And a state digest taken on a
thread that never keyed raised.
"""

from __future__ import annotations

import base64
import logging
import sys
import threading
import warnings

import pytest

from cash import kept_state
from cash.tracking import lookups_seen
from tests._cell_driver import run_cash_cell

NAME = "cash_test_lookups_watched"

needs_monitoring = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring needs Python 3.12")


@pytest.fixture(autouse=True)
def keep_everything(cash_instance, monkeypatch):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    monkeypatch.setattr(logging.root, "handlers", [])
    yield
    logging.getLogger(NAME).setLevel(logging.NOTSET)


def test_a_statement_reading_no_logger_reads_none(cash_magics, monkeypatch):
    """Loggers the notebook made are not read for a statement that reads
    none of them."""
    run_cash_cell(
        cash_magics,
        f"import logging\nlogs = [logging.getLogger('{NAME}.%d' % i) for i in range(50)]\n"
        "for lg in logs:\n    lg.setLevel(10)",
    )
    read = []
    digest = kept_state.state_digest
    monkeypatch.setattr(kept_state, "state_digest", lambda obj: read.append(obj) or digest(obj))
    run_cash_cell(cash_magics, "q = 1 + 1")
    assert cash_magics.shell.user_ns["q"] == 2
    assert [obj for obj in read if isinstance(obj, logging.Logger)] == []


@needs_monitoring
def test_a_loop_pays_for_one_event_per_call_site(cash_magics, monkeypatch):
    """A call site that can only call what it called is switched off once
    seen: the loop's ten thousand calls make a handful of events."""
    run_cash_cell(cash_magics, "def add(a, b):\n    return a + b")
    events = []
    on_call = lookups_seen._on_call

    def counted(*args):
        events.append(args[1])
        return on_call(*args)

    assert lookups_seen.available()
    sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, counted)
    try:
        run_cash_cell(cash_magics, "t = 0\nfor i in range(10_000):\n    t = add(t, i)")
    finally:
        sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, on_call)
    assert cash_magics.shell.user_ns["t"] == sum(range(10_000))
    assert len(events) < 20


SWAP = (
    "import logging, types\n"
    "def fake(name):\n"
    "    return types.SimpleNamespace(level=-1)\n"
    "getter = fake\n"
    "def swap(i):\n"
    "    global getter\n"
    "    getter = logging.getLogger if i == 0 else fake\n"
)


@needs_monitoring
def test_a_call_site_whose_callee_a_function_rebinds_is_watched_again():
    """``getter`` is a fake on the first call and ``logging.getLogger`` on
    the second, rebound by ``swap``: the site switched off after the first
    is switched back on when ``swap`` is seen to bind it, and the logger the
    second call returns is found."""
    namespace: dict = {}
    exec(SWAP, namespace)
    reader = f"levels = [(getter({NAME!r}).level, swap(i))[0] for i in range(2)]"
    code = compile(reader, "<cash-test-swap>", "exec")
    with lookups_seen.watching([namespace["swap"]], namespace) as found:
        lookups_seen.arm([code], namespace, reader, 0)
        exec(code, namespace)
    assert namespace["levels"][0] == -1
    recipes = [key for key in found if key]
    assert len(recipes) == 1
    assert NAME.encode() in base64.b64decode(recipes[0])


@needs_monitoring
def test_a_monitoring_id_freed_by_another_tool_breaks_no_cell(cash_magics):
    """Another tool frees the id cash took: the next cells run, and a
    reader of a logger it then finds is still not served stale."""
    setup = f"import logging\nlogging.getLogger({NAME!r}).setLevel(10)"
    reader = f"seen = logging.getLogger({NAME!r}).level"
    cells = [setup, reader]
    run_cash_cell(cash_magics, setup, cells=cells)
    run_cash_cell(cash_magics, reader, cells=cells)
    assert lookups_seen.available()
    sys.monitoring.free_tool_id(lookups_seen._TOOL)
    run_cash_cell(cash_magics, "z = 1 + 1", cells=[*cells, "z = 1 + 1"])
    assert cash_magics.shell.user_ns["z"] == 2
    run_cash_cell(cash_magics, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", cells=cells)
    run_cash_cell(cash_magics, reader, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30


def test_lookups_that_cannot_be_watched_are_named_once(cash_magics, monkeypatch):
    """Where a statement's lookups cannot be watched, a warning says what
    is not checked, once."""
    monkeypatch.setattr(lookups_seen, "_TOOL", None)
    monkeypatch.setattr(lookups_seen, "_REASON", "sys.monitoring needs Python 3.12 or newer")
    monkeypatch.setattr(lookups_seen, "_warned_unseen", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "a = 1 + 1")
        run_cash_cell(cash_magics, "b = 2 + 2")
    codes = [getattr(w.message, "code", None) for w in caught]
    assert codes.count("NOTEBOOK-LOOKUPS-UNSEEN") == 1
    (message,) = [str(w.message) for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-LOOKUPS-UNSEEN"]
    assert "Python 3.12" in message
    assert "getLogger" in message


def test_a_state_digest_on_a_thread_that_never_keyed():
    found = []
    thread = threading.Thread(target=lambda: found.append(kept_state.state_digest(logging.getLogger(NAME))))
    thread.start()
    thread.join()
    assert len(found) == 1
