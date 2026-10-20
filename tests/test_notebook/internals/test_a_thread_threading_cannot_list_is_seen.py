"""Review of claude/speed-model-once 8381d269 (one-read window) and the
base-gap fix: a thread `threading.enumerate()` does not list.

`other_thread_alive` looks at `threading.enumerate()`. A thread a C library
starts (an audio callback of sounddevice or PyAudio, a device driver's
reader, any ``ctypes``/cffi callback) runs Python code without ever being
listed there, unless that code calls `threading.current_thread()`. While
one is alive, the base-gap handoff and the one-read window both stand: a
change it makes between the in-place check and the session hash leaves the
session hash naming the model as it was.

Put this file in tests/test_notebook/internals/. Linux/macOS only
(pthread_create through ctypes).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import time

import pytest

from cash import value_hash, value_watch
from cash.tracking import trace_recorder
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="pthread_create through ctypes")

SETUP = (
    "import argparse, time\n"
    "def fit(model):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    model.fitted += 1\n"
    "    return model\n"
    "m = argparse.Namespace(rows=[[i, i * 2.0] for i in range(2000)], fitted=0)\n"
)

_START = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)


def _native_thread(work):
    """Start *work* on a thread made by the C library, as a C library's
    callback thread is: Python runs it, `threading` never hears of it."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None)
    if sys.platform != "darwin" and not hasattr(libc, "pthread_create"):
        libc = ctypes.CDLL(ctypes.util.find_library("pthread"))

    def run(_arg):
        work()
        return None

    start = _START(run)
    handle = ctypes.c_ulong()
    assert libc.pthread_create(ctypes.byref(handle), None, start, None) == 0
    return libc, handle, start  # *start* must outlive the thread


def test_a_c_librarys_thread_counts():
    flags = {"go": False, "done": False, "running": False}

    def work():
        flags["running"] = True
        while not flags["go"]:
            time.sleep(0.001)
        flags["done"] = True

    libc, handle, keep = _native_thread(work)
    try:
        deadline = time.monotonic() + 30
        while not flags["running"] and time.monotonic() < deadline:
            time.sleep(0.001)  # pthread_create returns before Python runs the callback
        assert trace_recorder.other_thread_alive({}), "a thread is alive that threading.enumerate() does not list"
    finally:
        flags["go"] = True
        libc.pthread_join(handle, None)


def test_a_c_librarys_thread_changing_the_model_between_two_checks_is_seen(cash_magics, mock_shell, monkeypatch):
    value_watch.reset()
    monkeypatch.setattr(value_watch, "WATCHING", False)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    flags = {"go": False, "done": False}

    def work():  # an audio callback filling the model's rows, say
        while not flags["go"]:
            time.sleep(0.001)
        mock_shell.user_ns["m"].rows.append([-1, -1.0])
        flags["done"] = True

    libc, handle, keep = _native_thread(work)
    routing = cash_magics._statement_processor._mutation_routing
    real_observe = routing.observe

    def observe(run):
        real_observe(run)  # reads m
        flags["go"] = True
        while not flags["done"]:
            time.sleep(0.001)

    monkeypatch.setattr(routing, "observe", observe)
    try:
        run_cash_cell(cash_magics, "fit(m)")
    finally:
        flags["go"] = True
        libc.pthread_join(handle, None)
    m = mock_shell.user_ns["m"]
    assert m.rows[-1] == [-1, -1.0]
    assert cash_magics.tracking_state.current_session_hashes["m"] == value_hash.compute_hash(m)
