"""A digest taken inside a `value_hash.one_read` window, or under a
`value_hash.watch`, is not handed on past code of the user's that ran in
between and changed the value.

Four ways it was, each from a review of the window:

- a user's done-callback parked on a process pool's manager thread made the
  pool look idle, so a window opened, and the callback then changed the value;
- a weakref callback that is C code (a bound ``list.append``) run by the
  collector starts no Python code, so nothing closed the window;
- a thread whose code had just been re-armed could switch it off again, and
  from then on its runs were not seen;
- a watch that lets the statement's own code start once let any cell's
  module code start, any number of times.

The user's code must be the user's: this file is a test module, which counts
as the host's, so the functions that change values live in a module written
for the test.
"""

from __future__ import annotations

import functools
import gc
import importlib.util
import os
import sys
import threading
import weakref

import pytest

from cash import value_hash
from cash._paths import CELL_NAME_PREFIX
from cash.decorator import code_trace
from cash.tracking.trace_recorder import other_thread_alive

pytestmark = pytest.mark.usefixtures("user_code_watched")

USER_MODULE = """
class Big:
    def __init__(self, n):
        self.rows = [[i, i * 2.0] for i in range(n)]


value = Big(50)
data = [1, 2, 3]


def touch():
    value.rows.append([-1, -1.0])


def parked_callback(parked, release, target):
    def callback(fut):
        parked.set()
        release.wait(30)
        target.rows.append([-1, -1.0])

    return callback
"""


@pytest.fixture
def user(tmp_path, monkeypatch):
    path = tmp_path / "users_own_code.py"
    path.write_text(USER_MODULE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("users_own_code", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # its values pickle
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector_off():
    gc.disable()
    yield
    gc.enable()


def test_a_pool_whose_callback_is_mid_run_counts_as_alive(user):
    from concurrent.futures import ProcessPoolExecutor

    parked, release = threading.Event(), threading.Event()
    with ProcessPoolExecutor(1) as pool:
        fut = pool.submit(os.getpid)
        fut.add_done_callback(user.parked_callback(parked, release, user.Big(5)))
        try:
            assert parked.wait(30)
            assert other_thread_alive({}), "the callback can change a value at any moment"
        finally:
            release.set()


@pytest.mark.usefixtures("collector_off")
def test_a_c_finalizer_run_by_the_collector_ends_the_window():
    value = [[i, i * 2.0] for i in range(50)]

    class Trigger:
        pass

    t = Trigger()
    t.me = t  # only the collector frees it
    keep = [weakref.ref(t, functools.partial(value.append))]
    del t
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
        gc.collect()
        assert len(value) == 51, "the finalizer ran"
        assert not value_hash.reading_once()
        assert value_hash.compute_hash(value) != before
    assert keep


def test_code_re_armed_while_another_thread_disables_it_stays_watched(user, monkeypatch):
    rec = code_trace._recorder()
    user.touch()
    user.touch()  # discovered, armed, run once more with its events off
    del user.value.rows[50:]
    reached, go = threading.Event(), threading.Event()
    real = rec._seen_in_window

    def seen(code, frame):
        real(code, frame)
        if code is user.touch.__code__ and threading.current_thread().name == "B":
            reached.set()
            go.wait(10)  # where a switch to the other thread can fall

    monkeypatch.setattr(rec, "_seen_in_window", seen)
    code_trace._RECORDER.watching()  # re-arms touch
    w = value_hash.watch()  # a watch is open: B's start is looked at
    b = threading.Thread(target=user.touch, name="B")
    b.start()
    try:
        assert reached.wait(10)
        with value_hash.one_read():  # re-arms while B is mid-callback
            go.set()
            b.join()
            del user.value.rows[50:]
            before = value_hash.compute_hash(user.value)
            user.touch()
            assert not value_hash.reading_once(), "touch ran: the window ends"
            assert value_hash.compute_hash(user.value) != before
    finally:
        go.set()
        b.join()
        w.close()


def test_another_cells_code_ends_a_watch_that_lets_the_statement_start(user):
    ns = user.__dict__  # the namespace of code of the user's
    other_cell = compile("data.append(99)\n", CELL_NAME_PREFIX + "7-0>", "exec")
    w = value_hash.watch(statement_code_ok=True)
    try:
        exec(other_cell, ns)  # not from cash's statement runner
        assert ns["data"][-1] == 99
        assert not w.open, "a cell's code ran outside the statement: the watch ends"
    finally:
        w.close()


def test_a_cell_started_on_another_thread_ends_the_watch(user):
    ns = user.__dict__
    cell = compile("data.append(99)\n", CELL_NAME_PREFIX + "8-0>", "exec")
    w = value_hash.watch(statement_code_ok=True)
    try:
        t = threading.Thread(target=exec, args=(cell, ns))
        t.start()
        t.join()
        assert ns["data"][-1] == 99
        assert not w.open
    finally:
        w.close()


def test_cashs_own_garbage_waits_while_a_watch_is_open():
    """A collection that frees only cash's own leftovers would end the watch
    as surely as a finalizer: fewer run while one is open, and the
    threshold is the collector's own again once it closes."""
    own = gc.get_threshold()
    w = value_hash.watch()
    try:
        assert gc.get_threshold()[0] > own[0]
        for _ in range(5000):  # cyclic garbage, as cash's syntax-tree walks leave
            a = []
            a.append(a)
        assert w.open
    finally:
        w.close()
    assert gc.get_threshold() == own


def test_a_cells_end_ends_every_watch_left_open():
    own = gc.get_threshold()
    w = value_hash.watch()
    value_hash.end_watches()
    assert not w.open
    assert gc.get_threshold() == own
