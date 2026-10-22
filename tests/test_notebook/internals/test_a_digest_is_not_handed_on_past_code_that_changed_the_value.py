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


def test_a_watch_leaves_the_collectors_threshold_alone():
    """The user's program runs as it would without cash: a watch open
    through a library's call must not let its garbage pile up."""
    own = gc.get_threshold()
    w = value_hash.watch()
    try:
        assert gc.get_threshold() == own
    finally:
        w.close()


def test_young_garbage_is_collected_before_a_watch_opens():
    """cash's own leftovers, collected before the watch rather than inside
    it, where the collection would end it."""
    threshold = gc.get_threshold()[0]
    for _ in range(threshold):  # cyclic garbage, as an import leaves
        a = []
        a.append(a)
    w = value_hash.watch()
    try:
        assert gc.get_count()[0] < threshold // 2
        kept = [[] for _ in range(threshold // 4)]  # what cash's checks make
        assert kept
        assert w.open, "no collection ended it"
    finally:
        w.close()


def _cycles(n):
    for _ in range(n):
        a = []
        a.append(a)


def test_cashs_own_garbage_inside_a_watch_does_not_end_it():
    """What cash's first-time work leaves (an import, a look at the installed
    packages) is collected without ending the watch, however much it is."""
    gc.collect()
    w = value_hash.watch()
    try:
        with value_hash.own_garbage():
            _cycles(gc.get_threshold()[0] * 3)
        assert w.open
        assert gc.isenabled()
    finally:
        w.close()


def test_garbage_left_before_cashs_own_work_still_ends_the_watch():
    """Only what the work itself leaves is passed over: a cycle that was
    garbage before it, whose collection may run the user's callbacks, ends
    the watch."""

    class Node:
        pass

    gc.collect()
    w = value_hash.watch()
    try:
        holder = Node()
        holder.me = holder
        weakref.finalize(holder, int)
        del holder
        with value_hash.own_garbage():
            pass
        assert not w.open
    finally:
        w.close()


def test_the_syntax_tree_fixer_leaves_no_garbage_and_fixes_as_the_stdlib():
    import ast

    from cash.analysis.ast_util import fix_locations

    source = "x = f(a, b=[1, 2])\nif x:\n    y = x + 1\n"
    ours, theirs = ast.parse(source), ast.parse(source)
    for tree in (ours, theirs):
        tree.body.append(ast.Expr(ast.Call(ast.Name("g", ast.Load()), [ast.Constant(1)], [])))
    gc.collect()
    gc.disable()
    try:
        fix_locations(ours)
        assert gc.collect() == 0, "the fixer left a cycle"
    finally:
        gc.enable()
    ast.fix_missing_locations(theirs)
    assert ast.dump(ours, include_attributes=True) == ast.dump(theirs, include_attributes=True)


def test_a_cells_end_ends_every_watch_left_open():
    w = value_hash.watch()
    value_hash.end_watches()
    assert not w.open
    assert not value_hash.OPEN_WATCHES


def test_code_run_while_a_watch_is_being_armed_ends_it(user, monkeypatch):
    """A watch is registered before the user's code is armed again: another
    thread's call in between finds it and ends it, rather than turning its
    code's events off again for the watch to miss."""
    user.touch()
    user.touch()  # discovered, armed, its events off after one run
    real = code_trace.watching_user_code

    def other_thread_runs_first():
        b = threading.Thread(target=user.touch)
        b.start()
        b.join()
        return real()

    monkeypatch.setattr(code_trace, "watching_user_code", other_thread_runs_first)
    with value_hash.one_read():
        monkeypatch.setattr(code_trace, "watching_user_code", real)
        del user.value.rows[50:]
        before = value_hash.compute_hash(user.value)
        user.touch()
        assert value_hash.compute_hash(user.value) != before


def test_a_finalizer_that_opens_a_watch_while_code_is_armed_does_not_hang():
    """Arming allocates, an allocation can start a collection, and a
    finalizer it runs may open a watch, which arms again on this thread."""
    left = [300]

    class Res:
        def __del__(self):
            value_hash.watch().close()
            if left[0] > 0:  # the next piece of cyclic garbage
                left[0] -= 1
                n = Res()
                n.me = n

    def opens():
        for _ in range(300):
            with value_hash.one_read():
                pass

    own = gc.get_threshold()
    r = Res()
    r.me = r
    del r
    gc.set_threshold(1)
    try:
        t = threading.Thread(target=opens, daemon=True)
        t.start()
        t.join(30)
    finally:
        gc.set_threshold(*own)
    assert not t.is_alive(), "hung re-arming inside its own re-arm"


class _Old:
    pass


def test_cash_letting_go_of_a_values_last_holder_ends_the_window():
    """After ``x = f(x)`` cash's hand-off may be all that holds the old
    ``x``; a finalizer the user gave it in C runs when cash lets go, with no
    code of the user's starting."""
    value = [[i, i * 2.0] for i in range(50)]
    old = _Old()
    weakref.finalize(old, value.append, [-1, -1.0])
    held = {id(old): (old, "digest")}
    del old
    with value_hash.one_read():
        before = value_hash.compute_hash(value)
        value_hash.letting_go(held)
        held.clear()
        assert len(value) == 51, "the finalizer ran"
        assert not value_hash.reading_once()
        assert value_hash.compute_hash(value) != before


def test_letting_go_of_a_value_held_elsewhere_keeps_the_window():
    kept = _Old()
    with value_hash.one_read():
        value_hash.letting_go({id(kept): (kept, "digest")})
        assert value_hash.reading_once()
