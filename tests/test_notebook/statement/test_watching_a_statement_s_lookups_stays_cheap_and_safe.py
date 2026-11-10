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
import collections
import gc
import logging
import sys
import threading
import types
import warnings
import weakref

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


LOOPS = {
    "a-global": "t = 0\nfor i in range(10_000):\n    t = add(t, i)",
    "a-loop-variable-s-method": "t = 0\nfor p in ps:\n    t += p.get()",
    "a-name-bound-in-the-loop": "t = 0\nfor i in range(10_000):\n    f = add\n    t = f(t, i)",
    "a-builtin-s-method": "out = []\nfor i in range(10_000):\n    out.append(str(i).split(','))",
    "a-generator-expression-made-per-call": "t = 0\nfor i in range(10_000):\n    t += length([i, i + 1, i * 2])",
    "a-nested-function-made-per-call": "t = 0\nfor i in range(10_000):\n    t = step(t, (i, i + 1))",
    "a-key-callback-making-a-generator-per-call": "best = min(range(40_000), key=score)",
}


@needs_monitoring
@pytest.mark.parametrize("loop", list(LOOPS))
def test_a_loop_pays_for_one_event_per_call_site(cash_magics, loop):
    """A call site is switched off once its callable is seen to: the loop's
    ten thousand calls make a handful of events, whatever names the callee.
    A generator expression or a nested function made again on each call,
    closing over numbers or lists of them, is the one seen first: no call
    switches the sites back on. Nor does one made by a key function that
    ``min`` calls from C, forty thousand times (the E16 notebook's
    ``min(tours, key=tour_length)``)."""
    run_cash_cell(
        cash_magics,
        "def add(a, b):\n    return a + b\n"
        "class P:\n    def get(self):\n        return 1\n"
        "ps = [P() for _ in range(10_000)]\n"
        "def length(xs):\n    return sum(add(xs[i], -xs[i - 1]) for i in range(len(xs)))\n"
        "def step(t, pair):\n    def inner():\n        return add(t, pair[0])\n    return inner()\n"
        "def score(i):\n    rows = [[i, 1], [2, (3, 4.0)]]\n    return sum(add(x, rows[0][0]) for x in rows[1][:1])",
    )
    events = []
    on_call = lookups_seen._on_call

    def counted(*args):
        events.append(args[1])
        return on_call(*args)

    assert lookups_seen.available()
    sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, counted)
    try:
        run_cash_cell(cash_magics, LOOPS[loop])
    finally:
        sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, on_call)
    assert len(events) < 20


@needs_monitoring
def test_a_generator_made_on_each_turn_of_a_loop_is_seen_once_per_call(cash_magics):
    """``mst``'s loop (the E16 notebook) makes a generator expression on
    each turn, closing over a dict too big to read through: each is the
    one made on the turn before, holding the same dict, and switches no
    site back on. A lambda closing over the dict, new on each call, switches
    them on as each call starts: three calls of 2,000 turns make a few
    dozen events, not one for each site on each turn."""
    run_cash_cell(
        cash_magics,
        "def first(items):\n    return next(iter(items))\n"
        "def pick(n):\n"
        "    seen = {i: [i] for i in range(n)}\n"
        "    out = [(lambda: len(seen))()]\n"
        "    for k in range(n):\n"
        "        out.append(first(seen[j][0] for j in (k,)))\n"
        "    return out",
    )
    events = []
    on_call = lookups_seen._on_call

    def counted(*args):
        events.append(args[1])
        return on_call(*args)

    assert lookups_seen.available()
    sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, counted)
    try:
        run_cash_cell(cash_magics, "picked = [len(pick(2_000 + i)) for i in range(3)]")
    finally:
        sys.monitoring.register_callback(lookups_seen._TOOL, sys.monitoring.events.CALL, on_call)
    assert cash_magics.shell.user_ns["picked"] == [2_001, 2_002, 2_003]
    assert len(events) < 100


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
        lookups_seen.arm([code], namespace)
        exec(code, namespace)
    assert namespace["levels"][0] == -1
    recipes = [key for key in found if key]
    assert len(recipes) == 1
    assert NAME.encode() in base64.b64decode(recipes[0])


@needs_monitoring
@pytest.mark.parametrize(
    "reader",
    [
        f"levels = [g({NAME!r}).level for g in (fake, fake, logging.getLogger)]",
        f"levels = []\nfor g in (fake, fake, logging.getLogger):\n    levels.append(g({NAME!r}).level)",
        f"levels = [(lambda: box[0][0]({NAME!r}))().level for box in ([[fake]], [[fake]], [[getattr(logging, 'get' + 'Logger')]])]",
        f"levels = [(lambda: get({NAME!r}))().level for get in (fake, fake, logging.getLogger)]",
    ],
    ids=["a-comprehension", "a-loop", "a-lambda-holding-a-nested-list", "a-lambda"],
)
def test_a_call_site_switched_off_still_sees_a_library_function_it_calls_next(reader, monkeypatch):
    """The site calling ``g`` is switched off once it called the fake
    twice: ``logging.getLogger``, which the statement reads, is called
    there next, and the logger it returns is found, though the process
    never saw it return one before. A lambda holding it in a list of
    lists, found by no name the statement reads, is not taken for one
    holding data: it is a new function, and is watched."""
    monkeypatch.setattr(lookups_seen, "_RETURNERS", weakref.WeakSet())
    namespace: dict = {}
    exec(SWAP, namespace)
    code = compile(reader, "<cash-test-switched-off>", "exec")
    with lookups_seen.watching([], namespace) as found:
        lookups_seen.arm([code], namespace)
        exec(code, namespace)
    assert namespace["levels"][:2] == [-1, -1]
    recipes = [key for key in found if key]
    assert len(recipes) == 1
    assert NAME.encode() in base64.b64decode(recipes[0])


@needs_monitoring
def test_a_function_made_again_holding_another_callable_is_watched(monkeypatch):
    """A lambda made on each turn of the loop closes over the function it
    calls: the third, holding ``logging.getLogger`` where the first two held
    a fake, is a new function, and the logger it returns is found."""
    monkeypatch.setattr(lookups_seen, "_RETURNERS", weakref.WeakSet())
    namespace: dict = {}
    exec(SWAP, namespace)
    reader = f"levels = [(lambda: get({NAME!r}))().level for get in (fake, fake, logging.getLogger)]"
    code = compile(reader, "<cash-test-made-again>", "exec")
    with lookups_seen.watching([], namespace) as found:
        lookups_seen.arm([code], namespace)
        exec(code, namespace)
    assert namespace["levels"][:2] == [-1, -1]
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
    is not checked, once: from the first statement that calls or reads an
    attribute."""
    monkeypatch.setattr(lookups_seen, "_install", lambda: False)
    monkeypatch.setattr(lookups_seen, "_REASON", "sys.monitoring needs Python 3.12 or newer")
    monkeypatch.setattr(lookups_seen, "_warned_unseen", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "a = 1 + 1")
        codes = [getattr(w.message, "code", None) for w in caught]
        assert "NOTEBOOK-LOOKUPS-UNSEEN" not in codes, "a statement that calls nothing finds nothing"
        run_cash_cell(cash_magics, "import os\nb = os.getcwd()")
        run_cash_cell(cash_magics, "c = os.getcwd()")
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


@needs_monitoring
def test_a_library_function_returning_a_logger_is_watched_from_any_call_site(cash_magics):
    """The call site calls a function of the notebook's first, and is
    switched off; ``logging.getLogger``, called there next, was seen
    returning a logger before, and is watched from the start."""
    setup = (
        f"import logging\nlogging.getLogger({NAME!r}).setLevel(10)\n"
        "import types\n"
        "def fake(name):\n    return types.SimpleNamespace(level=-1)\n"
        "getters = [fake, logging.getLogger]"
    )
    reader = f"seen = [get({NAME!r}).level for get in getters][1]"
    cells = [setup, reader, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", f"# again\n{reader}"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30


class Obj:
    """A class of the user's with a reduce of its own, counting its calls."""

    calls = 0

    def __init__(self, i):
        self.i = i

    def __reduce__(self):
        Obj.calls += 1
        return (Obj, (self.i,))


@needs_monitoring
def test_reading_a_module_s_list_runs_no_reduce_of_the_user_s():
    """``len(mod.OBJS)`` reads a list of objects of a class of the user's
    through a module: none of them is reduced to see whether it stands by
    reference."""
    mod = types.ModuleType("cash_test_objs")
    mod.OBJS = [Obj(i) for i in range(1_000)]
    namespace = {"mod": mod}
    code = compile("n = len(mod.OBJS)", "<cash-test-objs>", "exec")
    Obj.calls = 0
    with lookups_seen.watching((), {}) as found:
        lookups_seen.arm([code], namespace)
        exec(code, namespace)
    assert namespace["n"] == 1_000
    assert Obj.calls == 0
    assert found == {}


def _walk(namespace, source, monkeypatch):
    """Run *source* in *namespace*, watched: how many objects it reduced to
    see whether each stands by reference."""
    reduced = []
    note = lookups_seen._note_object
    monkeypatch.setattr(lookups_seen, "_note_object", lambda obj: reduced.append(obj) or note(obj))
    code = compile(source, "<cash-test-walk>", "exec")
    with lookups_seen.watching((), {}):
        lookups_seen.arm([code], namespace)
        exec(code, namespace)
    monkeypatch.undo()
    return len(reduced)


@needs_monitoring
def test_a_module_s_list_is_looked_through_once_while_it_holds_the_same_objects(monkeypatch):
    """Each object a module's list holds is reduced once while it stays
    there; one put in its place is reduced in turn, and a logger there is
    noted again on every statement, as its state may have changed."""
    mod = types.ModuleType("cash_test_walked")
    mod.ITEMS = [collections.OrderedDict(i=i) for i in range(1_000)] + [logging.getLogger(NAME)]
    namespace = {"mod": mod}
    assert _walk(namespace, "n = len(mod.ITEMS)", monkeypatch) == 1_001
    assert _walk(namespace, "n = len(mod.ITEMS)", monkeypatch) == 1
    mod.ITEMS[3] = collections.OrderedDict(i=-1)
    assert _walk(namespace, "n = len(mod.ITEMS)", monkeypatch) == 2
    # What it found holds none of them: dropped, each goes.
    gone = weakref.ref(mod.ITEMS[0])
    mod.ITEMS = None
    gc.collect()
    assert gone() is None


def test_a_timestamp_is_never_reduced_to_look_for_a_name():
    """A pandas Timestamp's reduce rebuilds it from numbers, whatever it
    holds: none in a module's list of them is reduced on a statement."""
    pd = pytest.importorskip("pandas")
    assert not kept_state.may_stand_by_reference(pd.Timestamp)
    assert not kept_state.may_stand_by_reference(pd.Timedelta)
    assert kept_state.may_stand_by_reference(logging.Logger)


def test_a_timestamp_is_not_taken_for_a_lookup():
    """A timestamp pickles as ``_unpickle_timestamp(<numbers>)``, which
    builds it again: not a lookup by a name, so a list of them is not found
    again item by item when hashed."""
    pd = pytest.importorskip("pandas")
    stamp = pd.Timestamp("2020-01-01")
    assert not kept_state.looks_up(stamp.__reduce_ex__(4))
    assert kept_state.looks_up(logging.getLogger(NAME).__reduce_ex__(4))
    assert kept_state.looks_up(logging.getLogger().__reduce_ex__(4))


@needs_monitoring
def test_monitoring_ids_freed_by_other_tools_are_taken_again(monkeypatch):
    """Both ids held by other tools: nothing can be watched. Once one is
    freed, it is taken."""
    mon = sys.monitoring
    assert lookups_seen.available()
    mine = lookups_seen._TOOL
    mon.free_tool_id(mine)
    taken = []
    try:
        for tool in (5, 2):
            if mon.get_tool(tool) is None:
                mon.use_tool_id(tool, "another tool")
                taken.append(tool)
        assert not lookups_seen.available()
        mon.free_tool_id(taken.pop())
        assert lookups_seen.available()
    finally:
        for tool in taken:
            mon.free_tool_id(tool)
    assert lookups_seen.available()
