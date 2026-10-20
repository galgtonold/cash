"""Which code ran in a unit (a statement, a cached call), on ``sys.monitoring``.

``PY_START`` and ``PY_RESUME`` callbacks that return ``DISABLE``: each code
object costs one callback until the events are armed again, which
:meth:`Recorder.begin` does with one ``restart_events()`` per unit. A unit
nested in another (a cached call inside a statement) arms them again for
itself, and what it saw is added to the unit around it when it ends.

Code is told apart by the module its frame's globals belong to
(``f_globals['__name__']``), never by its file name: IPython's autoreload
compiles a function it adds to a module under ``<string>``.

What makes a trace unusable is recorded as a *marker* on the unit:

* ``thread_alive``: a thread the unit started still runs after it ended;
* ``process_pool``: it started worker processes, whose code runs elsewhere;
* ``generator_resumed``: it resumed a generator or coroutine of the user's
  started before it, whose state is no input the trace names;
* ``tool_conflict``: another tool took the monitoring tool id or turned its
  events off while the unit ran.

Seen but not unusable (*notes*): ``other_thread`` (user code ran on another
thread during the unit and was recorded).
"""

from __future__ import annotations

import sys
import threading
import types
import weakref
from typing import Any

from ..tracking.tracker_context import untracked

__all__ = ["TOOL_NAME", "Recorder", "Unit", "other_thread_alive", "runs_user_code"]

TOOL_NAME = "cash-shadow"
#: Tried in this order; 0-2 belong to debuggers, coverage and profilers.
_TOOL_IDS = (4, 3)

RESUMABLE = 0x20 | 0x80 | 0x200  # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR

#: Library code whose start means the unit's work runs in other processes.
POOL_STARTS = frozenset(
    {
        ("multiprocessing.process", "BaseProcess.start"),
        ("multiprocessing.pool", "Pool.apply_async"),
        ("multiprocessing.pool", "Pool._map_async"),
        ("multiprocessing.pool", "Pool.imap"),
        ("multiprocessing.pool", "Pool.imap_unordered"),
        ("concurrent.futures.process", "ProcessPoolExecutor.submit"),
        # joblib's process backend; its sequential and threading backends
        # (sklearn's default) run in this process and are recorded.
        ("joblib.externals.loky.process_executor", "ProcessPoolExecutor.submit"),
    }
)


def inspect_unwrap(fn: Any) -> Any:
    """The function a bound method or partial runs."""
    for _ in range(4):
        inner = getattr(fn, "__func__", None) or getattr(fn, "func", None)
        if inner is None:
            break
        fn = inner
    return fn


def _target_name(t: threading.Thread) -> str:
    fn = getattr(t, "_target", None)
    return getattr(fn, "__qualname__", "") if fn is not None else ""


class Unit:
    """What one unit saw run."""

    __slots__ = (
        "cells",
        "codes",
        "extra",
        "kind",
        "markers",
        "notes",
        "pyplot_ran",
        "receivers",
        "snap",
        "thread",
        "threads_before",
    )

    def __init__(self, kind: str) -> None:
        self.kind = kind
        #: code -> (globals, module name) of the user code that ran.
        self.codes: dict[types.CodeType, tuple[dict, str]] = {}
        #: code -> the class of ``self`` (or ``cls``) the method first ran on.
        self.receivers: dict[types.CodeType, type] = {}
        #: code -> its closure cells' digests as it first started (`start_cells`).
        self.cells: dict[types.CodeType, tuple] = {}
        self.markers: set[str] = set()
        self.notes: set[str] = set()
        #: Item lists of cached calls served inside the unit (they did not run).
        self.extra: list[tuple] = []
        #: Code of ``matplotlib.pyplot`` ran in it: it read or changed
        #: pyplot's current figure.
        self.pyplot_ran = False
        #: What the caller noted before the unit ran (the lineages it read).
        self.snap: Any = None
        self.thread = threading.get_ident()
        self.threads_before: frozenset[int] = frozenset()

    def absorb(self, inner: Unit) -> None:
        for code, where in inner.codes.items():
            self.codes.setdefault(code, where)
        for code, cls in inner.receivers.items():
            self.receivers.setdefault(code, cls)
        for code, cells in inner.cells.items():
            self.cells.setdefault(code, cells)
        self.markers |= inner.markers
        self.notes |= inner.notes
        self.pyplot_ran = self.pyplot_ran or inner.pyplot_ran
        self.extra.extend(inner.extra)


class Recorder:
    """The monitoring tool: one per process."""

    tool_name = TOOL_NAME
    tool_ids = _TOOL_IDS

    def __init__(self, notebook_ns: Any = None) -> None:
        self.tool: int | None = None
        self.reason: str | None = None
        self.stack: list[Unit] = []
        #: id(globals) -> (globals, module name or None when not the user's)
        self._kinds: dict[int, tuple[dict, str | None, str]] = {}
        self.notebook_ns = notebook_ns
        self.notebook_name = "__main__"
        self.restarts = 0

    # -- the tool ---------------------------------------------------------
    def install(self) -> bool:
        mon = getattr(sys, "monitoring", None)
        if mon is None:
            self.reason = "sys.monitoring needs Python 3.12 or newer"
            return False
        for tool in self.tool_ids:
            if mon.get_tool(tool) is None:
                mon.use_tool_id(tool, self.tool_name)
                self.tool = tool
                break
        else:
            held = {t: mon.get_tool(t) for t in self.tool_ids}
            self.reason = f"monitoring tool ids {' and '.join(map(str, self.tool_ids))} are taken ({held})"
            return False
        ev = mon.events
        mon.register_callback(self.tool, ev.PY_START, self._on_start)
        mon.register_callback(self.tool, ev.PY_RESUME, self._on_resume)
        mon.set_events(self.tool, ev.PY_START | ev.PY_RESUME)
        return True

    def uninstall(self) -> None:
        mon = sys.monitoring
        if self.tool is None or mon.get_tool(self.tool) != self.tool_name:
            return
        mon.set_events(self.tool, 0)
        mon.register_callback(self.tool, mon.events.PY_START, None)
        mon.register_callback(self.tool, mon.events.PY_RESUME, None)
        mon.free_tool_id(self.tool)
        self.tool = None

    def other_tools(self) -> dict[int, str]:
        """Tools other than this one holding an id now: ``restart_events``
        arms their disabled events again too."""
        mon = sys.monitoring
        return {t: mon.get_tool(t) for t in range(6) if t != self.tool and mon.get_tool(t) is not None}

    def _intact(self) -> bool:
        mon = sys.monitoring
        if self.tool is None or mon.get_tool(self.tool) != self.tool_name:
            return False
        return mon.get_events(self.tool) == mon.events.PY_START | mon.events.PY_RESUME

    # -- units ------------------------------------------------------------
    def begin(self, kind: str) -> Unit:
        unit = Unit(kind)
        if not self._intact():
            unit.markers.add("tool_conflict")
            mon = sys.monitoring
            if self.tool is not None and mon.get_tool(self.tool) == self.tool_name:
                # Still ours, its events turned off: arm them again.
                mon.set_events(self.tool, mon.events.PY_START | mon.events.PY_RESUME)
        unit.threads_before = frozenset(t.ident for t in threading.enumerate() if t.ident is not None)
        self.stack.append(unit)
        sys.monitoring.restart_events()
        self.restarts += 1
        return unit

    def end(self, unit: Unit) -> Unit:
        stack = self.stack
        while stack:
            top = stack.pop()
            if top is unit:
                break
        if not self._intact():
            unit.markers.add("tool_conflict")
        for t in threading.enumerate():
            if t.ident not in unit.threads_before and t.is_alive():
                if self._runs_user_code(t):
                    unit.markers.add("thread_alive")
                else:
                    # A library's own thread (a writer, a pool's idle
                    # worker): what it runs is no code of the user's.
                    unit.notes.add("library_thread:" + type(t).__name__ + ":" + _target_name(t))
        if stack:
            stack[-1].absorb(unit)
        return unit

    def _runs_user_code(self, t: threading.Thread) -> bool:
        return runs_user_code(t, self.notebook_ns)

    # -- callbacks --------------------------------------------------------
    def _classify(self, g: dict) -> tuple[dict, str | None, str]:
        name = g.get("__name__")
        name = name if isinstance(name, str) else "?"
        user: str | None = None
        if g is self.notebook_ns:
            user = self.notebook_name
        elif name != "cash" and not name.startswith("cash."):
            # By the module the globals claim: ``exec``'d code (a dataclass's
            # generated ``__init__``) runs on its module's own dict.
            mod = sys.modules.get(name)
            file = g.get("__file__")
            if mod is not None and getattr(mod, "__dict__", None) is not g and isinstance(file, str):
                # A module that put another object in its place in
                # ``sys.modules`` (``torch._VF``): judged by its own file,
                # not by the replacement, which has none.
                if is_user_file(file):
                    user = name
            elif mod is not None and is_user_code(mod):
                user = name
        entry = (g, user, name)
        # Kept for a module's or the notebook's namespace, which live as long
        # as the process anyway; an ``exec``'d namespace is classified again
        # rather than kept alive here, with every function defined in it.
        mod = sys.modules.get(name)
        if g is self.notebook_ns or (mod is not None and getattr(mod, "__dict__", None) is g):
            if len(self._kinds) > 4096:
                self._kinds.clear()
            self._kinds[id(g)] = entry
        return entry

    def _on_start(self, code: types.CodeType, offset: int) -> Any:
        stack = self.stack
        if not stack:
            return sys.monitoring.DISABLE
        frame = sys._getframe(1)
        g = frame.f_globals
        kind = self._kinds.get(id(g))
        if kind is None or kind[0] is not g:
            kind = self._classify(g)
        unit = stack[-1]
        if kind[1] is None:
            if kind[2] == "matplotlib.pyplot":
                unit.pyplot_ran = True
            elif (kind[2], code.co_qualname) in POOL_STARTS:
                unit.markers.add("process_pool")
            return sys.monitoring.DISABLE
        unit.codes[code] = (g, kind[1])
        if threading.get_ident() != unit.thread:
            unit.notes.add("other_thread")
        if code.co_argcount and code.co_varnames[0] in ("self", "cls"):
            try:
                recv = frame.f_locals.get(code.co_varnames[0])
            except Exception:  # noqa: BLE001 - a frame whose locals cannot be read has no receiver
                recv = None
            if recv is not None:
                unit.receivers[code] = recv if isinstance(recv, type) else type(recv)
        return sys.monitoring.DISABLE

    def _on_resume(self, code: types.CodeType, offset: int) -> Any:
        stack = self.stack
        if not stack or not code.co_flags & RESUMABLE:
            return sys.monitoring.DISABLE
        unit = stack[-1]
        if code in unit.codes:
            return sys.monitoring.DISABLE
        g = sys._getframe(1).f_globals
        kind = self._kinds.get(id(g))
        if kind is None or kind[0] is not g:
            kind = self._classify(g)
        if kind[1] is not None:
            unit.codes[code] = (g, kind[1])
            unit.markers.add("generator_resumed")
        return sys.monitoring.DISABLE


def is_user_file(path: str) -> bool:
    """`is_user_path`, its first verdict's reads kept out as `is_user_code`'s."""
    from ..install_paths import is_user_path

    with untracked():
        return is_user_path(path)


def is_user_code(mod: types.ModuleType) -> bool:
    """`is_user_code_module`, its own reads (a listing of site-packages to
    place the file) kept out of the statement or call being recorded."""
    from ..install_paths import is_user_code_module

    with untracked():
        return is_user_code_module(mod)


def runs_user_code(t: threading.Thread, notebook_ns: Any) -> bool:
    """Does thread *t* run code of the user's: its target, or its own
    ``run`` method?"""
    fn = getattr(t, "_target", None)
    if fn is None and type(t).run is not threading.Thread.run:
        fn = type(t).run
    if fn is None:
        return False
    g = getattr(inspect_unwrap(fn), "__globals__", None)
    if g is notebook_ns:
        return True
    name = getattr(fn, "__module__", None) or (g or {}).get("__name__")
    if not isinstance(name, str) or name == "cash" or name.startswith("cash."):
        return False
    mod = sys.modules.get(name)
    return mod is not None and is_user_code(mod)


#: Threads of what hosts the notebook that change none of its values (a test
#: runner's watchdog), left out by `other_thread_alive`. Held weakly.
HOST_THREADS: weakref.WeakSet[threading.Thread] = weakref.WeakSet()

#: The packages of what hosts the notebook whose threads `threading` does
#: not list (a test runner's channel threads), left out by
#: `other_thread_alive`.
HOST_PACKAGES: set[str] = set()


def _unlisted_python_thread(listed: set[int | None]) -> bool:
    """Is a thread running Python that `threading` does not list, other
    than cash's own and the host's (`_hosts_unlisted`)? No frame is held
    past the look: this thread's own, kept in a local, is a cycle that keeps
    every caller's locals alive until a collection, and the checks that
    count a value's holders would count them."""
    frames = sys._current_frames()
    try:
        return any(ident not in listed and not _hosts_unlisted(frame) for ident, frame in frames.items())
    finally:
        frames.clear()


def _hosts_unlisted(frame: Any) -> bool:
    """Is the unlisted thread running *frame* cash's own or the host's, by
    the module of its outermost frame?"""
    while frame.f_back is not None:
        frame = frame.f_back
    name = frame.f_globals.get("__name__")
    if not isinstance(name, str):
        return False
    root = name.partition(".")[0]
    return root == "cash" or root in HOST_PACKAGES


def _code_of(t: threading.Thread) -> Any:
    """What *t* runs: a Timer's function, its target, or its own ``run``."""
    if isinstance(t, threading.Timer):
        return t.function
    return getattr(t, "_target", None) or type(t).run


#: The kernel's own threads, named exactly (``module:qualname`` of the
#: thread's class, or of its target), whenever they started: pyzmq's garbage
#: thread, started when a message is first sent without a copy, ipykernel's
#: iopub and pipe watchers, started again when the kernel restarts its
#: streams, and the rest of ipykernel's and IPython's. Not their packages,
#: and not "alive when cash was imported": IPython runs a user's background
#: job (``jobs.new(f)``) on its own Thread subclass, and a tornado loop a
#: user starts on a thread runs the user's callbacks, in a cell before
#: ``import cash`` as well as after.
_KERNEL_THREADS = frozenset(
    {
        "zmq.utils.garbage:GarbageCollectorThread",
        "ipykernel.iostream:IOPubThread._thread_main",
        "ipykernel.iostream:OutStream._watch_pipe_fd",
        "ipykernel.control:ControlThread",
        "ipykernel.heartbeat:Heartbeat",
        "ipykernel.shellchannel:ShellChannelThread",
        "ipykernel.parentpoller:ParentPollerUnix",
        "ipykernel.parentpoller:ParentPollerWindows",
        "IPython.core.history:HistorySavingThread",
    }
)


def _named(obj: Any) -> str | None:
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    return f"{module}:{qualname}" if isinstance(module, str) and isinstance(qualname, str) else None


def _kernel_code(t: threading.Thread) -> bool:
    """Is *t* one of the kernel's own threads (`_KERNEL_THREADS`), by its
    exact class or its target, whenever it started?"""
    target = getattr(t, "_target", None)
    if target is None:
        return not isinstance(t, threading.Timer) and _named(type(t)) in _KERNEL_THREADS
    return _named(getattr(target, "__func__", target)) in _KERNEL_THREADS


def other_thread_alive(notebook_ns: Any) -> bool:
    """Might another thread change a value between two checks that read it,
    so that a digest one took cannot stand for the other: any live thread
    but this one, cash's own (by the code it runs; a Timer by its function)
    and the kernel's (its own threads named exactly, `_kernel_code`), daemon
    or not. A pool's worker runs the user's code under a library's name,
    which `runs_user_code` cannot see, and a wrong yes only costs a read
    again. A thread a C library started (an audio or device callback) runs
    Python unlisted by `threading`: any thread running Python that it does
    not list counts too."""
    me = threading.current_thread()
    threads = threading.enumerate()
    listed = {t.ident for t in threads}
    if _unlisted_python_thread(listed):
        return True
    for t in threads:
        if t is me or t in HOST_THREADS or not t.is_alive():
            continue
        if runs_user_code(t, notebook_ns):
            return True
        if _kernel_code(t):
            continue
        name = getattr(_code_of(t), "__module__", None)
        if not (isinstance(name, str) and (name == "cash" or name.startswith("cash."))):
            return True
    return False
