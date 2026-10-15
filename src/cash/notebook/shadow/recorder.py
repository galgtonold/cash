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
from typing import Any

__all__ = ["TOOL_NAME", "Recorder", "Unit"]

TOOL_NAME = "cash-shadow"
#: Tried in this order; 0-2 belong to debuggers, coverage and profilers.
_TOOL_IDS = (4, 3)

_RESUMABLE = 0x20 | 0x80 | 0x200  # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR

#: Library code whose start means the unit's work runs in other processes.
_POOL_STARTS = frozenset(
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

    __slots__ = ("codes", "extra", "kind", "markers", "notes", "receivers", "snap", "thread", "threads_before")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        #: code -> (globals, module name) of the user code that ran.
        self.codes: dict[types.CodeType, tuple[dict, str]] = {}
        #: code -> the class of ``self`` (or ``cls``) the method first ran on.
        self.receivers: dict[types.CodeType, type] = {}
        self.markers: set[str] = set()
        self.notes: set[str] = set()
        #: Item lists of cached calls served inside the unit (they did not run).
        self.extra: list[tuple] = []
        #: What the caller noted before the unit ran (the lineages it read).
        self.snap: Any = None
        self.thread = threading.get_ident()
        self.threads_before: frozenset[int] = frozenset()

    def absorb(self, inner: Unit) -> None:
        for code, where in inner.codes.items():
            self.codes.setdefault(code, where)
        for code, cls in inner.receivers.items():
            self.receivers.setdefault(code, cls)
        self.markers |= inner.markers
        self.notes |= inner.notes
        self.extra.extend(inner.extra)


class Recorder:
    """The monitoring tool: one per process."""

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
        for tool in _TOOL_IDS:
            if mon.get_tool(tool) is None:
                mon.use_tool_id(tool, TOOL_NAME)
                self.tool = tool
                break
        else:
            held = {t: mon.get_tool(t) for t in _TOOL_IDS}
            self.reason = f"monitoring tool ids 4 and 3 are taken ({held})"
            return False
        ev = mon.events
        mon.register_callback(self.tool, ev.PY_START, self._on_start)
        mon.register_callback(self.tool, ev.PY_RESUME, self._on_resume)
        mon.set_events(self.tool, ev.PY_START | ev.PY_RESUME)
        return True

    def uninstall(self) -> None:
        mon = sys.monitoring
        if self.tool is None or mon.get_tool(self.tool) != TOOL_NAME:
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
        if self.tool is None or mon.get_tool(self.tool) != TOOL_NAME:
            return False
        return mon.get_events(self.tool) == mon.events.PY_START | mon.events.PY_RESUME

    # -- units ------------------------------------------------------------
    def begin(self, kind: str) -> Unit:
        unit = Unit(kind)
        if not self._intact():
            unit.markers.add("tool_conflict")
            mon = sys.monitoring
            if self.tool is not None and mon.get_tool(self.tool) == TOOL_NAME:
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
        """Does thread *t* run code of the user's: its target, or its own
        ``run`` method?"""
        from ...install_paths import is_user_code_module

        fn = getattr(t, "_target", None)
        if fn is None and type(t).run is not threading.Thread.run:
            fn = type(t).run
        if fn is None:
            return False
        g = getattr(inspect_unwrap(fn), "__globals__", None)
        if g is self.notebook_ns:
            return True
        name = getattr(fn, "__module__", None) or (g or {}).get("__name__")
        if not isinstance(name, str) or name == "cash" or name.startswith("cash."):
            return False
        mod = sys.modules.get(name)
        return mod is not None and is_user_code_module(mod)

    # -- callbacks --------------------------------------------------------
    def _classify(self, g: dict) -> tuple[dict, str | None, str]:
        from ...install_paths import is_user_code_module

        name = g.get("__name__")
        name = name if isinstance(name, str) else "?"
        user: str | None = None
        if g is self.notebook_ns:
            user = self.notebook_name
        elif name != "cash" and not name.startswith("cash."):
            # By the module the globals claim: ``exec``'d code (a dataclass's
            # generated ``__init__``) runs on its module's own dict.
            mod = sys.modules.get(name)
            if mod is not None and is_user_code_module(mod):
                user = name
        if len(self._kinds) > 4096:
            self._kinds.clear()
        entry = (g, user, name)
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
            if (kind[2], code.co_qualname) in _POOL_STARTS:
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
        if not stack or not code.co_flags & _RESUMABLE:
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
