"""A call that picks code by a run-time name, cached on a trace of what it ran.

``getattr(module, name)(x)``, ``importlib.import_module(name)`` and
``__import__(name)`` choose the code a call runs from a value, so a key built
from the code the function names cannot see an edit to it
(KEY-UNTRACKABLE-DEP). On Python 3.12 and newer such a call is keyed as any
other, and on a miss its body runs under ``sys.monitoring``: the user code
that ran, and each lookup by name, are recorded. The entry keeps that trace
as a list of *items*, each naming something by module and qualified name with
the digest it had, and a lookup serves the entry only when every item still
holds that digest (`verify`). The key is never replaced: a trace only adds a
check, since a trace cannot see a seed or a value a function holds.

``("code", module, qualname, digest)``
    a function of the user's that ran: its code and defaults.
``("held", module, name, qualname, digest)``
    code no qualified name finds (a lambda, a closure made outside the call),
    found through the global *name* holding it: its code and closure.
``("class", module, qualname, digest)``
    a class of the user's a method that ran was called on.
``("global", module, name, digest)``
    a global such a function reads by name, ``"absent"`` when unbound.
``("attr", module, "mod.a.b", digest)``
    an attribute chain it reads through a module of the user's.
``("import", name, digest)``
    a module imported by a run-time name: its file.
``("lookup", module, qualname, name, digest)``
    what ``getattr`` looked *name* up on: a module (*qualname* ``""``) or a
    class, by the digest of what it finds under *name*; ``"*"`` for a name
    the recorder could not read, by the digest of the whole namespace.
``("cached", func_name, state)``
    a cached function served inside the call: its state segment
    (`KeyBuilder.callee_state`).

The code the cached function names itself, and its helpers, are keyed and are
not in the trace. A call whose trace could not be checked again is not
stored, and runs again next time as before (`build` says why): a function no
name finds, a value too big to digest, a lookup on a library object by a name
the recorder could not read, a thread, worker process or generator the call
left behind, or a function it ran that has side effects, reads the clock,
draws from a generator of its own or runs ``eval``.

Threads: user code that starts while any traced call runs, on any thread, is
added to every traced call in progress, so the work of a pool a call waits on
is in its trace (and an unrelated thread's work only adds checks).
"""

from __future__ import annotations

import atexit
import builtins
import collections
import dis
import functools
import hashlib
import importlib
import itertools
import operator
import os
import sys
import threading
import types
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .._paths import is_pseudo_filename
from .._user_attrs import attribute_or, instance_of
from ..analysis.purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_DISCARDED_CALL,
    ISSUE_IMPURE_CALL,
    ISSUE_NETWORK_READ,
    ISSUE_SCOPE_MUTATION,
    ISSUE_UNTRACKABLE_DEP,
    PurityIssue,
    is_name_lookup,
)
from ..tracking.tracker_context import untracked

__all__ = [
    "TRACEABLE",
    "TraceUnit",
    "active",
    "begin",
    "build",
    "covers",
    "end",
    "note_cached_hit",
    "static_codes",
    "verify",
]

#: ``sys.monitoring`` is there: Python 3.12 or newer.
TRACEABLE = sys.version_info >= (3, 12)
_OFF = ("0", "off", "false", "no")


def on() -> bool:
    """Is every cached call recorded and served on its trace? On 3.12+
    unless ``CASH_TRACE=0``; a call that picks code by name is recorded
    either way."""
    return TRACEABLE and os.environ.get("CASH_TRACE", "").strip().lower() not in _OFF


#: Names a code object holds when it may look code up by name: CALL events
#: are armed on it (`_Recorder._on_start`).
_LOOKUP_NAMES = frozenset({"getattr", "__import__", "import_module"})
#: Names in a code object that may look a name up by a string
#: (`string_lookups`): ``globals()``, ``vars()``, ``sys.modules``, ...
_BY_STRING_NAMES = frozenset(
    {
        "globals",
        "eval",
        "exec",
        "__import__",
        "locals",
        "vars",
        "import_module",
        "f_globals",
        "f_locals",
        "user_ns",
        "user_global_ns",
        "modules",
    }
)


def _namespace_like(value: Any, g: dict) -> bool:
    """Is *value* a namespace a string is looked up in: *g* itself, the
    module table, or the module *g* is?"""
    return value is g or value is sys.modules or (instance_of(value, types.ModuleType) and value.__dict__ is g)


#: Markers that leave a trace nothing can check (`recorder` module).
_UNUSABLE = {
    "thread_alive": "a thread it started still runs",
    "process_pool": "it ran work in other processes",
    "generator_resumed": "it resumed a generator started before it",
    "tool_conflict": "another tool took the monitoring tool id",
    "trace_failed": "cash could not read code it saw run",
}
#: Findings in a function the call ran that a hit would skip or miss.
_REFUSED_KINDS = frozenset(
    {ISSUE_IMPURE_CALL, ISSUE_SCOPE_MUTATION, ISSUE_AMBIENT_READ, ISSUE_NETWORK_READ, ISSUE_DISCARDED_CALL}
)
#: Global names a function reads that are where cash put itself.
_SKIP_GLOBALS = frozenset({"__builtins__", "__name__", "__file__", "__spec__", "__loader__", "__package__"})
#: Namespaces bigger than this are not digested whole (an unread name on one).
_NAMESPACE_CAP = 2000
_ABSENT = "absent"


def covers(issues: Iterable[PurityIssue]) -> bool:
    """Can a trace stand in for *issues*, the run-time lookups of a call's
    tree: all of them lookups by name (`is_name_lookup`), on 3.12+?"""
    issues = list(issues)
    return TRACEABLE and bool(issues) and all(is_name_lookup(i) for i in issues)


# -- recording -------------------------------------------------------------
def _unit_class() -> type:
    from ..tracking.trace_recorder import Unit

    class TraceUnit(Unit):
        """What one traced call saw: `Unit`'s fields, its lookups by name
        and the items of what was served inside it."""

        __slots__ = ("cached_codes", "callees", "counters_watched", "lookups", "lookups_followed")

        def __init__(self, kind: str) -> None:
            super().__init__(kind)
            #: name -> code of the cached functions served inside (`note_cached_hit`).
            self.cached_codes: dict[str, types.CodeType] = {}
            #: ``(kind, id(target), name)`` -> ``(kind, target, name)``.
            self.lookups: dict[tuple, tuple] = {}
            #: A notebook unit's: ``id(callable)`` -> ``(callable, globals of
            #: the caller)``, for what code that looks names up by a string
            #: called (`_Recorder.by_string`).
            self.callees: dict[int, tuple[Any, dict]] = {}
            #: A notebook statement's: its own lookups by a string are ones
            #: its trace follows (``cache_key.lookups_followed``).
            self.lookups_followed = False
            #: A notebook unit with a memo of the user's about: its calls of
            #: one's counters are looked for (`_Recorder._on_call`).
            self.counters_watched = False

        def absorb(self, inner: Any) -> None:
            super().absorb(inner)
            for key, callee in getattr(inner, "callees", {}).items():
                self.callees.setdefault(key, callee)

    return TraceUnit


TraceUnit: Any = None


def _recorder_class() -> type:
    from ..install_paths import is_user_path, known_user_path
    from ..tracking import trace_recorder as rec

    DISABLE = sys.monitoring.DISABLE
    FUNCTION, LRU, STR, BUILTIN, PARTIAL = types.FunctionType, _LRU, str, _BUILTIN, functools.partial
    DESCRIPTORS = frozenset(_DESCRIPTORS)
    SITES_get = _SITES.get
    USER_FN_get = _USER_FN.get
    TAKERS = _TAKERS
    LOOKUPS = _LOOKUPS
    _POOL_QUALNAMES = frozenset(qualname for _, qualname in rec.POOL_STARTS)
    _PYPLOT_FILES = ("matplotlib/pyplot.py", "matplotlib\\pyplot.py")

    def _varies(code: types.CodeType, offset: int, fn: Any, arg0: Any) -> bool:
        """`callee_varies`, the code's globals read only the first time; a
        method the receiver's own class holds (``rows.append(x)``,
        ``model.fit(x)``) does not vary as the attribute it is called by
        would (`from_class`)."""
        sites = SITES_get(id(code))
        if sites is None:
            varies = callee_varies(code, offset, sys._getframe(rec.WATCHED_FRAME + 1).f_globals)
        else:
            varies = sites[1].get(offset, True)
        if varies is True or not varies:
            return bool(varies)
        return not from_class(fn, arg0, varies)

    class _Recorder(rec.Recorder):
        """The process's one recorder, for traced calls and for the
        notebook's units (`cash.notebook.shadow`): CALL events on code that
        may look a name up, every traced call in progress (on any thread)
        sees what runs, and the innermost notebook unit what its thread
        runs."""

        tool_name = "cash-trace"
        tool_ids = (3, 4)

        def __init__(self) -> None:
            super().__init__()
            self.active: list[Any] = []
            self._in_callback = False
            self._lock = threading.Lock()
            self._import_module = importlib.import_module

        def install(self) -> bool:
            # Two ids: one sees each code once (global events, disabled at
            # once), the other has local events on the user's code alone, so
            # arming them again before a call touches only that code, not
            # every function the process ran (`restart_events`).
            #
            # Everything a callback reads is set up before an event is on:
            # from then on every thread's code calls them. The base class's
            # install turned the events on with `_on_start` as the callback
            # before this one had made `rearm`, so a pool's other workers ran
            # it and raised AttributeError into whatever code they were
            # starting; a worker that died so left its task's future unset,
            # and the caller waited on it forever.
            mon = getattr(sys, "monitoring", None)
            if mon is None:
                self.reason = "sys.monitoring needs Python 3.12 or newer"
                return False
            # Held weakly: a notebook compiles each statement it runs into
            # code of its own, and a long session kept every one of them.
            # Code that is gone cannot run again, so nothing it would have
            # recorded is lost; code still alive stays.
            self.user_codes: weakref.WeakKeyDictionary[types.CodeType, int] = weakref.WeakKeyDictionary()
            self.rearm: set[types.CodeType] = set()
            #: The user's code that looks names up by a string in its own
            #: globals (`string_lookups`): what it calls is recorded for a
            #: notebook unit, the callable a string found among it.
            self.by_string: weakref.WeakSet[types.CodeType] = weakref.WeakSet()
            #: The user's code that names getattr or an import function.
            self.lookup_codes: weakref.WeakSet[types.CodeType] = weakref.WeakSet()
            free = [tool for tool in self.tool_ids if mon.get_tool(tool) is None]
            if len(free) < 2:
                held = {t: mon.get_tool(t) for t in self.tool_ids}
                self.reason = f"monitoring tool ids {' and '.join(map(str, self.tool_ids))} are taken ({held})"
                return False
            self.disc, self.tool = free[:2]
            mon.use_tool_id(self.disc, self.tool_name)
            mon.use_tool_id(self.tool, self.tool_name)
            ev = mon.events
            self.rearm_works = self._rearm_works()
            mon.register_callback(self.tool, ev.PY_START, self._guarded(self._on_start))
            mon.register_callback(self.tool, ev.PY_RESUME, self._guarded(self._on_resume))
            mon.register_callback(self.tool, ev.CALL, self._guarded(self._on_call))
            mon.register_callback(self.disc, ev.PY_START, self._guarded(self._discover))
            mon.register_callback(self.disc, ev.PY_RESUME, self._guarded(self._discover))
            mon.set_events(self.disc, ev.PY_START | ev.PY_RESUME)
            return True

        def _rearm_works(self) -> bool:
            """Does setting a code's local events again arm the ones its
            callback disabled? CPython does from 3.12 on; checked, not
            assumed, and `restart_events` does it where it does not."""
            mon = sys.monitoring
            seen: list[int] = []

            def probe() -> None:
                pass

            def on_start(code: types.CodeType, offset: int) -> Any:
                seen.append(1)
                return DISABLE

            code = probe.__code__
            mon.register_callback(self.tool, mon.events.PY_START, on_start)
            try:
                mon.set_local_events(self.tool, code, mon.events.PY_START)
                probe()
                probe()
                mon.set_local_events(self.tool, code, 0)
                mon.set_local_events(self.tool, code, mon.events.PY_START)
                probe()
            finally:
                mon.set_local_events(self.tool, code, 0)
                mon.register_callback(self.tool, mon.events.PY_START, None)
            return len(seen) == 2

        def uninstall(self) -> None:
            mon = sys.monitoring
            for tool in {getattr(self, "disc", None), self.tool} - {None}:
                if mon.get_tool(tool) == self.tool_name:
                    mon.set_events(tool, 0)
                    for ev in (mon.events.PY_START, mon.events.PY_RESUME, mon.events.CALL):
                        mon.register_callback(tool, ev, None)
                    mon.free_tool_id(tool)
            self.tool = None

        def _intact(self) -> bool:
            mon = sys.monitoring
            return (
                self.tool is not None
                and mon.get_tool(self.tool) == self.tool_name
                and mon.get_tool(self.disc) == self.tool_name
                and mon.get_events(self.disc) == mon.events.PY_START | mon.events.PY_RESUME
            )

        def _discover(self, code: types.CodeType, offset: int) -> Any:
            """First sight of *code*: arm local events on it when it is the
            user's (or starts a process pool), and record this run of it."""
            filename = code.co_filename
            if filename.endswith(_PYPLOT_FILES):
                # pyplot's own code: seen as it runs, once per unit, to tell
                # whether a unit read or changed pyplot's current figure
                # (`Unit.pyplot_ran`).
                return self._arm_pyplot(code)
            if filename and not is_pseudo_filename(filename) and code.co_qualname not in _POOL_QUALNAMES:
                # A library's file or cash's own, judged by its file as
                # `is_user_code_file` does, before reading the frame: a fresh
                # process sees thousands of them once each (an import of
                # pandas), and reading the frame is most of what each costs.
                # A first verdict on a file may list its import folder, which
                # is cash's own read, not the user's.
                user = known_user_path(filename)
                if user is None:
                    with untracked():
                        user = is_user_path(filename)
                if not user:
                    return DISABLE
            frame = sys._getframe(rec.WATCHED_FRAME)
            g = frame.f_globals
            kind = self._kinds.get(id(g))
            if kind is None or kind[0] is not g:
                kind = self._classify(g)
            pool = (kind[2], code.co_qualname) in rec.POOL_STARTS
            if kind[1] is None and not pool and not self._unregistered_user_code(kind[2], code):
                return DISABLE
            mon = sys.monitoring
            events = mon.events.PY_START | mon.events.PY_RESUME
            if not pool:
                # Every call the user's code makes is seen once a unit, and
                # again at each site that may read a memo's counters
                # (`_on_call`).
                events |= mon.events.CALL
                if _LOOKUP_NAMES.intersection(code.co_names):
                    self.lookup_codes.add(code)
            if not pool and self._looks_up_by_string(code, g):
                self.by_string.add(code)
            self.user_codes[code] = events
            mon.set_local_events(self.tool, code, events)
            if self.active or self.stack:
                if code.co_flags & rec.RESUMABLE and offset:
                    self._seen_resume(code, g, kind)
                else:
                    self._seen_start(code, frame, g, kind)
            return DISABLE

        def _arm_pyplot(self, code: types.CodeType) -> Any:
            mon = sys.monitoring
            self.user_codes[code] = mon.events.PY_START
            mon.set_local_events(self.tool, code, mon.events.PY_START)
            self._seen_pyplot()
            return DISABLE

        def _seen_pyplot(self) -> None:
            for unit in self.active:
                unit.pyplot_ran = True
            if self.stack:
                self.stack[-1].pyplot_ran = True

        def reclassify(self) -> None:
            """The notebook's namespace changed: what was not the user's
            code may be now, so every code is seen once more."""
            self._kinds.clear()
            sys.monitoring.restart_events()

        def begin(self, kind: str) -> Any:
            unit = TraceUnit(kind)
            if not self._intact():
                unit.markers.add("tool_conflict")
                mon = sys.monitoring
                if mon.get_tool(self.disc) == self.tool_name:
                    # Still ours, its events turned off: arm them again.
                    mon.set_events(self.disc, mon.events.PY_START | mon.events.PY_RESUME)
            unit.threads_before = frozenset(t.ident for t in threading.enumerate() if t.ident is not None)
            self._rearm()
            self.restarts += 1
            if kind != "decorator":
                # A notebook unit: the innermost one of its thread sees what
                # runs, and hands it to the one around it when it ends.
                if self.stack:
                    unit.counters_watched = self.stack[-1].counters_watched
                self.stack.append(unit)
                return unit
            with self._lock:
                self.active = [*self.active, unit]
            return unit

        def _rearm(self) -> None:
            rearm, self.rearm = self.rearm, set()
            mon = sys.monitoring
            if not self.rearm_works:
                if rearm:
                    mon.restart_events()
            else:
                for code in rearm:
                    events = self.user_codes.get(code)
                    if events is not None:
                        mon.set_local_events(self.tool, code, 0)
                        mon.set_local_events(self.tool, code, events)

        def end(self, unit: Any) -> Any:
            if unit.kind != "decorator":
                return super().end(unit)
            with self._lock:
                self.active = [u for u in self.active if u is not unit]
            if not self._intact():
                unit.markers.add("tool_conflict")
            for t in threading.enumerate():
                if t.ident not in unit.threads_before and t.is_alive() and self._runs_user_code(t):
                    unit.markers.add("thread_alive")
            return unit

        def _on_start(self, code: types.CodeType, offset: int) -> Any:
            self.rearm.add(code)
            if self.active or self.stack:
                frame = sys._getframe(rec.WATCHED_FRAME)
                g = frame.f_globals
                kind = self._kinds.get(id(g))
                if kind is None or kind[0] is not g:
                    kind = self._classify(g)
                self._seen_start(code, frame, g, kind)
            return DISABLE

        def _seen_start(self, code: types.CodeType, frame: Any, g: dict, kind: tuple) -> None:
            units = self.active
            stack = self.stack
            if stack:
                units = [*units, stack[-1]]
                if threading.get_ident() != stack[-1].thread:
                    stack[-1].notes.add("other_thread")
            module = kind[1]
            if module is None:
                if kind[2] == "matplotlib.pyplot":
                    self._seen_pyplot()
                    return
                if (kind[2], code.co_qualname) in rec.POOL_STARTS:
                    for unit in units:
                        unit.markers.add("process_pool")
                if stack and not self.active:
                    return
                if not self._unregistered_user_code(kind[2], code):
                    return
                # A file of the user's run as a module no name in sys.modules
                # finds (a plugin loaded from its path): `build` refuses it.
                module = kind[2]
            recv = None
            if code.co_argcount and code.co_varnames[0] in ("self", "cls"):
                try:
                    recv = frame.f_locals.get(code.co_varnames[0])
                except Exception:  # noqa: BLE001 - a frame whose locals cannot be read has no receiver
                    recv = None
            cells = None
            if code.co_freevars and not self._in_callback:
                self._in_callback = True
                try:
                    from ..tracking.trace_items import start_cells

                    with untracked():
                        cells = start_cells(frame, code)
                finally:
                    self._in_callback = False
            for unit in units:
                unit.codes.setdefault(code, (g, module))
                if cells is not None:
                    unit.cells.setdefault(code, cells)
                if recv is not None:
                    unit.receivers.setdefault(code, recv if issubclass(type(recv), type) else type(recv))

        @staticmethod
        def _unregistered_user_code(name: str, code: types.CodeType) -> bool:
            if name == "cash" or name.startswith("cash.") or sys.modules.get(name) is not None:
                return False
            filename = code.co_filename
            return bool(filename) and not is_pseudo_filename(filename) and is_user_path(filename)

        def _on_resume(self, code: types.CodeType, offset: int) -> Any:
            self.rearm.add(code)
            if (self.active or self.stack) and code.co_flags & rec.RESUMABLE:
                g = sys._getframe(rec.WATCHED_FRAME).f_globals
                kind = self._kinds.get(id(g))
                if kind is None or kind[0] is not g:
                    kind = self._classify(g)
                self._seen_resume(code, g, kind)
            return DISABLE

        def _seen_resume(self, code: types.CodeType, g: dict, kind: tuple) -> None:
            if kind[1] is not None:
                units = [*self.active, self.stack[-1]] if self.stack else self.active
                for unit in units:
                    if code not in unit.codes:
                        unit.codes[code] = (g, kind[1])
                        unit.markers.add("generator_resumed")

        @staticmethod
        def _looks_up_by_string(code: types.CodeType, g: dict) -> bool:
            """Might *code* look a name up by a string in *g*? Its names
            first, so most code costs a set intersection."""
            names = code.co_names
            if not _BY_STRING_NAMES.intersection(names):
                get = g.get
                if not any(_namespace_like(get(n), g) for n in names if n in g):
                    return False
            from ..notebook.string_lookups import code_looks_up_by_string

            try:
                return code_looks_up_by_string(code, g)
            except Exception:  # noqa: BLE001 - a check that fails arms the events, which cost only time
                return True

        def _on_call(self, code: types.CodeType, offset: int, fn: Any, arg0: Any) -> Any:
            units = self.active
            stack = self.stack
            if not units and stack and stack[-1].counters_watched:
                # The common call at a site kept seen (a loop calling a
                # function a variable holds): a Python function or a memo,
                # handed no counter nor a counter's name, at a site kept as
                # it was. `_plain_call` written out: it runs on every call.
                t = type(fn)
                if (
                    (t is FUNCTION and USER_FN_get(id(fn)) is True)
                    or t is LRU
                    or (t is BUILTIN and type(fn.__self__) is not LRU and id(fn) not in TAKERS)
                    or (t in DESCRIPTORS and fn.__objclass__ is not LRU and id(fn) not in TAKERS)
                ):
                    t = type(arg0)
                    if t is not STR and t is not BUILTIN and t is not PARTIAL and t not in DESCRIPTORS:
                        sites = SITES_get(id(code))
                        if sites is not None:
                            varies = sites[1].get(offset, True)
                            # An attribute site (``NS.fn(i)``) is kept as
                            # well, unless it calls the receiver's own
                            # method, which the slow path lets go.
                            if varies is True or (varies and not from_class(fn, arg0, varies)):
                                return None
            if not units and not stack:
                self.rearm.add(code)
                return DISABLE
            # Code that may look a name up keeps every call seen: one that
            # called something else first may call getattr next.
            keep = bool(units) and code in self.lookup_codes
            if stack and code in self.by_string:
                keep = True
                callees = stack[-1].callees
                if id(fn) not in callees:
                    callees[id(fn)] = (fn, sys._getframe(rec.WATCHED_FRAME).f_globals)
            if stack and stack[-1].counters_watched:
                try:
                    if reads_counters(fn, arg0):
                        stack[-1].counters_read = True
                        keep = True
                    elif id(fn) in LOOKUPS:
                        # ``getattr(f, name)``: a counter got by a name no
                        # text holds, handed on later where no trace runs.
                        keep = True
                        if (type(arg0) is LRU or arg0 is LRU) and names_a_counter(
                            code, offset, sys._getframe(rec.WATCHED_FRAME), arg0
                        ):
                            stack[-1].counters_read = True
                    elif not harmless(fn):
                        # Code the trace does not see (a builtin that calls
                        # what it is handed, a library's function or class):
                        # what it is handed is read at each call, every
                        # argument and keyword the frame holds, not only the
                        # first. A site handing on nothing callable
                        # (``np.mean(a)``, ``os.path.basename(s)``) is let go
                        # as a harmless one is, unless it calls what varies.
                        handed = what_is_handed(code, offset, sys._getframe(rec.WATCHED_FRAME))
                        if handed is HANDS_A_COUNTER:
                            stack[-1].counters_read = True
                        keep = id(fn) in TAKERS or handed is not None or _varies(code, offset, fn, arg0)
                    elif _varies(code, offset, fn, arg0):
                        # What it calls next may be a counter: seen every time.
                        keep = True
                except Exception:  # noqa: BLE001 - a call that cannot be judged stays seen
                    keep = True
            if units:
                try:
                    if fn is getattr:
                        if arg0 is not sys.monitoring.MISSING:
                            name = _last_argument(code, offset, sys._getframe(rec.WATCHED_FRAME))
                            for unit in units:
                                unit.lookups.setdefault(("getattr", id(arg0), name), ("getattr", arg0, name))
                        keep = True
                    elif fn is self._import_module or fn is builtins.__import__:
                        name = arg0 if isinstance(arg0, str) else None
                        for unit in units:
                            unit.lookups.setdefault(("import", 0, name), ("import", None, name))
                        keep = True
                except Exception:  # noqa: BLE001 - a lookup that cannot be read leaves the trace unusable
                    for unit in units:
                        unit.markers.add("lookup_unread")
                    keep = True
            if keep:
                return None
            self.rearm.add(code)
            return DISABLE

    return _Recorder


# -- a memo's counters ------------------------------------------------------
#: What reads or empties a memo's counters.
_COUNTERS = frozenset({"cache_info", "cache_clear"})
_LRU: Any = type(functools.lru_cache(None)(lambda: None))
_BUILTIN = types.BuiltinFunctionType
_DESCRIPTORS = (
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.ClassMethodDescriptorType,
)
#: Builtins that call or look up what they are handed: a site calling one
#: is seen every time, whatever it was handed first.
_TAKERS: frozenset[int] = frozenset(
    id(fn)
    for fn in (
        getattr,
        map,
        filter,
        sorted,
        min,
        max,
        iter,
        functools.reduce,
        functools.partial,
        operator.methodcaller,
        operator.attrgetter,
        getattr(operator, "call", None),
        object.__getattribute__,
        type.__getattribute__,
        list.sort,
        itertools.groupby,
        itertools.starmap,
        itertools.accumulate,
        itertools.filterfalse,
        itertools.takewhile,
        itertools.dropwhile,
        collections.defaultdict,
        functools.cmp_to_key,
    )
    if fn is not None
)
#: Builtins that look an attribute up by a name: one asked of a memo, by a
#: name that is a counter's or that cannot be read, hands one back.
_LOOKUPS: frozenset[int] = frozenset(id(fn) for fn in (getattr, object.__getattribute__, type.__getattribute__))
#: co_filename -> is it the user's code (`harmless`).
_USER_FILE: dict[str, bool] = {}
#: id(function) -> is it the user's code (`_user_function`); the weak
#: reference in `_USER_FN_REFS` takes it out as the function goes.
_USER_FN: dict[int, bool] = {}
_USER_FN_REFS: dict[int, Any] = {}


def _counter_method(obj: Any) -> bool:
    """Is *obj* a memo's ``cache_info`` or ``cache_clear``, bound or not?"""
    t = type(obj)
    if t is _BUILTIN:
        return type(obj.__self__) is _LRU and obj.__name__ in _COUNTERS
    if t in _DESCRIPTORS:
        return obj.__objclass__ is _LRU and obj.__name__ in _COUNTERS
    return False


def reads_counters(fn: Any, arg0: Any) -> bool:
    """Does the call of *fn* with first argument *arg0* read or empty a
    memo's counters: *fn* is one, or *arg0* is one handed on (``map``,
    ``operator.call``), or a counter's name (``methodcaller``), or *fn* is a
    ``methodcaller``, ``attrgetter`` or ``partial`` made for one."""
    if _counter_method(fn) or _counter_method(arg0):
        return True
    if type(arg0) is str and arg0 in _COUNTERS:
        return True
    t = type(fn)
    if t is functools.partial:
        return _counter_method(fn.func) or any(type(a) is str and a in _COUNTERS for a in fn.args)
    if t is operator.methodcaller or t is operator.attrgetter:
        made = fn.__reduce__()
        args = made[1] if len(made) > 1 and type(made[1]) is tuple else ()
        if made[0] is functools.partial and args:
            # A methodcaller with keywords: ``partial(methodcaller, name, ...)``.
            args = args[1:]
        return any(type(a) is str and (a in _COUNTERS or a.rsplit(".", 1)[-1] in _COUNTERS) for a in args)
    return False


def harmless(fn: Any) -> bool:
    """Is *fn* a callable that cannot reach a memo's counters other than by
    code of its own the trace sees: a function, method or class of the
    user's, a memo, a builtin or a builtin type that does not call or look
    up what it is handed, a method of a builtin type other than a memo's, a
    numpy ufunc. A library's function or class is not: the trace does not
    see its code, which may call what it is handed (a thread pool's
    ``submit``) in a thread of its own."""
    if id(fn) in _TAKERS:
        return False
    t = type(fn)
    if t is _LRU:
        # A memo's own call runs the user's code or none.
        return True
    if t is types.FunctionType:
        return _user_function(fn)
    if isinstance(fn, type):
        if not fn.__flags__ & _HEAP_TYPE:
            return True  # a builtin type: dict, list, a C library's
        # A class statement's: what its call runs is the ``__new__`` and
        # ``__init__`` it finds first, the user's or a library's.
        return all(_user_init(fn, name) for name in ("__new__", "__init__"))
    if t is types.MethodType:
        func = fn.__func__
        return type(func) is types.FunctionType and _user_function(func)
    if t is _BUILTIN:
        return type(fn.__self__) is not _LRU
    if t in _DESCRIPTORS:
        return fn.__objclass__ is not _LRU
    return t.__name__ == "ufunc" and t.__module__ == "numpy"


_HEAP_TYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class made by a ``class`` statement


def _user_init(cls: type, name: str) -> bool:
    """Is the *name* (``__new__`` or ``__init__``) *cls* runs the user's
    code or a builtin's (``object.__init__``), not a library's?"""
    for klass in cls.__mro__:
        found = klass.__dict__.get(name)
        if found is None:
            continue
        if type(found) is staticmethod:
            found = found.__func__
        code = attribute_or(found, "__code__", None)
        return code is None or _user_file(code.co_filename)
    return True


def _user_function(fn: types.FunctionType) -> bool:
    """Is *fn* the user's code? Kept by the function, for `_on_call`'s quick
    path: reading ``__code__`` there raises an audit event per call."""
    key = id(fn)
    known = _USER_FN.get(key)
    if known is None:
        known = _user_file(fn.__code__.co_filename)
        try:
            _USER_FN_REFS[key] = weakref.ref(
                fn, lambda _, key=key: (_USER_FN.pop(key, None), _USER_FN_REFS.pop(key, None))
            )
        except TypeError:
            return known
        _USER_FN[key] = known
    return known


def _user_file(filename: str) -> bool:
    """`is_user_code_file`, kept per name."""
    known = _USER_FILE.get(filename)
    if known is None:
        from ..install_paths import is_user_code_file

        if len(_USER_FILE) > 4096:
            _USER_FILE.clear()
        known = _USER_FILE[filename] = is_user_code_file(filename)
    return known


#: id(code) -> (a weak reference to it, {offset of a call: does the
#: callable it calls vary run to run}); by id, as `_on_call` reads it on
#: every call at a site it keeps seen. The reference takes the entry out as
#: the code goes, before its id can be another's: a notebook compiles each
#: statement into code of its own, which nothing here keeps alive.
_SITES: dict[int, tuple[Any, dict[int, bool | str]]] = {}
#: file name -> (its source, {end of each call in it: its node}).
_CALL_ENDS: dict[str, tuple[str, dict[tuple[int, int], Any]]] = {}


def callee_varies(code: types.CodeType, offset: int, g: Mapping[str, Any] | None = None) -> bool | str:
    """May the call at *offset* in *code* call something else next time?
    Its callee is a name *code* binds itself (a loop's variable: ``for g in
    [len, f.cache_info]: g()``), an argument or local, a closure's variable
    (a factory's closures share one code, each its own cell), an attribute
    (``self.g()``, ``o.stat()``: nothing ties one to one callable), an
    item, a call's result: anything but a global or builtin name, or an
    attribute of a module or class *g* (the code's globals) binds a global
    or builtin name to (``np.sqrt(x)``, ``Model.make()``, ``', '.join(x)``).
    An attribute's call that varies answers with the attribute's name, for
    `from_class`. Read from the source; a call it cannot place varies."""
    key = id(code)
    known = _SITES.get(key)
    if known is None:
        known = _SITES[key] = (
            weakref.ref(code, lambda _, key=key, sites=_SITES: sites.pop(key, None)),
            _varying_calls(code, g),
        )
    return known[1].get(offset, True)


def _fixed_base(node: Any, bound: set[str], g: Mapping[str, Any] | None) -> bool:
    """Is the attribute call's *node* (its ``.value``) a literal, or a module
    or class a global or builtin name binds (through modules: ``np.linalg``)?"""
    import ast

    if isinstance(node, ast.Constant):
        return True  # ``', '.join(parts)``
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id in bound or g is None:
        return False
    value = g.get(node.id, _MISSING)
    if value is _MISSING:
        value = builtins.__dict__.get(node.id, _MISSING)
    for attr in reversed(path):
        if type(value) is not types.ModuleType:
            return False
        value = value.__dict__.get(attr, _MISSING)
    return type(value) is types.ModuleType or isinstance(value, type)


def from_class(fn: Any, receiver: Any, name: str) -> bool:
    """Is the call of *fn* at an attribute site (``obj.name(...)``) a method
    the receiver's own class holds under *name*, the receiver handed as its
    first argument (``rows.append(x)``, ``model.fit(x)``)? What such a site
    calls next is the method of the next receiver's class: a counter only
    if a class holds one as a method, not an attribute an object was handed
    (``self.g()``, ``o.stat()``), which is not called this way."""
    if receiver is sys.monitoring.MISSING:
        return False
    for klass in type(receiver).__mro__:
        found = klass.__dict__.get(name)
        if found is not None:
            return found is fn
    return False


def _call_nodes(code: types.CodeType) -> dict[tuple[int, int], Any] | None:
    """The calls in *code*'s source, by where each ends; None when there is
    no source to read."""
    import ast
    import linecache

    filename = code.co_filename
    source = "".join(linecache.getlines(filename))
    if not source:
        return None
    known = _CALL_ENDS.get(filename)
    if known is None or known[0] != source:
        ends: dict[tuple[int, int], Any] = {}
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.end_lineno is not None:
                    ends[(node.end_lineno, node.end_col_offset or 0)] = node
        if len(_CALL_ENDS) > 256:
            _CALL_ENDS.clear()
        known = _CALL_ENDS[filename] = (source, ends)
    return known[1]


def _varying_calls(code: types.CodeType, g: Mapping[str, Any] | None = None) -> dict[int, bool | str]:
    import ast

    ends = _call_nodes(code)
    if ends is None:
        return {}
    bound = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_freevars)
    for ins in dis.get_instructions(code):
        if ins.opname.startswith(("STORE_", "DELETE_")):
            bound.update(ins.argval if isinstance(ins.argval, tuple) else (ins.argval,))
    sites: dict[int, bool | str] = {}
    for ins in dis.get_instructions(code):
        if ins.opname not in ("CALL", "CALL_KW", "CALL_FUNCTION_EX"):
            continue
        pos = ins.positions
        call = ends.get((pos.end_lineno, pos.end_col_offset)) if pos is not None else None
        func = call.func if call is not None else None
        if isinstance(func, ast.Name):
            sites[ins.offset] = func.id in bound
        elif isinstance(func, ast.Attribute):
            if func.attr in _COUNTERS:
                sites[ins.offset] = True
            else:
                sites[ins.offset] = not _fixed_base(func.value, bound, g) and func.attr
    return sites


#: id(code) -> (a weak reference to it, {offset of a call: what of its
#: arguments can be read from the frame}), as `_SITES`.
_HANDED: dict[int, tuple[Any, dict[int, tuple[str, ...]]]] = {}
#: How many items of a list, tuple or set handed on are looked at.
_HANDED_ITEMS = 64


def _handed_names(code: types.CodeType) -> dict[int, tuple[str, ...]]:
    """Per call in *code*, the names its arguments and keywords read whose
    values the frame holds: a name, a starred name, the names of a list,
    tuple or set display (``groupby([f], key=DESC)``). An argument the call
    computes (``submit(pick(f))``) is not among them: its counter was got
    by a lookup the trace sees (`names_a_counter`)."""
    import ast

    ends = _call_nodes(code) or {}
    out: dict[int, tuple[str, ...]] = {}
    for ins in dis.get_instructions(code):
        if ins.opname not in ("CALL", "CALL_KW", "CALL_FUNCTION_EX"):
            continue
        pos = ins.positions
        call = ends.get((pos.end_lineno, pos.end_col_offset)) if pos is not None else None
        if call is None:
            continue
        names: list[str] = []
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            if isinstance(arg, ast.Starred):
                arg = arg.value
            if isinstance(arg, ast.Name):
                names.append(arg.id)
            elif isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
                names.extend(item.id for item in arg.elts if isinstance(item, ast.Name))
        out[ins.offset] = tuple(names)
    return out


def handed_names(code: types.CodeType) -> dict[int, tuple[str, ...]]:
    """`_handed_names`, kept per code as `_SITES` is."""
    key = id(code)
    known = _HANDED.get(key)
    if known is None:
        known = _HANDED[key] = (
            weakref.ref(code, lambda _, key=key, handed=_HANDED: handed.pop(key, None)),
            _handed_names(code),
        )
    return known[1]


#: `what_is_handed`'s answers.
HANDS_A_COUNTER = "counter"
HANDS_A_CALLABLE = "callable"


def _handed(value: Any) -> str | None:
    """Is *value* a counter, a counter's name, a ``partial``,
    ``methodcaller`` or ``attrgetter`` made for one, or a list, tuple or set
    holding one (`HANDS_A_COUNTER`); else is it, or does such a container
    hold, something the call may call (`HANDS_A_CALLABLE`, `_may_call`)?"""
    if reads_counters(value, value):
        return HANDS_A_COUNTER
    if type(value) in (list, tuple, set, frozenset):
        found = None
        for item in itertools.islice(value, _HANDED_ITEMS):
            if reads_counters(item, item):
                return HANDS_A_COUNTER
            if found is None and _may_call(item):
                found = HANDS_A_CALLABLE
        return found
    return HANDS_A_CALLABLE if _may_call(value) else None


def _may_call(value: Any) -> bool:
    """Is *value* something the call it is handed to may call, and that
    could be a counter a later call swaps in: not a class (``dtype=float``),
    whose call runs code the trace sees or a builtin's."""
    return callable(value) and not isinstance(value, type)


def what_is_handed(code: types.CodeType, offset: int, frame: Any) -> str | None:
    """Does the call at *offset* in *code* hand a counter on, by any
    argument or keyword the frame holds (`HANDS_A_COUNTER`), or else
    something callable (`HANDS_A_CALLABLE`), which a later call at the site
    may swap for a counter? None when it hands on neither: data alone
    (``np.mean(a)``, ``os.path.basename(s)``). sys.monitoring tells the
    first argument only, which is ``self`` for a method
    (``pool.submit(...)``), and code the trace does not see may call any of
    them."""
    names = handed_names(code).get(offset)
    if not names:
        return None
    local = frame.f_locals
    g = frame.f_globals
    found = None
    for name in names:
        value = local.get(name, _MISSING)
        if value is _MISSING:
            value = g.get(name, _MISSING)
            if value is _MISSING:
                continue
        handed = _handed(value)
        if handed is HANDS_A_COUNTER:
            return handed
        found = found or handed
    return found


def names_a_counter(code: types.CodeType, offset: int, frame: Any, target: Any) -> bool:
    """Does the lookup (``getattr(target, name)``) at *offset* ask a memo,
    or the memos' class, for a counter: by the name of one, or by a name
    the frame does not hold (``'_'.join(PARTS)``)? What it hands back can
    then be called by code the trace does not see."""
    if type(target) is not _LRU and target is not _LRU:
        return False
    name = _last_argument(code, offset, frame)
    return name is None or name in _COUNTERS


_RECORDER: Any = None
_RECORDER_LOCK = threading.RLock()


def _recorder() -> Any:
    """The process's recorder, installed on first use; None when there is no
    tool id to be had."""
    global _RECORDER, TraceUnit
    if _RECORDER is not None:
        return _RECORDER if _RECORDER.tool is not None else None
    with _RECORDER_LOCK:
        if _RECORDER is None:
            TraceUnit = _unit_class()
            r = _recorder_class()()
            if r.install():
                # Before the interpreter tears modules down under the callbacks.
                atexit.register(r.uninstall)
            _RECORDER = r
    return _RECORDER if _RECORDER.tool is not None else None


def recorder_or_reason() -> Any:
    """The process's recorder, installed or not: its ``reason`` says why
    when its ``tool`` is None."""
    _recorder()
    return _RECORDER


def begin() -> Any:
    """Start recording a traced call's body; None when it cannot be."""
    rec = _recorder() if TRACEABLE else None
    return rec.begin("decorator") if rec is not None else None


def end(unit: Any) -> None:
    if unit is not None and _RECORDER is not None:
        _RECORDER.end(unit)


def active() -> bool:
    """Is a traced call recording now, on any thread?"""
    return _RECORDER is not None and bool(_RECORDER.active)


def note_cached_hit(func_name: str, code: types.CodeType | None, state: str | None, trace: list | None) -> None:
    """A cached function was served inside the traced calls in progress: its
    body did not run, so what it stands for goes in their traces. *code* is
    its code: a call whose key already covers it (`static_codes`) leaves the
    state out (`build`)."""
    rec = _RECORDER
    if rec is None:
        return
    for unit in list(rec.active):
        if state is None:
            unit.markers.add("cached_hit_unkeyed")
            continue
        unit.extra.append(("cached", func_name, state))
        if code is not None:
            unit.cached_codes[func_name] = code
        unit.extra.extend(tuple(item) for item in trace or ())


_INSTRUCTIONS: dict[types.CodeType, dict[int, tuple[str, Any] | None]] = {}
_FAST_LOADS = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_DEREF"})
_PAIRED_LOADS = frozenset({"LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"})
_SKIPPED = frozenset({"CACHE", "EXTENDED_ARG", "NOP"})


def _plan(code: types.CodeType, offset: int) -> tuple[str, Any] | None:
    """How to read the name a ``getattr(obj, name)`` or ``getattr(obj, name,
    default)`` at *offset* looks up: ``("const", value)``, ``("local",
    name)`` or ``("global", name)``, or None when an expression computed
    it."""
    plans = _INSTRUCTIONS.get(code)
    if plans is None:
        if len(_INSTRUCTIONS) > 4096:
            _INSTRUCTIONS.clear()
        plans = _INSTRUCTIONS[code] = {}
    if offset in plans:
        return plans[offset]
    plan = None
    previous = None
    for ins in dis.get_instructions(code):
        if ins.offset == offset:
            if ins.opname == "CALL" and ins.arg == 3:
                # A default follows the name, and may be an expression
                # (``repr(fn)``): the name is read from the source.
                plan = _plan_from_source(code, ins)
            elif ins.opname == "CALL" and ins.arg == 2 and previous is not None:
                op, val = previous.opname, previous.argval
                if op in ("LOAD_CONST", "LOAD_SMALL_INT"):
                    plan = ("const", val)
                elif op in _FAST_LOADS:
                    plan = ("local", val)
                elif op in _PAIRED_LOADS and isinstance(val, tuple) and len(val) == 2:
                    plan = ("local", val[1])
                elif op in ("LOAD_GLOBAL", "LOAD_NAME"):
                    plan = ("global", val)
            break
        if ins.opname not in _SKIPPED:
            previous = ins
    plans[offset] = plan
    return plan


def _plan_from_source(code: types.CodeType, ins: Any) -> tuple[str, Any] | None:
    """`_plan` for the call *ins*, its second argument read from the
    source."""
    import ast

    pos = ins.positions
    ends = _call_nodes(code) if pos is not None else None
    call = ends.get((pos.end_lineno, pos.end_col_offset)) if ends else None
    if call is None or len(call.args) < 2 or call.keywords:
        return None
    arg = call.args[1]
    if isinstance(arg, ast.Constant):
        return ("const", arg.value)
    if isinstance(arg, ast.Name):
        local = arg.id in code.co_varnames or arg.id in code.co_cellvars or arg.id in code.co_freevars
        return ("local", arg.id) if local else ("global", arg.id)
    return None


def _last_argument(code: types.CodeType, offset: int, frame: Any) -> str | None:
    """The name ``getattr(obj, name)`` at *offset* looks up, when a constant,
    a local or a global holds it; None otherwise."""
    plan = _plan(code, offset)
    if plan is None:
        return None
    how, what = plan
    if how == "const":
        value = what
    elif how == "local":
        value = frame.f_locals.get(what)
    else:
        value = frame.f_globals.get(what, builtins.__dict__.get(what))
    return value if isinstance(value, str) else None


# -- items -----------------------------------------------------------------
def _digest(value: Any) -> str | None:
    """*value*'s digest, the same in another process for an equal value; None
    when it is too big or cannot be read."""
    from ..install_paths import is_cash_path
    from ..tracking.trace_items import value_digest

    wrapped = getattr(value, "__wrapped__", None) if type(value) is types.FunctionType else None
    if wrapped is not None and is_cash_path(value.__code__.co_filename):
        # A ``@cash.cache`` wrapper: by the function it wraps (its state,
        # when it is served inside the call, is a "cached" item of its own).
        value = wrapped
    got = value_digest(value)
    return None if got.startswith("opaque:") else got


def _code_digest(code: types.CodeType, fn: Any) -> str | None:
    from ..tracking.trace_items import code_digest

    got = code_digest(code)
    if type(fn) is types.FunctionType and fn.__code__ is code and (fn.__defaults__ or fn.__kwdefaults__):
        defaults = _digest((fn.__defaults__, fn.__kwdefaults__))
        if defaults is None:
            return None
        got += "/" + defaults
    return got


def _namespace(module: str) -> Mapping[str, Any] | None:
    mod = sys.modules.get(module)
    ns = getattr(mod, "__dict__", None)
    return ns if isinstance(ns, dict) else None


def _find(module: str, qualname: str) -> Any:
    """The object *qualname* names in *module* now, or None."""
    from ..tracking.trace_items import inspect_getattr

    ns = _namespace(module)
    if ns is None or "<locals>" in qualname:
        return None
    parts = qualname.split(".")
    obj = ns.get(parts[0])
    for part in parts[1:]:
        try:
            obj = inspect_getattr(obj, part)
        except AttributeError:
            return None
    return obj


def _user_module(mod: Any) -> bool:
    from ..tracking.trace_recorder import is_user_code

    return issubclass(type(mod), types.ModuleType) and is_user_code(mod)


def _user_class(cls: Any) -> bool:
    return isinstance(cls, type) and _user_module(sys.modules.get(cls.__module__ or ""))


def _classes_digest(cls: type) -> str | None:
    """The user classes of *cls*'s MRO, each by what it holds."""
    from ..tracking.trace_items import class_digest

    parts = []
    for klass in cls.__mro__:
        if _user_class(klass):
            parts.append(class_digest(klass))
        else:
            parts.append(f"{klass.__module__}.{klass.__qualname__}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=12).hexdigest()


def _namespace_digest(ns: Mapping[str, Any]) -> str | None:
    """Every name a module binds, each by its digest."""
    try:
        pairs = sorted((k, v) for k, v in ns.items() if isinstance(k, str) and not k.startswith("__"))
    except TypeError:
        return None
    if len(pairs) > _NAMESPACE_CAP:
        return None
    h = hashlib.blake2b(digest_size=12)
    for k, v in pairs:
        got = _digest(v)
        if got is None:
            return None
        h.update(f"{k}={got};".encode())
    return h.hexdigest()


def _lookup_digest(module: str, qualname: str, name: str) -> str | None:
    """What a ``getattr(target, name)`` finds on the module or class the
    item names, now; None when that cannot be told."""
    from ..tracking.trace_items import inspect_getattr

    if qualname:
        target = _find(module, qualname)
        if not isinstance(target, type):
            return None
        if name == "*":
            return _classes_digest(target) if _user_class(target) else None
        try:
            found = inspect_getattr(target, name)
        except AttributeError:
            return _ABSENT
        return _digest(found)
    ns = _namespace(module)
    if ns is None:
        return None
    if name == "*":
        return _namespace_digest(ns) if _user_module(sys.modules.get(module)) else None
    return _digest(ns[name]) if name in ns else _ABSENT


_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    slot = (path, st.st_mtime_ns, st.st_size)
    got = _FILE_DIGESTS.get(slot)
    if got is None:
        try:
            with open(path, "rb") as fh:
                got = hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
        except OSError:
            return None
        if len(_FILE_DIGESTS) > 4096:
            _FILE_DIGESTS.clear()
        _FILE_DIGESTS[slot] = got
    return got


def _import_digest(name: str) -> str | None:
    """The file of module *name*: the loaded one's, else the one an import
    would load."""
    mod = sys.modules.get(name)
    if mod is not None:
        path = getattr(mod, "__file__", None)
        return _file_digest(path) if path else "builtin:" + name
    try:
        import importlib.util

        spec = importlib.util.find_spec(name)
    except Exception:  # noqa: BLE001 - a module that cannot be found is not the one recorded
        return None
    if spec is None:
        return _ABSENT
    if not spec.origin or spec.origin in ("built-in", "frozen"):
        return "builtin:" + name
    return _file_digest(spec.origin)


class _Build:
    """The items of one traced call (`build`)."""

    def __init__(self, unit: Any, static_codes: set[types.CodeType]) -> None:
        self.unit = unit
        self.static = static_codes
        self.items: dict[tuple, str] = {}
        self.functions: list[Any] = []

    def add(self, item: tuple, digest: str | None, why: str) -> None:
        if digest is None:
            raise _Untraceable(why)
        self.items.setdefault(item, digest)

    def run(self) -> list[tuple]:
        from ..tracking.trace_items import name_table

        unit = self.unit
        for marker in sorted(unit.markers):
            raise _Untraceable(_UNUSABLE.get(marker, marker.replace("_", " ")))
        ran = {c.co_qualname for c in unit.codes}
        # Where a function no qualified name finds may be held: its own
        # module, and every module the call looked a name up on.
        self.holders = [
            target.__dict__
            for kind, target, _name in unit.lookups.values()
            if kind == "getattr" and issubclass(type(target), types.ModuleType)
        ]
        for code, (g, module) in list(unit.codes.items()):
            if code in self.static:
                continue
            qn = code.co_qualname
            if sys.modules.get(module) is None:
                raise _Untraceable(f"it ran {qn} of {module}, a module sys.modules does not hold")
            if qn == "<module>" or not code.co_flags & 0x1:
                # A module imported, a class body: what they bound is read
                # through the items of the code that uses it.
                continue
            self._code(code, g, module, ran)
            recv = unit.receivers.get(code)
            if recv is not None and _user_class(recv):
                self._class(recv, ran)
            ns = g
            table = name_table(code)
            for name in table.reads:
                if not isinstance(name, str) or name in _SKIP_GLOBALS or name.startswith("__cash"):
                    continue
                value = ns.get(name, _MISSING)
                self.add(
                    ("global", module, name),
                    _ABSENT if value is _MISSING else _digest(value),
                    f"{qn} reads the global {name}, whose value cannot be checked",
                )
                if issubclass(type(value), types.ModuleType) and _user_module(value):
                    for chain in table.chains:
                        head, _, rest = chain.partition(".")
                        if head == name and rest:
                            self.add(("attr", module, chain), _chain_digest(module, chain), f"{qn} reads {chain}")
        for kind, target, name in unit.lookups.values():
            self._lookup(kind, target, name)
        out = [(*k, v) for k, v in self.items.items()]
        out.extend(
            tuple(x) for x in unit.extra if not (x[0] == "cached" and unit.cached_codes.get(x[1]) in self.static)
        )
        return list(dict.fromkeys(out))

    def _code(self, code: types.CodeType, g: Mapping[str, Any], module: str, ran: set[str]) -> None:
        from ..tracking.trace_items import candidates, resolve

        qn = code.co_qualname
        rc, fn, _owner = resolve(module, qn, {}, "\x00")
        if rc is code and set(code.co_freevars) <= {"__class__"}:
            self.add(("code", module, qn), _code_digest(code, fn), f"{qn} has defaults that cannot be checked")
            if type(fn) is types.FunctionType:
                self.functions.append(fn)
            return
        if "<locals>" in qn and qn.rsplit(".<locals>.", 1)[0] in ran:
            # Made by a function that ran in this call: its closure is this
            # call's own work, and its code is part of that function's.
            return
        for ns in (g, *self.holders):
            for name, value in list(ns.items()):
                if isinstance(name, str) and not name.startswith("__") and callable(value):
                    for c in candidates(value):
                        if c.__code__ is code:
                            where = ns.get("__name__")
                            self.add(
                                ("held", where, name, qn), _digest(value), f"{qn} holds values that cannot be checked"
                            )
                            self.functions.append(c)
                            return
        raise _Untraceable(f"it ran {module}.{qn}, which no name of {module} holds")

    def _class(self, cls: type, ran: set[str]) -> None:
        from ..tracking.trace_items import class_digest

        qn = cls.__qualname__
        if "<locals>" in qn:
            if qn.rsplit(".<locals>.", 1)[0] in ran:
                return
            raise _Untraceable(f"it ran a method of {qn}, a class no name holds")
        self.add(("class", cls.__module__, qn), class_digest(cls), "")

    def _lookup(self, kind: str, target: Any, name: str | None) -> None:
        if kind == "import":
            if name is None:
                raise _Untraceable("it imports a module whose name is not a string")
            self.add(("import", name), _import_digest(name), f"the module {name} has no file to check")
            return
        if issubclass(type(target), types.ModuleType):
            module, qualname = target.__name__, ""
            if name is None and not _user_module(target):
                raise _Untraceable(f"it looks up a computed name on {module}, a library module")
            # A later process may not have imported it yet: the check
            # imports it, as the call would, when its file is the same.
            self.add(("import", module), _import_digest(module), f"the module {module} has no file to check")
        else:
            cls = target if isinstance(target, type) else type(target)
            module, qualname = cls.__module__, cls.__qualname__
            if "<locals>" in qualname or _find(module, qualname) is not cls:
                raise _Untraceable(f"it looks up a name on {qualname}, a class no name holds")
            if name is None and not _user_class(cls):
                raise _Untraceable(f"it looks up a computed name on a {qualname}, a library class")
        key = name if name is not None else "*"
        self.add(
            ("lookup", module, qualname, key),
            _lookup_digest(module, qualname, key),
            f"what it looks up on {module}{'.' + qualname if qualname else ''} cannot be checked",
        )


class _Untraceable(Exception):
    """Why a traced call's trace could not be built."""


_MISSING = object()


def _chain_digest(module: str, chain: str) -> str | None:
    from ..tracking.trace_items import inspect_getattr

    ns = _namespace(module)
    parts = chain.split(".")
    if ns is None or parts[0] not in ns:
        return _ABSENT
    obj = ns[parts[0]]
    for p in parts[1:]:
        if not issubclass(type(obj), types.ModuleType):
            break
        try:
            obj = inspect_getattr(obj, p)
        except AttributeError:
            return _ABSENT
    return _digest(obj)


def _refusal_in(functions: Iterable[Any]) -> str | None:
    """Why a function the call ran makes its result unsafe to serve: a
    finding a hit would skip (a side effect, a clock read, eval), or a
    random generator of its own it draws from."""
    from ..analysis.purity_analyzer import get_analyzer
    from ..tracking.randomness.state import capture_reachable_carrier_states

    seen: set[int] = set()
    for fn in functions:
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        report = get_analyzer().analyze_reached(fn)
        if report.unwalkable:
            return f"it ran {fn.__qualname__}, whose helpers cash could not all walk"
        for issue in report.issues:
            if issue.kind in _REFUSED_KINDS or (issue.kind == ISSUE_UNTRACKABLE_DEP and not is_name_lookup(issue)):
                return f"it ran {fn.__qualname__}, which {issue.description} (line {issue.line})"
        if capture_reachable_carrier_states(fn):
            return f"it ran {fn.__qualname__}, which can draw from a random generator it holds"
    return None


def build(unit: Any, static_codes: set[types.CodeType], *, strict: bool = True) -> tuple[list[tuple] | None, str]:
    """The items of a traced call's finished *unit*, or None and why the
    call's result cannot be served from a trace. *static_codes* are the
    code the key covers, left out. *strict*: the call picks code by name,
    so code with effects a hit would skip is refused too; a call the key
    already accepted is not second-guessed for those."""
    from ..tracking.trace_items import call_starts

    try:
        with call_starts(unit.cells):
            b = _Build(unit, static_codes)
            items = b.run()
        why = _refusal_in(b.functions) if strict else None
        if why is not None:
            return None, why
        return items, ""
    except _Untraceable as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001 - a trace that cannot be built is not stored
        return None, f"its trace could not be built ({type(e).__name__})"


def static_codes(functions: Iterable[Any], reports: Iterable[Any]) -> set[types.CodeType]:
    """The code objects the key covers: those of *functions*, nested code
    included, and those of the helpers each report's walk read
    (``walked_code``, ``helper_bindings``)."""
    from ..tracking.trace_items import candidates

    out: set[types.CodeType] = set()
    stack: list[Any] = [getattr(getattr(f, "__func__", f), "__code__", None) for f in functions]
    for report in reports:
        for _ref, codes in getattr(report, "walked_code", ()) or ():
            stack.extend(codes)
        for _module, _chain, ref in getattr(report, "helper_bindings", ()) or ():
            held = ref() if callable(ref) else None
            stack.extend(c.__code__ for c in candidates(held))
    while stack:
        c = stack.pop()
        if isinstance(c, types.CodeType) and c not in out:
            out.add(c)
            stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return out


# -- checking --------------------------------------------------------------
def verify(items: list, cached_state: Callable[[str], str | None]) -> str | None:
    """None when every item of a stored trace still holds its digest, else
    what changed, in a few words. *cached_state* gives a cached function's
    state segment now."""
    try:
        items = [tuple(i) for i in items]
        # Modules imported by name first: the code items below find their
        # functions in them. One not loaded yet is imported when its file is
        # the one recorded, as the call would have.
        for item in items:
            if item[0] == "import" and item[1] not in sys.modules:
                # Loaded, it is what the call would get: the items below
                # check what of it ran. Not loaded, it is imported when its
                # file is the one recorded.
                name, was = item[1], item[2]
                if _import_digest(name) != was:
                    return f"the module {name} changed"
                if was != _ABSENT:
                    importlib.import_module(name)
        for item in items:
            kind = item[0]
            if kind == "import":
                continue
            if kind == "code":
                from ..tracking.trace_items import resolve

                rc, fn, _ = resolve(item[1], item[2], {}, "\x00")
                now = None if rc is None else _code_digest(rc, fn)
                what = f"{item[1]}.{item[2]}"
            elif kind == "held":
                ns = _namespace(item[1])
                now = None if ns is None else (_digest(ns[item[2]]) if item[2] in ns else _ABSENT)
                what = f"{item[1]}.{item[2]}"
            elif kind == "class":
                from ..tracking.trace_items import class_digest

                cls = _find(item[1], item[2])
                now = class_digest(cls) if isinstance(cls, type) else None
                what = f"the class {item[1]}.{item[2]}"
            elif kind == "global":
                ns = _namespace(item[1])
                now = None if ns is None else (_digest(ns[item[2]]) if item[2] in ns else _ABSENT)
                what = f"the global {item[1]}.{item[2]}"
            elif kind == "attr":
                now = _chain_digest(item[1], item[2])
                what = item[2]
            elif kind == "lookup":
                now = _lookup_digest(item[1], item[2], item[3])
                where = f"{item[1]}.{item[2]}" if item[2] else item[1]
                what = f"what {where} holds" + (f" under {item[3]!r}" if item[3] != "*" else "")
            elif kind == "cached":
                now = cached_state(item[1])
                what = f"the cached function {item[1]}"
            else:
                return f"an item cash does not know ({kind})"
            if now is None or now != item[-1]:
                return f"{what} changed"
        return None
    except Exception as e:  # noqa: BLE001 - a trace that cannot be checked is not served
        return f"its trace could not be checked ({type(e).__name__})"
