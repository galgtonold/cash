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
``("first_import", name, digest)``
    a module of the user's whose top level ran inside the call (a lazy
    ``import`` in a method): its file, loaded or not.
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

import builtins
import dis
import functools
import hashlib
import importlib
import os
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .. import _weak
from .._gc_context import collecting
from .._own_work import inside as own_work_inside
from .._own_work import left_out as own_work_left_out
from .._own_work import rearm_with as own_work_rearm_with
from .._paths import is_pseudo_filename
from .._user_attrs import instance_of
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
from ..source_reading import SETTLED_SECONDS, stat_has_settled
from ..tracking.process_registries import register_atexit
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

        __slots__ = ("cached_codes", "callees", "lookups", "lookups_followed", "sites")

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
            #: A traced call's: code of a function that writes through its
            #: parameters -> ``(caller code, offset of the call)`` for each
            #: start of it (`_Recorder.site_codes`, `cash.analysis.out_arguments`).
            self.sites: dict[types.CodeType, set[tuple[types.CodeType, int]]] = {}

        def absorb(self, inner: Any) -> None:
            super().absorb(inner)
            for key, callee in getattr(inner, "callees", {}).items():
                self.callees.setdefault(key, callee)

    return TraceUnit


TraceUnit: Any = None


def _recorder_class() -> type:
    from ..analysis.out_arguments import writes_through_parameters
    from ..install_paths import is_user_path, known_user_path
    from ..tracking import trace_recorder as rec

    DISABLE = sys.monitoring.DISABLE
    _POOL_QUALNAMES = frozenset(qualname for _, qualname in rec.POOL_STARTS)
    _PYPLOT_FILES = ("matplotlib/pyplot.py", "matplotlib\\pyplot.py")

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
            self.user_codes: weakref.WeakKeyDictionary[types.CodeType, int] = _weak.WeakKeyDictionary()
            #: The user's top-level functions that write through their
            #: parameters (`out_arguments.writes_through_parameters`): their
            #: start event stays armed while a call is traced, and each start
            #: records where it was called from (`TraceUnit.sites`).
            self.site_codes: weakref.WeakSet[types.CodeType] = _weak.WeakSet()
            self.rearm: set[types.CodeType] = set()
            #: The user's code that looks names up by a string in its own
            #: globals (`string_lookups`): what it calls is recorded for a
            #: notebook unit, the callable a string found among it.
            self.by_string: weakref.WeakSet[types.CodeType] = _weak.WeakSet()
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
            own_work_rearm_with(self._rearm_codes)
            mon.register_callback(self.tool, ev.PY_START, self._guarded(self._on_start))
            mon.register_callback(self.tool, ev.PY_RESUME, self._guarded(self._on_resume))
            mon.register_callback(self.tool, ev.CALL, self._guarded(self._on_call))
            mon.register_callback(self.disc, ev.PY_START, self._guarded(self._discover))
            # A partial adds no Python frame: `_discover` reads the watched
            # frame at `WATCHED_FRAME`.
            mon.register_callback(
                self.disc, ev.PY_RESUME, self._guarded(functools.partial(self._discover, resumed=True))
            )
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

        def _discover(self, code: types.CodeType, offset: int, *, resumed: bool = False) -> Any:
            """First sight of *code*: arm local events on it when it is the
            user's (or starts a process pool), and record this run of it.

            Told a start from a resume by the event, not by *offset*: a
            generator's ``PY_START`` comes from the ``RESUME`` after its
            ``RETURN_GENERATOR``, at a non-zero offset, so a generator a call
            made and ran itself read as one it resumed (`generator_resumed`)."""
            filename = code.co_filename
            if filename.endswith(_PYPLOT_FILES):
                # pyplot's own code: seen as it runs, once per unit, to tell
                # whether a unit read or changed pyplot's current figure
                # (`Unit.pyplot_ran`).
                return self._arm_pyplot(code)
            if (
                filename
                and not is_pseudo_filename(filename)
                and code.co_qualname not in _POOL_QUALNAMES
                and code is not rec.FINALIZE_CALL
            ):
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
            if (
                kind[1] is None
                and not pool
                and code is not rec.FINALIZE_CALL
                and not self._unregistered_user_code(kind[2], code)
            ):
                return DISABLE
            mon = sys.monitoring
            events = mon.events.PY_START | mon.events.PY_RESUME
            if not pool and _LOOKUP_NAMES.intersection(code.co_names):
                events |= mon.events.CALL
            if not pool and self._looks_up_by_string(code, g):
                self.by_string.add(code)
                events |= mon.events.CALL
            self.user_codes[code] = events
            mon.set_local_events(self.tool, code, events)
            if not pool and writes_through_parameters(code):
                self.site_codes.add(code)
                if self.active:
                    self._note_site(code, frame)
            if self.active or self.stack:
                if resumed and code.co_flags & rec.RESUMABLE:
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
                self.stack.append(unit)
                return unit
            with self._lock:
                self.active = [*self.active, unit]
            return unit

        def _rearm(self) -> None:
            rearm, self.rearm = self.rearm, set()
            self._rearm_codes(rearm)

        def _rearm_codes(self, rearm: set[types.CodeType]) -> None:
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
            inside_collection = collecting()
            if not inside_collection:
                self.rearm.add(code)
            site = self.active and code in self.site_codes
            if site:
                # Every start, not the first one: where each was called
                # from decides what its writes change.
                frame = sys._getframe(rec.WATCHED_FRAME)
                self._note_site(code, frame)
                if not self.stack and all(code in u.codes for u in self.active):
                    return None
            if self.active or self.stack:
                frame = sys._getframe(rec.WATCHED_FRAME)
                g = frame.f_globals
                kind = self._kinds.get(id(g))
                if kind is None or kind[0] is not g:
                    kind = self._classify(g)
                self._seen_start(code, frame, g, kind)
            # Code a garbage collection runs (a ``__del__``) is left armed: a
            # run of it by the unit's own code is told apart (`Unit.collected`).
            return None if inside_collection or site else DISABLE

        def _note_site(self, code: types.CodeType, frame: Any) -> None:
            caller = frame.f_back
            site = (caller.f_code, caller.f_lasti) if caller is not None else (None, -1)
            for unit in self.active:
                unit.sites.setdefault(code, set()).add(site)

        def _seen_start(self, code: types.CodeType, frame: Any, g: dict, kind: tuple) -> None:
            if own_work_inside():
                # Run by cash's own pickling of a value, not by the user's
                # program (`cash._own_work`): armed again once that ends.
                own_work_left_out(code)
                return
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
                if code is rec.FINALIZE_CALL:
                    rec.note_finalizer(units, frame)
                    return
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
            inside_collection = collecting()
            for unit in units:
                rec.note_collected(unit, code, inside_collection)
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
            if own_work_inside():
                own_work_left_out(code)
                return
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
            if stack and code in self.by_string:
                callees = stack[-1].callees
                if id(fn) not in callees:
                    callees[id(fn)] = (fn, sys._getframe(rec.WATCHED_FRAME).f_globals)
            if not units:
                if stack and code in self.by_string:
                    return None
                self.rearm.add(code)
                return DISABLE
            try:
                if fn is getattr:
                    if arg0 is sys.monitoring.MISSING:
                        return None
                    name = _last_argument(code, offset, sys._getframe(rec.WATCHED_FRAME))
                    entry = ("getattr", arg0, name)
                    slot = ("getattr", id(arg0), name)
                elif fn is self._import_module or fn is builtins.__import__:
                    entry = ("import", None, arg0 if isinstance(arg0, str) else None)
                    slot = ("import", 0, entry[2])
                else:
                    return None
                for unit in units:
                    unit.lookups.setdefault(slot, entry)
            except Exception:  # noqa: BLE001 - a lookup that cannot be read leaves the trace unusable
                for unit in units:
                    unit.markers.add("lookup_unread")
            return None

    return _Recorder


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
                register_atexit(r.uninstall)
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


def _load_plan(ins: Any, pair_slot: int) -> tuple[str, Any] | None:
    """How to read the value instruction *ins* pushes last (*pair_slot* 1)
    or first (0, of a paired load): a constant, a local or a global."""
    op, val = ins.opname, ins.argval
    if op in ("LOAD_CONST", "LOAD_SMALL_INT"):
        return ("const", val)
    if op in _FAST_LOADS:
        return ("local", val)
    if op in _PAIRED_LOADS and isinstance(val, tuple) and len(val) == 2:
        return ("local", val[pair_slot])
    if op in ("LOAD_GLOBAL", "LOAD_NAME"):
        return ("global", val)
    return None


def _plan(code: types.CodeType, offset: int) -> tuple[str, Any] | None:
    """How to read the name argument of the ``getattr`` call at *offset*:
    the last of two arguments, or the middle of three (``getattr(obj, name,
    default)``): ``("const", value)``, ``("local", name)`` or ``("global",
    name)``, or None when an expression computed it."""
    plans = _INSTRUCTIONS.get(code)
    if plans is None:
        if len(_INSTRUCTIONS) > 4096:
            _INSTRUCTIONS.clear()
        plans = _INSTRUCTIONS[code] = {}
    if offset in plans:
        return plans[offset]
    plan = None
    previous = before = None
    for ins in dis.get_instructions(code):
        if ins.offset == offset:
            if ins.opname == "CALL" and previous is not None:
                if ins.arg == 2:
                    plan = _load_plan(previous, 1)
                elif ins.arg == 3:
                    # The default is pushed last: by a paired load with the
                    # name, or by a load of its own after the name's.
                    if previous.opname in _PAIRED_LOADS:
                        plan = _load_plan(previous, 0)
                    elif before is not None and _load_plan(previous, 1) is not None:
                        plan = _load_plan(before, 1)
            break
        if ins.opname not in _SKIPPED:
            previous, before = ins, previous
    plans[offset] = plan
    return plan


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


def _frame_namespace(module: str, filename: str) -> dict | None:
    """The globals of the innermost running frame of *filename* whose
    ``__name__`` is *module*: the namespace of a script that is running but
    that no module in ``sys.modules`` holds."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename == filename and frame.f_globals.get("__name__") == module:
            return frame.f_globals
        frame = frame.f_back
    return None


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


def _holding_class(code: types.CodeType, g: Mapping[str, Any], module: str) -> type | None:
    """The class of module *module* (globals *g*) that a name there holds and
    that holds a function running *code* as an attribute of its own: a method
    a decorator built and set on it, whose qualified name finds nothing."""
    from ..tracking.trace_items import candidates

    for value in list(g.values()):
        if not isinstance(value, type) or value.__module__ != module or "<locals>" in value.__qualname__:
            continue
        try:
            members = list(vars(value).values())
        except TypeError:
            continue
        for member in members:
            if any(c.__code__ is code for c in candidates(member)):
                return value
    return None


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


#: The attributes a C type defines: compiled, so nothing edits them.
_C_DESCRIPTORS = frozenset(
    {
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        types.ClassMethodDescriptorType,
    }
)


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
        if type(found) in _C_DESCRIPTORS:
            # A slot of a C type (``ndarray.dtype``): no edit can change
            # it, so it stands for itself by where it is defined.
            owner = getattr(found, "__objclass__", None)
            return f"c:{getattr(owner, '__module__', '?')}.{getattr(owner, '__qualname__', '?')}.{name}"
        return _digest(found)
    ns = _namespace(module)
    if ns is None:
        return None
    if name == "*":
        return _namespace_digest(ns) if _user_module(sys.modules.get(module)) else None
    return _digest(ns[name]) if name in ns else _ABSENT


#: ``(path, size, mtime_ns, ctime_ns, inode) -> digest`` of files that had
#: settled when read. The mtime alone let a same-size save that put it back
#: (``cp -p``, ``os.utime``) keep the old digest, and one inside the tick the
#: digest was read in kept it too: the racy-git rule, as `stat_has_settled`.
_FILE_DIGESTS: dict[tuple[str, int, int, int, int], str] = {}


def _file_digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    slot = (path, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    got = _FILE_DIGESTS.get(slot)
    if got is None:
        settled = stat_has_settled(st) and time.time() - st.st_ctime > SETTLED_SECONDS
        try:
            with open(path, "rb") as fh:
                got = hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
        except OSError:
            return None
        if not settled:
            return got
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
        for code, (g, _module) in list(unit.codes.items()):
            if code.co_flags & 0x1:
                # Keyed or not: a module the key could not read before it
                # was loaded hands the call what it binds through these.
                self._imported(code, g)
        for code, (g, module) in list(unit.codes.items()):
            if code in self.static:
                continue
            qn = code.co_qualname
            if sys.modules.get(module) is None:
                raise _Untraceable(f"it ran {qn} of {module}, a module sys.modules does not hold")
            if qn == "<module>" or not code.co_flags & 0x1:
                # A module imported, a class body: what they bound is read
                # through the items of the code that uses it, and what a
                # function takes from a module it imports by the value it
                # takes (`_imported`). A module of the user's first imported
                # inside the call (``from ._tags import X`` in a method) is
                # checked by its file too, so the check imports it when a
                # later process has not.
                if qn == "<module>" and _user_module(sys.modules.get(module)):
                    self.add(
                        ("first_import", module),
                        _import_digest(module),
                        f"the module {module} has no file to check",
                    )
                continue
            if _namespace(module) is not g:
                self._hidden(code, g, module)
                continue
            # A later process may not have imported the module yet (``import
            # helpers`` inside the body): the check imports it, as the call
            # would, when its file is the same; without the item, its code
            # is not found there and the entry is never served. A module
            # whose file cannot be read is left to its code items.
            loaded_from = _import_digest(module)
            if loaded_from is not None:
                self.add(("import", module), loaded_from, "")
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

    def _imported(self, code: types.CodeType, g: Mapping[str, Any]) -> None:
        """What *code* takes from a user module it imports, by value:
        ``from m import X`` and ``m.X``, as a global read is. Its top level
        may have computed it from outside its source (the environment, a
        file, another module's value), which the module's file does not
        show. A module the code hands on whole is checked by every name it
        binds."""
        for imported, module, chain in import_reads(code, g):
            if not _user_module(sys.modules.get(module)):
                continue
            # Loaded before the call, a later process may not have it yet:
            # the check imports it, as the call would, when its file is the
            # one recorded.
            # ``from pkg import sub`` loads ``pkg.sub``, read through ``pkg``.
            subs = [f"{module}.{'.'.join(chain[: n + 1])}" for n in range(len(chain))]
            for name in [imported, *(n for n in subs if n in sys.modules)]:
                loaded_from = _import_digest(name)
                if loaded_from is not None:
                    self.add(("import", name), loaded_from, "")
            target = _chain_target(module, chain)
            if chain and not issubclass(type(target), types.ModuleType):
                path = ".".join(chain)
                self.add(
                    ("attr", module, path),
                    _chain_digest(module, path),
                    f"{code.co_qualname} takes {module}.{path}, whose value cannot be checked",
                )
                continue
            # The module itself: what it holds, name by name.
            name = getattr(target, "__name__", None)
            if not isinstance(name, str) or not _user_module(target):
                continue
            self.add(
                ("lookup", name, "", "*"),
                _lookup_digest(name, "", "*"),
                f"{code.co_qualname} hands on the module {name}, which cash cannot check name by name",
            )

    def _hidden(self, code: types.CodeType, g: Mapping[str, Any], module: str) -> None:
        """Code whose globals are not those of the module its ``__name__``
        names: a script ``runpy.run_path(path, run_name="__main__")`` runs,
        or the script that ran it while the other one holds ``__main__``.
        Named by module, it was looked up in the wrong namespace in the next
        process and read as changed every time. Its items name its file
        instead, and the check finds the namespace through a frame of that
        file running under that name (`_frame_namespace`): the script that
        is running the call."""
        qn = code.co_qualname
        filename = code.co_filename
        if is_pseudo_filename(filename) or _frame_namespace(module, filename) is not g:
            raise _Untraceable(f"it ran {qn} of a {module} no module holds and no running frame shows")
        from ..tracking.trace_items import name_table, resolve

        rc, fn, _owner = resolve(module, qn, g, module)
        if rc is not code:
            raise _Untraceable(f"it ran {module}.{qn} of {filename}, which no name there holds")
        self.add(
            ("file_code", module, filename, qn), _code_digest(code, fn), f"{qn} has defaults that cannot be checked"
        )
        table = name_table(code)
        for name in table.reads:
            if not isinstance(name, str) or name in _SKIP_GLOBALS or name.startswith("__cash"):
                continue
            value = g.get(name, _MISSING)
            self.add(
                ("file_global", module, filename, name),
                _ABSENT if value is _MISSING else _digest(value),
                f"{qn} reads the global {name}, whose value cannot be checked",
            )

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
        if qn == "__create_fn__" and is_pseudo_filename(code.co_filename):
            # ``@dataclass`` building a class's methods from its fields (a
            # module first imported inside the call): what it builds is
            # held by the class, whose digest a call of them checks.
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
        cls = _holding_class(code, g, module)
        if cls is not None:
            # Made for a class and set on it (``@dataclass``'s ``__init__``
            # and ``__eq__``, built from source of its own): the class's
            # digest holds it, by code and defaults.
            self._class(cls, ran)
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


def _chain_target(module: str, chain: tuple[str, ...]) -> Any:
    """What *chain* reaches from module *module* through modules, as
    `_chain_digest` follows it: the first value that is not a module, or the
    module it ends on; None when a part is unbound."""
    from ..tracking.trace_items import inspect_getattr

    obj = sys.modules.get(module)
    for p in chain:
        if not issubclass(type(obj), types.ModuleType):
            break
        try:
            obj = inspect_getattr(obj, p)
        except AttributeError:
            return None
    return obj


#: Instructions that read a local of the function's own.
_LOAD_LOCAL = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"})
#: The same for two locals at once (3.13+).
_LOAD_TWO_LOCALS = frozenset({"LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"})


def import_reads(code: types.CodeType, g: Mapping[str, Any]) -> list[tuple[str, str, tuple[str, ...]]]:
    """What *code*'s own imports take, read from its bytecode: ``(imported,
    module, chain)`` for each ``from module import name`` used (chain
    ``(name,)``) and each ``module.a.b`` read through a local an import
    bound; chain ``()`` for a module the code uses otherwise (hands on,
    stores). *imported* is the module the import loads (``import a.b``
    loads ``a.b`` and binds ``a``). Only the code's own instructions: a
    nested function's are its own code."""
    try:
        ins = list(dis.get_instructions(code))
    except Exception:  # noqa: BLE001 - code dis cannot read takes nothing cash can name
        return []
    package = g.get("__package__") or ""
    bound: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    out: list[tuple[str, str, tuple[str, ...]]] = []

    def bind(store: Any, ref: tuple[str, str, tuple[str, ...]]) -> None:
        if store.opname == "STORE_FAST":
            bound[store.argval] = ref
        else:
            # A cell a nested function reads, a global, a name of a class
            # body: read where these instructions do not show.
            out.append(ref)

    for i, op in enumerate(ins):
        name = op.opname
        if name == "IMPORT_NAME" and i >= 2:
            level, fromlist = ins[i - 2].argval, ins[i - 1].argval
            module = op.argval
            try:
                if isinstance(level, int) and level > 0:
                    module = importlib.util.resolve_name("." * level + module, package or None)
            except (ImportError, ValueError):
                continue
            j = i + 1
            if fromlist:
                while j + 1 < len(ins) and ins[j].opname == "IMPORT_FROM":
                    bind(ins[j + 1], (module, module, (ins[j].argval,)))
                    j += 2
                continue
            # ``import a.b as c`` reads ``a.b`` through IMPORT_FROM; ``import
            # a.b`` binds ``a``.
            whole = j < len(ins) and ins[j].opname == "IMPORT_FROM"
            while j < len(ins) and ins[j].opname in ("IMPORT_FROM", "SWAP", "POP_TOP"):
                j += 1
            if j < len(ins) and ins[j].opname.startswith("STORE_"):
                bind(ins[j], (module, module if whole else module.split(".")[0], ()))
        elif name in _LOAD_LOCAL and op.argval in bound:
            imported, module, chain = bound[op.argval]
            j = i + 1
            while j < len(ins) and ins[j].opname == "LOAD_ATTR":
                chain = (*chain, ins[j].argval)
                j += 1
            out.append((imported, module, chain))
        elif name in _LOAD_TWO_LOCALS:
            out.extend(bound[v] for v in op.argval if v in bound)
    return list(dict.fromkeys(out))


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


def _refusal_in(functions: Iterable[Any], unit: Any = None) -> str | None:
    """Why a function the call ran makes its result unsafe to serve: a
    finding a hit would skip (a side effect, a clock read, eval), or a
    random generator of its own it draws from. A finding in a helper that
    did not run (`_helpers_not_run`), and one on a parameter every caller
    handed an object it made itself (`_OutArguments`), is not one."""
    from ..analysis.purity_analyzer import get_analyzer
    from ..tracking.randomness.state import capture_reachable_carrier_states

    out = _OutArguments(unit)
    for fn in dict.fromkeys(functions):
        report = get_analyzer().analyze_reached(fn)
        if report.unwalkable:
            return f"it ran {fn.__qualname__}, whose helpers cash could not all walk"
        idle = _helpers_not_run(report, unit)
        for issue in report.issues:
            if issue.kind in _REFUSED_KINDS or (issue.kind == ISSUE_UNTRACKABLE_DEP and not is_name_lookup(issue)):
                if issue.where in idle or out.excuses(issue):
                    continue
                return f"it ran {fn.__qualname__}, which {issue.description} (line {issue.line})"
        if capture_reachable_carrier_states(fn):
            return f"it ran {fn.__qualname__}, which can draw from a random generator it holds"
    return None


def _helpers_not_run(report: Any, unit: Any) -> set[str]:
    """The names findings go under (`qualname_of`) of the helpers *report*
    walked that did not run in the call: what a hit skips is what ran, and
    a helper of the user's that ran is in the trace (`Unit.codes`). The
    same arguments take the same path while the trace's items hold, so a
    finding in a branch the call did not take is one no hit of it skips;
    a call that takes it records it and is judged on it."""
    if unit is None:
        return set()
    from ..analysis.helper_code import qualname_of

    codes = unit.codes
    idle: set[str] = set()
    ran: set[str] = set()
    for ref, walked in getattr(report, "walked_code", ()) or ():
        fn = ref()
        if type(fn) is not types.FunctionType or not _user_module(sys.modules.get(fn.__module__)):
            continue
        name = qualname_of(fn)
        (ran if any(c in codes for c in walked) else idle).add(name)
    return idle - ran


class _OutArguments:
    """Which findings on a parameter the call sites a traced call ran
    excuse (`cash.analysis.out_arguments`): the trace recorded where each
    function that writes through its parameters was called from
    (`TraceUnit.sites`)."""

    def __init__(self, unit: Any) -> None:
        self.unit = unit
        self._by_name: dict[str, Any] | None = None
        self._judged: dict[tuple[str, str], bool] = {}

    def _functions(self) -> dict[str, Any]:
        """``qualname_of(fn) -> fn`` for each function with recorded sites."""
        if self._by_name is None:
            from ..analysis.helper_code import qualname_of

            self._by_name = {}
            for code in getattr(self.unit, "sites", {}) if self.unit is not None else ():
                fn = self._function_of(code)
                if fn is not None:
                    self._by_name[qualname_of(fn)] = fn
        return self._by_name

    def _function_of(self, code: types.CodeType) -> Any:
        from ..tracking.trace_items import resolve

        where = self.unit.codes.get(code)
        if where is None:
            return None
        g, module = where
        _rc, fn, _owner = resolve(module, code.co_qualname, g, module)
        return fn if type(fn) is types.FunctionType and fn.__code__ is code else None

    def excuses(self, issue: Any) -> bool:
        from ..analysis.out_arguments import mutated_parameter

        if self.unit is None:
            return False
        param = mutated_parameter(issue)
        if param is not None:
            return self._excused(issue.where, param)
        if issue.kind == ISSUE_DISCARDED_CALL:
            return self._discard_excused(issue)
        return False

    def _excused(self, where: str, param: str) -> bool:
        key = (where, param)
        if key not in self._judged:
            self._judged[key] = self._judge(where, param)
        return self._judged[key]

    def _judge(self, where: str, param: str) -> bool:
        from ..analysis.out_arguments import site_hands_own

        fn = self._functions().get(where)
        if fn is None:
            return False
        sites = self.unit.sites.get(fn.__code__)
        if not sites:
            return False
        for caller_code, lasti in sites:
            caller = self._function_of(caller_code)
            if caller is None or not site_hands_own(fn, param, caller, caller_code, lasti):
                return False
        return True

    def _discard_excused(self, issue: Any) -> bool:
        """``fill(x, buf)`` drops what ``fill`` returns: what such a call is
        for is its effect. Excused when that function's every finding on a
        parameter is excused; any other finding of its own is reported as
        itself."""
        from ..analysis.out_arguments import mutated_parameter
        from ..analysis.purity_analyzer import get_analyzer

        text = getattr(issue, "description", "")
        if not (text.startswith("discards return of ") and text.endswith("(...)")):
            return False
        name = text[len("discards return of ") : -len("(...)")]
        for qualname, fn in self._functions().items():
            if fn.__name__ != name:
                continue
            own = [i for i in get_analyzer().analyze_reached(fn).issues if i.where == qualname]
            params = [mutated_parameter(i) for i in own]
            if params and all(p is not None and self._excused(qualname, p) for p in params):
                return True
        return False


def build(
    unit: Any,
    static_codes: set[types.CodeType],
    *,
    strict: bool = True,
    accepted: set[types.CodeType] | None = None,
) -> tuple[list[tuple] | None, str]:
    """The items of a traced call's finished *unit*, or None and why the
    call's result cannot be served from a trace. *static_codes* are the
    code the key covers, left out. *strict*: the call picks code by name,
    so code with effects a hit would skip is refused too; a call the key
    already accepted is not second-guessed for those, nor is *accepted*,
    code the key folded from the call's arguments (a method of an
    argument's class), which a call without the lookup runs as well."""
    from ..tracking.trace_items import call_starts

    try:
        with call_starts(unit.cells):
            b = _Build(unit, static_codes)
            items = b.run()
        judged = [f for f in b.functions if f.__code__ not in accepted] if accepted else b.functions
        why = _refusal_in(judged, unit) if strict else None
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


def folded_codes(folded: Any) -> set[types.CodeType] | None:
    """The code objects a key folded (`cash.code_swaps.FoldedCode`): each
    function's, and each class's with its user base classes' methods, which
    its key part holds too (`CodeSurface.class_surface_parts`)."""
    if folded is None or not folded.owners:
        return None
    from ..loaded_code import code_objects_of
    from .user_code import is_user_code_object

    out: set[types.CodeType] = set()
    for owner, codes in list(folded.owners.values()):
        out.update(codes)
        if isinstance(owner, type):
            for base in owner.__mro__[1:]:
                if base is not object and is_user_code_object(base):
                    out.update(code_objects_of(base))
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
            if item[0] in ("import", "first_import") and item[1] not in sys.modules:
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
            if kind == "first_import":
                # Loaded already, by this process or the check above: its
                # file is the one the call imported.
                now = _import_digest(item[1])
                what = f"the module {item[1]}"
            elif kind == "code":
                from ..tracking.trace_items import resolve

                rc, fn, _ = resolve(item[1], item[2], {}, "\x00")
                now = None if rc is None else _code_digest(rc, fn)
                what = f"{item[1]}.{item[2]}"
            elif kind in ("file_code", "file_global"):
                ns = _frame_namespace(item[1], item[2])
                if kind == "file_code":
                    from ..tracking.trace_items import resolve

                    rc, fn, _ = resolve(item[1], item[3], ns, item[1]) if ns is not None else (None, None, None)
                    now = None if rc is None else _code_digest(rc, fn)
                    what = f"{item[3]} of {item[2]}"
                else:
                    now = None if ns is None else (_digest(ns[item[3]]) if item[3] in ns else _ABSENT)
                    what = f"the global {item[3]} of {item[2]}"
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
