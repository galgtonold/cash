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
import dis
import hashlib
import importlib
import os
import sys
import threading
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

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

#: Names a code object holds when it may look code up by name: CALL events
#: are armed on it (`_Recorder._on_start`).
_LOOKUP_NAMES = frozenset({"getattr", "__import__", "import_module"})
#: Markers that leave a trace nothing can check (`recorder` module).
_UNUSABLE = {
    "thread_alive": "a thread it started still runs",
    "process_pool": "it ran work in other processes",
    "generator_resumed": "it resumed a generator started before it",
    "tool_conflict": "another tool took the monitoring tool id",
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
    from ..notebook.shadow.recorder import Unit

    class TraceUnit(Unit):
        """What one traced call saw: `Unit`'s fields, its lookups by name
        and the items of what was served inside it."""

        __slots__ = ("lookups",)

        def __init__(self, kind: str) -> None:
            super().__init__(kind)
            #: ``(kind, id(target), name)`` -> ``(kind, target, name)``.
            self.lookups: dict[tuple, tuple] = {}

    return TraceUnit


TraceUnit: Any = None


def _recorder_class() -> type:
    from ..install_paths import is_user_path
    from ..notebook.shadow import recorder as rec

    DISABLE = sys.monitoring.DISABLE

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
            if not super().install():
                return False
            mon = sys.monitoring
            self.disc = self.tool
            mon.register_callback(self.disc, mon.events.PY_START, self._discover)
            mon.register_callback(self.disc, mon.events.PY_RESUME, self._discover)
            self.tool = None
            for tool in self.tool_ids:
                if mon.get_tool(tool) is None:
                    mon.use_tool_id(tool, self.tool_name)
                    self.tool = tool
                    break
            if self.tool is None:
                self.tool = self.disc
                self.uninstall()
                return False
            ev = mon.events
            self.rearm_works = self._rearm_works()
            mon.register_callback(self.tool, ev.PY_START, self._on_start)
            mon.register_callback(self.tool, ev.PY_RESUME, self._on_resume)
            mon.register_callback(self.tool, ev.CALL, self._on_call)
            self.user_codes: dict[types.CodeType, int] = {}
            self.rearm: set[types.CodeType] = set()
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
            frame = sys._getframe(1)
            g = frame.f_globals
            kind = self._kinds.get(id(g))
            if kind is None or kind[0] is not g:
                kind = self._classify(g)
            pool = (kind[2], code.co_qualname) in rec.POOL_STARTS
            if kind[1] is None and not pool and not self._unregistered_user_code(kind[2], code):
                return DISABLE
            mon = sys.monitoring
            events = mon.events.PY_START | mon.events.PY_RESUME
            if not pool and _LOOKUP_NAMES.intersection(code.co_names):
                events |= mon.events.CALL
            self.user_codes[code] = events
            mon.set_local_events(self.tool, code, events)
            if self.active or self.stack:
                if code.co_flags & rec.RESUMABLE and offset:
                    self._seen_resume(code, g, kind)
                else:
                    self._seen_start(code, frame, g, kind)
            return DISABLE

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
                frame = sys._getframe(1)
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
                    from ..notebook.shadow.items import start_cells

                    cells = start_cells(frame, code)
                finally:
                    self._in_callback = False
            for unit in units:
                unit.codes.setdefault(code, (g, module))
                if cells is not None:
                    unit.cells.setdefault(code, cells)
                if recv is not None:
                    unit.receivers.setdefault(code, recv if isinstance(recv, type) else type(recv))

        @staticmethod
        def _unregistered_user_code(name: str, code: types.CodeType) -> bool:
            if name == "cash" or name.startswith("cash.") or sys.modules.get(name) is not None:
                return False
            filename = code.co_filename
            return bool(filename) and not filename.startswith("<") and is_user_path(filename)

        def _on_resume(self, code: types.CodeType, offset: int) -> Any:
            self.rearm.add(code)
            if (self.active or self.stack) and code.co_flags & rec.RESUMABLE:
                g = sys._getframe(1).f_globals
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

        def _on_call(self, code: types.CodeType, offset: int, fn: Any, arg0: Any) -> Any:
            units = self.active
            if not units:
                self.rearm.add(code)
                return DISABLE
            try:
                if fn is getattr:
                    if arg0 is sys.monitoring.MISSING:
                        return None
                    name = _last_argument(code, offset, sys._getframe(1))
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
_RECORDER_LOCK = threading.Lock()


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


def note_cached_hit(func_name: str, state: str | None, trace: list | None) -> None:
    """A cached function was served inside the traced calls in progress: its
    body did not run, so what it stands for goes in their traces."""
    rec = _RECORDER
    if rec is None:
        return
    for unit in list(rec.active):
        if state is None:
            unit.markers.add("cached_hit_unkeyed")
            continue
        unit.extra.append(("cached", func_name, state))
        unit.extra.extend(tuple(item) for item in trace or ())


_INSTRUCTIONS: dict[types.CodeType, dict[int, tuple[str, Any] | None]] = {}
_FAST_LOADS = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_DEREF"})
_PAIRED_LOADS = frozenset({"LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"})
_SKIPPED = frozenset({"CACHE", "EXTENDED_ARG", "NOP"})


def _plan(code: types.CodeType, offset: int) -> tuple[str, Any] | None:
    """How to read the last argument of the two-argument call at *offset*:
    ``("const", value)``, ``("local", name)`` or ``("global", name)``, or
    None when an expression computed it."""
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
            if ins.opname == "CALL" and ins.arg == 2 and previous is not None:
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
    from ..notebook.shadow.items import value_digest

    wrapped = getattr(value, "__wrapped__", None) if isinstance(value, types.FunctionType) else None
    if wrapped is not None and is_cash_path(value.__code__.co_filename):
        # A ``@cash.cache`` wrapper: by the function it wraps (its state,
        # when it is served inside the call, is a "cached" item of its own).
        value = wrapped
    got = value_digest(value)
    return None if got.startswith("opaque:") else got


def _code_digest(code: types.CodeType, fn: Any) -> str | None:
    from ..notebook.shadow.items import code_digest

    got = code_digest(code)
    if isinstance(fn, types.FunctionType) and fn.__code__ is code and (fn.__defaults__ or fn.__kwdefaults__):
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
    from ..notebook.shadow.items import inspect_getattr

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
    from ..install_paths import is_user_code_module

    return isinstance(mod, types.ModuleType) and is_user_code_module(mod)


def _user_class(cls: Any) -> bool:
    return isinstance(cls, type) and _user_module(sys.modules.get(cls.__module__ or ""))


def _classes_digest(cls: type) -> str | None:
    """The user classes of *cls*'s MRO, each by what it holds."""
    from ..notebook.shadow.items import class_digest

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
    from ..notebook.shadow.items import inspect_getattr

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
        from ..notebook.shadow.items import name_table

        unit = self.unit
        for marker in sorted(unit.markers):
            raise _Untraceable(_UNUSABLE.get(marker, marker.replace("_", " ")))
        ran = {c.co_qualname for c in unit.codes}
        # Where a function no qualified name finds may be held: its own
        # module, and every module the call looked a name up on.
        self.holders = [
            target.__dict__
            for kind, target, _name in unit.lookups.values()
            if kind == "getattr" and isinstance(target, types.ModuleType)
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
                if isinstance(value, types.ModuleType) and _user_module(value):
                    for chain in table.chains:
                        head, _, rest = chain.partition(".")
                        if head == name and rest:
                            self.add(("attr", module, chain), _chain_digest(module, chain), f"{qn} reads {chain}")
        for kind, target, name in unit.lookups.values():
            self._lookup(kind, target, name)
        out = [(*k, v) for k, v in self.items.items()]
        out.extend(tuple(x) for x in unit.extra)
        return list(dict.fromkeys(out))

    def _code(self, code: types.CodeType, g: Mapping[str, Any], module: str, ran: set[str]) -> None:
        from ..notebook.shadow.items import candidates, resolve

        qn = code.co_qualname
        rc, fn, _owner = resolve(module, qn, {}, "\x00")
        if rc is code and set(code.co_freevars) <= {"__class__"}:
            self.add(("code", module, qn), _code_digest(code, fn), f"{qn} has defaults that cannot be checked")
            if isinstance(fn, types.FunctionType):
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
        from ..notebook.shadow.items import class_digest

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
        if isinstance(target, types.ModuleType):
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
    from ..notebook.shadow.items import inspect_getattr

    ns = _namespace(module)
    parts = chain.split(".")
    if ns is None or parts[0] not in ns:
        return _ABSENT
    obj = ns[parts[0]]
    for p in parts[1:]:
        if not isinstance(obj, types.ModuleType):
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


def build(unit: Any, static_codes: set[types.CodeType]) -> tuple[list[tuple] | None, str]:
    """The items of a traced call's finished *unit*, or None and why the
    call's result cannot be served from a trace. *static_codes* are the
    code the key covers, left out."""
    from ..notebook.shadow.items import call_starts

    try:
        with call_starts(unit.cells):
            b = _Build(unit, static_codes)
            items = b.run()
        why = _refusal_in(b.functions)
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
    from ..notebook.shadow.items import candidates

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
                from ..notebook.shadow.items import resolve

                rc, fn, _ = resolve(item[1], item[2], {}, "\x00")
                now = None if rc is None else _code_digest(rc, fn)
                what = f"{item[1]}.{item[2]}"
            elif kind == "held":
                ns = _namespace(item[1])
                now = None if ns is None else (_digest(ns[item[2]]) if item[2] in ns else _ABSENT)
                what = f"{item[1]}.{item[2]}"
            elif kind == "class":
                from ..notebook.shadow.items import class_digest

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
