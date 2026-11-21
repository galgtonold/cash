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
``("alias", module, name, digest)``
    a ``type name = ...`` alias whose value was built in the call: what it
    stands for.
``("class", module, qualname, digest)``
    a class of the user's a method that ran was called on.
``("surface", module, qualname, digest)``
    the same, by its surface (`trace_items.class_surface_digest`), in a
    trace built for a key that holds the classes of module data by their
    surface (`build`'s *narrow*).
``("full_state", func_name, digest)``
    the state segment of the cached function *func_name* built with every
    class of module data keyed by all its code, for an entry whose key
    held them by their surface and whose call left no trace to check.
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
left behind, a function it ran that has side effects, reads the clock,
draws from a generator of its own or runs ``eval`` on a line that ran (the
lines of code holding such a finding are watched on a miss, one LINE
callback per line; a line no watch saw run keeps its finding), or a global the code it
ran reads that the call changed in place (``heapq.heappush(HEAP, x)``): its
item holds the state the call left, and a hit would not change it again.

Threads: user code that starts while any traced call runs, on any thread, is
added to every traced call in progress, so the work of a pool a call waits on
is in its trace (and an unrelated thread's work only adds checks).
"""

from __future__ import annotations

import builtins
import contextlib
import dis
import functools
import hashlib
import importlib
import logging
import os
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from .. import _weak
from .._gc_context import collecting
from .._memo import CODE_OBJECTS, LruMemo
from .._own_work import inside as own_work_inside
from .._own_work import left_out as own_work_left_out
from .._own_work import rearm_with as own_work_rearm_with
from .._paths import is_pseudo_filename
from .._user_attrs import NOT_HELD, held_attribute, instance_of, module_file
from ..analysis.purity_flow import LOG_METHODS as _LOG_METHOD_NAMES
from ..analysis.purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_IMPURE_CALL,
    ISSUE_NETWORK_READ,
    ISSUE_SCOPE_MUTATION,
    ISSUE_UNTRACKABLE_DEP,
    UNREAD_MODULE,
    PurityIssue,
    is_name_lookup,
)
from ..analysis.static_dispatch import any_name_stdlib
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
    "first_import_digest",
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
_REFUSED_KINDS = frozenset({ISSUE_IMPURE_CALL, ISSUE_SCOPE_MUTATION, ISSUE_AMBIENT_READ, ISSUE_NETWORK_READ})
#: Global names a function reads that are where cash put itself.
_SKIP_GLOBALS = frozenset({"__builtins__", "__name__", "__file__", "__spec__", "__loader__", "__package__"})
#: Namespaces bigger than this are not digested whole (an unread name on one).
_NAMESPACE_CAP = 2000
_ABSENT = "absent"
#: Code of the user's -> did a strict traced call judge it to hold a finding
#: a hit would skip (`_refusal_in`)? Code judged to hold none has its lines
#: left unwatched (`_Recorder._arm_lines`).
_LINE_STATUS: weakref.WeakKeyDictionary[types.CodeType, bool] = _weak.WeakKeyDictionary()


def covers(issues: Iterable[PurityIssue]) -> bool:
    """Can a trace stand in for *issues*, the run-time lookups of a call's
    tree: all of them lookups by name (`is_name_lookup`), on 3.12+?

    A lazy import of a module not loaded yet (UNREAD_MODULE) beside them
    is covered too while the trace is on: the lookups trace the call
    anyway, and an import that runs is in that trace (``first_import``),
    with the code of the module that ran and the globals it read, as for
    the same import in an argument's code (`CallRunner._carrier_refusal`).
    Alone, with no lookup to trace the call, it still refuses."""
    issues = list(issues)
    if not TRACEABLE or not issues:
        return False
    lookups = sum(1 for i in issues if is_name_lookup(i))
    if lookups == len(issues):
        return True
    return bool(lookups) and on() and all(is_name_lookup(i) or UNREAD_MODULE in i.description for i in issues)


# -- recording -------------------------------------------------------------
def _unit_class() -> type:
    from ..tracking.trace_recorder import Unit

    class TraceUnit(Unit):
        """What one traced call saw: `Unit`'s fields, its lookups by name
        and the items of what was served inside it."""

        __slots__ = (
            "attr_params",
            "attr_targets",
            "cached_codes",
            "callees",
            "global_starts",
            "line_codes",
            "line_ran",
            "lookups",
            "lookups_followed",
            "lib_cells",
            "lib_codes",
            "lib_first",
            "lib_items",
            "lib_keys",
            "lib_receivers",
            "sites",
            "surface",
        )

        def __init__(self, kind: str) -> None:
            super().__init__(kind)
            #: A traced call whose key holds the classes of module data by
            #: their surface (`begin`): its cells are digested as they start
            #: with classes by their surface too, as `build` digests them.
            self.surface = False
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
            #: A traced call's: a library's code that ran -> its globals,
            #: and the items of the entries served inside it (`library_reads`).
            self.lib_codes: dict[types.CodeType, dict] = {}
            self.lib_items: list[tuple] = []
            #: A library's code that reads one settings entry by a key it is
            #: handed (`library_reads.KEYED_READERS`) -> the keys its starts
            #: read (None for a start whose key could not be read).
            self.lib_keys: dict[types.CodeType, set[str | None]] = {}
            #: User code -> the library modules its closure holds under the
            #: names a chain it reads starts from (`library_reads.deref_roots`).
            self.lib_cells: dict[types.CodeType, dict[str, Any]] = {}
            #: ``(kind, module, dotted)`` of a library setting -> its
            #: fingerprint as the call first read it (`library_reads.first_reads`).
            self.lib_first: dict[tuple[str, str, str], str | None] = {}
            #: A library's method that ran -> the class of the object (or the
            #: class) it first ran on, whose attributes it reads off
            #: ``self``/``cls`` (`library_reads.self_names`).
            self.lib_receivers: dict[types.CodeType, type] = {}
            #: A traced call's: code of a function that writes through its
            #: parameters -> ``(caller code, offset of the call)`` for each
            #: start of it (`_Recorder.site_codes`, `cash.analysis.out_arguments`).
            self.sites: dict[types.CodeType, set[tuple[types.CodeType, int]]] = {}
            #: A traced call's: code of the user's that binds attributes of
            #: its parameters (`out_arguments.binds_parameter_attributes`)
            #: -> ``{parameter: ids of what each start of it bound it to}``
            #: (`_Recorder.attr_codes`, `_AttributeOwners`).
            self.attr_params: dict[types.CodeType, dict[str, set[int]]] = {}
            #: The same starts' objects, held weakly: code -> ``{parameter:
            #: {id: weak reference, or False for an object that takes
            #: none}}``. A binding made only on objects that are all gone
            #: when the call is judged changed nothing a hit skips
            #: (`_OutArguments.excuses`).
            self.attr_targets: dict[types.CodeType, dict[str, dict[int, Any]]] = {}
            #: A strict traced call's (`begin`): ``(module, name)`` of a
            #: global or ``(module, "mod.a.b")`` of an attribute chain the
            #: code that ran reads -> ``(digest, qualname)`` as the first
            #: code reading it started; None when not watched.
            self.global_starts: dict[tuple[str, str], tuple[str, str]] | None = None
            #: A strict traced call's: the code whose every line that ran in
            #: the call was recorded (`_Recorder._arm_lines`), and code ->
            #: the lines of it that ran (`_Recorder._on_line`); None when
            #: lines are not watched. A finding on a line that did not run
            #: is no effect of the call (`_line_unrun`).
            self.line_codes: set[types.CodeType] | None = None
            self.line_ran: dict[types.CodeType, set[int]] | None = None

        def absorb(self, inner: Any) -> None:
            super().absorb(inner)
            for key, callee in getattr(inner, "callees", {}).items():
                self.callees.setdefault(key, callee)

    return TraceUnit


TraceUnit: Any = None


class _CodeIds:
    """Code objects held by identity (see `library_reads.CodeSet`): what a
    set of them keeps of two equal ones is the first."""

    __slots__ = ("_codes",)

    def __init__(self) -> None:
        self._codes: dict[int, types.CodeType] = {}

    def add(self, code: types.CodeType) -> None:
        self._codes[id(code)] = code

    def __iter__(self) -> Iterator[types.CodeType]:
        return iter(list(self._codes.values()))

    def __len__(self) -> int:
        return len(self._codes)

    def __bool__(self) -> bool:
        return bool(self._codes)


class _CodeEvents:
    """Code object -> the local events armed on it, held weakly and by
    identity (see `library_reads.CodeSet`)."""

    __slots__ = ("_held",)

    def __init__(self) -> None:
        self._held: dict[int, tuple[Any, int]] = {}

    def __setitem__(self, code: types.CodeType, events: int) -> None:
        key = id(code)
        held = self._held
        found = held.get(key)
        if found is not None and found[0]() is code:
            held[key] = (found[0], events)
            return

        def gone(ref: Any, key: int = key) -> None:
            entry = held.get(key)
            if entry is not None and entry[0] is ref:
                del held[key]

        held[key] = (_weak.ref(code, gone), events)

    def get(self, code: types.CodeType, default: Any = None) -> Any:
        found = self._held.get(id(code))
        if found is None or found[0]() is not code:
            return default
        return found[1]

    def __contains__(self, code: Any) -> bool:
        found = self._held.get(id(code))
        return found is not None and found[0]() is code

    def __len__(self) -> int:
        return len(self._held)

    def keys(self) -> list[types.CodeType]:
        """The code held, alive."""
        return [code for code in (ref() for ref, _events in list(self._held.values())) if code is not None]

    def __iter__(self) -> Iterator[types.CodeType]:
        return iter(self.keys())


def _recorder_class() -> type:
    from ..analysis.out_arguments import binds_parameter_attributes, writes_through_parameters
    from ..install_paths import is_user_path, known_user_path
    from ..tracking import trace_recorder as rec
    from .library_reads import KEYED_NAMES as library_keyed_names
    from .library_reads import QUIET as library_quiet
    from .library_reads import CodeMap as LibraryCodeMap
    from .library_reads import CodeSet as LibraryCodes
    from .library_reads import counts_file
    from .library_reads import deref_roots as library_deref_roots
    from .library_reads import first_reads as library_first_reads
    from .library_reads import keyed_argument as library_keyed_argument
    from .library_reads import receiver_class as library_receiver_class

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
            #: Is a strict traced call among `active` (`_strict_active`):
            #: kept as `active` changes, read on every start.
            self.strict = False
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
            # By identity (`_CodeEvents`): two code objects compare equal
            # whatever file they come from, and a module imported again
            # from another folder had its code's events kept under the
            # first one's, gone with it.
            self.user_codes: _CodeEvents = _CodeEvents()
            #: The user's top-level functions that write through their
            #: parameters (`out_arguments.writes_through_parameters`): their
            #: start event stays armed while a call is traced, and each start
            #: records where it was called from (`TraceUnit.sites`).
            self.site_codes: weakref.WeakSet[types.CodeType] = _weak.WeakSet()
            #: The user's code that binds attributes of its parameters
            #: (`out_arguments.binds_parameter_attributes`) -> those
            #: parameters: armed the same way, and each start records what
            #: they were bound to (`TraceUnit.attr_params`).
            self.attr_codes: weakref.WeakKeyDictionary[types.CodeType, tuple[str, ...]] = _weak.WeakKeyDictionary()
            #: The code to arm again at the next call's start, by identity:
            #: two code objects compare equal whatever file they come from,
            #: and a set kept a module's code imported before in place of
            #: the code running now, which stayed disabled.
            self.rearm: _CodeIds = _CodeIds()
            #: The user's code that looks names up by a string in its own
            #: globals (`string_lookups`): what it calls is recorded for a
            #: notebook unit, the callable a string found among it.
            self.by_string: weakref.WeakSet[types.CodeType] = _weak.WeakSet()
            #: The user's code with LINE events on for a strict traced call
            #: (`_arm_lines`): each line costs one callback, which disables
            #: it; `line_dirty` holds the code a line of fired in since, armed
            #: again when the next strict call begins.
            self.line_on = LibraryCodes()
            self.line_dirty: set[types.CodeType] = set()
            #: A library's code, armed so each traced call sees it run
            #: (`_arm_library`); its events are in `user_codes` too.
            self.lib_codes = LibraryCodes()
            #: The library's code of `library_reads.KEYED_READERS` seen ->
            #: the argument naming the entry it reads.
            self.keyed_codes = LibraryCodeMap()
            #: LINE callbacks run, for the tests of what watching lines costs.
            self.line_events = 0
            #: id(code) -> (a weak reference to it, is it in `lib_codes`,
            #: in `site_codes`, its `attr_codes` entry, `_start_gen` then):
            #: read on every start of code left armed, which a strict call
            #: does thousands of times (lmfit's ``Parameter._getval``),
            #: without a weak reference made for each look into the three.
            #: `_start_gen` moves after code joins one of them.
            self._start_kinds: dict[int, tuple[Any, bool, bool, tuple[str, ...] | None, int]] = {}
            self._start_gen = 0
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
            mon.register_callback(self.tool, ev.LINE, self._guarded(self._on_line))
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
                    for ev in (mon.events.PY_START, mon.events.PY_RESUME, mon.events.CALL, mon.events.LINE):
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
            if filename.startswith("<frozen "):
                # The interpreter's frozen modules (importlib's bootstrap,
                # ntpath, posixpath): never the user's (`is_user_code_file`).
                return DISABLE
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
                    with untracked():
                        library = code.co_name != "<module>" and counts_file(filename)
                    if library:
                        self._arm_library(code)
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
            if not pool and _LOOKUP_NAMES.intersection(code.co_names) and looks_up_by_name(code):
                events |= mon.events.CALL
            if not pool and self._looks_up_by_string(code, g):
                self.by_string.add(code)
                events |= mon.events.CALL
            self.user_codes[code] = events
            mon.set_local_events(self.tool, code, events | self._line_events(code))
            if not pool and (writes_through_parameters(code) or code.co_name in _LOG_METHOD_NAMES):
                # A log method's sites too: what its callers do with what it
                # returns decides whether a clock read in it is an input
                # (`_LogClock`).
                self.site_codes.add(code)
                if self.active:
                    self._note_site(code, frame)
            bound = () if pool else binds_parameter_attributes(code)
            if bound:
                self.attr_codes[code] = bound
            # After the sets it joined: a start on another thread meanwhile
            # works its kind out again (`_start_kinds`).
            self._start_gen += 1
            if bound and self._strict_active():
                self._note_bound(code, frame, bound, code in self.site_codes)
            if self.active or self.stack:
                if resumed and code.co_flags & rec.RESUMABLE:
                    self._seen_resume(code, g, kind)
                else:
                    self._seen_start(code, frame, g, kind)
            return DISABLE

        def _arm_library(self, code: types.CodeType) -> None:
            """First sight of a library's *code*: armed, so each traced
            call that runs it sees it once (`library_reads`), and noted for
            the calls in progress, whose run of it this is."""
            if code in library_quiet:
                # Seen again after a restart: it reads no setting.
                return
            frame = sys._getframe(rec.WATCHED_FRAME + 1)
            if code not in self.lib_codes:
                mon = sys.monitoring
                events = mon.events.PY_START | mon.events.PY_RESUME
                self.user_codes[code] = events
                self.lib_codes.add(code)
                self._start_gen += 1
                if code.co_qualname in library_keyed_names:
                    arg = library_keyed_argument(code, frame.f_globals)
                    if arg is not None:
                        self.keyed_codes[code] = arg
                mon.set_local_events(self.tool, code, events | self._line_events(code))
            if self.active:
                self._seen_library(code, frame)
                if code in self.keyed_codes:
                    self._note_key(code, frame)

        def _note_key(self, code: types.CodeType, frame: Any) -> None:
            """A start of a library's code that reads the settings entry its
            caller names (`library_reads.KEYED_READERS`): which one, for
            each call in progress."""
            if own_work_inside():
                return
            try:
                key = frame.f_locals.get(self.keyed_codes[code])
            except Exception:  # noqa: BLE001 - unreadable: the whole dict is checked
                key = None
            if not isinstance(key, str):
                key = None
            fresh = []
            for unit in self.active:
                read = unit.lib_keys.setdefault(code, set())
                if key not in read:
                    read.add(key)
                    fresh.append(unit)
            if fresh:
                self._first_reads(fresh, code, frame.f_globals, False, keys={key})

        def _seen_library(self, code: types.CodeType, frame: Any) -> None:
            if own_work_inside():
                own_work_left_out(code)
                return
            if self._for_no_call(frame):
                return
            g = frame.f_globals
            fresh = [unit for unit in self.active if code not in unit.lib_codes]
            cls = library_receiver_class(code, frame) if fresh else None
            for unit in fresh:
                unit.lib_codes[code] = g
                if cls is not None:
                    unit.lib_receivers[code] = cls
            if fresh:
                self._first_reads(fresh, code, g, False, receiver=cls)

        def _first_reads(
            self,
            units: list[Any],
            code: types.CodeType,
            g: dict,
            user: bool,
            cells: Any = None,
            keys: Any = None,
            receiver: type | None = None,
        ) -> None:
            """*code* starts in *units* for the first time: the library
            settings it reads, as they are before it runs, for those that
            have not read them yet (`library_reads.first_reads`). Taken on
            the thread a call runs on only: what another thread runs
            meanwhile (a test's in-process S3 server counting requests,
            moto's ``request_count += 1``) is no read of the call's."""
            me = threading.get_ident()
            units = [unit for unit in units if getattr(unit, "thread", me) == me]
            if not units or self._in_callback:
                return
            self._in_callback = True
            try:
                with untracked():
                    found = library_first_reads(code, g, user, cells, keys, receiver)
            except Exception:  # noqa: BLE001 - not taken: read after the body
                found = ()
            finally:
                self._in_callback = False
            for unit in units:
                first = unit.lib_first
                for place, fp in found:
                    first.setdefault(place, fp)

        def _on_library(self, code: types.CodeType) -> Any:
            if code in library_quiet:
                # Reads no setting (a restart armed it again): left disabled.
                return DISABLE
            if not collecting():
                self.rearm.add(code)
            if self.active:
                frame = sys._getframe(rec.WATCHED_FRAME + 1)
                if self._for_no_call(frame):
                    return None  # left armed: the call's own start of it is still seen
                self._seen_library(code, frame)
                if code in self.keyed_codes:
                    # Each start reads its own entry: left armed while a
                    # call is in progress.
                    self._note_key(code, frame)
                    return None
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

        def begin(self, kind: str, *, watch_globals: bool = False, surface: bool = False) -> Any:
            unit = TraceUnit(kind)
            unit.surface = surface
            if watch_globals:
                unit.global_starts = {}
                unit.line_codes = set()
                unit.line_ran = {}
                self._rearm_lines()
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
                self.strict = any(u.global_starts is not None for u in self.active)
            return unit

        def _rearm(self) -> None:
            rearm, self.rearm = self.rearm, _CodeIds()
            self._rearm_codes(rearm)

        def _rearm_codes(self, rearm: Iterable[types.CodeType]) -> None:
            mon = sys.monitoring
            if not self.rearm_works:
                if rearm:
                    mon.restart_events()
            else:
                for code in rearm:
                    events = self.user_codes.get(code)
                    if events is not None:
                        mon.set_local_events(self.tool, code, 0)
                        mon.set_local_events(self.tool, code, events | self._line_events(code))

        def _line_events(self, code: types.CodeType) -> int:
            """LINE when *code*'s lines are watched (`_arm_lines`): every
            setting of its local events keeps them."""
            return sys.monitoring.events.LINE if code in self.line_on else 0

        def _rearm_lines(self) -> None:
            """A strict traced call begins: arm again the lines that fired
            (and so were disabled) since the last one, so each line of
            watched code that runs in this call is seen; and, while no
            strict call runs, stop watching code judged to hold no finding
            a hit would skip (`note_line_status`)."""
            mon = sys.monitoring
            if not any(u.line_ran is not None for u in self.active):
                for code in [c for c in self.line_on if _LINE_STATUS.get(c) is False]:
                    self.line_on.discard(code)
                    self.line_dirty.discard(code)
                    mon.set_local_events(self.tool, code, self.user_codes.get(code, 0))
            dirty, self.line_dirty = self.line_dirty, set()
            if not dirty:
                return
            if not self.rearm_works:
                mon.restart_events()
                return
            for code in dirty:
                if code in self.line_on:
                    mon.set_local_events(self.tool, code, 0)
                    mon.set_local_events(self.tool, code, self.user_codes.get(code, 0) | mon.events.LINE)

        def _arm_lines(self, code: types.CodeType, units: list[Any]) -> None:
            """*code* starts for the first time in each of *units*, strict
            traced calls: watch its lines, unless a call judged it to hold
            no finding a hit would skip. Armed here, before its first line
            runs, every line of it that runs in those calls is recorded;
            code armed no other way is judged as a whole (`_line_unrun`).
            Code no finding was judged on yet is armed too: its lines cost
            one callback each, once per call."""
            if _LINE_STATUS.get(code) is False:
                return
            if code not in self.line_on:
                self.line_on.add(code)
                mon = sys.monitoring
                mon.set_local_events(self.tool, code, 0)
                mon.set_local_events(self.tool, code, self.user_codes.get(code, 0) | mon.events.LINE)
            for unit in units:
                unit.line_codes.add(code)

        def _on_line(self, code: types.CodeType, line: int) -> Any:
            self.line_events += 1
            for unit in self.active:
                ran = unit.line_ran
                if ran is not None:
                    held = ran.get(code)
                    if held is None:
                        ran[code] = {line}
                    else:
                        held.add(line)
            self.line_dirty.add(code)
            return DISABLE

        def end(self, unit: Any) -> Any:
            if unit.kind != "decorator":
                return super().end(unit)
            with self._lock:
                self.active = [u for u in self.active if u is not unit]
                self.strict = any(u.global_starts is not None for u in self.active)
            if not self._intact():
                unit.markers.add("tool_conflict")
            for t in threading.enumerate():
                if t.ident not in unit.threads_before and t.is_alive() and self._runs_user_code(t):
                    unit.markers.add("thread_alive")
            return unit

        def _start_kind(self, code: types.CodeType) -> tuple[Any, bool, bool, tuple[str, ...] | None, int]:
            """`_start_kinds`' entry for *code*, worked out now."""
            gen = self._start_gen
            try:
                ref = _weak.ref(code)
            except TypeError:
                ref = None
            kind = (ref, code in self.lib_codes, code in self.site_codes, self.attr_codes.get(code), gen)
            if ref is not None:
                if len(self._start_kinds) > 65536:
                    self._start_kinds.clear()
                self._start_kinds[id(code)] = kind
            return kind

        def _on_start(self, code: types.CodeType, offset: int) -> Any:
            kind = self._start_kinds.get(id(code))
            if kind is None or kind[4] != self._start_gen or kind[0] is None or kind[0]() is not code:
                kind = self._start_kind(code)
            if kind[1]:
                return self._on_library(code)
            inside_collection = collecting()
            if not inside_collection:
                self.rearm.add(code)
            active = self.active
            site = active and kind[2]
            bound = kind[3] if self.strict else None
            watched = bool(site or bound)
            if watched:
                # Every start, not the first one: where each was called
                # from decides what its writes change, and what it was
                # handed whose attributes it binds.
                frame = sys._getframe(rec.WATCHED_FRAME)
                if site:
                    self._note_site(code, frame)
                if bound:
                    self._note_bound(code, frame, bound, kind[2])
                if not self.stack:
                    for u in active:
                        if code not in u.codes:
                            break
                    else:
                        return None
            if self.active or self.stack:
                frame = sys._getframe(rec.WATCHED_FRAME)
                if not self.stack and self._for_no_call(frame):
                    return None  # left armed: the call's own start of it is still seen
                g = frame.f_globals
                kind = self._kinds.get(id(g))
                if kind is None or kind[0] is not g:
                    kind = self._classify(g)
                self._seen_start(code, frame, g, kind)
            # Code a garbage collection runs (a ``__del__``) is left armed: a
            # run of it by the unit's own code is told apart (`Unit.collected`).
            return None if inside_collection or watched else DISABLE

        def _for_no_call(self, frame: Any) -> bool:
            """Does this start run on a thread none of the calls in progress
            runs on, that does none of the program's work
            (`trace_recorder.works_for_no_call`)? What pytest-xdist's
            channel thread ran while a call was in progress was taken for
            the call's: its code put in the trace and its class attributes
            in the library settings the entry is checked on, which a hit
            then found moved (or code no scan watches), and the call was
            computed again, or never served from its memo."""
            me = threading.get_ident()
            for unit in self.active:
                if unit.thread == me:
                    return False
            return rec.works_for_no_call(frame)

        def _note_site(self, code: types.CodeType, frame: Any) -> None:
            caller = frame.f_back
            site = (caller.f_code, caller.f_lasti) if caller is not None else (None, -1)
            for unit in self.active:
                unit.sites.setdefault(code, set()).add(site)

        def _strict_active(self) -> bool:
            """Is a strict traced call (`begin` with *watch_globals*) in
            progress: only its check reads `TraceUnit.attr_params`."""
            return self.strict

        def _note_bound(self, code: types.CodeType, frame: Any, params: tuple[str, ...], is_site: bool) -> None:
            """What this start of *code* was handed for *params*, whose
            attributes it binds; and, for an ``__init__`` (*is_site*: one
            in `site_codes`, which notes its sites itself), where it was
            called from, which tells whether its instance is new
            (`out_arguments.site_constructs`)."""
            f_locals = frame.f_locals
            init = code.co_name == "__init__" and not is_site
            site = None
            if init:
                caller = frame.f_back
                site = (caller.f_code, caller.f_lasti) if caller is not None else (None, -1)
            for unit in self.active:
                if unit.global_starts is None:
                    continue
                held = unit.attr_params.get(code)
                if held is None:
                    held = unit.attr_params[code] = {}
                targets = unit.attr_targets.get(code)
                if targets is None:
                    targets = unit.attr_targets[code] = {}
                for name in params:
                    obj = f_locals.get(name)
                    refs = targets.get(name)
                    if refs is not None:
                        ref = refs.get(id(obj))
                        if ref and ref() is obj:
                            # Seen in this unit already: its id is held.
                            continue
                    else:
                        refs = targets[name] = {}
                    held.setdefault(name, set()).add(id(obj))
                    # An id seen before whose object is gone is a new
                    # object: the dead one's reference has nothing to add.
                    try:
                        refs[id(obj)] = _weak.ref(obj)
                    except TypeError:
                        refs[id(obj)] = False
                if init:
                    unit.sites.setdefault(code, set()).add(site)

        def _seen_start(self, code: types.CodeType, frame: Any, g: dict, kind: tuple) -> None:
            if own_work_inside():
                # Run by cash's own pickling of a value, not by the user's
                # program (`cash._own_work`): armed again once that ends.
                own_work_left_out(code)
                return
            if not self.stack and self._for_no_call(frame):
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
            cells = surface_cells = None
            if code.co_freevars and self.active:
                self._closure_modules(code, frame)
            # A generator expression's cells are left out: what they held is
            # read only when `build` digests a function of this code
            # (`trace_items.call_starts`), and no function of a generator
            # expression's code outlives the expression that made it; its
            # cells hold the frame's locals (a ``self``), which cost the most
            # to digest.
            if code.co_freevars and not self._in_callback and not _is_genexpr(code):
                self._in_callback = True
                try:
                    from ..tracking.trace_items import SURFACE, start_cells

                    modes = {getattr(u, "surface", False) for u in units}
                    with untracked():
                        if False in modes:
                            cells = start_cells(frame, code)
                        if True in modes:
                            token = SURFACE.set(True)
                            try:
                                surface_cells = start_cells(frame, code)
                            finally:
                                SURFACE.reset(token)
                finally:
                    self._in_callback = False
            watching = [u for u in self.active if u.global_starts is not None and code not in u.codes]
            if watching and not self._in_callback:
                self._in_callback = True
                try:
                    with untracked():
                        _note_global_starts(watching, code, g, module)
                finally:
                    self._in_callback = False
            if code.co_flags & 0x1:
                lined = [u for u in self.active if u.line_codes is not None and code not in u.codes]
                if lined:
                    self._arm_lines(code, lined)
            inside_collection = collecting()
            fresh = [unit for unit in self.active if code not in unit.codes]
            for unit in units:
                rec.note_collected(unit, code, inside_collection)
                unit.codes.setdefault(code, (g, module))
                started = surface_cells if getattr(unit, "surface", False) else cells
                if started is not None:
                    unit.cells.setdefault(code, started)
                if recv is not None:
                    unit.receivers.setdefault(code, recv if issubclass(type(recv), type) else type(recv))
            fresh = [unit for unit in fresh if hasattr(unit, "lib_first")]
            if fresh:
                cls = None if recv is None else recv if issubclass(type(recv), type) else type(recv)
                self._first_reads(fresh, code, g, True, fresh[0].lib_cells.get(code), receiver=cls)

        def _closure_modules(self, code: types.CodeType, frame: Any) -> None:
            """The library modules *code*'s closure holds that a chain it
            reads starts from (``np.errstate`` with ``np`` captured), for
            `library_reads`."""
            roots = library_deref_roots(code)
            if not roots:
                return
            try:
                f_locals = frame.f_locals
                held = {name: f_locals.get(name) for name in roots}
            except Exception:  # noqa: BLE001 - unreadable locals: nothing followed from them
                return
            held = {k: v for k, v in held.items() if isinstance(v, types.ModuleType)}
            if held:
                for unit in self.active:
                    unit.lib_cells.setdefault(code, held)

        @staticmethod
        def _unregistered_user_code(name: str, code: types.CodeType) -> bool:
            if name == "cash" or name.startswith("cash.") or sys.modules.get(name) is not None:
                return False
            filename = code.co_filename
            return bool(filename) and not is_pseudo_filename(filename) and is_user_path(filename)

        def _on_resume(self, code: types.CodeType, offset: int) -> Any:
            if code in self.lib_codes:
                return self._on_library(code)
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
                    # What `_lookup` reads it by: a module, or the class of
                    # what it looks the name up on. Not the object itself,
                    # which the unit would keep alive until the call ends.
                    target = arg0 if issubclass(type(arg0), (type, types.ModuleType)) else type(arg0)
                    entry = ("getattr", target, name)
                    slot = ("getattr", id(target), name)
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


def begin(*, watch_globals: bool = False, surface: bool = False) -> Any:
    """Start recording a traced call's body; None when it cannot be.
    *watch_globals*: the call picks code by name, so the globals the code
    it runs reads are digested as that code starts (`_note_global_starts`)
    and `build` refuses a call that changed one. *surface*: its trace will
    be built with *narrow* (`build`), so the closure cells of the code that
    starts are digested with classes by their surface."""
    rec = _recorder() if TRACEABLE else None
    return rec.begin("decorator", watch_globals=watch_globals, surface=surface) if rec is not None else None


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


def note_library_items(items: list[tuple]) -> None:
    """An entry whose run read *items* (`library_reads`) was served inside
    the traced calls in progress."""
    rec = _RECORDER
    if rec is None:
        return
    for unit in list(rec.active):
        unit.lib_items.extend(items)


def library_codes(unit: Any) -> dict[types.CodeType, tuple[dict, bool]]:
    """What `library_reads.build` reads of *unit*: the library code that
    ran in it, and the user's."""
    out: dict[types.CodeType, tuple[dict, bool]] = {code: (g, False) for code, g in unit.lib_codes.items()}
    for code, (g, _module) in unit.codes.items():
        out.setdefault(code, (g, True))
    return out


#: code -> does it call ``getattr`` with a name that is not a constant, or
#: import by a name (`looks_up_by_name`).
_BY_NAME: dict[types.CodeType, bool] = {}


#: ``CO_GENERATOR``: a generator expression's code is a generator's.
_CO_GENERATOR = 0x20


def _is_genexpr(code: types.CodeType) -> bool:
    """Is *code* a generator expression's? Its function is made and called
    at once, never bound to a name or stored."""
    return code.co_name == "<genexpr>" and bool(code.co_flags & _CO_GENERATOR)


def looks_up_by_name(code: types.CodeType) -> bool:
    """Does *code* look a name up at run time: ``import_module`` or
    ``__import__`` by name, or ``getattr`` other than called with a constant
    string as its name? ``getattr(obj, "c")`` and ``getattr(obj, "c", d)``
    read ``obj.c``, as the attribute read does, which no event follows; only
    the others are armed for CALL events, so an ndarray subclass's
    ``__array_finalize__`` copying ``getattr(obj, "_opts", {})`` on every
    slice costs the traced call nothing. ``getattr`` read other than as the
    callee of such a call (bound to a name, handed on) counts as a lookup."""
    got = _BY_NAME.get(code)
    if got is not None:
        return got
    got = _by_name(code)
    if len(_BY_NAME) > 4096:
        _BY_NAME.clear()
    _BY_NAME[code] = got
    return got


def _by_name(code: types.CodeType) -> bool:
    if "__import__" in code.co_names or "import_module" in code.co_names:
        return True
    try:
        ins = [i for i in dis.get_instructions(code) if i.opname not in _SKIPPED]
    except Exception:  # noqa: BLE001 - code dis cannot read is armed, as before
        return True
    for k, load in enumerate(ins):
        if load.opname not in ("LOAD_GLOBAL", "LOAD_NAME") or load.argval != "getattr":
            continue
        start = load.positions
        if start is None or start.lineno is None or start.col_offset is None:
            return True
        # The call of it: the first CALL after it that starts where it does
        # (a call in its arguments starts further right; a call of what it
        # returns, ``getattr(o, "f")(x)``, comes after it).
        call = next(
            (
                c
                for c in ins[k + 1 :]
                if c.opname == "CALL"
                and c.positions is not None
                and (c.positions.lineno, c.positions.col_offset) == (start.lineno, start.col_offset)
            ),
            None,
        )
        if call is None or call.arg not in (2, 3):
            return True
        plan = _plan(code, call.offset)
        if plan is None or plan[0] != "const" or not isinstance(plan[1], str):
            return True
    return False


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


def _empty_build(ins: Any) -> bool:
    """Does *ins* build an empty dict, list or tuple: a default such as
    ``getattr(obj, "opts", {})``?"""
    return ins.opname in ("BUILD_MAP", "BUILD_LIST", "BUILD_TUPLE") and ins.arg == 0


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
                    elif before is not None and (_load_plan(previous, 1) is not None or _empty_build(previous)):
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
    if got.startswith("opaque:") and wrapped is not None:
        return _wrapping_cached_digest(value)
    return None if got.startswith("opaque:") else got


#: How many ``__wrapped__`` links `_wrapping_cached_digest` follows.
_WRAPPED_LINKS_MAX = 16


def _wrapping_cached_digest(value: Any) -> str | None:
    """The digest of a function whose ``__wrapped__`` chain reaches a
    ``@cash.cache`` wrapper (``@deprecated_arg`` over ``@cash.cache``), else
    None: the code and bindings of each layer above the cached function, and
    that function's own state (``_cash_state``, what a nested cached call's
    "cached" item holds) -- its code, helpers and the globals they read. By
    value the layers' closures hold the cached wrapper, whose `Cash` cannot
    be digested, so a function importing such a name ran unstored."""
    from .user_code import is_cash_wrapper

    parts: list[str] = []
    link = value
    for _ in range(_WRAPPED_LINKS_MAX):
        if is_cash_wrapper(link):
            try:
                state = link._cash_state()
            except Exception:  # noqa: BLE001 - a state that cannot be built cannot be checked
                return None
            if not isinstance(state, str):
                return None
            parts.append(f"cached:{state}")
            return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]
        nxt = getattr(link, "__wrapped__", None) if type(link) is types.FunctionType else None
        if nxt is None:
            return None
        layer = _code_digest(link.__code__, link)
        if layer is None:
            return None
        parts.append(f"layer:{layer}")
        for cell in link.__closure__ or ():
            try:
                held = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            if held is nxt:
                continue
            got = _digest(held)
            if got is None:
                return None
            parts.append(f"cell:{got}")
        link = nxt
    return None


def _type_alias(ns: Mapping[str, Any] | None, module: str, name: str) -> Any:
    """The ``type`` statement's alias *ns* holds under *name*, made in
    *module*, else None."""
    alias = None if ns is None else ns.get(name)
    if (
        type(alias).__name__ == "TypeAliasType"
        and type(alias).__module__ == "typing"
        and getattr(alias, "__name__", None) == name
        and getattr(alias, "__module__", None) == module
    ):
        return alias
    return None


def _alias_digest(ns: Mapping[str, Any] | None, module: str, name: str) -> str | None:
    """The digest of what the alias *name* stands for (its ``__value__``)."""
    alias = _type_alias(ns, module, name)
    return None if alias is None else _digest(alias.__value__)


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


def _instance_holder(cls: type, holders: list) -> tuple[str, str] | None:
    """``(module, name)`` of a module global holding an instance of *cls*
    (or *cls* itself): the namespaces the call used first, then every
    module of the user's. None when there is none."""
    spaces = [*holders, *(vars(m) for m in list(sys.modules.values()) if _user_module(m))]
    for ns in spaces:
        module = ns.get("__name__")
        if not isinstance(module, str) or _namespace(module) is not ns:
            continue
        for name, value in list(ns.items()):
            if isinstance(name, str) and not name.startswith("__") and (value is cls or type(value) is cls):
                return module, name
    return None


def _type_of_digest(module: str, name: str, kind: str) -> str | None:
    """The digest (*kind* ``surface`` or ``class``) of the class of global
    *name* of *module*, or of that global, a class."""
    from ..tracking.trace_items import class_surface_digest, remembered_class_digest

    ns = _namespace(module)
    if ns is None or name not in ns:
        return None
    held = ns[name]
    cls = held if isinstance(held, type) else type(held)
    return class_surface_digest(cls) if kind == "surface" else remembered_class_digest(cls)


def _type_held_digest(module: str, name: str, qualname: str, attr: str) -> str | None:
    """The digest of attribute *attr* of the class *qualname* in the MRO of
    the class of global *name* of *module* (or of that global, a class)."""
    ns = _namespace(module)
    if ns is None or name not in ns:
        return None
    held = ns[name]
    cls = held if isinstance(held, type) else type(held)
    for klass in cls.__mro__:
        if klass.__qualname__ == qualname and attr in vars(klass):
            return _digest(vars(klass)[attr])
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
        if not isinstance(value, type) or "<locals>" in value.__qualname__:
            continue
        if value.__module__ != module and _find(value.__module__, value.__qualname__) is not value:
            # Defined here and renamed to where it is exported from
            # (``@set_module("pkg")``): kept only where that name finds it.
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
    from ..tracking.trace_items import remembered_class_digest

    parts = []
    for klass in cls.__mro__:
        if _user_class(klass):
            parts.append(remembered_class_digest(klass))
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
        mod = sys.modules.get(module)
        return _namespace_digest(ns) if _user_module(mod) or any_name_stdlib(mod) else None
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
        path = module_file(mod)
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


def first_import_digest(name: str, was: str | None = None) -> str | None:
    """The file of module *name*, a module of the user's first imported
    inside a unit (`_import_digest`). Not loaded, and its file the one *was*
    recorded, it is imported, as the unit's own import would: the items that
    name its code then find it."""
    if name in sys.modules or was is None:
        return _import_digest(name)
    now = _import_digest(name)
    if now == was and was != _ABSENT:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - not imported: its code items say it is gone
            return None
    return now


class _Build:
    """The items of one traced call (`build`)."""

    def __init__(self, unit: Any, static_codes: set[types.CodeType], strict: bool = True, narrow: bool = False) -> None:
        self.unit = unit
        self.static = static_codes
        self.strict = strict
        #: The key holds the classes of module data by their surface: a
        #: class is digested by its surface (`trace_items.SURFACE`), its
        #: methods that ran by their own code items, and nothing that ran
        #: may be left out (`left_out`).
        self.narrow = narrow
        #: Code that ran was left out, no name holding it (`_unless_unnamed`).
        self.left_out = False
        self.items: dict[tuple, str] = {}
        self.functions: list[Any] = []

    def add(self, item: tuple, digest: str | None, why: str) -> None:
        if digest is None:
            raise _Untraceable(why)
        if self.narrow and item[0] not in _FILE_KINDS:
            digest = _NARROW + digest
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
        # And the namespace of every function that ran: a decorated function
        # it calls by a global name (``check_valid``, a stub
        # ``boltons.funcutils.wraps`` built) holds the decorator's wrapper.
        held = {id(h) for h in self.holders}
        for _code, (g, _module) in unit.codes.items():
            if id(g) not in held and _namespace(_module) is g:
                held.add(id(g))
                self.holders.append(g)
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
                with self._unless_unnamed():
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
            try:
                self._code(code, g, module, ran)
            except _Unnamed:
                if self.strict:
                    raise
                self.left_out = True
                # Left out whole, with the globals it reads: they are found
                # again only through the code, and what it reads may be no
                # value a later process rebuilds the same (a
                # ``dynamic_depends_on=`` lambda reading a dict of sources
                # that carry fresh bytes).
                continue
            recv = unit.receivers.get(code)
            if recv is not None and _user_class(recv):
                with self._unless_unnamed():
                    self._class(recv, ran, receiver=True)
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

    @contextlib.contextmanager
    def _unless_unnamed(self) -> Iterator[None]:
        """Leave out code no name holds, for a call that does not need its
        trace (*strict* off): the key already accepted the call, and each
        item only adds a check, so the rest of the trace still stands --
        dropped whole, it took with it the items that would see an edit
        elsewhere (``PROBLEMS = {"a": lambda v: ...}`` beside a library
        function the call reached). The code left out, and the globals it
        reads, are read through the global that holds it (``PROBLEMS``),
        which the reading code's items check. A call that needs its trace
        is refused as before."""
        try:
            yield
        except _Unnamed:
            if self.strict:
                raise
            self.left_out = True

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
                if _log_call_chain(module, chain):
                    # A logger's method, called for a log line: a hit
                    # skipping it is what caching means (`is_log_line`),
                    # whichever logger the chain finds then.
                    continue
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
            raise _Unnamed(f"it ran {qn} of a {module} no module holds and no running frame shows")
        from ..tracking.trace_items import name_table, resolve

        rc, fn, _owner = resolve(module, qn, g, module)
        if rc is not code:
            raise _Unnamed(f"it ran {module}.{qn} of {filename}, which no name there holds")
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
                # The function that holds the code that ran: for a method a
                # ``@cash.cache`` (or ``functools.wraps``) wrapper replaced on
                # its class, the one it wraps, judged and accepted by that
                # code (`build`), not the wrapper's.
                self.functions.append(next((c for c in candidates(fn) if c.__code__ is code), fn))
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
        if qn == code.co_name and _type_alias(g, module, qn) is not None:
            # A ``type X = ...`` statement's value, built on its first read:
            # all the code does is build ``X.__value__``, so that value
            # stands for it.
            self.add(("alias", module, qn), _alias_digest(g, module, qn), f"the type alias {qn} cannot be checked")
            return
        cls = _holding_class(code, g, module)
        if cls is not None:
            # Made for a class and set on it (``@dataclass``'s ``__init__``
            # and ``__eq__``, built from source of its own): the class's
            # digest holds it, by code and defaults.
            self._class(cls, ran)
            return
        if self._class_held(code):
            return
        raise _Unnamed(f"it ran {module}.{qn}, which no name of {module} holds")

    def _class_held(self, code: types.CodeType) -> bool:
        """Is *code* that of a function held as an attribute of a class a
        method ran on (``compute = staticmethod(stub)`` on an indicator
        class ``type(...)`` built)? Then it is checked by that attribute,
        found again through a module global holding an instance of the
        class (``type_held``): such a class often has no name of its own.
        True when an item was added."""
        from ..tracking.trace_items import candidates

        classes = list(dict.fromkeys(self.unit.receivers.values()))
        for cls in classes:
            for klass in cls.__mro__:
                if not _user_class(klass):
                    continue
                for attr, value in list(vars(klass).items()):
                    if not isinstance(attr, str) or attr.startswith("__") or isinstance(value, type):
                        continue
                    if not any(c.__code__ is code for c in candidates(value)):
                        continue
                    where = _instance_holder(cls, self.holders)
                    if where is None:
                        return False
                    module, name = where
                    self.add(
                        ("type_held", module, name, klass.__qualname__, attr),
                        _type_held_digest(module, name, klass.__qualname__, attr),
                        f"{code.co_qualname} holds values that cannot be checked",
                    )
                    self.functions.extend(c for c in candidates(value) if c.__code__ is code)
                    return True
        return False

    def _class(self, cls: type, ran: set[str], *, receiver: bool = False) -> None:
        from ..tracking.trace_items import class_surface_digest, remembered_class_digest

        qn = cls.__qualname__
        if "<locals>" in qn:
            if qn.rsplit(".<locals>.", 1)[0] in ran:
                return
            raise _Unnamed(f"it ran a method of {qn}, a class no name holds")
        where = _instance_holder(cls, self.holders) if _find(cls.__module__, qn) is not cls else None
        if where is not None:
            # No name finds it (``type(...)`` built it, as xclim builds each
            # indicator's class): found again through a module global that
            # holds an instance of it.
            kind = "surface" if receiver and self.narrow and class_surface_digest(cls) is not None else "class"
            self.add(("type_of", *where, kind), _type_of_digest(*where, kind), "")
            return
        if receiver and self.narrow:
            # The class a method ran on: its methods that ran have code
            # items of their own, so the rest go by name. A class that
            # holds a method made for it (`_holding_class`) is checked by
            # all its code below.
            surface = class_surface_digest(cls)
            if surface is not None:
                self.add(("surface", cls.__module__, qn), surface, "")
                return
        self.add(("class", cls.__module__, qn), remembered_class_digest(cls), "")

    def _lookup(self, kind: str, target: Any, name: str | None) -> None:
        if kind == "import":
            if name is None:
                raise _Untraceable("it imports a module whose name is not a string")
            self.add(("import", name), _import_digest(name), f"the module {name} has no file to check")
            return
        if issubclass(type(target), types.ModuleType):
            module, qualname = target.__name__, ""
            if name is None and not (_user_module(target) or any_name_stdlib(target)):
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


#: Items whose digest is a file's, the same whatever classes it names.
_FILE_KINDS = frozenset({"import", "first_import"})
#: What a digest taken with classes by their surface starts with
#: (`_Build.add`): `verify` digests such an item the same way again.
_NARROW = "n:"


class _Unnamed(_Untraceable):
    """Code ran that no name holds, so no item can find it again: the
    trace leaves it out unless the call needs its trace (`_Build`)."""


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


def import_reads(code: types.CodeType, g: Mapping[str, Any]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """`import_reads_now`, remembered per code object and package: it
    reads nothing else, and a miss's trace asked it of every helper the
    body ran, each disassembled again on every call."""
    package = g.get("__package__") or ""
    key = (code, package)
    found = _IMPORT_READS.get(key)
    if found is None:
        found = _IMPORT_READS[key] = tuple(import_reads_now(code, g))
    return found


#: `import_reads` by ``(code, package)``.
_IMPORT_READS: LruMemo[tuple, tuple] = LruMemo(CODE_OBJECTS)


def import_reads_now(code: types.CodeType, g: Mapping[str, Any]) -> list[tuple[str, str, tuple[str, ...]]]:
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


#: Globals not digested as code starts: what they hold is code, keyed by
#: its own items, not data a call changes in place.
_CODE_TYPES = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)


def _note_global_starts(units: list[Any], code: types.CodeType, g: Mapping[str, Any], module: str) -> None:
    """Digest the globals *code* reads (and the attribute chains it reads
    through a module of the user's) as it starts, for each of *units* that
    has not seen them yet. A hit serves the entry without running the call,
    and the trace's items are digested once it ended: a global the call
    changed in place (``heapq.heappush(HEAP, x)``, ``np.copyto(ARR, x)``)
    is checked by the state the call left it in, and a hit would not
    change it again (`build`)."""
    from ..tracking.trace_items import name_table

    table = name_table(code)
    qn = code.co_qualname
    for name in table.reads:
        if not isinstance(name, str) or name in _SKIP_GLOBALS or name.startswith("__cash"):
            continue
        key = (module, name)
        todo = [u for u in units if key not in u.global_starts]
        if not todo:
            continue
        value = g.get(name, _MISSING)
        if value is _MISSING:
            continue
        if issubclass(type(value), types.ModuleType):
            if _user_module(value):
                for chain in table.chains:
                    head, _, rest = chain.partition(".")
                    if head == name and rest:
                        ckey = (module, chain)
                        cunits = [u for u in units if ckey not in u.global_starts]
                        if cunits:
                            digest = _chain_digest(module, chain)
                            if digest is not None:
                                for u in cunits:
                                    u.global_starts[ckey] = (digest, qn)
            continue
        if issubclass(type(value), _CODE_TYPES):
            continue
        digest = _digest(value)
        if digest is None:
            continue
        for u in todo:
            u.global_starts[key] = (digest, qn)


def _changed_global(items: list[tuple], starts: Mapping[tuple[str, str], tuple[str, str]] | None) -> str | None:
    """Why a strict traced call changed a global its code read, by the
    items `_Build` digested after it against the digests as that code
    started (`_note_global_starts`); None when none moved."""
    if not starts:
        return None
    for item in items:
        if item[0] not in ("global", "attr"):
            continue
        start = starts.get((item[1], item[2]))
        if start is None or start[0] == item[-1]:
            continue
        what = f"{item[1]}.{item[2]}" if item[0] == "global" else item[2]
        return (
            f"it ran {start[1]}, and calling it modifies the module global '{what}' "
            "in place, which a cache hit would not repeat"
        )
    return None


def _log_call_chain(module: str, chain: tuple[str, ...]) -> bool:
    """Is *chain*, read from *module*, a logger's log method
    (``settings._root_logger.info``): named as `is_log_line` names one,
    and, as far as held data reaches, a method of a ``logging.Logger``?"""
    from ..analysis.purity_flow import LOG_METHODS

    if len(chain) < 2 or chain[-1] not in LOG_METHODS or "log" not in chain[-2].lower():
        return False
    obj: Any = sys.modules.get(module)
    from ..tracking.trace_items import inspect_getattr

    for depth, name in enumerate(chain[:-1]):
        if issubclass(type(obj), types.ModuleType):
            try:
                obj = inspect_getattr(obj, name)
            except AttributeError:
                return False
            continue
        held = held_attribute(obj, name)
        if held is NOT_HELD:
            # Past what reads as data (a ``__getattr__``): by its names
            # alone, as the log line in the body is.
            return depth > 0
        obj = held
    return isinstance(obj, logging.Logger)


def _chain_digest(module: str, chain: str) -> str | None:
    from ..tracking.trace_items import inspect_getattr, logger_method_digest

    ns = _namespace(module)
    parts = chain.split(".")
    if ns is None or parts[0] not in ns:
        return _ABSENT
    obj = ns[parts[0]]
    for i, p in enumerate(parts[1:], 2):
        if not issubclass(type(obj), types.ModuleType):
            # Past a module, the attribute the code read where it is data
            # the object holds (``settings.preset.leiden.flavor``): that
            # is all the read took. Anything a getter would compute stops
            # the walk, and the object is digested whole, as before.
            held = held_attribute(obj, p)
            if held is NOT_HELD:
                if i == len(parts):
                    method = logger_method_digest(obj, p)
                    if method is not None:
                        return method
                break
            obj = held
            continue
        try:
            obj = inspect_getattr(obj, p)
        except AttributeError:
            return _ABSENT
    return _digest(obj)


def _refusal_in(functions: Iterable[Any], unit: Any = None, owners: frozenset[int] = frozenset()) -> str | None:
    """Why a function the call ran makes its result unsafe to serve: a
    finding a hit would skip (a side effect, a clock read, eval), or a
    random generator of its own it draws from. A finding in a helper that
    did not run (`_helpers_not_run`), one on a line of a function that ran
    that did not run itself (a branch the call did not take, `_line_unrun`),
    one on a parameter every caller
    handed an object it made itself (`_OutArguments`), and an attribute
    bound on a parameter that every start of its function was handed one
    of *owners* for, is not one: *owners* are the ids of the instances the
    cached call's attributes were recorded on (`attribute_replay`), whose
    verdict stores and replays that binding or refuses it with its own
    reason."""
    from ..analysis.purity_analyzer import get_analyzer
    from ..tracking.randomness.state import capture_reachable_carrier_states

    out = _OutArguments(unit, owners)
    log_clock = _LogClock(out)
    for fn in dict.fromkeys(functions):
        report = get_analyzer().analyze_reached(fn)
        if report.unwalkable:
            return f"it ran {fn.__qualname__}, whose helpers cash could not all walk"
        idle = _helpers_not_run(report, unit)
        refused = [
            issue
            for issue in report.issues
            if issue.kind in _REFUSED_KINDS or (issue.kind == ISSUE_UNTRACKABLE_DEP and not is_name_lookup(issue))
        ]
        families = _families(report, {issue.where for issue in refused})
        for issue in refused:
            if (
                issue.where in idle
                or out.excuses(issue)
                or log_clock.excuses(issue)
                or _line_unrun(issue, families, unit)
            ):
                continue
            return f"it ran {fn.__qualname__}, which {issue.description} (line {issue.line})"
        if capture_reachable_carrier_states(fn):
            return f"it ran {fn.__qualname__}, which can draw from a random generator it holds"
    return None


def _families(report: Any, flagged: set[str]) -> dict[str, set[types.CodeType]]:
    """``qualname_of(helper) -> its code objects``, nested code included (a
    lambda, a nested function, a generator expression), for each function
    of the user's *report* walked whose name is in *flagged*, the helpers
    with a finding a hit would skip; those codes are noted as holding one,
    and every other code walked as holding none (`_LINE_STATUS`), so a
    later strict call watches the lines of the first alone."""
    from ..analysis.helper_code import qualname_of

    out: dict[str, set[types.CodeType]] = {}
    clean: set[types.CodeType] = set()
    for ref, walked in getattr(report, "walked_code", ()) or ():
        fn = ref()
        if type(fn) is not types.FunctionType or not _user_module(sys.modules.get(fn.__module__)):
            continue
        name = qualname_of(fn)
        codes = _nested_codes(walked)
        if name in flagged:
            out.setdefault(name, set()).update(codes)
        else:
            clean |= codes
    flagged_codes = set().union(*out.values()) if out else set()
    for code in flagged_codes:
        _LINE_STATUS[code] = True
    for code in clean - flagged_codes:
        _LINE_STATUS.setdefault(code, False)
    return out


def _nested_codes(codes: Iterable[Any]) -> set[types.CodeType]:
    out: set[types.CodeType] = set()
    stack = [c for c in codes if isinstance(c, types.CodeType)]
    while stack:
        c = stack.pop()
        if c not in out:
            out.add(c)
            stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return out


def _code_lines(code: types.CodeType) -> set[int]:
    return {line for _start, _end, line in code.co_lines() if line is not None}


def _line_unrun(issue: Any, families: Mapping[str, set[types.CodeType]], unit: Any) -> bool:
    """Did the line *issue* is on not run in the strict traced call *unit*
    (`TraceUnit.line_ran`)? Its function ran, but not the branch it is in.

    The line is looked for in the code of the helper the finding names
    (`_families`), nested code included: each code that holds an
    instruction on it and started in the call must have had its lines
    watched from its first start in it (`_Recorder._arm_lines`) and not
    run that one. Anything less sure keeps the finding: a finding about a
    whole function (line 0), a line no instruction is on (a decorator, a
    constant folded away, a line of a statement spanning several that only
    another line's instructions run), code from another file, or a
    definition not where its file says (an edit since it was loaded)."""
    if unit is None or getattr(unit, "line_ran", None) is None or issue.line <= 0:
        return False
    family = families.get(issue.where)
    if not family:
        return False
    holders = [c for c in family if issue.line in _code_lines(c)]
    if not holders:
        return False
    if issue.filename:
        want = os.path.normcase(os.path.abspath(issue.filename))
        if any(os.path.normcase(os.path.abspath(c.co_filename)) != want for c in holders):
            return False
    for code in holders:
        if code not in unit.codes:
            # Never started in the call: none of its lines ran.
            continue
        if code not in unit.line_codes or issue.line in unit.line_ran.get(code, ()):
            return False
    from ..source_reading import code_is_where_it_says

    return all(code_is_where_it_says(c) for c in holders)


def note_line_status(codes: Iterable[types.CodeType]) -> None:
    """*codes* (the code the key covers, or folded) are not judged by a
    strict traced call: their lines are not watched, unless a call judged
    them to hold a finding (`_families`)."""
    for code in codes:
        _LINE_STATUS.setdefault(code, False)


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

    def __init__(self, unit: Any, owners: frozenset[int] = frozenset()) -> None:
        self.unit = unit
        self.owners = owners
        self._by_name: dict[str, Any] | None = None
        self._binders: dict[str, dict[str, set[int]]] | None = None
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
        from ..tracking.trace_items import candidates, resolve

        where = self.unit.codes.get(code)
        if where is None:
            return None
        g, module = where
        _rc, fn, _owner = resolve(module, code.co_qualname, g, module)
        # A method a decorator wrapped (``@_fit_context``) is found as the
        # wrapper: the function is the one whose code ran.
        return next((c for c in candidates(fn) if c.__code__ is code), None) if fn is not None else None

    def _bound(self) -> dict[str, tuple[dict[str, set[int]], dict[str, dict[int, Any]]]]:
        """``qualname_of(fn) -> ({parameter: ids}, {parameter: {id: weak
        reference}})`` for each function the trace recorded binding
        attributes of its parameters."""
        if self._binders is None:
            from ..analysis.helper_code import qualname_of

            self._binders = {}
            targets = getattr(self.unit, "attr_targets", {})
            for code, params in getattr(self.unit, "attr_params", {}).items():
                fn = self._function_of(code)
                if fn is not None:
                    self._binders[qualname_of(fn)] = (params, targets.get(code, {}))
        return self._binders

    def excuses(self, issue: Any) -> bool:
        from ..analysis.out_arguments import attribute_bound_parameter, mutated_parameter

        if self.unit is None:
            return False
        param = attribute_bound_parameter(issue)
        if param is not None:
            params, targets = self._bound().get(issue.where, ({}, {}))
            held = params.get(param)
            if held and self.owners and held <= self.owners:
                return True
            # Every object it was bound on is gone (a clone the call made
            # and dropped, ``clone(est).fit(X)``): nothing after the call
            # can read what the binding set, so a hit skips nothing. One
            # still alive (returned, stored, on a cycle not collected yet)
            # or one no weak reference reaches keeps the refusal.
            refs = targets.get(param)
            if held and refs and len(refs) >= len(held) and all(ref and ref() is None for ref in refs.values()):
                return True
        param = mutated_parameter(issue)
        if param is not None:
            return self._excused(issue.where, param)
        return False

    def _excused(self, where: str, param: str) -> bool:
        key = (where, param)
        if key not in self._judged:
            self._judged[key] = self._judge(where, param)
        return self._judged[key]

    def _judge(self, where: str, param: str) -> bool:
        from ..analysis.out_arguments import site_constructs, site_hands_own

        fn = self._functions().get(where)
        if fn is None:
            return False
        sites = self.unit.sites.get(fn.__code__)
        if not sites:
            return False
        for caller_code, lasti in sites:
            caller = self._function_of(caller_code)
            if caller is None:
                return False
            if not (
                site_hands_own(fn, param, caller, caller_code, lasti)
                or (param == fn.__code__.co_varnames[0] and site_constructs(fn, caller, caller_code, lasti))
            ):
                return False
        return True


class _LogClock:
    """Which clock reads the call sites a traced call ran excuse: one in a
    log method (``Logger.log`` returning the time it logged at) whose every
    start in the call was a log line whose value was dropped, reached only
    other log lines (``start = logg.info(...)``, then ``logg.info("done",
    time=start)``), or was returned by a log method whose own starts are
    the same (`out_arguments.log_site_use`). The time then reaches nothing
    but log output, which a hit skipping is what caching means. Anything
    else, a start no site was recorded for included, keeps the finding."""

    def __init__(self, out: _OutArguments) -> None:
        self.out = out

    def excuses(self, issue: Any) -> bool:
        from ..analysis.out_arguments import clock_read_on

        if issue.kind != ISSUE_AMBIENT_READ or self.out.unit is None:
            return False
        fn = self.out._functions().get(issue.where)
        if fn is None or fn.__code__.co_name not in _LOG_METHOD_NAMES or not clock_read_on(fn, issue.line):
            return False
        return self._only_logged(fn, set())

    def _only_logged(self, fn: types.FunctionType, seen: set[types.CodeType]) -> bool:
        from ..analysis.out_arguments import RETURNED, log_site_use

        if fn.__code__ in seen:
            return True  # already being followed up the chain
        seen.add(fn.__code__)
        sites = self.out.unit.sites.get(fn.__code__)
        if not sites:
            return False
        for caller_code, lasti in sites:
            caller = self.out._function_of(caller_code) if caller_code is not None else None
            use = log_site_use(caller, caller_code, lasti) if caller is not None else None
            if use is None:
                return False
            if use == RETURNED and (
                caller_code.co_name not in _LOG_METHOD_NAMES or not self._only_logged(caller, seen)
            ):
                return False
        return True


def build(
    unit: Any,
    static_codes: set[types.CodeType],
    *,
    strict: bool = True,
    accepted: set[types.CodeType] | None = None,
    owners: frozenset[int] = frozenset(),
    narrow: bool = False,
    also_accepted: Callable[[], set[types.CodeType] | None] | None = None,
) -> tuple[list[tuple] | None, str]:
    """The items of a traced call's finished *unit*, or None and why the
    call's result cannot be served from a trace. *static_codes* are the
    code the key covers, left out. *strict*: the call picks code by name,
    so code with effects a hit would skip is refused too; a call the key
    already accepted is not second-guessed for those, nor is *accepted*,
    code the key folded from the call's arguments (a method of an
    argument's class), which a call without the lookup runs as well.
    *owners*: the ids of the instances the call's attribute changes were
    recorded on (`_refusal_in`). *narrow*: the call's key holds the classes
    of module data by their surface only (`HELD_BY_NAME`), so every class
    here is digested by its surface too, and its methods that ran are
    checked by their code items; code that ran and that no name holds
    leaves no trace at all then, since the key no longer stands for it.
    *also_accepted*: more code to accept, asked for only when the strict
    check refused what ran against *accepted* (a fold built late, at a
    cost), and judged again without it."""
    from ..tracking.trace_items import SURFACE, call_starts

    try:
        surface = SURFACE.set(narrow)
        try:
            with call_starts(unit.cells):
                b = _Build(unit, static_codes, strict, narrow)
                items = b.run()
        finally:
            SURFACE.reset(surface)
        if b.left_out and narrow:
            return None, "it ran code no name holds, which its key does not hold all of"
        judged = [f for f in b.functions if f.__code__ not in accepted] if accepted else b.functions
        if strict:
            note_line_status(static_codes)
            if accepted:
                note_line_status(accepted)
        why = _refusal_in(judged, unit, owners) if strict else None
        if why is not None and also_accepted is not None:
            more = also_accepted()
            if more:
                note_line_status(more)
                rest = [f for f in judged if f.__code__ not in more]
                if len(rest) < len(judged):
                    why = _refusal_in(rest, unit, owners)
        if why is None and strict:
            why = _changed_global(items, unit.global_starts)
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
        if not isinstance(owner, type):
            out.update(codes)
            continue
        # A method a library's decorator wrapped is the wrapper's code
        # here, not the method's below it: what the method's helpers do
        # (an attribute bound on a global's object) is judged by the
        # strict check, which a method accepted here would skip.
        out.update(set(codes) & set(code_objects_of(owner, follow_wrapped=False)))
        for base in owner.__mro__[1:]:
            if base is not object and is_user_code_object(base):
                out.update(code_objects_of(base, follow_wrapped=False))
    return out


# -- checking --------------------------------------------------------------
def verify(
    items: list,
    cached_state: Callable[[str], str | None],
    full_state: Callable[[str], str | None] | None = None,
) -> str | None:
    """None when every item of a stored trace still holds its digest, else
    what changed, in a few words. *cached_state* gives a cached function's
    state segment now; *full_state* the same with every class of module
    data keyed by all its code (a ``full_state`` item)."""
    from ..tracking.trace_items import SURFACE

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
            if item[0] == "import":
                continue
            was = item[-1]
            narrow = isinstance(was, str) and was.startswith(_NARROW)
            surface = SURFACE.set(narrow)
            try:
                now, what = _item_now(item, cached_state, full_state)
            finally:
                SURFACE.reset(surface)
            if now is _UNKNOWN:
                return f"an item cash does not know ({item[0]})"
            if now is None or (_NARROW + now if narrow else now) != was:
                return f"{what} changed"
        return None
    except Exception as e:  # noqa: BLE001 - a trace that cannot be checked is not served
        return f"its trace could not be checked ({type(e).__name__})"


_UNKNOWN = object()


def _item_now(
    item: tuple, cached_state: Callable[[str], str | None], full_state: Callable[[str], str | None] | None
) -> tuple[Any, str]:
    """``(the digest *item* names now, what it names)`` for `verify`;
    `_UNKNOWN` for an item of a kind cash does not know."""
    kind = item[0]
    if kind == "first_import":
        # Loaded already, by this process or the check in `verify`: its
        # file is the one the call imported.
        return _import_digest(item[1]), f"the module {item[1]}"
    if kind == "code":
        from ..tracking.trace_items import resolve

        rc, fn, _ = resolve(item[1], item[2], {}, "\x00")
        return (None if rc is None else _code_digest(rc, fn)), f"{item[1]}.{item[2]}"
    if kind in ("file_code", "file_global"):
        ns = _frame_namespace(item[1], item[2])
        if kind == "file_code":
            from ..tracking.trace_items import resolve

            rc, fn, _ = resolve(item[1], item[3], ns, item[1]) if ns is not None else (None, None, None)
            return (None if rc is None else _code_digest(rc, fn)), f"{item[3]} of {item[2]}"
        now = None if ns is None else (_digest(ns[item[3]]) if item[3] in ns else _ABSENT)
        return now, f"the global {item[3]} of {item[2]}"
    if kind in ("held", "global"):
        ns = _namespace(item[1])
        now = None if ns is None else (_digest(ns[item[2]]) if item[2] in ns else _ABSENT)
        return now, (f"{item[1]}.{item[2]}" if kind == "held" else f"the global {item[1]}.{item[2]}")
    if kind == "type_of":
        return _type_of_digest(*item[1:4]), f"the class of {item[1]}.{item[2]}"
    if kind == "type_held":
        return _type_held_digest(*item[1:5]), f"{item[1]}.{item[2]}: {item[3]}.{item[4]}"
    if kind == "alias":
        return _alias_digest(_namespace(item[1]), item[1], item[2]), f"the type alias {item[1]}.{item[2]}"
    if kind in ("class", "surface"):
        from ..tracking.trace_items import class_surface_digest, remembered_class_digest

        cls = _find(item[1], item[2])
        if not isinstance(cls, type):
            now = None
        else:
            now = class_surface_digest(cls) if kind == "surface" else remembered_class_digest(cls)
        return now, f"the class {item[1]}.{item[2]}"
    if kind == "attr":
        return _chain_digest(item[1], item[2]), item[2]
    if kind == "lookup":
        where = f"{item[1]}.{item[2]}" if item[2] else item[1]
        what = f"what {where} holds" + (f" under {item[3]!r}" if item[3] != "*" else "")
        return _lookup_digest(item[1], item[2], item[3]), what
    if kind == "cached":
        return cached_state(item[1]), f"the cached function {item[1]}"
    if kind == "full_state" and full_state is not None:
        return full_state(item[1]), f"code the classes {item[1]} reads from module data hold"
    return _UNKNOWN, kind
