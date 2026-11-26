"""The objects pickled by reference a statement finds by a lookup as it runs.

``seen = logging.getLogger('app').level`` holds no logger in any input: the
statement looks it up by its name, reads it and keeps an int. Its inputs
then say nothing of the logger (`reference_states`), and a change to it
elsewhere (``logging.getLogger('app').setLevel(10)`` in another cell) left
the entry served with the level read before. So while a statement runs,
what a call made from the user's code returns is looked at: an object
pickled by reference that its recipe finds again (`kept_state.looks_up`,
`kept_state.finds_itself`) is recorded with its state as found, as an
input's is, and the entry is served only while each is so.

The calls are seen with ``sys.monitoring``: a CALL
event on the statement's own code, and on each function of the user's
that it calls (`_is_users`), names the callee, and a PY_RETURN event on
the callee's code hands over what it returns. A callee written in C hands
nothing over; the user's code that called it is seen, and what it returns
reaches the statement as a value, which a digest reads. What a module's
first import finds is the module's, as what it registers with the process
is (`process_registries`): ``logging.getLogger('lib').setLevel(10)`` at
its top level is not the statement changing a logger.
"""

from __future__ import annotations

import base64
import builtins
import collections
import contextlib
import dis
import functools
import logging
import sys
import threading
import types
import weakref
from collections.abc import Iterable, Iterator, Mapping
from time import thread_time
from typing import Any

from .. import _weak, kept_state
from .._paths import is_pseudo_filename
from .._type_memo import class_flags, mro_of
from .._user_attrs import attribute_or, own_dict
from ..analysis.analysis_budget import ANALYSIS_BUDGET_FLOOR_S
from ..diagnostics import log_diagnostic, warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning, CashWarning
from ..install_paths import is_cash_path, is_user_path
from ..value_hash import note_known_by_reference, recipes_met
from .process_registries import importing

__all__ = ["available", "collecting", "note", "watching"]

_logger = logging.getLogger(__name__)

#: The warning that what a statement finds by a lookup is not checked here.
CODE_UNSEEN = "NOTEBOOK-LOOKUPS-UNSEEN"

_TOOL_NAME = "cash-lookups"
#: The ids asked for, in turn: the optimizer's, then the profiler's. The
#: debugger's and coverage's are left to them, and cash's own trace takes
#: 3 and 4 (`code_trace`).
_TOOL_IDS = (5, 2)

#: Types whose instances never stand by reference: no reduce is asked.
_PLAIN = frozenset({type(None), bool, int, float, complex, str, bytes, tuple, list, dict, set, frozenset})


class _State(threading.local):
    """Whether a callback runs on this thread: what it runs is cash's own."""

    busy = False
    #: The code whose reads are to be noted, with its globals, while
    #: `_note_reads` runs on this thread: code met on a chain (a property's
    #: getter) is noted after the code being noted, not inside it.
    pending: list[tuple[types.CodeType, Mapping[str, Any]]] | None = None


_THREAD = _State()


class _Watch:
    """One statement's run: the code armed, and the collectors open."""

    def __init__(self) -> None:
        #: Code with local events set: code -> the events set on it.
        self.armed: dict[types.CodeType, int] = {}
        #: ``{recipe key: digest}`` per collector, innermost last: the
        #: statement's, and a kept call's while it runs (`collecting`).
        self.collectors: list[dict[str, str]] = [{}]
        #: The notebook's namespace: what its globals hold is keyed by
        #: their digests already (`reference_states.states`).
        self.user_ns: Mapping[str, Any] | None = None
        #: The functions a call of armed code was seen to call (`_on_call`).
        self.called: set[types.FunctionType] = set()
        #: What the last functions seen called for each code held, as they
        #: were called, newest last (`_called_already`).
        self.made: dict[types.CodeType, list[tuple[Any, ...]]] = {}
        #: The globals, and the attributes, armed code binds or deletes
        #: (`_note_rebinds`).
        self.rebound: set[str] = set()
        self.rebound_attrs: set[str] = set()
        #: The code with a call site switched off since sites were last
        #: switched back on (`_switch_on`).
        self.switched_off: set[types.CodeType] = set()
        #: The notebook's functions not armed yet, by their code: armed when
        #: first started (`_on_start`), with the globals they read.
        self.starts: dict[types.CodeType, Mapping[str, Any]] = {}
        #: The ids of the items of containers noted (`_note_value`), and
        #: whether each may stand by reference: each
        #: reduced once.
        self.noted: dict[int, bool] = {}
        #: ``id -> object`` of the objects whose own attributes were looked
        #: through for what code of their class reads (`_note_held`).
        self.held_read: dict[int, Any] = {}
        #: The reads of containers too big to look through (`_too_big`).
        self.too_big: list[str] = []
        #: The CPU seconds noting may take (`_note_object`), what it took,
        #: and whether it ran past them.
        self.budget = ANALYSIS_BUDGET_FLOOR_S
        self.noting = 0.0
        self.past_budget = False


_ACTIVE: _Watch | None = None
#: The key and digest of what could not be seen: no recipe finds it, and an
#: entry holding it is never served (`reference_states.UNFINDABLE`).
UNSEEN = ""
#: The events set: ``sys.monitoring.events.CALL`` and ``PY_RETURN``, once
#: installed.
_CALL = 0
_RETURN = 0
_START = 0
_LOCK = threading.RLock()
_TOOL: int | None = None
_REASON = ""


def available() -> bool:
    """Can a statement's lookups be seen here (`watching`)?"""
    return _install()


def _install() -> bool:
    """Take a monitoring tool id, or check the one taken is still cash's.
    Ids other tools hold are asked for again each time: one may be freed."""
    global _TOOL, _REASON, _CALL, _RETURN, _START
    if _TOOL is not None:
        if _ours():
            return True
        # Freed by another tool (``sys.monitoring.free_tool_id``), and maybe
        # taken since: asked for again.
        _TOOL = None
    mon = attribute_or(sys, "monitoring", None)
    if mon is None:
        _REASON = "this Python has no sys.monitoring"
        return False
    for tool in _TOOL_IDS:
        if mon.get_tool(tool) is None:
            mon.use_tool_id(tool, _TOOL_NAME)
            mon.register_callback(tool, mon.events.CALL, _on_call)
            mon.register_callback(tool, mon.events.PY_RETURN, _return_callback())
            mon.register_callback(tool, mon.events.PY_START, _on_start)
            _TOOL = tool
            _CALL, _RETURN, _START = mon.events.CALL, mon.events.PY_RETURN, mon.events.PY_START
            _REASON = ""
            return True
    _REASON = f"the monitoring tool ids {' and '.join(map(str, _TOOL_IDS))} are taken by other tools"
    return False


def _return_callback() -> Any:
    """The PY_RETURN callback: `_on_return`, behind the C extension's
    filter (`cwatch.plain_return_filter`) where it is loaded, which answers
    a value of a class in `_PLAIN` itself, as `_on_return` would, without
    calling it. A library function a loop calls (``random.randint`` in a
    simulation) returns millions of plain values, each of which cost a
    Python call; every return still reaches the filter, and every other
    value `_on_return`."""
    from .._native import cwatch

    make = attribute_or(cwatch, "plain_return_filter", None)
    return _on_return if make is None else make(_on_return)


_warned_unseen = False


def _warn_unseen_once() -> None:
    """NOTEBOOK-LOOKUPS-UNSEEN, once per session: what a statement finds
    by a lookup is not checked here (`_REASON`)."""
    global _warned_unseen
    if _warned_unseen:
        return
    _warned_unseen = True
    what = (
        f"cash cannot watch what a statement finds by a lookup as it runs, as {_REASON}. "
        "An object found by its name, such as a logger in "
        "`seen = logging.getLogger('app').level`, is not checked: once another cell "
        "changes it, the statement is served what it read before."
    )
    fix = (
        "bind the object to a variable in a cell of its own "
        "(`log = logging.getLogger('app')`) and read it through that, or run "
        "with the monitoring tool ids 5 or 2 free."
    )
    log_diagnostic(_logger, CODE_UNSEEN, what, fix)
    warn_diagnostic(CashWarning, CODE_UNSEEN, what, fix, location=("<cash>", 0))


def _ours() -> bool:
    """Does the tool id cash took still hold cash's watch? Another tool
    may have freed it, and taken it since."""
    try:
        return _TOOL is not None and sys.monitoring.get_tool(_TOOL) == _TOOL_NAME
    except ValueError:
        return False


#: The reads `_warn_too_big` told of, once each.
_TOO_BIG_SAID: set[str] = set()


def _warn_too_big(read: str) -> None:
    """NOTEBOOK-ITEMS-UNREAD, once for each container read: the statement
    ran uncached, as *read* holds more items than cash looks through for an
    object found by its name."""
    if read in _TOO_BIG_SAID:
        return
    _TOO_BIG_SAID.add(read)
    from ..analysis.handed_callables import ITEM_TYPES_AT

    what = (
        f"this statement reads {read}, which holds more than {ITEM_TYPES_AT:,} items to look through for "
        "objects found by their names (such as loggers): the statement runs every time rather than be "
        "served from the cache"
    )
    fix = (
        "read what the statement needs from it in a cell of its own, or bind it to a variable and read "
        "that; then the statement is cached again."
    )
    log_diagnostic(_logger, "NOTEBOOK-ITEMS-UNREAD", what, fix)
    warn_diagnostic(CashWarning, "NOTEBOOK-ITEMS-UNREAD", what, fix)


_warned_past_budget = False


def _warn_past_budget(watch: _Watch) -> None:
    """NOTEBOOK-ANALYSIS-BUDGET, once per session for this cause: checking
    the objects found by their names the statement met took longer than
    its budget, so it is not served."""
    global _warned_past_budget
    if _warned_past_budget:
        return
    _warned_past_budget = True
    what = (
        f"this statement was not cached: checking the objects found by their names it met (such as "
        f"loggers) took more than its {watch.budget:.1f}s budget (four times what it took the last time it "
        "ran, at least half a second). It runs every time; this is said once."
    )
    fix = (
        "bind the objects it reads to variables in a cell of their own and read them through those, or "
        "wrap the work in a function decorated with @cash.cache."
    )
    log_diagnostic(_logger, "NOTEBOOK-ANALYSIS-BUDGET", what, fix)
    warn_diagnostic(CashCacheIneffectiveWarning, "NOTEBOOK-ANALYSIS-BUDGET", what, fix)


@contextlib.contextmanager
def watching(
    functions: Iterable[Any] = (),
    user_ns: Mapping[str, Any] | None = None,
    budget: float = ANALYSIS_BUDGET_FLOOR_S,
) -> Iterator[dict[str, str] | None]:
    """Within it, a statement runs whose code `arm` arms, and that may call
    the user's *functions* (from a library's code too, as a callback): yields
    the ``{recipe key: digest}`` of each object pickled by reference it
    found -- a call's result, a module's attribute or a global its code
    reads -- with its state when first found; None when nothing can be seen
    here (`available`). Noting them may take *budget* CPU seconds, the
    statement's analysis budget: past it, it is never served
    (NOTEBOOK-ANALYSIS-BUDGET)."""
    global _ACTIVE, _UNWATCHED
    if not _install():
        # Said once, by the first statement that may find one (`arm`).
        _UNWATCHED += 1
        try:
            yield None
        finally:
            _UNWATCHED -= 1
        return
    watch = _Watch()
    watch.user_ns = user_ns
    watch.budget = budget
    with _LOCK:
        outer, _ACTIVE = _ACTIVE, watch
    try:
        for fn in functions:
            if type(fn) is types.FunctionType and _is_users(fn.__code__):
                _arm_users(watch, fn.__code__, fn.__globals__)
        if user_ns is not None:
            _arm_starts(watch, user_ns)
        # A library's function seen returning one: what it returns is seen
        # from wherever it is called, a call site switched off included.
        with _LOCK:
            for code in list(_RETURNERS):
                _set(watch, code, _RETURN)
        yield watch.collectors[0]
    finally:
        with _LOCK:
            _ACTIVE = outer
            if not _ours():
                # The id was freed as it ran: what it found since is unseen.
                watch.collectors[0][UNSEEN] = UNSEEN
            _disarm(watch)
    for read in watch.too_big:
        _warn_too_big(read)
    if watch.past_budget:
        _warn_past_budget(watch)


def arm(units: list[types.CodeType], namespace: Mapping[str, Any]) -> None:
    """Have the calls *units*, a statement's compiled code run in
    *namespace*, make seen, and those of the code nested in it (a
    comprehension, a lambda, a function it defines); and note what the
    globals and module attributes it reads hold now. Nothing when no
    statement is watched; where none can be, the first that calls or reads
    an attribute says so (NOTEBOOK-LOOKUPS-UNSEEN)."""
    watch = _ACTIVE
    if watch is None:
        if _UNWATCHED and not _warned_unseen and any(_may_look_up(code) for unit in units for code in _nested(unit)):
            _warn_unseen_once()
        return
    for unit in units:
        _arm_users(watch, unit, namespace)


#: Statements running unwatched, as nothing can be (`watching`).
_UNWATCHED = 0

#: What a statement does when it may find an object by a lookup: a call, or
#: a read of an attribute.
_LOOKS = frozenset({"CALL", "CALL_KW", "CALL_FUNCTION_EX", "CALL_FUNCTION", "CALL_METHOD", "LOAD_ATTR", "LOAD_METHOD"})


def _may_look_up(code: types.CodeType) -> bool:
    return any(ins.opname in _LOOKS for ins in dis.get_instructions(code))


def _arm_starts(watch: _Watch, user_ns: Mapping[str, Any]) -> None:
    """Have the notebook's functions, and the methods of its classes,
    armed when first started (`_on_start`), from wherever they are called:
    a call site switched off after calling one (`_site_done`) may call
    another next."""
    for value in list(user_ns.values()):
        kind = type(value)
        if kind is types.FunctionType:
            _arm_start(watch, value)
        elif kind is type and _is_users_class(value):
            for held in list(_CLASS_DICT(value).values()):
                fn = held.__func__ if type(held) in (staticmethod, classmethod) else held
                if type(fn) is types.FunctionType:
                    _arm_start(watch, fn)


def _arm_start(watch: _Watch, fn: types.FunctionType) -> None:
    code = fn.__code__
    if code in watch.armed or not _is_users(code):
        return
    watch.starts[code] = fn.__globals__
    with _LOCK:
        if _ACTIVE is watch:
            _set(watch, code, _START)


def _on_start(code: types.CodeType, offset: int) -> Any:
    """A function of the notebook's armed to start (`_arm_starts`) runs:
    its own calls and reads are seen from now on, as a call to it from an
    armed call site would have it."""
    watch = _ACTIVE
    if watch is not None and not _THREAD.busy:
        namespace = watch.starts.pop(code, None)
        if namespace is not None and not watch.armed.get(code, 0) & _CALL:
            try:
                _arm_users(watch, code, namespace)
            except Exception:  # noqa: BLE001 - whatever arming it raises: never served
                _unreadable()
    return sys.monitoring.DISABLE


def _arm_users(watch: _Watch, unit: types.CodeType, namespace: Mapping[str, Any]) -> None:
    """Arm the user's code *unit* and the code nested in it, before it runs:
    the calls it makes are seen, and what the globals it reads in
    *namespace*, and the module attributes it reads through them, hold now
    is noted (`_note_reads`)."""
    for code in _nested(unit):
        if watch.armed.get(code, 0) & _CALL:
            continue
        with _LOCK:
            if _ACTIVE is not watch:
                return
            _note_rebinds(watch, code)
            _set(watch, code, _CALL)
        _note_reads(code, namespace)


def _note_reads(code: types.CodeType, namespace: Mapping[str, Any]) -> None:
    """Note each object pickled by reference that a global *code* reads in
    *namespace* holds, or an attribute of a module it reads through one
    (``logging.root``, ``pkg.conf.LOG``), as each is before it runs. Read
    from the module's own dict: nothing runs (a module's ``__getattr__``
    would)."""
    if _THREAD.busy:
        pending = _THREAD.pending
        if pending is not None:
            # Met on a chain being noted: noted once that one is.
            pending.append((code, namespace))
        return
    _THREAD.busy = True
    pending = _THREAD.pending = [(code, namespace)]
    try:
        watch = _ACTIVE
        while pending:
            code, namespace = pending.pop()
            _note_globals(code, namespace, watch is not None and namespace is watch.user_ns)
    except Exception:  # noqa: BLE001 - whatever reading them raises: never served
        _unreadable()
    finally:
        _THREAD.pending = None
        _THREAD.busy = False


def _note_globals(code: types.CodeType, namespace: Mapping[str, Any], notebooks: bool) -> None:
    """`_note_reads`, unguarded. A global of the notebook's own
    (*notebooks*) is a variable whose digest notes what it holds: only what
    is read through a module of it is noted. From a module, the chain goes
    on through classes and objects (``logging.Logger.manager.loggerDict``);
    a dict, list or tuple met on it has the objects it holds noted, as an
    item or a ``.get`` of it is read by C. Where code decides what an
    attribute on it is (a property, a ``__getattr__``), the chain is
    followed as `_follow` says."""
    for name, attrs in _global_reads(code):
        value = namespace.get(name, _MISSING)
        if value is _MISSING:
            value = vars(builtins).get(name, _MISSING)
        through_module = False
        for depth, attr in enumerate((None, *attrs)):
            if attr is not None:
                if not through_module:
                    if not issubclass(type(value), types.ModuleType):
                        break
                    through_module = True
                value = _follow(value, attr, ".".join((name, *attrs[:depth])), depth == len(attrs))
            if value is _MISSING:
                break
            _arm_returns_of(value)
            if attr is None and notebooks:
                continue
            _note_value(value, ".".join((name, *attrs[:depth])))


def _arm_returns_of(value: Any) -> None:
    """Have what a library's function *value* returns seen, or each one a
    tuple, list or dict *value* holds: a call site switched off after
    calling another (``for g in (fake, fake, logging.getLogger)``), or C
    (``sorted(names, key=lib.find)``), may call it, and no CALL event
    names it then (`_on_call`)."""
    watch = _ACTIVE
    if watch is None:
        return
    kind = type(value)
    if kind is _FUNCTION or kind is _METHOD:
        items: Iterable[Any] = (value,)
    elif kind is tuple or kind is list or kind is dict:
        from ..analysis.handed_callables import ITEM_TYPES_AT

        items = value.values() if kind is dict else value
        if len(items) > ITEM_TYPES_AT or not {_FUNCTION, _METHOD} & set(map(type, items)):
            return
    else:
        return
    for item in list(items):
        if type(item) is _METHOD:
            item = item.__func__
        if type(item) is _FUNCTION and not _is_users(item.__code__):
            _arm_callee(watch, item)


def _note_value(value: Any, read: str) -> None:
    """Note *value*, which the statement reads as *read*, when it stands by
    reference, or what a dict, list, tuple or set *value* holds that does
    (``mod.CFG["a"]["log"]``, ``mod.ITEMS[0].log``), at any depth, as
    `_note_within` looks through it. An item is reduced only when its
    class, read in C, may pickle by reference and is not the user's
    (`_may_note`): no code of the user's runs. Past `ITEM_TYPES_AT` items
    in all, the statement is never served (`_too_big`)."""
    kind = type(value)
    if kind not in _CONTAINERS:
        if kind in _SETTINGS_KINDS:
            _note_settings(value)
            return
        if (_may_note(kind) and _note_object(value)) or _container_base(kind) is None:
            return
    watch = _ACTIVE
    if watch is None or id(value) in watch.held_read:
        return
    watch.held_read[id(value)] = value
    _note_within(watch, value, read)


#: The settings a library keeps in a mapping of its own class, read by
#: subscript (``plt.rcParams["font.size"]``): ``(module, attribute)`` of
#: each. No call of the library's reads them, so nothing else sees the
#: read; the mapping is noted with its state as a logger is
#: (`_note_settings`), found again by its module and name.
SETTINGS_MAPPINGS = (("matplotlib", "rcParams"),)


def _settings_kinds() -> frozenset[type]:
    """The classes of `SETTINGS_MAPPINGS` loaded now: a set lookup decides
    whether a value read may be one."""
    found = set()
    for module, attr in SETTINGS_MAPPINGS:
        held = vars(sys.modules[module]).get(attr) if module in sys.modules else None
        if held is not None:
            found.add(type(held))
    return frozenset(found)


class _SettingsKinds:
    """`_settings_kinds`, read again once a module of `SETTINGS_MAPPINGS`
    is imported after the last read."""

    def __init__(self) -> None:
        self._loaded: tuple[bool, ...] = ()
        self._kinds: frozenset[type] = frozenset()

    def __contains__(self, kind: type) -> bool:
        loaded = tuple(module in sys.modules for module, _attr in SETTINGS_MAPPINGS)
        if loaded != self._loaded:
            self._loaded, self._kinds = loaded, _settings_kinds()
        return kind in self._kinds


_SETTINGS_KINDS = _SettingsKinds()


def _note_settings(value: Any) -> None:
    """Note *value*, when it is one of `SETTINGS_MAPPINGS`, with its state,
    in each collector open, found again by its module and name
    (`kept_state.by_name`): the statement reading it is served only while
    it holds what it held then (`reference_states.changed`)."""
    import pickle

    for module, attr in SETTINGS_MAPPINGS:
        held = sys.modules.get(module)
        if held is None or vars(held).get(attr) is not value:
            continue
        recipe = pickle.dumps((kept_state.by_name, (module, attr)), protocol=4)
        note_known_by_reference(recipe)
        watch = _ACTIVE
        key = base64.b64encode(recipe).decode("ascii")
        missing = [] if watch is None else [found for found in watch.collectors if key not in found]
        if not missing:
            return
        try:
            digest = kept_state.state_digest(value)
        except Exception:  # noqa: BLE001 - whatever pickling it raises: never served
            digest = ""
        for found in missing:
            found[key] = digest
        return


#: ``id -> object`` of the objects a module's container held, found not to
#: stand by reference (`_note_value`), held weakly: each is reduced once
#: while it lives, not on every statement reading the container. One that
#: dies leaves, so its id is never taken for another's.
_PLAIN_ITEMS: weakref.WeakValueDictionary[int, Any] = _weak.WeakValueDictionary()


def forget() -> None:
    """Forget what was found of the objects modules hold, as a fresh kernel
    would (`TrackingState.reset_session_state`)."""
    _PLAIN_ITEMS.clear()
    kept_state.forget_streams()


def _note_item(watch: _Watch, item: Any) -> bool:
    """Note *item* of a container once per statement; whether it may stand
    by reference (past the budget, unknown: it may)."""
    found = watch.noted.get(id(item))
    if found is None:
        found = watch.noted[id(item)] = _note_object(item) or watch.past_budget
    return found


def _too_big(watch: _Watch, read: str) -> None:
    """A container the statement reads, as *read*, holds more than
    `ITEM_TYPES_AT` items: it is never served (`_unreadable`), rather than
    served on what was not looked at, and `watching` says so once it ran."""
    watch.too_big.append(read)
    _unreadable()


def _may_note(kind: type) -> bool:
    """May an object of *kind* found be reduced, to see whether it stands
    by reference? Not one of a class of the user's: its reduce is their
    code."""
    if kind in _PLAIN:
        return False
    try:
        may = _MAY_NOTE[kind]
    except KeyError:
        may = _MAY_NOTE[kind] = kept_state.may_stand_by_reference(kind) and not _is_users_class(kind)
    except TypeError:  # a class whose metaclass makes it unhashable
        may = False
    return may or kept_state.seen_by_reference(kind)


#: ``class -> whether one of it may stand by reference, not the user's``.
_MAY_NOTE: weakref.WeakKeyDictionary[type, bool] = _weak.WeakKeyDictionary()


def _is_users_class(kind: type) -> bool:
    from ..analysis.handed_callables import is_users_class

    return is_users_class(kind)


def _attribute(value: Any, attr: str) -> Any:
    """``value.attr`` as a plain lookup finds it, with no code of the
    object's run: a module's or an object's own dict, then its class's.
    `_MISSING` when a descriptor (a property), a ``__getattr__`` or a
    metaclass would decide what it is."""
    if issubclass(type(value), types.ModuleType):
        return vars(value).get(attr, _MISSING)
    if issubclass(type(value), type):
        return _class_attribute(value, attr) if type(value) is type else _MISSING
    if any(
        _CLASS_DICT(base).get("__getattribute__", _OBJECTS_LOOKUP) not in _GENERIC_GETATTRIBUTE
        for base in mro_of(type(value))
    ):
        return _MISSING
    found = _class_attribute(type(value), attr)
    if found is not _MISSING and _is_descriptor(found):
        return _MISSING
    own = own_dict(value)
    if own is not None and attr in own:
        return own[attr]
    return found


def _class_attribute(klass: type, attr: str) -> Any:
    """*attr* as *klass*'s MRO holds it: `_MISSING` when it is a descriptor,
    or when no class of it holds it."""
    for base in mro_of(klass):
        held = _CLASS_DICT(base)
        if attr in held:
            found = held[attr]
            return _MISSING if _is_descriptor(found) else found
    return _MISSING


def _is_descriptor(value: Any) -> bool:
    return any("__get__" in _CLASS_DICT(base) for base in mro_of(type(value)))


_MISSING = object()
#: A class's ``__dict__`` as ``type`` reads it: no metaclass's code runs.
_CLASS_DICT = type.__dict__["__dict__"].__get__
#: The ``__getattribute__`` slot wrappers that are ``object``'s lookup: a C
#: class that sets it explicitly holds a wrapper of its own for it, which
#: reads attributes as ``object`` does (``SimpleNamespace`` below Python
#: 3.13, ``functools.partial``, and ``tuple``, ``dict`` and ``list``, which a
#: named tuple, a struct sequence such as ``sys.version_info`` and a
#: ``dict`` subclass such as matplotlib's ``rcParams`` reach). Any other
#: decides what a read finds.
_OBJECTS_LOOKUP = object.__getattribute__
_GENERIC_GETATTRIBUTE = frozenset(
    {
        _OBJECTS_LOOKUP,
        *(
            _CLASS_DICT(kind)["__getattribute__"]
            for kind in (
                types.SimpleNamespace,
                functools.partial,
                tuple,
                list,
                dict,
                set,
                int,
                bytes,
                bytearray,
                complex,
                BaseException,
                collections.deque,
                collections.defaultdict,
            )
            if "__getattribute__" in _CLASS_DICT(kind)
        ),
    }
)
#: The lookup of an attribute of a class that ``type`` makes: any other
#: a metaclass defines decides what a read finds.
_TYPES_LOOKUP = _CLASS_DICT(type)["__getattribute__"]

#: How code decides an attribute `_attribute` does not read (`_decided_by`):
#: read after all, without code of the object's run; a function written in
#: Python; a method, which the statement calls next; code written in C; or
#: no attribute there (the read raises).
_READ, _PYTHON, _CALLED, _COMPILED, _ABSENT = range(5)

_MEMBER = type(type.__dict__["__weakrefoffset__"])
_GETSET = type(type.__dict__["__dict__"])
#: The descriptors that make a method when read on an object: called next,
#: so what the call does is seen as a call (`_on_call`).
_METHODS_PYTHON = (types.FunctionType, classmethod, staticmethod)
_METHODS_C = frozenset(
    {
        types.BuiltinFunctionType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
        types.WrapperDescriptorType,
        types.MethodWrapperType,
    }
)


def _follow(value: Any, attr: str, read: str, last: bool) -> Any:
    """``value.attr``, read through a module as *read*, the last of its
    chain when *last*: as `_attribute` reads it, as `_decided_by` reads it
    where that runs none of the object's code, or `_MISSING`. Code that
    decides it is followed as a call is: a library's function has what it
    returns seen; a function of the user's (a property's getter, a
    ``__getattr__``, a method) has its own calls and reads seen, and what
    the object or class of the user's it reads holds noted (`_note_held`).
    Where the chain goes on past what other compiled code decides
    (``mod.local.log.level``), what it finds cannot be seen, and the
    statement is never served (`_cannot_follow`)."""
    found = _attribute(value, attr)
    if found is not _MISSING:
        return found
    how, what = _decided_by(value, attr)
    if how == _READ:
        return what
    watch = _ACTIVE
    if watch is None:
        return _MISSING
    if how == _PYTHON:
        # No CALL event names a getter: armed here, as a callee is (a
        # library's has what it returns seen).
        _arm_callee(watch, what)
    if (how == _PYTHON or how == _CALLED) and what is not None and _is_users(what.__code__):
        if _is_users_class(value if issubclass(type(value), type) else type(value)):
            _note_held(watch, value, read)
    elif how == _COMPILED and not last:
        _cannot_follow(read)
    return _MISSING


def _decided_by(value: Any, attr: str) -> tuple[int, Any]:
    """How ``value.attr`` is found where `_attribute` does not read it, as
    Python's own lookup finds it: ``(_READ, the value)`` where it is read
    with no code of the object's run after all (a slot, an attribute of an
    object shadowing a method, a class whose metaclass looks up as
    ``type`` does); ``(_PYTHON, function)`` for a function written in
    Python that decides it (a property's getter, a ``__getattr__``, a
    ``__getattribute__``); ``(_CALLED, function or None)`` for a method;
    ``(_COMPILED, None)`` for code written in C; ``(_ABSENT, None)``."""
    kind = type(value)
    if issubclass(kind, types.ModuleType):
        return _decided_on_module(value, kind, attr)
    if issubclass(kind, type):
        return _decided_on_class(value, kind, attr)
    hook = _own_lookup(kind, _GENERIC_GETATTRIBUTE)
    if hook is not None:
        return _decider(hook)
    held = _in_mro(kind, attr)
    if held is not _MISSING and _is_data_descriptor(held):
        if type(held) is _GETSET and attr == "__dict__":
            own = own_dict(value)
            return (_COMPILED, None) if own is None else (_READ, own)
        if _reads_a_slot(held, kind):
            try:
                return _READ, held.__get__(value, kind)
            except AttributeError:  # an empty slot: ``__getattr__`` decides
                return _hook(kind)
        return _decided_by_descriptor(held)
    own = own_dict(value)
    if own is not None and attr in own:
        return _READ, own[attr]
    if held is not _MISSING:
        return _decided_by_descriptor(held) if _is_descriptor(held) else (_READ, held)
    return _hook(kind)


_MODULES_LOOKUP = frozenset({_CLASS_DICT(types.ModuleType)["__getattribute__"]})


def _decided_on_module(module: Any, kind: type, attr: str) -> tuple[int, Any]:
    if kind is not types.ModuleType:
        # A module of a class of its own (a lazy loader's): its class decides.
        hook = _own_lookup(kind, _MODULES_LOOKUP)
        if hook is not None:
            return _decider(hook)
        held = _in_mro(kind, attr)
        if held is not _MISSING and _is_data_descriptor(held):
            return _decided_by_descriptor(held)
    if attr in vars(module):
        return _READ, vars(module)[attr]
    hook = vars(module).get("__getattr__", _MISSING)
    return (_ABSENT, None) if hook is _MISSING else _decider(hook)


def _decided_on_class(klass: type, meta: type, attr: str) -> tuple[int, Any]:
    hook = _own_lookup(meta, frozenset({_TYPES_LOOKUP}))
    if hook is not None:
        return _decider(hook)
    on_meta = _in_mro(meta, attr)
    if on_meta is not _MISSING and _is_data_descriptor(on_meta):
        if _reads_a_slot(on_meta, meta):
            try:
                return _READ, on_meta.__get__(klass, meta)
            except AttributeError:
                return _hook(meta)
        return _decided_by_descriptor(on_meta)
    held = _in_mro(klass, attr)
    if held is not _MISSING:
        if not _is_descriptor(held):
            return _READ, held
        kind = type(held)
        if kind is types.FunctionType or kind is property:
            # Read on its class, a function or a property is itself.
            return _READ, held
        if kind is staticmethod:
            return _READ, held.__func__
        return _decided_by_descriptor(held)
    if on_meta is not _MISSING:
        return _decided_by_descriptor(on_meta) if _is_descriptor(on_meta) else (_READ, on_meta)
    return _hook(meta)


def _decided_by_descriptor(held: Any) -> tuple[int, Any]:
    """What decides an attribute *held*, a descriptor, gives."""
    kind = type(held)
    if kind in _METHODS_PYTHON:
        fn = held if kind is types.FunctionType else held.__func__
        return _CALLED, fn if type(fn) is types.FunctionType else None
    if kind in _METHODS_C:
        return _CALLED, None
    getter = _in_mro(kind, "__get__")
    if getter is _PROPERTY_GET and isinstance(held, property):
        return _decider(held.fget)
    if getter is _CACHED_GET and isinstance(held, functools.cached_property):
        return _decider(held.func)
    return _decider(getter)


_PROPERTY_GET = _CLASS_DICT(property)["__get__"]
_CACHED_GET = _CLASS_DICT(functools.cached_property)["__get__"]


def _reads_a_slot(held: Any, owner: type) -> bool:
    """Is *held*, an attribute of an object of class *owner*, a slot, or one
    that Python's own classes, or numpy's, define in C (``__class__``,
    ``__name__``, a struct sequence's field, an array's ``dtype``, ``T`` or
    ``flags``), whose read changes nothing and runs no code of the user's?
    numpy's only on an object of numpy's own class: a view of a subclass's
    runs its ``__array_finalize__``."""
    kind = type(held)
    if kind is _MEMBER:
        return True
    if kind is not _GETSET:
        return False
    module = attribute_or(held.__objclass__, "__module__", None)
    return module == "builtins" or (module == "numpy" and attribute_or(owner, "__module__", None) == "numpy")


def _hook(kind: type) -> tuple[int, Any]:
    """What a class's ``__getattr__`` decides: an attribute not found."""
    hook = _in_mro(kind, "__getattr__")
    return (_ABSENT, None) if hook is _MISSING else _decider(hook)


def _decider(fn: Any) -> tuple[int, Any]:
    if fn is None or fn is _MISSING:  # a property with no getter: the read raises
        return _ABSENT, None
    if type(fn) is types.MethodType:
        fn = fn.__func__
    return (_PYTHON, fn) if type(fn) is types.FunctionType else (_COMPILED, None)


def _own_lookup(kind: type, generic: frozenset[Any]) -> Any:
    """The ``__getattribute__`` the first class of *kind*'s MRO that has one
    defines, when not one of *generic*; else None."""
    for base in mro_of(kind):
        held = _CLASS_DICT(base).get("__getattribute__", _MISSING)
        if held is not _MISSING:
            return None if held in generic else held
    return None


def _in_mro(kind: type, attr: str) -> Any:
    for base in mro_of(kind):
        held = _CLASS_DICT(base)
        if attr in held:
            return held[attr]
    return _MISSING


def _is_data_descriptor(value: Any) -> bool:
    return any("__set__" in held or "__delete__" in held for held in map(_CLASS_DICT, mro_of(type(value))))


def _note_held(watch: _Watch, obj: Any, read: str) -> None:
    """Note each object pickled by reference that *obj*, read as *read*,
    holds, as code of its class the user wrote reads what it holds (a
    property returning ``self._log``, a method reading
    ``self.cfg["log"].level``, ``self.LOG.level`` of its class): its own
    attributes and its class's, looked through as `_note_within` says."""
    if id(obj) in watch.held_read:
        return
    watch.held_read[id(obj)] = obj
    _note_within(watch, obj, read)


def _note_within(watch: _Watch, root: Any, read: str) -> bool:
    """Note each object pickled by reference *root*, read as *read*, holds
    at any depth; whether one was. Looked through (`_looked_through`): the
    items of dicts (their keys too), lists, tuples, sets and deques, and of
    their subclasses; the attributes and slots of ``SimpleNamespace``, of
    objects of the user's classes, and of a library's objects pickled by
    what they hold (``object``'s own reduce: a dataclass), each with no
    code of theirs run; and the class's attributes of an object of the
    user's. A library's object that pickles itself its own way (a frame,
    a figure) is not looked into. Past
    `ITEM_TYPES_AT` items in all, the statement is never served
    (`_too_big`); a container holding only scalars and such containers is
    read at once (`_pure_data`) and not counted. The time it takes counts
    toward the noting budget (`_note_object`)."""
    from ..analysis.handed_callables import ITEM_TYPES_AT

    left = ITEM_TYPES_AT
    seen = {id(root)}
    classes: set[type] = set()
    stack = [root]
    plain = _PLAIN_ITEMS
    found = False
    started, before = thread_time(), watch.noting
    try:
        while stack:
            held = stack.pop()
            if type(held) in _CONTAINERS and _pure_data(held):
                # Numbers, strings and containers of them, read in C:
                # nothing in them stands by reference.
                continue
            items = _held_items(held, classes)
            left -= len(items)
            if left < 0:
                _too_big(watch, read)
                return found
            # Which classes the items are, read in C: nothing of theirs runs.
            if set(map(type, items)) <= _SCALARS:
                continue
            for item in items:
                kind = type(item)
                if kind in _SCALARS or id(item) in seen:
                    continue
                seen.add(id(item))
                if kind in _CONTAINERS:
                    stack.append(item)
                    continue
                if kind in _SETTINGS_KINDS:
                    _note_settings(item)
                    found = True
                    continue
                if _may_note(kind) and plain.get(id(item)) is not item:
                    if _note_item(watch, item):
                        found = True
                        continue
                    try:
                        plain[id(item)] = item
                    except TypeError:  # one that takes no weak reference: reduced each time
                        pass
                if _looked_through(kind):
                    stack.append(item)
        return found
    finally:
        watch.noting = max(watch.noting, before + thread_time() - started)
        if watch.noting > watch.budget and not watch.past_budget:
            watch.past_budget = True
            _unreadable()


#: How many items `_pure_data` reads in all before it answers no: about
#: 25 ms in C.
PURE_DATA_ITEMS = 5_000_000


def _pure_data(value: Any) -> bool:
    """Is *value* exactly a dict, list, tuple, set or frozenset holding
    only scalars (`_SCALARS`) and such containers, keys too, at any depth,
    in at most `PURE_DATA_ITEMS` items in all (a dict's key and value count
    two)? Read in C where the C extension is loaded (`cwatch.pure_data`),
    which answers as `_pure_data_py` does."""
    global _PURE_DATA_C
    if _PURE_DATA_C is None:
        from .._native import cwatch

        _PURE_DATA_C = attribute_or(cwatch, "pure_data", False)
    if _PURE_DATA_C is False:
        return _pure_data_py(value, PURE_DATA_ITEMS)
    return _PURE_DATA_C(value, PURE_DATA_ITEMS)


_PURE_DATA_C: Any = None


def _pure_data_py(value: Any, left: int) -> bool:
    """`_pure_data`, read in Python, with *left* items to read."""
    if type(value) not in _CONTAINERS:
        return False
    stack = [value]
    while stack:
        held = stack.pop()
        items = [*dict.keys(held), *dict.values(held)] if type(held) is dict else held
        left -= len(items)
        if left < 0:
            return False
        for item in items:
            kind = type(item)
            if kind in _SCALARS:
                continue
            if kind not in _CONTAINERS:
                return False
            stack.append(item)
    return True


#: The classes whose instances hold their items as these do: read with
#: their own methods (`_held_items`).
_CONTAINER_BASES = (dict, list, tuple, set, frozenset, collections.deque)


def _container_base(kind: type) -> type | None:
    """The first of `_CONTAINER_BASES` *kind* is a subclass of, or None."""
    for base in _CONTAINER_BASES:
        if issubclass(kind, base):
            return base
    return None


#: ``class -> whether `_note_within` looks through one`` (`_looked_through`).
_LOOKED_THROUGH: weakref.WeakKeyDictionary[type, bool] = _weak.WeakKeyDictionary()
_HEAPTYPE = 1 << 9


def _looked_through(kind: type) -> bool:
    """Does `_note_within` look through an object of *kind*: a container
    or a subclass of one, a ``SimpleNamespace``, or a class defined in
    Python, not a class or module, that is the user's or pickles what it
    holds (`_pickles_its_attributes`)?"""
    try:
        return _LOOKED_THROUGH[kind]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    if kind in _CONTAINERS or kind is types.SimpleNamespace or _container_base(kind) is not None:
        found = True
    elif issubclass(kind, (type, types.ModuleType)) or not class_flags(kind) & _HEAPTYPE:
        found = False
    else:
        found = _is_users_class(kind) or _pickles_its_attributes(kind)
    _LOOKED_THROUGH[kind] = found
    return found


_OBJECTS_OWN = {name: _CLASS_DICT(object).get(name) for name in ("__reduce_ex__", "__reduce__", "__getstate__")}


def _pickles_its_attributes(kind: type) -> bool:
    """Does an object of *kind* pickle as ``object`` has it, by its
    attributes and slots: no reduce or ``__getstate__`` of its own along its
    MRO but ``object``'s, or the ``__getstate__`` a slotted dataclass is
    given?"""
    for name, objects in _OBJECTS_OWN.items():
        held = _in_mro(kind, name)
        if held is _MISSING or held is objects:
            continue
        if name == "__getstate__" and attribute_or(held, "__module__", None) == "dataclasses":
            continue
        return False
    return True


def _held_items(obj: Any, classes: set[type]) -> list[Any]:
    """What *obj* holds, read with no code of its run: a container's items
    (a dict's keys too, unless all scalars), a class's own attributes along
    its MRO, an object's own attributes and slots, and its class's when it
    is the user's and not in *classes*."""
    kind = type(obj)
    if kind is dict:
        return _mapping_items(obj)
    if kind in _CONTAINERS:
        return list(obj)
    if issubclass(kind, type):
        return _class_items(obj)
    base = _container_base(kind)
    if base is None:
        items = []
    elif base is dict:
        items = _mapping_items(obj)
    else:
        items = list(base.__iter__(obj))
    own = own_dict(obj)
    if own is not None:
        items.extend(own.values())
    for base in mro_of(kind):
        for held in _CLASS_DICT(base).values():
            if type(held) is _MEMBER:
                try:
                    items.append(held.__get__(obj, kind))
                except AttributeError:  # an empty slot
                    pass
    if kind not in classes and _is_users_class(kind):
        # What its methods read as ``self.LOG`` may be its class's.
        classes.add(kind)
        items.extend(_class_items(kind))
    return items


def _mapping_items(held: dict) -> list[Any]:
    """A dict's values, and its keys unless all are scalars, read with
    ``dict``'s own methods."""
    items = list(dict.values(held))
    keys = dict.keys(held)
    if not set(map(type, keys)) <= _SCALARS:
        items.extend(keys)
    return items


def _class_items(klass: type) -> list[Any]:
    """The attributes *klass* and its bases but ``object`` hold."""
    return [held for base in mro_of(klass) if base is not object for held in _CLASS_DICT(base).values()]


#: The reads `_cannot_follow` told of, once each.
_UNFOLLOWED_SAID: set[str] = set()


def _cannot_follow(read: str) -> None:
    """The chain goes on past *read*, an attribute code written in C
    decides: what it finds is not seen, so the statement is never served,
    and NOTEBOOK-LOOKUPS-UNSEEN says so once for each such read."""
    _unreadable()
    if read in _UNFOLLOWED_SAID:
        return
    _UNFOLLOWED_SAID.add(read)
    what = (
        f"this statement reads `{read}` and attributes of what it gives, which compiled code decides: cash "
        "cannot look through it for an object found by its name (such as a logger), so the statement runs "
        "every time rather than be served what it read before"
    )
    fix = "bind what it gives to a variable in a cell of its own and read that; then the statement is cached again."
    log_diagnostic(_logger, CODE_UNSEEN, what, fix)
    warn_diagnostic(CashWarning, CODE_UNSEEN, what, fix)


#: ``code -> ((global name, (attribute, ...)), ...)`` (`_global_reads`).
_READS: weakref.WeakKeyDictionary[types.CodeType, tuple[tuple[str, tuple[str, ...]], ...]] = _weak.WeakKeyDictionary()


def _global_reads(code: types.CodeType) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The globals *code* reads, each with the attributes it then reads of
    it in a chain (``logging.root.level``: ``("logging", ("root",
    "level"))``), from its bytecode. Its nested code is armed on its own."""
    found = _READS.get(code)
    if found is not None:
        return found
    chains: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            current = []
            chains.append((ins.argval, current))
        elif ins.opname == "LOAD_ATTR" and current is not None:
            current.append(ins.argval)
        elif ins.opname not in ("CACHE", "EXTENDED_ARG"):
            current = None
    found = _READS[code] = tuple(
        (name, tuple(attrs)) for name, attrs in dict.fromkeys((name, tuple(attrs)) for name, attrs in chains)
    )
    return found


def _nested(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _nested(const)


def _set(watch: _Watch, code: types.CodeType, events: int) -> None:
    """Set *events* on *code*. When it cannot be (the tool id was freed as
    the statement ran), what the run finds is not all seen: never served."""
    held = watch.armed.get(code, 0)
    if held & events == events:
        return
    watch.armed[code] = held | events
    try:
        sys.monitoring.set_local_events(_TOOL, code, held | events)
    except ValueError:
        _unreadable()


def _disarm(watch: _Watch) -> None:
    """Clear the events *watch* set: not when the id is no longer cash's,
    as freeing it cleared them, and they may be another tool's now."""
    if _ours():
        for code in watch.armed:
            try:
                sys.monitoring.set_local_events(_TOOL, code, 0)
            except ValueError:
                break
    watch.armed.clear()


def _on_call(code: types.CodeType, offset: int, callable_: Any, arg0: Any) -> Any:
    """A call made by code armed (`arm`): the returns of a library's
    callee are seen, and the calls and reads of a function of the user's,
    as the statement's own. A site is switched off once it calls a C
    function, which returns nothing seen, or a function seen already in
    the statement, so a loop pays for one or two events. A function seen
    first switches every site switched off back on (`_switch_on`): a site
    that called others may call it next (``for f in (fake,
    logging.getLogger)``). A generator expression made again on each call
    of the function around it, or on each turn of a loop, is one seen
    already (`_called_already`): switching every site back on for each
    made a loop calling that function run several times slower."""
    kind = type(callable_)
    if kind is _METHOD:
        callable_ = callable_.__func__
        kind = type(callable_)
    watch = _ACTIVE
    if watch is None:
        return None
    if kind is not _FUNCTION or callable_ in watch.called or _called_already(watch, callable_):
        watch.switched_off.add(code)
        return sys.monitoring.DISABLE
    if _THREAD.busy:
        return None
    watch.called.add(callable_)
    _keep_made(watch, callable_)
    with _LOCK:
        if _ACTIVE is watch:
            _switch_on(watch)
    try:
        _arm_callee(watch, callable_)
    except Exception:  # noqa: BLE001 - whatever arming it raises: never served
        _unreadable()
    return None


_FUNCTION = types.FunctionType
_METHOD = types.MethodType


def _called_already(watch: _Watch, fn: types.FunctionType) -> bool:
    """Is *fn* a copy of a function seen called already in the statement,
    as a generator expression, a lambda or a nested ``def`` is made again
    on each call of the code around it? It is when it runs the same code
    in the same globals, holds no attribute of its own (a ``__wrapped__``),
    and each variable it closes over, and each default, holds plain data
    (`_data`) or what one of the last seen held when it was called
    (`_keep_made`): whatever it can call through them, that one could, or
    is written in C. Each is compared with what the one seen held then,
    not now: a variable both close over may have been bound to another
    function since. ``mst``'s ``first(... for ... if (A in tree) ^ (B in
    tree))`` in its loop closes over a dict of lists, the same on each
    turn: it is seen once for each call of ``mst``, not on each turn."""
    made = watch.made.get(fn.__code__)
    if not made or fn.__dict__:
        return False
    held = _held(fn)
    if held is None:
        return False
    for seen in reversed(made):
        if _holds_as(held, seen):
            if any(now is not then for now, then in zip(held, seen)):
                # Kept, as the data it holds may grow past `_DATA_ITEMS`.
                _keep(made, held)
            return True
    return False


def _held(fn: types.FunctionType) -> tuple[Any, ...] | None:
    """What *fn* holds that a call of it may call through: its globals,
    defaults and the contents of the variables it closes over, as they are
    now. None when one is empty: never taken for another's."""
    cells = fn.__closure__
    contents: tuple[Any, ...] = ()
    if cells:
        try:
            contents = tuple(cell.cell_contents for cell in cells)
        except ValueError:  # an empty cell
            return None
    kwdefaults = fn.__kwdefaults__
    return (
        fn.__globals__,
        *(fn.__defaults__ or ()),
        *(() if kwdefaults is None else (*kwdefaults, *kwdefaults.values())),
        len(contents),
        *contents,
    )


def _holds_as(held: tuple[Any, ...], seen: tuple[Any, ...]) -> bool:
    """Does *held* hold what *seen* did, or plain data where it did not?
    The globals, first, are the same dict."""
    if len(held) != len(seen) or held[0] is not seen[0]:
        return False
    return all(now is then or _data(now) for now, then in zip(held, seen))


#: How many functions made from one code `_called_already` compares with.
_MADE_KEPT = 8


def _keep_made(watch: _Watch, fn: types.FunctionType) -> None:
    """Keep what *fn*, seen called first, holds (`_held`)."""
    held = _held(fn)
    if held is not None:
        _keep(watch.made.setdefault(fn.__code__, []), held)


def _keep(made: list[tuple[Any, ...]], held: tuple[Any, ...]) -> None:
    made.append(held)
    if len(made) > _MADE_KEPT:
        del made[0]


#: The data `_data` takes: nothing in it can be called but C.
_SCALARS = frozenset({type(None), bool, int, float, complex, str, bytes})
_CONTAINERS = frozenset({list, tuple, set, frozenset, dict})
#: How many items `_data` reads through in the containers a container
#: holds, and how deep: past them, a value is not taken for data, and a
#: function holding it is compared by what it holds (`_called_already`).
_DATA_ITEMS = 1024
_DATA_DEPTH = 3


def _data(value: Any) -> bool:
    """Is *value* a number, a string, or a list, tuple, set or dict of
    them, or of such containers (``{city: [city, ...]}``), at most
    `_DATA_DEPTH` deep and `_DATA_ITEMS` items in those held: nothing a
    call can reach through it runs Python code? Read in C, by exact
    class: nothing of the user's runs."""
    if type(value) in _SCALARS:
        return True
    try:
        return _data_within(value, [_DATA_ITEMS], _DATA_DEPTH)
    except RuntimeError:  # changed in size by another thread as read
        return False


def _data_within(value: Any, left: list[int], depth: int) -> bool:
    kind = type(value)
    if kind in _SCALARS:
        return True
    if kind not in _CONTAINERS:
        return False
    if kind is dict:
        return _items_data(value.keys(), left, depth) and _items_data(value.values(), left, depth)
    return _items_data(value, left, depth)


def _items_data(items: Any, left: list[int], depth: int) -> bool:
    if set(map(type, items)) <= _SCALARS:
        return True
    left[0] -= len(items)
    if depth <= 1 or left[0] < 0:
        return False
    return all(_data_within(item, left, depth - 1) for item in items if type(item) not in _SCALARS)


def _note_rebinds(watch: _Watch, code: types.CodeType) -> None:
    """Note the globals and attributes *code* binds or deletes: the call
    sites switched off already are switched back on, as one may name what
    it binds."""
    names, attrs = _rebinds(code)
    if names <= watch.rebound and attrs <= watch.rebound_attrs:
        return
    watch.rebound |= names
    watch.rebound_attrs |= attrs
    _switch_on(watch)


def _switch_on(watch: _Watch) -> None:
    """Switch the call sites switched off back on: setting a code's events
    again does."""
    switched_off, watch.switched_off = watch.switched_off, set()
    for done in switched_off:
        events = watch.armed.get(done, 0)
        try:
            sys.monitoring.set_local_events(_TOOL, done, 0)
            sys.monitoring.set_local_events(_TOOL, done, events)
        except ValueError:
            _unreadable()


_STORES_NAME = frozenset({"STORE_NAME", "STORE_GLOBAL", "DELETE_NAME", "DELETE_GLOBAL"})
_STORES_ATTR = frozenset({"STORE_ATTR", "DELETE_ATTR"})

#: ``code -> (names it binds, attributes it binds)`` (`_rebinds`).
_REBINDS: weakref.WeakKeyDictionary[types.CodeType, tuple[frozenset[str], frozenset[str]]] = _weak.WeakKeyDictionary()


def _rebinds(code: types.CodeType) -> tuple[frozenset[str], frozenset[str]]:
    found = _REBINDS.get(code)
    if found is None:
        names, attrs = set(), set()
        for ins in dis.get_instructions(code):
            if ins.opname in _STORES_NAME:
                names.add(ins.argval)
            elif ins.opname in _STORES_ATTR:
                attrs.add(ins.argval)
        found = _REBINDS[code] = (frozenset(names), frozenset(attrs))
    return found


def _arm_callee(watch: _Watch, fn: types.FunctionType) -> None:
    """`_on_call`'s arming of the function *fn* calls, unguarded."""
    # A wrapper (the call cache's, a decorator's) and what it wraps: the
    # function under it runs from the wrapper's code, which is not armed.
    for _ in range(_WRAPS_FOLLOWED):
        if type(fn) is not types.FunctionType:
            return
        callee = fn.__code__
        if _is_users(callee):
            # What it finds, its own calls and reads see: its returns are
            # not watched, as a loop calling it would pay for each.
            _arm_users(watch, callee, fn.__globals__)
        elif not watch.armed.get(callee, 0) & _RETURN and not _is_cash(callee):
            # Not cash's own (the call cache's wrapper): what it returns is
            # what the function under it returned, armed below.
            with _LOCK:
                if _ACTIVE is watch:
                    _set(watch, callee, _RETURN)
        fn = attribute_or(fn, "__wrapped__", None)


#: How many wrappers deep `_on_call` follows ``__wrapped__``.
_WRAPS_FOLLOWED = 8


#: ``co_filename -> whether it is cash's own code``.
_CASH: dict[str, bool] = {}


def _is_cash(code: types.CodeType) -> bool:
    path = code.co_filename
    found = _CASH.get(path)
    if found is None:
        found = _CASH[path] = is_cash_path(path)
    return found


#: ``co_filename -> whether it is code the user writes``.
_USERS: dict[str, bool] = {}


def _is_users(code: types.CodeType) -> bool:
    """Is *code* the user's: a cell's (a pseudo-filename, as a statement
    compiles to) or a file `is_user_path` takes for theirs?"""
    path = code.co_filename
    found = _USERS.get(path)
    if found is None:
        found = _USERS[path] = is_pseudo_filename(path) or is_user_path(path)
    return found


def _on_return(code: types.CodeType, offset: int, retval: Any, _type: type = type, _plain: frozenset = _PLAIN) -> Any:
    """A callee armed returned *retval*: noted when it stands by reference,
    or what it holds that does when it is a container or an object
    `_note_within` looks through (``lib.settings()["log"]``,
    ``lib.settings().log``). No return is switched off, as the next value
    it returns may be one (``random.choice``); a value of a class that
    cannot stand by reference costs a set or dict lookup, and a scalar, or
    a tuple, list or dict of them, no call of this function where the C
    extension filters them (`_return_callback`)."""
    kind = _type(retval)
    if kind in _plain:
        if kind in _SCALARS or _all_scalars(retval):
            return None
    elif not _may_note(kind) and not _looked_through(kind):
        return None
    watch = _ACTIVE
    if watch is None or _THREAD.busy or importing():
        # Found by a module's first import: the module's, set up once a
        # process as its registrations are (``process_registries``).
        return None
    _THREAD.busy = True
    try:
        noted = kind not in _plain and _may_note(kind) and _note_object(retval)
        if not noted and (kind in _plain or _looked_through(kind)):
            noted = _note_within(watch, retval, f"what {code.co_qualname}() returns")
    except Exception:  # noqa: BLE001 - whatever reading it raises: never served
        _unreadable()
        return None
    finally:
        _THREAD.busy = False
    if noted:
        _RETURNERS.add(code)
    return None


def _all_scalars(value: Any) -> bool:
    """Does *value*, a container of `_PLAIN`, hold only scalars, keys too?
    Read in C."""
    if type(value) is dict and not set(map(type, value.values())) <= _SCALARS:
        return False
    return set(map(type, value)) <= _SCALARS


#: The code of a library's functions seen returning an object that stands
#: by reference (``logging.getLogger``): armed for each statement.
_RETURNERS: weakref.WeakSet[types.CodeType] = _weak.WeakSet()


def _unreadable() -> None:
    """What the run found could not be read: each collector open holds a
    recipe no lookup serves (`reference_states.UNFINDABLE`), so neither the
    statement nor a kept call around it is served from an entry. A callback
    that raised would raise into the user's code."""
    watch = _ACTIVE
    if watch is not None:
        for found in watch.collectors:
            found[UNSEEN] = UNSEEN


def _note_object(obj: Any) -> bool:
    """Note *obj*, with its state, in each collector open when it stands by
    reference to a lookup; whether it does. What noting takes, in this
    thread's CPU time, is held to the statement's analysis budget
    (`watching`): past it, the statement is never served and nothing more
    is noted."""
    watch = _ACTIVE
    if watch is None:
        return _note_now(obj)
    if watch.past_budget:
        return True
    started = thread_time()
    try:
        return _note_now(obj)
    finally:
        watch.noting += thread_time() - started
        if watch.noting > watch.budget:
            watch.past_budget = True
            _unreadable()


def _note_now(obj: Any) -> bool:
    """`_note_object`, untimed."""
    try:
        reduced = obj.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - one that cannot be reduced is pickled by value or not at all
        return False
    if not kept_state.stands_by_reference(reduced) or not kept_state.looks_up(reduced):
        return False
    kept_state.note_kind_by_reference(type(obj), reduced)
    recipes = recipes_met([(reduced, obj)])
    if not recipes:
        return True
    recipe = recipes[0]
    note_known_by_reference(recipe)
    key = "" if recipe is None else base64.b64encode(recipe).decode("ascii")
    watch = _ACTIVE
    if watch is None:
        return True
    missing = [found for found in watch.collectors if key not in found]
    if not missing:
        return True
    try:
        digest = kept_state.state_digest(obj) if key else ""
    except Exception:  # noqa: BLE001 - whatever its own code raises: not read
        digest = ""
    for found in missing:
        found[key] = digest
    return True


def note(states: Mapping[str, str] | None) -> None:
    """Have each collector open hold *states* as well, those a kept call
    served now recorded: the statement around it reads them."""
    watch = _ACTIVE
    if watch is None or not states:
        return
    for found in watch.collectors:
        for key, digest in states.items():
            found.setdefault(key, digest)


@contextlib.contextmanager
def collecting() -> Iterator[dict[str, str]]:
    """Within it, a kept call runs: yields the ``{recipe key: digest}`` of
    each object pickled by reference it found, with its state when first
    found within it. Empty when no statement is watched."""
    watch = _ACTIVE
    found: dict[str, str] = {}
    if watch is None:
        yield found
        return
    if watch.past_budget:
        # Nothing is noted past the budget: what the call finds is unseen.
        found[UNSEEN] = UNSEEN
    watch.collectors.append(found)
    try:
        yield found
    finally:
        # By identity: an empty dict is equal to the statement's own.
        watch.collectors[:] = [held for held in watch.collectors if held is not found]
