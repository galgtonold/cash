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

The calls are seen with ``sys.monitoring`` (Python 3.12 or newer): a CALL
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
import contextlib
import dis
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
from .._type_memo import mro_of
from .._user_attrs import attribute_or, own_dict
from ..analysis.analysis_budget import ANALYSIS_BUDGET_FLOOR_S
from ..diagnostics import log_diagnostic, warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning, CashWarning
from ..install_paths import is_cash_path, is_user_path
from ..value_hash import HASH_ERRORS, note_known_by_reference, recipes_met
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


_OLD_PYTHON = "sys.monitoring needs Python 3.12 or newer"


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
    if _REASON == _OLD_PYTHON:
        return False
    mon = attribute_or(sys, "monitoring", None)
    if mon is None:
        _REASON = _OLD_PYTHON
        return False
    for tool in _TOOL_IDS:
        if mon.get_tool(tool) is None:
            mon.use_tool_id(tool, _TOOL_NAME)
            mon.register_callback(tool, mon.events.CALL, _on_call)
            mon.register_callback(tool, mon.events.PY_RETURN, _on_return)
            mon.register_callback(tool, mon.events.PY_START, _on_start)
            _TOOL = tool
            _CALL, _RETURN, _START = mon.events.CALL, mon.events.PY_RETURN, mon.events.PY_START
            _REASON = ""
            return True
    _REASON = f"the monitoring tool ids {' and '.join(map(str, _TOOL_IDS))} are taken by other tools"
    return False


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
        "(`log = logging.getLogger('app')`) and read it through that, or run on "
        "Python 3.12 or newer with the monitoring tool ids 5 or 2 free."
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
        return
    _THREAD.busy = True
    try:
        watch = _ACTIVE
        _note_globals(code, namespace, watch is not None and namespace is watch.user_ns)
    except Exception:  # noqa: BLE001 - whatever reading them raises: never served
        _unreadable()
    finally:
        _THREAD.busy = False


def _note_globals(code: types.CodeType, namespace: Mapping[str, Any], notebooks: bool) -> None:
    """`_note_reads`, unguarded. A global of the notebook's own
    (*notebooks*) is a variable whose digest notes what it holds: only what
    is read through a module of it is noted. From a module, the chain goes
    on through classes and objects (``logging.Logger.manager.loggerDict``);
    a dict, list or tuple met on it has the objects it holds noted, as an
    item or a ``.get`` of it is read by C."""
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
                value = _attribute(value, attr)
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
    reference, or what a dict, list or tuple *value* holds that does, one
    level down. An item is reduced only when its class, read in C, may
    pickle by reference and is not the user's (`_may_note`): no code of the
    user's runs. One of more than `ITEM_TYPES_AT` items is not looked
    through: the statement is never served (`_too_big`)."""
    kind = type(value)
    if kind is dict:
        items = value.values()
    elif kind is list or kind is tuple:
        items = value
    else:
        if _may_note(kind):
            _note_object(value)
        return
    watch = _ACTIVE
    if watch is None or not items:
        return
    from ..analysis.handed_callables import ITEM_TYPES_AT

    if len(items) > ITEM_TYPES_AT:
        _too_big(watch, read)
        return
    # Which classes the items are, read in C: nothing of theirs runs.
    met = [kind for kind in set(map(type, items)) if _may_note(kind)]
    if not met:
        return
    plain = _PLAIN_ITEMS
    for item in list(items):
        if type(item) not in met or plain.get(id(item)) is item:
            continue
        if not _note_item(watch, item):
            try:
                plain[id(item)] = item
            except TypeError:  # one that takes no weak reference: reduced each time
                pass


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
    if any("__getattribute__" in _CLASS_DICT(base) for base in mro_of(type(value)) if base is not object):
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
    """A callee armed returned *retval*: noted when it stands by reference.
    No return is switched off, as the next value it returns may be one
    (``random.choice``); a value of a class that cannot stand by reference
    costs a set or dict lookup."""
    kind = _type(retval)
    if kind in _plain or _ACTIVE is None:
        return None
    if not _may_note(kind) or _THREAD.busy or importing():
        # Found by a module's first import: the module's, set up once a
        # process as its registrations are (``process_registries``).
        return None
    _THREAD.busy = True
    try:
        noted = _note_object(retval)
    except Exception:  # noqa: BLE001 - whatever reading it raises: never served
        _unreadable()
        return None
    finally:
        _THREAD.busy = False
    if noted:
        _RETURNERS.add(code)
    return None


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
    except (*HASH_ERRORS, MemoryError):
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
