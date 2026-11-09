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
reaches the statement as a value, which a digest reads.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import dis
import sys
import threading
import types
import weakref
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .. import kept_state
from .._paths import is_pseudo_filename
from .._user_attrs import attribute_or
from ..install_paths import is_cash_path, is_user_path
from ..value_hash import HASH_ERRORS, note_known_by_reference, recipes_met

__all__ = ["available", "collecting", "note", "watching"]

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
        self.callables: set[types.FunctionType] = set()


_ACTIVE: _Watch | None = None
#: The events set: ``sys.monitoring.events.CALL`` and ``PY_RETURN``, once
#: installed.
_CALL = 0
_RETURN = 0
_LOCK = threading.RLock()
_TOOL: int | None = None
_REASON = ""


def available() -> bool:
    """Can a statement's lookups be seen here (`watching`)?"""
    return _install()


def _install() -> bool:
    global _TOOL, _REASON, _CALL, _RETURN
    if _TOOL is not None:
        return True
    if _REASON:
        return False
    mon = attribute_or(sys, "monitoring", None)
    if mon is None:
        _REASON = "sys.monitoring needs Python 3.12 or newer"
        return False
    for tool in _TOOL_IDS:
        if mon.get_tool(tool) is None:
            mon.use_tool_id(tool, _TOOL_NAME)
            mon.register_callback(tool, mon.events.CALL, _on_call)
            mon.register_callback(tool, mon.events.PY_RETURN, _on_return)
            _TOOL = tool
            _CALL, _RETURN = mon.events.CALL, mon.events.PY_RETURN
            return True
    _REASON = f"monitoring tool ids {' and '.join(map(str, _TOOL_IDS))} are taken"
    return False


@contextlib.contextmanager
def watching(
    functions: Iterable[Any] = (), user_ns: Mapping[str, Any] | None = None
) -> Iterator[dict[str, str] | None]:
    """Within it, a statement runs whose code `arm` arms, and that may call
    the user's *functions* (from a library's code too, as a callback): yields
    the ``{recipe key: digest}`` of each object pickled by reference it
    found -- a call's result, a module's attribute or a global its code
    reads -- with its state when first found; None when nothing can be seen
    here (`available`)."""
    global _ACTIVE
    if not _install():
        yield None
        return
    watch = _Watch()
    watch.user_ns = user_ns
    with _LOCK:
        outer, _ACTIVE = _ACTIVE, watch
    try:
        for fn in functions:
            if type(fn) is types.FunctionType and _is_users(fn.__code__):
                _arm_users(watch, fn.__code__, fn.__globals__)
        yield watch.collectors[0]
    finally:
        with _LOCK:
            _ACTIVE = outer
            _disarm(watch)


def arm(units: list[types.CodeType], namespace: Mapping[str, Any]) -> None:
    """Have the calls *units*, a statement's compiled code run in
    *namespace*, make seen, and those of the code nested in it (a
    comprehension, a lambda, a function it defines); and note what the
    globals and module attributes it reads hold now. Nothing when no
    statement is watched."""
    watch = _ACTIVE
    if watch is None:
        return
    for unit in units:
        _arm_users(watch, unit, namespace)


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
    is read through a module of it is noted."""
    for name, attrs in _global_reads(code):
        value = namespace.get(name, _MISSING)
        if value is _MISSING:
            value = vars(builtins).get(name, _MISSING)
        for attr in (None, *attrs):
            if attr is not None:
                if not issubclass(type(value), types.ModuleType):
                    break
                value = vars(value).get(attr, _MISSING)
            if value is _MISSING:
                break
            if attr is None and notebooks:
                continue
            if type(value) not in _PLAIN and kept_state.may_stand_by_reference(type(value)):
                _note_object(value)


_MISSING = object()

#: ``code -> ((global name, (attribute, ...)), ...)`` (`_global_reads`).
_READS: weakref.WeakKeyDictionary[types.CodeType, tuple[tuple[str, tuple[str, ...]], ...]] = weakref.WeakKeyDictionary()


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
    held = watch.armed.get(code, 0)
    if held & events == events:
        return
    watch.armed[code] = held | events
    sys.monitoring.set_local_events(_TOOL, code, held | events)


def _disarm(watch: _Watch) -> None:
    for code in watch.armed:
        sys.monitoring.set_local_events(_TOOL, code, 0)
    watch.armed.clear()


def _on_call(code: types.CodeType, offset: int, callable_: Any, arg0: Any) -> Any:
    """A call made by code armed (`arm`): the returns of a library's
    callee are seen, and the calls and reads of a function of the user's,
    as the statement's own. Once per callable in a statement."""
    watch = _ACTIVE
    if watch is None:
        return None
    # A function hashes by its identity: no code of the user's runs.
    fn = callable_.__func__ if type(callable_) is types.MethodType else callable_
    if type(fn) is not types.FunctionType or fn in watch.callables or _THREAD.busy:
        return None
    watch.callables.add(fn)
    try:
        _arm_callee(watch, fn)
    except Exception:  # noqa: BLE001 - whatever arming it raises: never served
        _unreadable()
    return None


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


def _on_return(code: types.CodeType, offset: int, retval: Any) -> Any:
    """A callee armed returned *retval*: noted when it stands by reference."""
    if _ACTIVE is None or _THREAD.busy or type(retval) in _PLAIN:
        return None
    if not kept_state.may_stand_by_reference(type(retval)):
        return None
    _THREAD.busy = True
    try:
        _note_object(retval)
    except Exception:  # noqa: BLE001 - whatever reading it raises: never served
        _unreadable()
    finally:
        _THREAD.busy = False
    return None


def _unreadable() -> None:
    """What the run found could not be read: each collector open holds a
    recipe no lookup serves (`reference_states.UNFINDABLE`), so neither the
    statement nor a kept call around it is served from an entry. A callback
    that raised would raise into the user's code."""
    watch = _ACTIVE
    if watch is not None:
        for found in watch.collectors:
            found[""] = ""


def _note_object(obj: Any) -> None:
    try:
        reduced = obj.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - one that cannot be reduced is pickled by value or not at all
        return
    if not kept_state.stands_by_reference(reduced) or not kept_state.looks_up(reduced):
        return
    recipes = recipes_met([(reduced, obj)])
    if not recipes:
        return
    recipe = recipes[0]
    note_known_by_reference(recipe)
    key = "" if recipe is None else base64.b64encode(recipe).decode("ascii")
    watch = _ACTIVE
    if watch is None:
        return
    missing = [found for found in watch.collectors if key not in found]
    if not missing:
        return
    try:
        digest = kept_state.state_digest(obj) if key else ""
    except (*HASH_ERRORS, MemoryError):
        digest = ""
    for found in missing:
        found[key] = digest


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
    watch.collectors.append(found)
    try:
        yield found
    finally:
        # By identity: an empty dict is equal to the statement's own.
        watch.collectors[:] = [held for held in watch.collectors if held is not found]
