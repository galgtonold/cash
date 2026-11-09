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

import ast
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
from typing import Any

from .. import kept_state
from .._paths import is_pseudo_filename
from .._type_memo import mro_of
from .._user_attrs import attribute_or, own_dict
from ..diagnostics import log_diagnostic, warn_diagnostic
from ..exceptions import CashWarning
from ..install_paths import is_cash_path, is_user_path
from ..value_hash import HASH_ERRORS, note_known_by_reference, recipes_met

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
        self.callables: set[types.FunctionType] = set()
        #: The globals, and the attributes, armed code binds or deletes: a
        #: call site naming one may call another callable next time, and is
        #: not switched off (`_site_done`).
        self.rebound: set[str] = set()
        self.rebound_attrs: set[str] = set()
        #: The code with a call site switched off since `rebound` last grew.
        self.switched_off: set[types.CodeType] = set()


_ACTIVE: _Watch | None = None
#: The key and digest of what could not be seen: no recipe finds it, and an
#: entry holding it is never served (`reference_states.UNFINDABLE`).
UNSEEN = ""
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
        if _ours():
            return True
        # Freed by another tool (``sys.monitoring.free_tool_id``), and maybe
        # taken since: asked for again.
        _TOOL = None
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
        _warn_unseen_once()
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
            if not _ours():
                # The id was freed as it ran: what it found since is unseen.
                watch.collectors[0][UNSEEN] = UNSEEN
            _disarm(watch)


def arm(units: list[types.CodeType], namespace: Mapping[str, Any], source: str | None = None, shift: int = 0) -> None:
    """Have the calls *units*, a statement's compiled code run in
    *namespace*, make seen, and those of the code nested in it (a
    comprehension, a lambda, a function it defines); and note what the
    globals and module attributes it reads hold now. Nothing when no
    statement is watched. *source* is the text they were compiled from,
    *shift* lines further down (`compiled_source.shifted`): what each call
    in it calls is read from it (`_static_sites`)."""
    watch = _ACTIVE
    if watch is None:
        return
    if source is not None:
        lines = source.encode("utf-8").splitlines(keepends=True)
        for unit in units:
            for code in _nested(unit):
                _SOURCES[code] = (lines, shift)
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
        for attr in (None, *attrs):
            if attr is not None:
                if not through_module:
                    if not issubclass(type(value), types.ModuleType):
                        break
                    through_module = True
                value = _attribute(value, attr)
            if value is _MISSING:
                break
            if attr is None and notebooks:
                continue
            _note_value(value)


#: How many items of a container a chain meets are looked through
#: (`_note_value`): a bigger one's are not noted.
_ITEMS_LOOKED_AT = 10_000


def _note_value(value: Any) -> None:
    """Note *value* when it stands by reference, or what a dict, list or
    tuple *value* holds that does, one level down."""
    kind = type(value)
    if kind is dict:
        if len(value) > _ITEMS_LOOKED_AT:
            return
        items = list(value.values())
    elif kind is list or kind is tuple:
        if len(value) > _ITEMS_LOOKED_AT:
            return
        items = list(value)
    else:
        if kind not in _PLAIN and kept_state.may_stand_by_reference(kind):
            _note_object(value)
        return
    for item in items:
        if type(item) not in _PLAIN and kept_state.may_stand_by_reference(type(item)):
            _note_object(item)


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
    as the statement's own. Once per callable in a statement; a call site
    that can only call the one it called is then switched off
    (`_site_done`)."""
    watch = _ACTIVE
    if watch is None or _THREAD.busy:
        return None
    # A function hashes by its identity: no code of the user's runs.
    fn = callable_.__func__ if type(callable_) is types.MethodType else callable_
    if type(fn) is types.FunctionType and fn not in watch.callables:
        watch.callables.add(fn)
        try:
            _arm_callee(watch, fn)
        except Exception:  # noqa: BLE001 - whatever arming it raises: never served
            _unreadable()
            return None
    return _site_done(watch, code, offset)


def _site_done(watch: _Watch, code: types.CodeType, offset: int) -> Any:
    """Once the callable a call site at *offset* of *code* called is seen to:
    ``sys.monitoring.DISABLE`` when the site names it by a global, or an
    attribute of one (``add``, ``random.randint``, ``p.get``), that no code
    the statement runs of the user's binds (`_note_rebinds`), so it calls
    the same each time and a loop pays for one event. A site whose callee
    is anything else (a local, an item, a call's result) may call another
    each time, and stays."""
    try:
        site = _static_sites(code).get(offset)
    except Exception:  # noqa: BLE001 - a site that cannot be read stays
        return None
    if site is None:
        return None
    root, attrs = site
    if root in watch.rebound or not watch.rebound_attrs.isdisjoint(attrs):
        return None
    # Switched back on when its code's events are set again: by the next
    # statement's arming, or by `_note_rebinds`.
    watch.switched_off.add(code)
    return sys.monitoring.DISABLE


def _note_rebinds(watch: _Watch, code: types.CodeType) -> None:
    """Note the globals and attributes *code* binds or deletes: a call site
    naming one is not switched off, and those switched off already are
    switched back on."""
    names, attrs = _rebinds(code)
    if names <= watch.rebound and attrs <= watch.rebound_attrs:
        return
    watch.rebound |= names
    watch.rebound_attrs |= attrs
    # Setting a code's events again switches its sites back on.
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
_REBINDS: weakref.WeakKeyDictionary[types.CodeType, tuple[frozenset[str], frozenset[str]]] = weakref.WeakKeyDictionary()


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


_CALLS = frozenset({"CALL", "CALL_KW", "CALL_FUNCTION_EX"})
_ROOTS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
#: What may sit among a callee's own instructions without changing what it
#: loads.
_INERT = frozenset({"PUSH_NULL", "NOP", "CACHE", "EXTENDED_ARG"})

#: ``code -> {call offset: (global, (attribute, ...))}`` (`_static_sites`).
_SITES: weakref.WeakKeyDictionary[types.CodeType, dict[int, tuple[str, tuple[str, ...]]]] = weakref.WeakKeyDictionary()


def _static_sites(code: types.CodeType) -> dict[int, tuple[str, tuple[str, ...]]]:
    """The call sites of *code* whose callee is a global, or a chain of
    attributes of one, each with the global and the attributes. Read from
    the source positions of its instructions: a callee starts where its
    call does and ends before it, and is made of loads of that name and
    attributes and nothing else (``a[0].f()``, ``f(x)(y)`` and ``(a or
    b)()`` are not)."""
    found = _SITES.get(code)
    if found is not None:
        return found
    found = {}
    instructions = list(dis.get_instructions(code))
    source = _SOURCES.get(code)
    # A name the code holds as its own variable (a parameter, a loop's)
    # may be another callable at each call.
    own = {*code.co_varnames, *code.co_cellvars, *code.co_freevars}
    for k, call in enumerate(instructions):
        if call.opname in _CALLS:
            site = _callee(instructions, k) if source is None else _callee_in_source(call, *source)
            if site is not None and site[0] not in own:
                found[call.offset] = site
    _SITES[code] = found
    return found


#: ``code -> (the UTF-8 lines it was compiled from, how far down)`` (`arm`).
_SOURCES: weakref.WeakKeyDictionary[types.CodeType, tuple[list[bytes], int]] = weakref.WeakKeyDictionary()

#: The names the call cache routes a call through (`call_interception`):
#: bound by cash for the statement, never by the user's code.
_ROUTING = frozenset({"__cash_call__", "__cash_plain__", "__cash_count__"})


def _callee_in_source(call: dis.Instruction, lines: list[bytes], shift: int) -> tuple[str, tuple[str, ...]] | None:
    """`_static_sites` for *call*, read from the text it was compiled from:
    the call parsed, its callee is a name or a chain of attributes of one,
    as written or as the call cache routes it (``(f if f is
    __cash_plain__[0] and __cash_count__[0]() else __cash_call__(f, 0))``,
    ``__cash_call__(random.randint, 0)``, ``__cash_count__[0]``)."""
    span = _span(call)
    if span is None:
        return None
    (first, col), (last, end) = span
    first, last = first - shift - 1, last - shift - 1
    if not 0 <= first <= last < len(lines):
        return None
    if first == last:
        text = lines[first][col:end]
    else:
        text = lines[first][col:] + b"".join(lines[first + 1 : last]) + lines[last][:end]
    try:
        node = ast.parse(b"(" + text + b")", mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if type(node) is not ast.Call:
        return None
    return _chain(_unrouted(node.func))


def _unrouted(func: ast.expr) -> ast.expr:
    """*func* as the user wrote it, when the call cache routes it."""
    if (
        type(func) is ast.IfExp
        and type(func.body) is ast.Name
        and type(func.orelse) is ast.Call
        and _routes(func.orelse)
        and type(func.orelse.args[0]) is ast.Name
        and func.orelse.args[0].id == func.body.id
        and type(func.test) is ast.BoolOp
        and type(func.test.op) is ast.And
        and len(func.test.values) == 2
        and _is_plain_check(func.test.values[0], func.body.id)
        and _is_slot(func.test.values[1], "__cash_count__", call=True)
    ):
        return func.body
    if type(func) is ast.Call and _routes(func):
        return func.args[0]
    return func


def _routes(node: ast.Call) -> bool:
    """Is *node* ``__cash_call__(callee, <site>)``?"""
    return (
        type(node.func) is ast.Name
        and node.func.id == "__cash_call__"
        and len(node.args) == 2
        and not node.keywords
        and type(node.args[1]) is ast.Constant
    )


def _is_plain_check(node: ast.expr, name: str) -> bool:
    """Is *node* ``name is __cash_plain__[<site>]``?"""
    return (
        type(node) is ast.Compare
        and type(node.left) is ast.Name
        and node.left.id == name
        and len(node.ops) == 1
        and type(node.ops[0]) is ast.Is
        and _is_slot(node.comparators[0], "__cash_plain__")
    )


def _is_slot(node: ast.expr, name: str, call: bool = False) -> bool:
    """Is *node* ``name[<site>]``, or ``name[<site>]()`` with *call*?"""
    if call:
        if type(node) is not ast.Call or node.args or node.keywords:
            return False
        node = node.func
    return (
        type(node) is ast.Subscript
        and type(node.value) is ast.Name
        and node.value.id == name
        and type(node.slice) is ast.Constant
    )


def _chain(func: ast.expr) -> tuple[str, tuple[str, ...]] | None:
    """The global and attributes of a callee written ``name.a.b``, or of
    a routing slot (``__cash_count__[0]``); None for any other."""
    attrs: list[str] = []
    while type(func) is ast.Attribute:
        attrs.append(func.attr)
        func = func.value
    if type(func) is ast.Subscript and not attrs:
        name = func.value.id if type(func.value) is ast.Name else None
        if name in _ROUTING and type(func.slice) is ast.Constant:
            return name, ()
        return None
    if type(func) is not ast.Name:
        return None
    return func.id, tuple(reversed(attrs))


def _span(ins: dis.Instruction) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pos = ins.positions
    if pos is None or None in (pos.lineno, pos.col_offset, pos.end_lineno, pos.end_col_offset):
        return None
    return (pos.lineno, pos.col_offset), (pos.end_lineno, pos.end_col_offset)


def _callee(instructions: list[dis.Instruction], k: int) -> tuple[str, tuple[str, ...]] | None:
    """`_static_sites` for the call at *k*. The names of a call's keyword
    arguments, loaded just before it at its callee's place, are no part of
    the callee."""
    call = _span(instructions[k])
    if call is None:
        return None
    start = call[0]
    first = None
    for j in range(k - 1, -1, -1):
        ins = instructions[j]
        span = _span(ins)
        if ins.opname in _ROOTS and span is not None and span[0] == start and span[1] < call[1]:
            first = j
            break
    if first is None:
        return None
    end = None
    for ins in instructions[first:k]:
        span = _span(ins)
        if span is not None and span[0] == start and span[1] < call[1] and (end is None or span[1] > end):
            end = span[1]
    root = instructions[first].argval
    attrs: list[str] = []
    for j in range(first + 1, k):
        ins = instructions[j]
        span = _span(ins)
        if span is None:
            if ins.opname not in _INERT:
                return None
            continue
        if not (span[0] >= start and span[1] <= end):
            continue  # an argument, or the call's own
        if j == k - 1 and ins.opname == "LOAD_CONST" and instructions[k].opname == "CALL_KW":
            continue
        if ins.opname == "LOAD_ATTR" and span[0] == start:
            attrs.append(ins.argval)
        elif ins.opname not in _INERT:
            return None
    return root, tuple(attrs)


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
            found[UNSEEN] = UNSEEN


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
