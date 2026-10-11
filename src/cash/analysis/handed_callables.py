"""The callables a statement hands to a call, found in the live namespace.

``s.apply(f)`` runs ``f`` as much as ``f(v)`` does, so what ``f`` changes is
what the statement changes. A bare name is the simple case; the callable can
as well be an entry of a dict (``ops['dbl']``), a bound method
(``tracker.record``), an object whose class defines ``__call__``
(``counter``), a closure (``memo``), a function of the user's module
(``helpers.record``), or sit in a list a helper runs (``steps = [f]``).

:func:`handed_callables` maps each argument of each call in a tree to the
values it holds, looked up statically: a name, a chain of attributes
(``inspect.getattr_static``: no property or ``__getattr__`` runs), a
constant subscript of a builtin dict, list or tuple, and the elements of a
list, tuple, set or dict display. A builtin list, tuple, set or dict found
that way is looked into, however deeply nested, where the callee calls what
it is handed there: a function of the user's whose body calls the parameter
or what it holds (``st(x)`` for ``st`` in ``steps``), or a library's
``agg`` / ``aggregate`` / ``transform``. Elsewhere a container is data, and
a large one is not walked on every statement. What it reports for each user
callable:

* its source, for the globals it changes (``source_global_mutations``) and
  the functions it calls by name;
* the names of the notebook's objects calling it changes in place: the
  object a method is bound to when the method changes it, an object whose
  ``__call__`` changes it, a closure that changes what it closes over, and
  the name the argument was read from;
* why it could not be read, for a user callable with no source.

Only the user's own code is followed: a function defined in a cell or in a
local module. A library's callables are not, as a library function's
effects are not anywhere else. Both engines call this with the same
statement and the live namespace.
"""

from __future__ import annotations

import ast
import builtins
import collections
import functools
import inspect
import operator
import sys
import types
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from itertools import chain, compress, repeat
from typing import Any, NamedTuple

from .. import _plain_data
from .._memo import CODE_OBJECTS, LruMemo
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import own_source, wrapper_layers
from ..tracking.function_tracker import is_local_module
from .ast_util import CallScope, calls_in
from .callee_effects import (
    all_param_names,
    callee_global_mutations,
    iterated_sources,
    params_mutated_in_function,
    parse_function_source,
    source_global_mutations,
)
from .mutations import MUTATING_METHODS, chain_is_pure

__all__ = [
    "HandedCallables",
    "handed_callable_values",
    "handed_callables",
    "handed_values",
    "hands_a_state_changing_callable",
    "users_object_attributes",
]


class HandedCallables(NamedTuple):
    """What :func:`handed_callables` found."""

    #: Sources of the user functions handed over (a method's and a
    #: ``__call__``'s included), each once.
    sources: tuple[str, ...] = ()
    #: Names of the notebook's objects the handed callables change in place.
    receivers: frozenset[str] = frozenset()
    #: Why a handed user callable could not be read, one line each.
    unreadable: tuple[str, ...] = ()
    #: The functions whose sources are in `sources`, each with the globals
    #: its names are read from.
    functions: tuple[types.FunctionType, ...] = ()


_NONE = HandedCallables()

#: Containers opened for the callables they hold: their items are read at C
#: speed and no code of the user's runs. A ``UserList`` or ``UserDict`` is
#: read through its ``.data``.
_OPENED = frozenset(
    {
        list,
        tuple,
        set,
        frozenset,
        dict,
        OrderedDict,
        defaultdict,
        collections.deque,
        collections.UserList,
        collections.UserDict,
    }
)
_HELD_IN_DATA = (collections.UserList, collections.UserDict)

#: How many objects of the user's classes one statement's handed arguments
#: may lead through (`_Found.add`). Not a depth: each is opened once. Past
#: it the statement is told the arguments cannot be read, and runs uncached.
_INSTANCE_LIMIT = 10_000


def handed_callables(
    tree: ast.AST | None, namespace: Mapping[str, Any] | None, scope: CallScope = "all"
) -> HandedCallables:
    """The user callables the calls in *tree* (within *scope*) are handed, as
    *namespace* holds them. See the module docstring."""
    if tree is None or not namespace:
        return _NONE
    calls = calls_in(tree, scope)
    if not calls:
        return _NONE
    found = _Found(namespace)
    for call in calls:
        if isinstance(call.func, (ast.Name, ast.Attribute)):
            # ``dbl(3)`` with ``dbl`` a ``functools.wraps`` wrapper: its
            # source, read by name, is the wrapped function's, so what the
            # wrapper's own body does is read here.
            wrapper = _wrapper_called(call.func, namespace)
            if wrapper is not None:
                found.add(*wrapper, False)
            elif isinstance(call.func, ast.Attribute):
                found.class_changed_by(call.func, namespace)
        if not call.args and not call.keywords and not isinstance(call.func, ast.Subscript):
            continue
        calling: _Opened | None = None
        for position, keyword, arg in _arguments(call):
            resolved = _resolve_argument(arg, namespace)
            for value, root in resolved.values:
                opens = False
                if _openable(value):
                    if calling is None:
                        calling = _calling_positions(call, namespace)
                    opens = calling.opens(position, keyword)
                found.add(value, root, opens)
            if resolved.any_of or resolved.unknown:
                if calling is None:
                    calling = _calling_positions(call, namespace)
                if calling.opens(position, keyword):
                    for value, root in resolved.any_of:
                        found.add(value, root, True)
                    for node in resolved.unknown:
                        found.unreadable.append(_cannot_resolve(node, call))
        if isinstance(call.func, ast.Subscript):
            # ``ops['dbl'](v)``: a callable read out of a container is called.
            for value, root in handed_values(call.func, namespace):
                found.add(value, root, False)
    if not (found.sources or found.receivers or found.unreadable):
        return _NONE
    return HandedCallables(
        tuple(found.sources), frozenset(found.receivers), tuple(found.unreadable), tuple(found.functions)
    )


def handed_callable_values(call: ast.Call, namespace: Mapping[str, Any]) -> list[Any]:
    """The callables *call* is handed, as *namespace* holds them: its
    arguments (`handed_values`), the ones inside a builtin container or an
    object of the user's when the callee calls what it is handed there, and
    the functions a lambda among them calls. Functions, methods, classes and
    callable objects, each once; a partial's function."""
    calling: _Opened | None = None
    out: dict[int, Any] = {}
    for position, keyword, arg in _arguments(call):
        resolved = _resolve_argument(arg, namespace)
        candidates = [(value, False) for value, _root in resolved.values]
        if resolved.any_of:
            if calling is None:
                calling = _calling_positions(call, namespace)
            if calling.opens(position, keyword):
                candidates.extend((value, True) for value, _root in resolved.any_of)
        for value, any_of in candidates:
            items = [value]
            if _openable(value):
                if not any_of:
                    if calling is None:
                        calling = _calling_positions(call, namespace)
                    any_of = calling.opens(position, keyword)
                items = _callables_reached(value)[0] if any_of else []
                if callable(value) and not isinstance(value, type):
                    items.append(value)
            for item in items:
                while isinstance(item, functools.partial):
                    item = item.func
                if callable(item) and not isinstance(item, types.BuiltinFunctionType):
                    out.setdefault(id(item), item)
    return list(out.values())


def _arguments(call: ast.Call) -> Iterable[tuple[int | None, str | None, ast.expr]]:
    """``(position, keyword, expression)`` of each argument of *call*; the
    position is None from a ``*args`` on, the keyword None for a ``**kw``."""
    position: int | None = 0
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            position = None
        yield position, None, arg
        if position is not None:
            position += 1
    for kw in call.keywords:
        yield None, kw.arg, kw.value


def _callee_of(func: ast.expr, namespace: Mapping[str, Any]) -> Any:
    if isinstance(func, (ast.Name, ast.Attribute, ast.Subscript)):
        found = _static_value(func, namespace)
        if found is not None:
            return found[0]
    return None


def _calling_positions(call: ast.Call, namespace: Mapping[str, Any]) -> _Opened:
    """The arguments of *call* the callee calls, or calls what is inside
    of: those `_opened_by` opens, and the function argument of a library's
    higher-order call (``s.apply(f)``, ``map(f, rows)``, ``sorted(xs,
    key=f)``)."""
    opened = _opened_by(_callee_of(call.func, namespace), call.func)
    name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
    where = _FUNCTION_ARGUMENT.get(name) if name else None
    if where is None:
        return opened
    if isinstance(where, int):
        return opened._replace(positions=opened.positions | {where}, keywords=opened.keywords | {"func"})
    return opened._replace(keywords=opened.keywords | {where})


def _cannot_resolve(node: ast.expr, call: ast.Call) -> str:
    """The reason given for an argument part *node* of *call* that cannot be
    resolved to what it hands."""
    return (
        f"{_short(node)} is handed to {_short(call.func)}(), and cash cannot tell which function it is "
        "without running it"
    )


def _short(node: ast.expr) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 60 else text[:57] + "..."


def handed_values(arg: ast.expr, namespace: Mapping[str, Any]) -> list[tuple[Any, str | None]]:
    """``(value, root)`` for each value the argument expression *arg* hands
    over that can be found without running code, *root* the name it was read
    from (None for a literal's element built in place). Both branches of a
    conditional (``f if flag else g``) and of ``or``/``and``; ``getattr(obj,
    'name')`` and ``d.get('key')`` read like ``obj.name`` and ``d['key']``;
    and what a lambda among them calls (`_resolve_argument`)."""
    return _resolve_argument(arg, namespace).values


class _Resolved(NamedTuple):
    """What an argument expression hands over (`_resolve_argument`)."""

    #: ``(value, root)`` handed over for certain.
    values: list[tuple[Any, str | None]]
    #: ``(value, root)`` whose items or attributes, any of them, may be what
    #: is handed: ``ops[key]`` with a variable key, ``random.choice(fns)``.
    any_of: list[tuple[Any, str | None]]
    #: Parts that cannot be resolved without running code: the result of a
    #: call of the user's (``make_scorer(2)``), a property of the user's.
    unknown: list[ast.expr]


#: What a constant argument hands over. Never changed: the lists stay empty.
_NOTHING_HANDED = _Resolved([], [], [])


def _resolve_argument(arg: ast.expr, namespace: Mapping[str, Any]) -> _Resolved:
    """What *arg* hands over, read without running code (see `_Resolved`).

    A lambda is transparent: what its body calls, and the names it reads,
    are handed over with it. ``any_of`` and ``unknown`` matter only where the
    callee calls the argument (`_calling_positions`); elsewhere an
    argument is data."""
    kind = type(arg)
    if kind is ast.Name:  # the common case, read without the walk
        value = namespace.get(arg.id, _MISSING)  # type: ignore[attr-defined]
        return _Resolved([] if value is _MISSING else [(value, arg.id)], [], [])  # type: ignore[attr-defined]
    if kind is ast.Constant:
        return _NOTHING_HANDED
    out = _Resolved([], [], [])
    pending: list[ast.expr] = [arg]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Starred, ast.NamedExpr)):
            pending.append(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            pending.extend(node.elts)
        elif isinstance(node, ast.Dict):
            pending.extend(v for v in node.values if v is not None)
            pending.extend(k for k in node.keys if k is not None)
        elif isinstance(node, ast.IfExp):
            pending.extend((node.body, node.orelse))
        elif isinstance(node, ast.BoolOp):
            pending.extend(node.values)
        elif isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            found = _static_value(node, namespace)
            if found is not None:
                out.values.append(found)
            else:
                _unresolved_reference(node, namespace, out, pending)
        elif isinstance(node, ast.Call):
            _resolve_call(node, namespace, out, pending)
        elif isinstance(node, ast.Lambda):
            _lambda_reach(node, namespace, out)
        elif isinstance(node, ast.BinOp):
            pending.extend((node.left, node.right))
        elif isinstance(node, ast.UnaryOp):
            pending.append(node.operand)
        elif not isinstance(node, (ast.Constant, ast.JoinedStr, ast.Compare)):
            out.unknown.append(node)  # a comprehension, an ``await``: built by running code
    return out


def _lambda_reach(node: ast.Lambda, namespace: Mapping[str, Any], out: _Resolved) -> None:
    """What calling the lambda *node* runs: the functions its body calls
    (``tally.record`` in ``lambda v: tally.record(v)``, ``ops[k]`` in
    ``lambda v: ops[k](v)``) and the callables it reads by a free name, each
    handed over as the lambda is. A function of the user's that a call in
    it returns is unknown."""
    params = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
    params.update(a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None)
    for sub in ast.walk(node.body):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in params:
            value = namespace.get(sub.id, _MISSING)
            if value is not _MISSING and callable(value):
                out.values.append((value, sub.id))
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, (ast.Attribute, ast.Subscript)):
            root: ast.expr = func
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if not isinstance(root, ast.Name) or root.id in params:
                continue  # a method of a value the lambda computes or is given
            found = _static_value(func, namespace)
            if found is not None:
                out.values.append(found)
            else:
                _unresolved_reference(func, namespace, out, [])
        elif isinstance(func, ast.Call) and _is_users_callable(_callee_of(func.func, namespace)):
            out.unknown.append(func)


def _unresolved_reference(
    node: ast.Name | ast.Attribute | ast.Subscript, namespace: Mapping[str, Any], out: _Resolved, pending: list
) -> None:
    """Sort out a name, attribute chain or subscript `_static_value` could
    not read: an item of a container under a key known only when it runs
    (``ops[key]``: any of its items), a property or ``__getitem__`` of the
    user's (unknown), or a part of a library object (data)."""
    steps: list[ast.expr] = []
    root_node: ast.expr = node
    while isinstance(root_node, (ast.Attribute, ast.Subscript)):
        steps.append(root_node)
        root_node = root_node.value
    if not isinstance(root_node, ast.Name):
        # ``make()[0]``, ``df.set_index('k')['v']``: what the call hands over.
        pending.append(root_node)
        for step in steps:
            if isinstance(step, ast.Subscript):
                pending.append(step.slice)
        return
    value = namespace.get(root_node.id, _MISSING)
    if value is _MISSING:
        return  # not bound yet: a builtin, or a name the statement binds
    root = root_node.id
    for step in reversed(steps):
        if isinstance(step, ast.Subscript):
            following = _item(value, step.slice)
            if following is _MISSING:
                if type(value) in _OPENED:
                    out.any_of.append((value, root))
                elif _is_users_object(value) and _class_member(type(value), "__getitem__") is not None:
                    out.unknown.append(step)
                return
        else:
            following = _attribute(value, step.attr)
            if following is _MISSING:
                if not isinstance(value, types.ModuleType) and (
                    _is_users_object(value) or (isinstance(value, type) and _is_users_class(value))
                ):
                    out.unknown.append(step)
                return
        value = following


def _resolve_call(node: ast.Call, namespace: Mapping[str, Any], out: _Resolved, pending: list) -> None:
    """What the result of the call *node* can hand over."""
    callee = _callee_of(node.func, namespace)
    if callee is None and isinstance(node.func, ast.Name) and node.func.id not in namespace:
        callee = vars(builtins).get(node.func.id)
    if callee is builtins.getattr and len(node.args) >= 2 and not node.keywords:
        name = node.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str) and name.value.isidentifier():
            pending.append(ast.Attribute(value=node.args[0], attr=name.value, ctx=ast.Load()))
            pending.extend(node.args[2:])
        else:
            out.unknown.append(node)
        return
    if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
        owner = _static_value(node.func.value, namespace)
        if owner is not None and isinstance(owner[0], (dict, collections.UserDict)):
            # ``ops.get('dbl')`` as ``ops['dbl']``; under a key known only
            # when it runs, any of its items.
            container, root = owner
            key = node.args[0]
            if isinstance(key, ast.Constant):
                item = _item(container, key)
                if item is not _MISSING:
                    out.values.append((item, root))
            else:
                out.any_of.append((container, root))
            pending.extend(node.args[1:])
            return
    if callee is None or _is_users_callable(callee):
        # What a function of the user's returns is known only by running
        # it; so is what a call of a value that cannot be read returns.
        if callee is not None or not _unbound(node.func, namespace):
            out.unknown.append(node)
        return
    # A library's call (``functools.partial(f, 2)``, ``random.choice(fns)``,
    # ``np.vectorize(f)``) hands over what it was given, or a part of it.
    inner = _Resolved([], [], [])
    parts: list[ast.expr] = [*node.args, *(kw.value for kw in node.keywords)]
    if isinstance(node.func, ast.Attribute):
        parts.append(node.func.value)
    for part in parts:
        found = _resolve_argument(part, namespace)
        inner.any_of.extend(found.values)
        inner.any_of.extend(found.any_of)
        inner.unknown.extend(found.unknown)
    out.any_of.extend(inner.any_of)
    out.unknown.extend(inner.unknown)


def _unbound(func: ast.expr, namespace: Mapping[str, Any]) -> bool:
    """Whether the call target *func* starts at a name *namespace* does not
    hold: a builtin, or a name bound by the statement itself."""
    while isinstance(func, (ast.Attribute, ast.Subscript)):
        func = func.value
    return isinstance(func, ast.Name) and func.id not in namespace


def _is_users_callable(value: Any) -> bool:
    """A function, method, class or callable object of the user's: calling
    it runs their code."""
    if isinstance(value, functools.partial):
        return _is_users_callable(value.func)
    if isinstance(value, types.FunctionType):
        return _is_users(value)
    if isinstance(value, types.MethodType):
        fn = value.__func__
        return isinstance(fn, types.FunctionType) and _is_users(fn)
    if isinstance(value, type):
        return _is_users_class(value)
    return _is_users_object(value) and callable(value)


#: ``class -> whether it is the user's`` (`_is_users_class`).
_USERS_CLASSES: LruMemo[type, bool] = LruMemo(CODE_OBJECTS)


def _is_users_class(cls: type) -> bool:
    """A class defined in a cell or in a local module: by its module, or by
    the functions it defines (a class made by ``exec`` names no module)."""
    known = _USERS_CLASSES.get(cls)
    if known is None:
        name = getattr(cls, "__module__", None)
        try:
            module = sys.modules.get(name) if isinstance(name, str) else None
            known = name == "__main__" or (module is not None and is_local_module(module))
            if not known and (module is None or name == "builtins"):
                own = vars(cls).values()
                known = any(isinstance(v, types.FunctionType) and _is_users(v) for v in own)
        except (TypeError, AttributeError):
            known = False
        _USERS_CLASSES[cls] = known
    return known


def _is_users_object(value: Any) -> bool:
    """An instance of a class of the user's (not a class, function or module)."""
    if isinstance(value, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        return False
    return _is_users_class(type(value))


def _openable(value: Any) -> bool:
    """A value `_Found.add` looks inside of where the callee calls what it
    is handed: a builtin container, or an object of the user's."""
    return type(value) in _OPENED or _is_users_object(value)


def _wraps_another(value: Any) -> bool:
    """A function of the user's that wraps another (``functools.wraps``)."""
    return isinstance(value, types.FunctionType) and "__wrapped__" in vars(value) and _is_users(value)


def _wrapper_called(func: ast.Name | ast.Attribute, namespace: Mapping[str, Any]) -> tuple[Any, str] | None:
    """``(callable, root)`` when the call target *func* is a ``functools.wraps``
    wrapper of the user's, or a method that is one."""
    found = _users_target(func, namespace)
    if found is None:
        return None
    value = found[0]
    fn = value.__func__ if isinstance(value, types.MethodType) else value
    return found if _wraps_another(fn) else None


def _users_target(func: ast.Name | ast.Attribute, namespace: Mapping[str, Any]) -> tuple[Any, str] | None:
    """``(value, root)`` of the call target *func*, followed through modules
    and the user's own classes and objects only: a library object's
    attributes are not looked up for every call made on it."""
    attrs: list[str] = []
    node: ast.expr = func
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    value = namespace.get(node.id, _MISSING)
    for attr in reversed(attrs):
        if isinstance(value, types.ModuleType):
            value = vars(value).get(attr, _MISSING)
        elif (isinstance(value, type) and _is_users_class(value)) or _is_users_object(value):
            value = _attribute(value, attr)
        else:
            return None
        if value is _MISSING:
            return None
    return value, node.id


_MISSING = object()


def _static_value(node: ast.expr, namespace: Mapping[str, Any]) -> tuple[Any, str] | None:
    """``(value, root name)`` of a name, an attribute chain or a constant
    subscript, read without running code; None when that cannot be done.

    An attribute holding a plain function, read from an object, is that
    function bound to the object, as Python would hand it over."""
    steps: list[ast.expr] = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        steps.append(node)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    value = namespace.get(node.id, _MISSING)
    if value is _MISSING:
        return None
    for step in reversed(steps):
        if isinstance(step, ast.Subscript):
            value = _item(value, step.slice)
        else:
            value = _attribute(value, step.attr)
        if value is _MISSING:
            return None
    return value, node.id


def _item(container: Any, index: ast.expr) -> Any:
    if not isinstance(index, ast.Constant) or type(container) not in _OPENED:
        return _MISSING
    if isinstance(container, _HELD_IN_DATA):
        container = container.data
    try:
        if isinstance(container, dict):
            # ``dict.get``: a defaultdict's factory does not run.
            return dict.get(container, index.value, _MISSING)
        if isinstance(container, (list, tuple, collections.deque)):
            return container[index.value]
    except (IndexError, TypeError):
        pass
    return _MISSING


def _attribute(owner: Any, attr: str) -> Any:
    if isinstance(owner, types.ModuleType):
        return vars(owner).get(attr, _MISSING)
    try:
        member = inspect.getattr_static(owner, attr)
    except Exception:  # noqa: BLE001 - an object's attribute lookup
        return _MISSING
    if isinstance(member, staticmethod):
        return member.__func__
    if isinstance(member, classmethod):
        return types.MethodType(member.__func__, owner if isinstance(owner, type) else type(owner))
    if isinstance(member, types.FunctionType):
        in_instance = not isinstance(owner, type) and attr in _instance_dict(owner)
        return member if isinstance(owner, type) or in_instance else types.MethodType(member, owner)
    if isinstance(member, (types.MethodDescriptorType, types.WrapperDescriptorType)) and not isinstance(owner, type):
        # ``results.append`` handed over: the builtin method, bound.
        try:
            return getattr(owner, attr)
        except Exception:  # noqa: BLE001 - an object's attribute lookup
            return _MISSING
    if hasattr(type(member), "__get__") and not isinstance(member, type):
        # A property or another descriptor computes the value: not read.
        return _MISSING
    return member


def _instance_dict(owner: Any) -> dict:
    try:
        own = object.__getattribute__(owner, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return own if isinstance(own, dict) else {}


class _Found:
    """What the values handed over lead to, collected over one statement."""

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self.namespace = namespace
        self.sources: dict[str, None] = {}
        self.receivers: set[str] = set()
        self.unreadable: list[str] = []
        #: The functions whose sources are in `sources`.
        self.functions: list[types.FunctionType] = []
        #: Whether a callable found changes an object in place.
        self.changes = False
        self._seen: set[tuple[int, str | None]] = set()

    def add(self, value: Any, root: str | None, open_containers: bool) -> None:
        """Add what calling *value* changes. A builtin container, or an
        object of the user's class, is opened for the callables it holds
        only with *open_containers*: when the callee calls what it is handed
        there (`_callables_reached`)."""
        if open_containers and _openable(value):
            if (id(value), root, True) in self._seen:
                return
            self._seen.add((id(value), root, True))
            items, complete = _callables_reached(value)
            if not complete:
                label = repr(root) if root else "an argument"
                self.unreadable.append(
                    f"{label} is handed to a call that calls what it holds, and holds more than "
                    f"{_INSTANCE_LIMIT} objects of your classes to look through"
                )
            for item in items:
                self.add(item, root, False)
            if type(value) in _OPENED or not callable(value):
                return
        elif type(value) in _OPENED:
            return
        if (id(value), root) in self._seen:
            return
        self._seen.add((id(value), root))
        self._callable(value, root)

    def class_changed_by(self, func: ast.Attribute, namespace: Mapping[str, Any]) -> None:
        """A method of the user's called by name (``m.fit(3)``) whose body
        changes the class of its object (``type(self).fitted += 1``) changes
        the class every instance shares: the names bound to it are changed.
        What it changes on the object itself is seen when it runs."""
        found = _users_target(func, namespace)
        if found is None or not isinstance(found[0], types.MethodType):
            return
        owner, fn = found[0].__self__, found[0].__func__
        if isinstance(owner, type) or not isinstance(fn, types.FunctionType) or not self._users(fn):
            return
        for layer in wrapper_layers(fn):
            source = _source_of(layer) if isinstance(layer, types.FunctionType) else None
            if source is not None and _changes_its_class(source):
                self._changed(type(owner), None)
                return

    def _changed(self, obj: Any, root: str | None) -> None:
        """*obj*, which *root* holds or is, is changed in place by the call."""
        self.changes = True
        if root is not None and not isinstance(self.namespace.get(root), types.ModuleType):
            self.receivers.add(root)
        if obj is not None and not isinstance(obj, types.ModuleType):
            # The other names of the notebook bound to that very object.
            names = list(self.namespace)
            self.receivers.update(compress(names, map(operator.is_, list(self.namespace.values()), repeat(obj))))

    def _source(self, fn: types.FunctionType, label: str) -> str | None:
        source = _source_of(fn)
        if source is None:
            self.unreadable.append(f"{label} is handed to a call and its source cannot be read")
            return None
        self.sources.setdefault(source)
        self.functions.append(fn)
        return source

    def _callable(self, value: Any, root: str | None) -> None:
        label = repr(root) if root else "a function"
        if isinstance(value, functools.partial):
            self.add(value.func, root, False)
            return
        if isinstance(value, types.FunctionType):
            # A ``functools.wraps`` wrapper and each function under it: the
            # call runs the wrapper's body as well as the wrapped one's.
            for layer in wrapper_layers(value):
                if not isinstance(layer, types.FunctionType) or not self._users(layer):
                    continue
                source = self._source(layer, label)
                if source is not None and layer.__closure__:
                    free = set(layer.__code__.co_freevars)
                    if free & source_global_mutations(source):
                        self._changed(value, root)
            return
        if isinstance(value, types.MethodType):
            owner = value.__self__
            cls = owner if isinstance(owner, type) else type(owner)
            for fn in wrapper_layers(value.__func__):
                if not isinstance(fn, types.FunctionType) or not self._users(fn):
                    continue
                source = self._source(fn, label)
                if source is not None and not isinstance(owner, types.ModuleType):
                    if _changes_its_first_argument(fn, cls, source):
                        self._changed(owner, root)
                        if not isinstance(owner, type) and _reaches_its_class(source):
                            self._changed(cls, None)
            return
        if isinstance(value, (types.BuiltinMethodType, types.MethodWrapperType)):
            owner = getattr(value, "__self__", None)
            if owner is None or isinstance(owner, (types.ModuleType, type)):
                return
            if getattr(value, "__name__", "") in MUTATING_METHODS:
                self._changed(owner, root)
            return
        if isinstance(value, type):
            init = _class_member(value, "__init__")
            if isinstance(init, types.FunctionType) and self._users(init):
                self._source(init, label)
            return
        if not callable(value) or isinstance(value, types.ModuleType):
            return
        call = _class_member(type(value), "__call__")
        if isinstance(call, types.FunctionType) and self._users(call):
            source = self._source(call, label)
            if source is not None and _changes_its_first_argument(call, type(value), source):
                self._changed(value, root)
                if _reaches_its_class(source):
                    self._changed(type(value), None)

    def _users(self, fn: types.FunctionType) -> bool:
        return fn.__globals__ is self.namespace or _is_users(fn)


def _is_users(fn: types.FunctionType) -> bool:
    """Defined in a cell (``__main__``, or globals of no module at all, as
    code run with ``exec``) or in a local module."""
    name = fn.__globals__.get("__name__")
    if name is None or name == "__main__":
        return True
    module = sys.modules.get(name) if isinstance(name, str) else None
    try:
        return module is not None and is_local_module(module)
    except (TypeError, AttributeError):
        return False


#: ``code object -> source`` (None when it cannot be read) of the functions
#: met, so a call made once per element does not read its source each time.
_SOURCES: LruMemo[Any, str | None] = LruMemo(CODE_OBJECTS)


def _source_of(fn: types.FunctionType) -> str | None:
    """*fn*'s own source: a ``functools.wraps`` wrapper's, not the wrapped
    function's (`own_source`), as the memo is keyed by the code that runs."""
    code = fn.__code__
    if code in _SOURCES:
        return _SOURCES.get(code)
    try:
        source: str | None = own_source(fn)
    except SOURCE_RETRIEVAL_ERRORS:
        source = None
    _SOURCES[code] = source
    return source


#: Argument values with nothing inside to hand over a callable, by exact type.
_ATOMS = frozenset({int, float, complex, str, bytes, bool, type(None)})


def hands_a_state_changing_callable(fn: Any, args: tuple, kwargs: Mapping[str, Any]) -> bool:
    """Whether calling *fn* with *args* and *kwargs* hands it a user callable
    that changes state when called: a function that changes a global or what
    it closes over, a method or a callable object that changes its object,
    or one whose source cannot be read. A builtin container or an object of
    the user's class among the arguments is looked into when *fn* calls
    what it is handed there.

    ``run_steps(steps, data)`` with ``steps = [f]`` runs ``f``; served from
    the cache, ``f``'s writes would not happen."""
    found: _Found | None = None
    opened: _Opened | None = None
    for position, keyword, value in chain(
        zip(range(len(args)), repeat(None), args), zip(repeat(None), kwargs.keys(), kwargs.values())
    ):
        if type(value) in _ATOMS:
            continue
        opens = False
        if _openable(value):
            if opened is None:
                opened = _opened_by(fn, None)
            opens = opened.opens(position, keyword)
            if not opens and not callable(value):
                continue
        elif not callable(value):
            continue
        if found is None:
            found = _Found({})
        found.add(value, None, opens)
    if found is None:
        return False
    if found.changes or found.unreadable:
        return True
    for function in found.functions:
        source = _source_of(function)
        resolve = functools.partial(_global_source, function.__globals__)
        if callee_global_mutations(None, resolve, extra_sources=(source,)):
            return True
    return False


def _global_source(namespace: Mapping[str, Any], name: str) -> str | None:
    value = namespace.get(name)
    return _source_of(value) if isinstance(value, types.FunctionType) else None


class _Opened(NamedTuple):
    """Which arguments of a call the callee calls what is inside of."""

    positions: frozenset[int] = frozenset()
    keywords: frozenset[str] = frozenset()
    #: The position a ``*args`` the callee calls into starts at, if any.
    star_from: int | None = None
    #: Whether a ``**kwargs`` the callee calls into takes the other keywords.
    double_star: bool = False

    def opens(self, position: int | None, keyword: str | None) -> bool:
        if keyword is not None:
            return keyword in self.keywords or self.double_star
        if position is None:  # after a ``*args`` unpacking: any position
            return bool(self.positions or self.keywords) or self.star_from is not None
        return position in self.positions or (self.star_from is not None and position >= self.star_from)


_OPENS_NOTHING = _Opened()

#: A library call that takes a list or dict of functions, by method name:
#: ``df.agg([f, g])``, ``df.agg({'a': f})``. Its first argument, or ``func=``.
_TAKES_FUNCTIONS = frozenset({"agg", "aggregate", "transform"})

#: Where a higher-order function takes the function it calls: by position,
#: or by keyword. ``map(f, rows)`` calls ``f``; its other arguments are data.
_FUNCTION_ARGUMENT: dict[str, int | str] = {
    "map": 0,
    "filter": 0,
    "starmap": 0,
    "reduce": 0,
    "apply": 0,
    "applymap": 0,
    "agg": 0,
    "aggregate": 0,
    "transform": 0,
    "pipe": 0,
    "sorted": "key",
    "min": "key",
    "max": "key",
}


def _opened_by(callee: Any, func: ast.expr | None) -> _Opened:
    """The arguments whose builtin containers the call opens for callables:
    those a user callee calls what is inside of (`_called_params`), or the
    function list a library ``agg``/``transform`` takes."""
    fn, offset = _user_function_of(callee)
    if fn is not None:
        return _opened_by_function(fn, offset)
    attr = func.attr if isinstance(func, ast.Attribute) else getattr(callee, "__name__", None)
    if attr in _TAKES_FUNCTIONS:
        return _Opened(frozenset({0}), frozenset({"func"}))
    return _OPENS_NOTHING


def _opened_by_function(fn: types.FunctionType, offset: int) -> _Opened:
    called = _called_params(fn)
    if not called:
        return _OPENS_NOTHING
    code = fn.__code__
    names = code.co_varnames[: code.co_argcount]
    positions = frozenset(i - offset for i, name in enumerate(names) if name in called and i >= offset)
    keywords = frozenset(n for n in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount] if n in called)
    extra = code.co_argcount + code.co_kwonlyargcount
    star_from = None
    if code.co_flags & inspect.CO_VARARGS:
        if code.co_varnames[extra] in called:
            star_from = max(0, code.co_argcount - offset)
        extra += 1
    double_star = bool(code.co_flags & inspect.CO_VARKEYWORDS) and code.co_varnames[extra] in called
    return _Opened(positions, keywords, star_from, double_star)


def _user_function_of(callee: Any) -> tuple[types.FunctionType | None, int]:
    """``(function, offset)``: the user function calling *callee* runs and
    how many of its leading parameters the call does not fill (``self``)."""
    if isinstance(callee, types.MethodType):
        fn, offset = callee.__func__, 1
    elif isinstance(callee, types.FunctionType):
        fn, offset = callee, 0
    elif isinstance(callee, type):
        fn, offset = _class_member(callee, "__init__"), 1
    elif callee is not None and callable(callee) and not isinstance(callee, types.ModuleType):
        fn, offset = _class_member(type(callee), "__call__"), 1
    else:
        return None, 0
    if not isinstance(fn, types.FunctionType) or not _is_users(fn):
        return None, 0
    return fn, offset


class _CallFacts(NamedTuple):
    """What a function's own text says it calls of its parameters."""

    #: Parameters whose values, or what they hold, the body calls itself.
    called: frozenset[str]
    #: ``(name, arguments)`` of each call by a bare name, which may be a
    #: user function that calls what it is handed: per argument
    #: ``(position, keyword, parameters it hands over)``.
    handoffs: tuple[tuple[str, tuple[tuple[int | None, str | None, frozenset[str]], ...]], ...]


#: ``code object -> _CallFacts``: the text alone decides them.
_CALL_FACTS: LruMemo[Any, _CallFacts | None] = LruMemo(CODE_OBJECTS)


def _call_facts(fn: types.FunctionType) -> _CallFacts | None:
    code = fn.__code__
    if code in _CALL_FACTS:
        return _CALL_FACTS.get(code)
    source = _source_of(fn)
    node = parse_function_source(source) if source is not None else None
    facts = None if node is None else _read_call_facts(node)
    _CALL_FACTS[code] = facts
    return facts


def _read_call_facts(node: ast.FunctionDef | ast.AsyncFunctionDef) -> _CallFacts:
    params = all_param_names(node)
    iterated = iterated_sources(node)

    def roots(expr: ast.expr) -> frozenset[str]:
        while True:
            if isinstance(expr, (ast.Subscript, ast.Attribute)):
                # ``ops['dbl'](x)``; ``pipe.steps[0](x)``, or ``st(x)`` for
                # ``st`` in ``pipe.steps``: what the parameter holds.
                expr = expr.value
            elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
                expr = expr.func.value  # ``ops.get('dbl')(x)``
            elif (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id == "getattr"
                and expr.args
            ):
                expr = expr.args[0]  # ``getattr(obj, name)(x)``
            elif isinstance(expr, ast.Starred):
                expr = expr.value
            else:
                break
        if not isinstance(expr, ast.Name):
            return frozenset()
        return frozenset({expr.id, *iterated.get(expr.id, ())}) & params

    called: set[str] = set()
    handoffs = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            called |= roots(call.func)
        name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
        where = _FUNCTION_ARGUMENT.get(name) if name else None
        if isinstance(where, int) and len(call.args) > where:
            called |= roots(call.args[where])
        for kw in call.keywords:
            if kw.arg is not None and (kw.arg == where or kw.arg == "func"):
                called |= roots(kw.value)
        if isinstance(call.func, ast.Name):
            handed = tuple((p, k, r) for p, k, arg in _arguments(call) if (r := roots(arg)))
            if handed:
                handoffs.append((call.func.id, handed))
    return _CallFacts(frozenset(called), tuple(handoffs))


def _called_params(fn: types.FunctionType, seen: frozenset[Any] = frozenset()) -> frozenset[str]:
    """The parameters of *fn* whose values, or what they hold, *fn* calls:
    ``st(x)`` for ``st`` in ``steps``, ``ops['dbl'](x)``, ``fn(x)``; handed
    on as the function of a higher-order call (``map(fn, rows)``); or handed
    to a user function that calls them, as that function is now."""
    facts = _call_facts(fn)
    if facts is None:
        return frozenset()
    called = set(facts.called)
    seen = seen | {fn.__code__}
    for name, handed in facts.handoffs:
        target = fn.__globals__.get(name)
        if not isinstance(target, types.FunctionType) or target.__code__ in seen or not _is_users(target):
            continue
        inner = _called_params(target, seen)
        if not inner:
            continue
        inner_names = target.__code__.co_varnames[: target.__code__.co_argcount]
        for position, keyword, params in handed:
            if (
                (keyword is not None and keyword in inner)
                or (position is not None and position < len(inner_names) and inner_names[position] in inner)
                or (position is None and keyword is None)
            ):
                called |= params
    return frozenset(called)


def _class_member(cls: type, name: str) -> Any:
    try:
        return inspect.getattr_static(cls, name)
    except AttributeError:
        return None


def _callables_reached(value: Any) -> tuple[list[Any], bool]:
    """``(callables, complete)``: the callables *value* holds -- a builtin
    container's items, an object of the user's class's attributes (its
    ``__dict__`` and ``__slots__``, read without running code) -- and those
    the containers and objects of the user's among them hold, however deeply
    nested, each once. A level at a time, the types sorted at C speed.
    *complete* is False when more than `_INSTANCE_LIMIT` objects of the
    user's classes had to be opened, and the walk stopped."""
    if _plain_data.is_tree(value):
        return [], True  # lists and tuples of plain values only
    found: list[Any] = []
    seen = {id(value)}
    level = [value]
    users_objects = 0
    while level:
        items = list(chain.from_iterable(map(_items, level)))
        if not items:
            break
        found.extend(filter(callable, items))
        kinds = set(map(type, items))
        opened = {kind for kind in kinds if kind in _OPENED or _users_instance_type(kind)}
        if not opened:
            break
        nested = [c for c in compress(items, map(opened.__contains__, map(type, items))) if id(c) not in seen]
        seen.update(map(id, nested))
        users_objects += sum(1 for c in nested if type(c) not in _OPENED)
        if users_objects > _INSTANCE_LIMIT:
            return found, False
        level = nested
    return found, True


def _users_instance_type(kind: type) -> bool:
    """Whether objects of the type *kind* are objects of the user's class:
    not classes, functions or modules."""
    if issubclass(kind, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        return False
    return _is_users_class(kind)


def _items(container: Any) -> Iterable[Any]:
    """What `_callables_reached` looks at in *container*: a builtin
    container's items (a dict's keys and values), an object's attributes."""
    if type(container) in _OPENED:
        if isinstance(container, _HELD_IN_DATA):
            container = container.data
        if isinstance(container, dict):
            return chain(dict.keys(container), dict.values(container))
        return container
    return _attributes_of(container)


def users_object_attributes(value: Any) -> list[Any]:
    """The values of the instance attributes of *value* when it is an object
    of the user's class (its ``__dict__`` and ``__slots__``, read without
    running code); empty for anything else."""
    return _attributes_of(value) if _is_users_object(value) else []


def _attributes_of(obj: Any) -> list[Any]:
    """The values of *obj*'s instance attributes: its ``__dict__`` and its
    ``__slots__``, read through the slot descriptors so no property or
    ``__getattr__`` runs."""
    values = list(_instance_dict(obj).values())
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__")
        if slots is None:
            continue
        for name in (slots,) if isinstance(slots, str) else slots:
            member = cls.__dict__.get(name)
            if type(member) is types.MemberDescriptorType:
                try:
                    values.append(member.__get__(obj, cls))
                except AttributeError:
                    pass  # an unset slot
    return values


@functools.lru_cache(maxsize=4096)
def _changes_its_class(source: str) -> bool:
    """Whether the method *source* changes the class of its first argument
    in place: a store, a delete or an augmented assignment under, or a
    mutating method called on, ``type(self)`` or ``self.__class__``."""
    node = parse_function_source(source)
    if node is None:
        return False
    params = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
    if not params:
        return False
    first = params[0]

    def through_class(expr: ast.expr) -> bool:
        while isinstance(expr, (ast.Attribute, ast.Subscript)):
            if isinstance(expr, ast.Attribute) and expr.attr == "__class__":
                return isinstance(expr.value, ast.Name) and expr.value.id == first
            expr = expr.value
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "type"
            and len(expr.args) == 1
            and isinstance(expr.args[0], ast.Name)
            and expr.args[0].id == first
        )

    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.Delete)):
            targets = sub.targets
        elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
            targets = [sub.target]
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in MUTATING_METHODS:
            if through_class(sub.func.value):
                return True
            continue
        else:
            continue
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)) and through_class(target.value):
                return True
    return False


@functools.lru_cache(maxsize=4096)
def _reaches_its_class(source: str) -> bool:
    """Whether the method *source* reaches the class of its first argument
    (``type(self)``, ``self.__class__``): a change made there is a change to
    the class every instance shares, and to the names bound to it."""
    node = parse_function_source(source)
    if node is None:
        return True
    params = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
    if not params:
        return False
    first = params[0]
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "type"
            and len(sub.args) == 1
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id == first
        ):
            return True
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "__class__"
            and isinstance(sub.value, ast.Name)
            and sub.value.id == first
        ):
            return True
    return False


def _changes_its_first_argument(
    fn: types.FunctionType, cls: type, source: str, seen: frozenset[int] = frozenset()
) -> bool:
    """Whether calling the method *fn* of *cls* changes the object it is
    bound to (``self.seen.append(v)``, ``self.n += 1``), itself or through
    another method of the class it calls on it. A call on something the
    object holds (``self.model.fit(x)``) counts unless the method is known
    not to change anything: what it does is not read."""
    node = parse_function_source(source)
    if node is None:
        return True
    params = [a.arg for a in (*node.args.posonlyargs, *node.args.args)]
    if not params:
        return False
    first = params[0]
    if first in params_mutated_in_function(node):
        return True
    seen = seen | {id(fn)}
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        chain_root = call.func.value
        depth = 1
        while isinstance(chain_root, (ast.Attribute, ast.Subscript)):
            chain_root = chain_root.value
            depth += 1
        if not (isinstance(chain_root, ast.Name) and chain_root.id == first):
            continue
        method = call.func.attr
        if depth > 1:
            if not chain_is_pure(method, frozenset()):
                return True
            continue
        member = _class_member(cls, method)
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if isinstance(member, types.FunctionType):
            if id(member) in seen:
                continue
            inner = _source_of(member)
            if inner is None or _changes_its_first_argument(member, cls, inner, seen):
                return True
        elif method in MUTATING_METHODS or member is None:
            return True
    return False
