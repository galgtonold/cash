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
import contextlib
import dataclasses
import functools
import inspect
import operator
import sys
import types
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import chain, compress, repeat
from typing import Any, NamedTuple

from .. import _plain_data
from .._memo import CODE_OBJECTS, LruMemo
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import own_source, wrapper_layers
from ..tracking.function_tracker import is_local_module
from .ast_util import CallScope, calls_in, nodes_in, same_as_all
from .callee_effects import (
    all_param_names,
    callee_global_mutations,
    iterated_sources,
    params_mutated_in_function,
    parse_function_source,
    source_called_names,
    source_global_mutations,
)
from .mutations import MUTATING_METHODS, chain_is_pure

__all__ = [
    "HandedCallables",
    "answers_shared",
    "handed_callable_values",
    "handed_callables",
    "handed_values",
    "hands_a_state_changing_callable",
    "iteration_bindings",
    "new_analysis_round",
    "one_reading",
    "stop_sharing_answers",
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


#: While `one_reading` is open: ``(id(tree), id(namespace), all) -> (tree,
#: namespace, answer)`` of `handed_callables`.
_READING: list[dict[tuple[int, int, bool], tuple[Any, Any, HandedCallables]] | None] = [None]


@contextlib.contextmanager
def one_reading() -> Iterator[None]:
    """Answer `handed_callables` once for each tree and namespace while
    open: the analyses of one statement that ask it the same question, with
    no code run between them, share the answer, and the walk through what
    the statement hands over is made once."""
    outer = _READING[0]
    if outer is None:
        _READING[0] = {}
    try:
        yield
    finally:
        if outer is None:
            _READING[0] = None


def handed_callables(
    tree: ast.AST | None, namespace: Mapping[str, Any] | None, scope: CallScope = "all"
) -> HandedCallables:
    """The user callables the calls in *tree* (within *scope*) are handed, as
    *namespace* holds them. See the module docstring."""
    reading = _READING[0]
    if reading is None or tree is None or not namespace:
        return _handed_callables(tree, namespace, scope)
    every = same_as_all(tree, scope)
    key = (id(tree), id(namespace), every)
    kept = reading.get(key)
    if kept is not None and kept[0] is tree and kept[1] is namespace:
        return kept[2]
    answer = _handed_callables(tree, namespace, "all" if every else scope)
    reading[key] = (tree, namespace, answer)
    return answer


def _handed_callables(
    tree: ast.AST | None, namespace: Mapping[str, Any] | None, scope: CallScope = "all"
) -> HandedCallables:
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()
    if tree is None or not namespace:
        return _NONE
    found = _Found(namespace)
    # ``a = memo['k']``, ``with timer('fit'):``: what Python runs for the
    # statement without a call in its text.
    found.run_implicitly(tree, namespace, scope)
    calls = calls_in(tree, scope)
    if not calls and not found.sources:
        return _NONE
    bound = iteration_bindings(tree)
    for call in calls:
        if isinstance(call.func, (ast.Name, ast.Attribute)) and not _bound_root(call.func, bound):
            # ``dbl(3)`` with ``dbl`` a ``functools.wraps`` wrapper: its
            # source, read by name, is the wrapped function's, so what the
            # wrapper's own body does is read here.
            wrapper = _wrapper_called(call.func, namespace)
            if wrapper is not None:
                found.add(*wrapper, False)
            elif isinstance(call.func, ast.Attribute):
                found.class_changed_by(call.func, namespace)
                found.held_by_its_object(call.func, namespace)
            if wrapper is None:
                found.called_by_name(call.func, namespace)
            found.wrapping_object_called(call, namespace, bound)
        else:
            # ``ops[key](v)``, ``ops.get(key)(v)``, ``f(v) for f in fns``:
            # what is called is read out of a container, or out of what
            # the statement iterates.
            target = _resolve_argument(call.func, namespace, bound)
            found.methodcaller_called(call, namespace, bound)
            for value, root in target.values:
                found.add(value, root, False)
            for value, root in target.any_of:
                found.add(value, root, True)
            for node in target.unknown:
                found.unreadable.append(_cannot_tell_called(node))
        if not call.args and not call.keywords:
            continue
        calling: _Opened | None = None
        for position, keyword, arg in _arguments(call):
            resolved = _resolve_argument(arg, namespace, bound)
            for value, root in resolved.values:
                opens = False
                if _openable(value):
                    if calling is None:
                        calling = _calling_positions(call, namespace)
                    opens = calling.opens(position, keyword)
                    methods = calling.calls_methods(position, keyword)
                    if methods:
                        found.with_methods(value, root, calling.attributes, methods == 2)
                found.add(value, root, opens)
            if resolved.any_of or resolved.unknown:
                if calling is None:
                    calling = _calling_positions(call, namespace)
                if calling.opens(position, keyword):
                    for value, root in resolved.any_of:
                        found.add(value, root, True)
                    for node in resolved.unknown:
                        found.unreadable.append(_cannot_resolve(node, call))
    if not (found.sources or found.receivers or found.unreadable):
        return _NONE
    return HandedCallables(
        tuple(found.sources), frozenset(found.receivers), tuple(found.unreadable), tuple(found.functions)
    )


def handed_callable_values(
    call: ast.Call, namespace: Mapping[str, Any], bound: Mapping[str, tuple[ast.expr, ...]] | None = None
) -> list[Any]:
    """The callables *call* is handed, as *namespace* holds them: its
    arguments (`handed_values`), the ones inside a builtin container or an
    object of the user's when the callee calls what it is handed there, and
    the functions a lambda among them calls. Functions, methods, classes and
    callable objects, each once; a partial's function and the callables it
    holds.

    The callables its target may be as well, when that is not a name
    called directly: ``ops[key](v)``, ``ops.get(key)(v)``, ``f(v)`` for
    ``f`` in what the statement iterates (*bound*, the statement's
    `iteration_bindings`), what a method of the user's reaches through its
    object (``bus.emit(3)``)."""
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()
    calling: _Opened | None = None
    out: dict[int, Any] = {}
    bound = bound or {}
    if isinstance(call.func, (ast.Subscript, ast.Call)) or _bound_root(call.func, bound):
        target = _resolve_argument(call.func, namespace, bound)
        for value, _root in target.values:
            _add_callable(out, value)
        for value, _root in target.any_of:
            for item in _callables_reached(value)[0] if _openable(value) else [value]:
                _add_callable(out, item)
    elif isinstance(call.func, ast.Attribute):
        # ``bus.emit(3)`` calling ``cb(x)`` for ``cb`` in ``self.subs``.
        method = _users_target(call.func, namespace)
        if method is not None and isinstance(method[0], types.MethodType):
            for item in _held_callables(method[0]):
                _add_callable(out, item)
    for position, keyword, arg in _arguments(call):
        resolved = _resolve_argument(arg, namespace, bound)
        candidates = [(value, False) for value, _root in resolved.values]
        if resolved.any_of:
            if calling is None:
                calling = _calling_positions(call, namespace)
            if calling.opens(position, keyword):
                candidates.extend((value, True) for value, _root in resolved.any_of)
        for value, any_of in candidates:
            items = [value]
            if _openable(value):
                if calling is None:
                    calling = _calling_positions(call, namespace)
                if not any_of:
                    any_of = calling.opens(position, keyword)
                items = _callables_reached(value)[0] if any_of else []
                if callable(value) and not isinstance(value, type):
                    items.append(value)
                methods = calling.calls_methods(position, keyword)
                if methods:
                    items.extend(_methods_called(value, calling.attributes, methods == 2))
            for item in items:
                _add_callable(out, item)
    return list(out.values())


def _held_callables(method: types.MethodType, seen: set[tuple[int, Any]] | None = None) -> list[Any]:
    """The callables the user method *method* reaches through its object:
    what the object holds when the method calls it, and the methods it
    calls on the object (`_Found._held_by`), and theirs in turn. Worked out
    once per analysis round (`_this_round`): ``sim.step()`` is asked about
    by several analyses of the statement before it runs, and each answer
    opens every agent the simulation holds."""
    if seen is None:
        return _this_round(
            ("held", id(method.__self__), id(method.__func__)),
            (method.__self__, method.__func__),
            lambda: _held_callables_of(method, set()),
        )
    return _held_callables_of(method, seen)


def _held_callables_of(method: types.MethodType, seen: set[tuple[int, Any]]) -> list[Any]:
    return _held_group(method.__func__, [method.__self__], seen)


def _held_group(fn: Any, owners: list[Any], seen: set[tuple[int, Any]]) -> list[Any]:
    """`_held_callables` of the user function *fn* bound to each of
    *owners*, read once for all of them: ``a.act()`` for each of a
    simulation's agents is one function, whose source says the same of
    every agent -- that it calls ``self.strategy`` -- so the question per
    agent is only what each holds there, asked of all at once
    (`_methods_called_by_many`). Then the methods found are grouped by
    their function in turn. Each owner and function is looked at once
    (*seen*), as when each was asked by itself; what is found is what the
    questions asked one owner at a time find, since the facts read from
    *fn* are the same for every owner and the walk of all owners at once
    reaches what each of theirs does (`_reached_from`).

    What the owners hold is still read from the owners, not from the
    collector's view of who holds a function (``gc.get_referrers``): that
    view misses what ``gc.freeze`` moved out of it, says nothing of whether
    a holder is reachable from these owners, and a holder found is not only
    a notebook function (a builtin's bound method, a class, an object with a
    ``__call__``). The walk reads each object's items once, at C speed."""
    if not isinstance(fn, types.FunctionType) or not _is_users(fn):
        return []
    code = fn.__code__
    keys = list(zip(map(id, owners), repeat(code)))
    fresh = [
        owner for owner, key in zip(owners, keys) if key not in seen and not isinstance(owner, (type, types.ModuleType))
    ]
    seen.update(keys)
    facts = _call_facts(fn)
    if not fresh or facts is None or not code.co_argcount:
        return []
    first = code.co_varnames[0]
    out: list[Any] = []
    if first in _called_params(fn):
        out.extend(_callables_reached_by_many(fresh))
    if facts.attributes and (first in facts.attribute_called or first in facts.held_attribute_called):
        members = _methods_called_by_many(fresh, facts.attributes, first in facts.held_attribute_called)
        out.extend(members)
        # By function (a function hashes by identity); a method of anything
        # else reaches nothing more (`_held_group` reads functions only).
        bound: dict[types.FunctionType, list[Any]] = {}
        for member in members:
            if type(member) is types.MethodType and type(member.__func__) is types.FunctionType:
                bound.setdefault(member.__func__, []).append(member.__self__)
        for func, selves in bound.items():
            out.extend(_held_group(func, selves, seen))
    return out


def _callables_reached_by_many(values: list[Any]) -> list[Any]:
    """The callables any of *values* holds (`_callables_reached` of each),
    with one walk: each object reached from several is opened once."""
    if len(values) == 1:
        return _callables_reached(values[0])[0]
    return _reached_from(values, False, None, None, _INSTANCE_LIMIT * len(values))[0]


def _methods_called(value: Any, names: frozenset[str], held: bool) -> list[Any]:
    """The methods and stored functions called *names* of *value*, an
    object of the user's, and with *held* of those it holds
    (`_Found.with_methods`). With *held*, worked out once per analysis
    round (`_this_round`): several analyses of one statement ask it before
    it runs."""
    if not held:
        return _methods_called_by_many([value], names, False)
    return _this_round(("methods", id(value), names), (value,), lambda: _methods_called_by_many([value], names, True))


def _methods_called_by_many(values: list[Any], names: frozenset[str], held: bool) -> list[Any]:
    """`_methods_called` of each of *values*: the objects of the user's
    among them, and with *held* those any of them holds (one walk,
    `_reached_from`), each object's methods read a class at a time
    (`_members_by_class`)."""
    objects = _users_objects(values)
    if held:
        if len(values) == 1:
            reached = _reached(values[0])[0]
        else:
            reached = _reached_from(values, True, None, None, _INSTANCE_LIMIT * len(values))[0]
        objects.extend(_users_objects(reached))
    return _members_by_class(objects, names)


def _plain_own(kind: type, name: str, value: Any) -> bool:
    """Whether *value*, held as *name* in an object of *kind*'s own
    ``__dict__``, is what `_attribute` reads of it unchanged -- the class
    has no data descriptor of that name in the way (`_static_attribute`,
    already asked of an object of *kind* holding the name) -- and is not a
    ``classmethod``, which `_attribute` binds. Such a value that is not
    callable needs no reading."""
    return not isinstance(value, classmethod) and _CLASS_ATTRIBUTES.get((kind, name, True)) is _OWN


def _members_by_class(objects: list[Any], names: frozenset[str]) -> list[Any]:
    """The methods and stored functions called *names* of each of
    *objects*, as `_attribute` reads each, a class at a time.

    What ``o.total`` is depends on ``type(o)`` -- read as the class is now,
    so ``Order.total = f`` after the orders were made is what is found --
    and on whether ``total`` is in ``o``'s own ``__dict__`` (``o.total =
    f`` on one order). An object holding the name itself is read by itself.
    For the rest of a class the class's member is the same, so it is read
    once: a function is bound to each object (``o.total`` of each, since
    which object it is bound to is part of what is found), unless calling
    it does the same whatever the object (`_same_for_every_object`), when
    one is enough (``r.value()`` for ``r`` in rows); anything else the
    class holds is the same object for all."""
    out: list[Any] = []
    #: ids of what the objects hold themselves that is found: one strategy
    #: function held by many agents is found once.
    held: set[int] = set()
    groups: dict[type, list[Any]] = {}
    for obj in objects:
        groups.setdefault(type(obj), []).append(obj)
    for kind, group in groups.items():
        dicts = list(map(_instance_dict, group))
        for name in names:
            rest = []
            for obj, own in zip(group, dicts):
                if name not in own:
                    rest.append(obj)
                    continue
                value = own[name]
                if _plain_own(kind, name, value):
                    if not callable(value):
                        continue  # not callable, nor what `_attribute` makes of it
                    if type(value) is types.FunctionType:
                        if id(value) not in held:  # what `_attribute` reads: the function itself
                            held.add(id(value))
                            out.append(value)
                        continue
                member = _attribute(obj, name)
                if member is not _MISSING and callable(member):
                    if type(member) is types.MethodType:
                        out.append(member)
                    elif id(member) not in held:
                        held.add(id(member))
                        out.append(member)
            if not rest:
                continue
            sample = rest[0]
            member = _attribute(sample, name)
            if member is _MISSING or not callable(member):
                continue
            if type(member) is types.MethodType and member.__self__ is sample:
                if _same_for_every_object(member):
                    out.append(member)
                else:
                    func = member.__func__
                    out.extend(types.MethodType(func, obj) for obj in rest)
            elif getattr(member, "__self__", None) is sample:
                # A builtin's method bound to each object (a subclass of
                # ``list``'s ``append``): each one's own.
                out.extend(m for obj in rest if (m := _attribute(obj, name)) is not _MISSING and callable(m))
            else:
                out.append(member)  # the class's own callable, the same for every object
    return out


def _same_for_every_object(method: types.MethodType) -> bool:
    """Whether calling the user method *method* does the same whatever object
    of its class it is bound to: it neither changes the object nor reaches
    through it (`_Found._held_by`), each layer's source read."""
    owner = method.__self__
    cls = type(owner)
    for fn in wrapper_layers(method.__func__):
        if not isinstance(fn, types.FunctionType):
            return False
        source = _source_of(fn)
        if source is None or _reaches_through_its_object(fn):
            return False
        key = (fn.__code__, cls)
        changes = _CHANGES_ITS_OBJECT.get(key)
        if changes is None:
            changes = _CHANGES_ITS_OBJECT[key] = _changes_its_first_argument(fn, cls, source)
        if changes:
            return False
    return True


def _add_callable(out: dict[int, Any], item: Any) -> None:
    """Add *item* to *out* when calling it may run code of the user's; a
    partial as its function and the callables it is given."""
    if isinstance(item, functools.partial):
        for part in _partial_parts(item):
            _add_callable(out, part)
        return
    if callable(item) and not isinstance(item, types.BuiltinFunctionType):
        out.setdefault(id(item), item)


def _partial_parts(value: functools.partial) -> list[Any]:
    """The function of the partial *value* and the callables, containers and
    objects of the user's it was given: ``partial(apply_, record)`` runs
    ``record`` when ``apply_`` calls its argument."""
    parts = [value.func]
    for arg in chain(value.args, value.keywords.values()):
        if type(arg) not in _ATOMS and (callable(arg) or _openable(arg)):
            parts.append(arg)
    return parts


def iteration_bindings(tree: ast.AST | None) -> dict[str, tuple[ast.expr, ...]]:
    """The names *tree* binds by iterating, each with the expressions it
    iterates: ``f`` in ``[f(3) for f in fns]`` or ``for f in fns:`` is one
    of the items of ``fns``. A comprehension's target, and a ``for``
    target."""
    out: dict[str, list[ast.expr]] = {}
    if tree is None:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.comprehension, ast.For, ast.AsyncFor)):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node.iter)
    return {name: tuple(iters) for name, iters in out.items()}


def _bound_root(func: ast.expr, bound: Mapping[str, tuple[ast.expr, ...]]) -> bool:
    """Whether the call target *func* starts at a name the statement binds
    by iterating (`iteration_bindings`)."""
    if not bound:
        return False
    while isinstance(func, (ast.Attribute, ast.Subscript)):
        func = func.value
    return isinstance(func, ast.Name) and func.id in bound


def _cannot_tell_called(node: ast.expr) -> str:
    """The reason given for a call target part *node* that cannot be resolved."""
    return f"{_short(node)} is called, and cash cannot tell which function it is without running it"


#: The keyword `_arguments` gives a ``**kw`` unpacking.
DOUBLE_STAR = "**"


def _arguments(call: ast.Call) -> Iterable[tuple[int | None, str | None, ast.expr]]:
    """``(position, keyword, expression)`` of each argument of *call*; the
    position is None from a ``*args`` on, the keyword `DOUBLE_STAR` for a
    ``**kw``."""
    position: int | None = 0
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            position = None
        yield position, None, arg
        if position is not None:
            position += 1
    for kw in call.keywords:
        yield None, kw.arg if kw.arg is not None else DOUBLE_STAR, kw.value


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
        function = call.args[where] if len(call.args) > where else None
        opened = opened._replace(positions=opened.positions | {where}, keywords=opened.keywords | {"func"})
    else:
        function = next((kw.value for kw in call.keywords if kw.arg == where), None)
        opened = opened._replace(keywords=opened.keywords | {where})
    if function is not None and _calls_its_parameters(function, namespace):
        # ``map(lambda f: f(3), fns)``, ``reduce(lambda acc, f: f(acc), fns,
        # x)``, ``sorted(fns, key=lambda f: f(0))``: the function calls what
        # it is given, the items of the other arguments.
        return opened._replace(star_from=0, double_star=True)
    return opened


def _calls_its_parameters(function: ast.expr, namespace: Mapping[str, Any]) -> bool:
    """Whether the function argument *function* of a higher-order call
    calls (a method of) what it is given: a lambda whose body does, or a
    function of the user's whose parameters `_called_params` finds."""
    if isinstance(function, ast.Lambda):
        return bool(_read_call_facts(function).called)
    fn, _offset = _user_function_of(_callee_of(function, namespace))
    return fn is not None and bool(_called_params(fn))


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
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()
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


def _resolve_argument(
    arg: ast.expr,
    namespace: Mapping[str, Any],
    bound: Mapping[str, tuple[ast.expr, ...]] | None = None,
    following: frozenset[str] = frozenset(),
) -> _Resolved:
    """What *arg* hands over, read without running code (see `_Resolved`).

    A lambda is transparent: what its body calls, and the names it reads,
    are handed over with it. ``any_of`` and ``unknown`` matter only where the
    callee calls the argument (`_calling_positions`); elsewhere an
    argument is data. A name in *bound* (`iteration_bindings`) is any of
    the items of what the statement iterates; *following* are such names
    being read already."""
    kind = type(arg)
    if kind is ast.Name and not (bound and arg.id in bound):  # the common case, read without the walk
        value = namespace.get(arg.id, _MISSING)  # type: ignore[attr-defined]
        return _Resolved([] if value is _MISSING else [(value, arg.id)], [], [])  # type: ignore[attr-defined]
    if kind is ast.Constant:
        return _NOTHING_HANDED
    out = _Resolved([], [], [])
    pending: list[ast.expr] = [arg]
    while pending:
        node = pending.pop()
        if bound and isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and _bound_root(node, bound):
            _iterated_reference(node, namespace, bound, following, out)
        elif isinstance(node, (ast.Starred, ast.NamedExpr)):
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
            _resolve_call(node, namespace, out, pending, bound, following)
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
    # ``lambda v, g=record: g(v)``: a default is handed over with the lambda.
    for default in (*node.args.defaults, *node.args.kw_defaults):
        if default is not None:
            inner = _resolve_argument(default, namespace)
            out.values.extend(inner.values)
            out.any_of.extend(inner.any_of)
            out.unknown.extend(inner.unknown)
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


def _resolve_call(
    node: ast.Call,
    namespace: Mapping[str, Any],
    out: _Resolved,
    pending: list,
    bound: Mapping[str, tuple[ast.expr, ...]] | None = None,
    following: frozenset[str] = frozenset(),
) -> None:
    """What the result of the call *node* can hand over."""
    callee = _callee_of(node.func, namespace)
    if callee is None and isinstance(node.func, ast.Name) and node.func.id not in namespace:
        callee = vars(builtins).get(node.func.id)
    if callee is builtins.getattr and len(node.args) >= 2 and not node.keywords:
        name = node.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str) and name.value.isidentifier():
            pending.append(ast.Attribute(value=node.args[0], attr=name.value, ctx=ast.Load()))
            pending.extend(node.args[2:])
        elif _of_the_users(_static_value(node.args[0], namespace), node.args[0], bound):
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
        found = _resolve_argument(part, namespace, bound, following)
        inner.any_of.extend(found.values)
        inner.any_of.extend(found.any_of)
        inner.unknown.extend(found.unknown)
    out.any_of.extend(inner.any_of)
    out.unknown.extend(inner.unknown)


def _of_the_users(found: tuple[Any, str] | None, node: ast.expr, bound) -> bool:
    """Whether an attribute of the object *found* (read from *node*) under a
    name known only when it runs may be code of the user's: the user's
    object, class or local module, or one cash cannot read. A library's
    (``getattr(np, name)``) is not."""
    if found is None:
        return not (isinstance(node, ast.Name) and not (bound and node.id in bound))
    value = found[0]
    if isinstance(value, types.ModuleType):
        try:
            return is_local_module(value)
        except (TypeError, AttributeError):
            return True
    return _is_users_object(value) or (isinstance(value, type) and _is_users_class(value))


def _iterated_reference(
    node: ast.Name | ast.Attribute | ast.Subscript,
    namespace: Mapping[str, Any],
    bound: Mapping[str, tuple[ast.expr, ...]],
    following: frozenset[str],
    out: _Resolved,
) -> None:
    """What *node*, rooted at a name the statement binds by iterating, may
    be: the attributes and items it names of each item of the user's (a
    function, an object of their class) that the iterated values hold."""
    steps: list[ast.expr] = []
    root_node: ast.expr = node
    while isinstance(root_node, (ast.Attribute, ast.Subscript)):
        steps.append(root_node)
        root_node = root_node.value
    name = root_node.id  # type: ignore[attr-defined]
    if name in following:
        return  # ``for x in x``: what it iterates is read already
    values: list[tuple[Any, str | None]] = []
    for iterated in bound[name]:
        resolved = _resolve_argument(iterated, namespace, bound, following | {name})
        out.unknown.extend(resolved.unknown)
        for value, root in chain(resolved.values, resolved.any_of):
            if _openable(value) and (type(value) in _OPENED or not callable(value)):
                if not steps:
                    # ``f`` in ``fns``: any of its items, looked into only
                    # where the call calls it (`_Resolved.any_of`).
                    out.any_of.append((value, root))
                    continue
                items, complete = _reached_this_round(value)
                if not complete:
                    out.unknown.append(iterated)
                followed, unknown = _steps_followed(items, steps)
                values.extend((item, root) for item in followed)
                out.unknown.extend(steps[index] for index in unknown)
            else:
                followed, unknown = _follow_steps([value], steps)
                values.extend((item, root) for item in followed)
                out.unknown.extend(steps[index] for index in unknown)
    out.values.extend(values)


def _follow_steps(values: list[Any], steps: list[ast.expr]) -> tuple[list[Any], list[int]]:
    """What *steps* (``.name`` and ``[...]``, the last first) lead to from
    each of *values*, and the indices of the steps that lead somewhere that
    cannot be read without running code of the user's."""
    unknown: list[int] = []
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        following_values: list[Any] = []
        #: ``class -> the function its objects' attribute binds``.
        methods: dict[type, Any] = {}
        for value in values:
            if isinstance(step, ast.Subscript):
                # ``row[0](x) for row in rows``: any item it holds.
                if _openable(value):
                    items, complete = _reached_this_round(value)
                    if not complete:
                        unknown.append(index)
                    following_values.extend(items)
                continue
            kind = type(value)
            method = methods.get(kind, _MISSING)
            if method is not _MISSING and step.attr not in _instance_dict(value):  # type: ignore[attr-defined]
                following_values.append(types.MethodType(method, value))
                continue
            member = _attribute(value, step.attr)  # type: ignore[attr-defined]
            if type(member) is types.MethodType and member.__self__ is value:
                # A function of the class bound to the object: the same for
                # every object of the class whose own dict does not hide it.
                methods[kind] = member.__func__
            if member is not _MISSING:
                following_values.append(member)
            elif _is_users_object(value) or (isinstance(value, type) and _is_users_class(value)):
                unknown.append(index)  # a property or ``__getattr__`` of the user's
        values = following_values
    return _one_per_class(values), unknown


def _one_per_class(values: list[Any]) -> list[Any]:
    """*values* with one of the bound methods of the same function on
    objects of one class whose effect does not depend on the object
    (`_same_for_every_object`): ``r.value`` for each of many rows is
    followed once."""
    out: list[Any] = []
    kept: dict[tuple[Any, type], bool] = {}
    for value in values:
        if type(value) is types.MethodType and not isinstance(value.__self__, type):
            key = (value.__func__, type(value.__self__))
            same = kept.get(key)
            if same:
                continue
            if same is None:
                kept[key] = _is_users(value.__func__) and _same_for_every_object(value)
        out.append(value)
    return out


#: `_follow_steps` of the items `_reached_this_round` found, for the same
#: steps: ``id(items) -> (items, {steps: answer})``. Cleared with it.
_STEPS_THIS_ROUND: dict[int, tuple[list[Any], dict[str, tuple[list[Any], list[int]]]]] = {}


def _steps_followed(items: list[Any], steps: list[ast.expr]) -> tuple[list[Any], list[int]]:
    """`_follow_steps` from the items of a container read this round, once
    per round and steps: ``r.value() for r in rows`` is asked about several
    times before it runs."""
    kept = _STEPS_THIS_ROUND.get(id(items))
    if kept is None or kept[0] is not items:
        if len(_STEPS_THIS_ROUND) > 256:
            _STEPS_THIS_ROUND.clear()
        kept = (items, {})
        _STEPS_THIS_ROUND[id(items)] = kept
    key = "|".join(ast.dump(step) if isinstance(step, ast.Subscript) else step.attr for step in steps)  # type: ignore[attr-defined]
    answer = kept[1].get(key)
    if answer is None:
        answer = _follow_steps(items, steps)
        kept[1][key] = answer
    return answer


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


def _users_objects(values: Iterable[Any]) -> list[Any]:
    """Those of *values* that are objects of the user's classes
    (`_is_users_object`), asked once per type: whether a value is one
    depends on its type alone."""
    values = list(values)
    users = {kind for kind in set(map(type, values)) if _users_instance_type(kind)}
    return list(compress(values, map(users.__contains__, map(type, values)))) if users else []


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


#: ``(class, attribute, in the object's own __dict__) -> what
#: inspect.getattr_static finds``: for objects of one class, whether the
#: attribute is in the object's own ``__dict__`` decides where it is found
#: (the class's data descriptor, the class's member, or the object's own
#: value, `_OWN`). Read once by each question asked of this module (no code
#: of the user's runs while it is answered), not once per object
#: (``r.value() for r in rows``).
_CLASS_ATTRIBUTES: dict[tuple[type, str, bool], Any] = {}
#: ``(method code, class) -> whether it changes its object``, for
#: `_same_for_every_object`; cleared with `_CLASS_ATTRIBUTES`.
_CHANGES_ITS_OBJECT: dict[tuple[Any, type], bool] = {}
_OWN = object()


def _static_attribute(owner: Any, attr: str) -> Any:
    if isinstance(owner, type):
        return inspect.getattr_static(owner, attr)
    own = _instance_dict(owner)
    in_own = attr in own
    key = (type(owner), attr, in_own)
    member = _CLASS_ATTRIBUTES.get(key)
    if member is None:
        try:
            member = inspect.getattr_static(owner, attr)
        except AttributeError:
            member = _MISSING
        if len(_CLASS_ATTRIBUTES) > 1024:
            _CLASS_ATTRIBUTES.clear()
        _CLASS_ATTRIBUTES[key] = _OWN if in_own and member is own[attr] else member
        if member is None:
            return None
    elif member is _OWN:
        return own[attr]
    if member is _MISSING:
        raise AttributeError(attr)
    return member


def _attribute(owner: Any, attr: str) -> Any:
    if isinstance(owner, types.ModuleType):
        return vars(owner).get(attr, _MISSING)
    try:
        member = _static_attribute(owner, attr)
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
        if not isinstance(owner, type) and _instance_dict(owner).get(attr, _MISSING) is member:
            # The object's own value (a bound method or a partial it holds):
            # Python binds no descriptor held there, so it is what is called.
            return member
        # A property or another descriptor computes the value: not read.
        return _MISSING
    return member


def _instance_dict(owner: Any) -> dict:
    try:
        own = object.__getattribute__(owner, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return own if isinstance(own, dict) else {}


#: Special methods a statement naming an object does not run through it:
#: making, setting up and dropping it (an ``__init__`` runs on a call of the
#: class, which is read as such).
_NOT_RUN_BY_A_STATEMENT = frozenset(
    {"__init__", "__new__", "__del__", "__init_subclass__", "__class_getitem__", "__set_name__", "__post_init__"}
)


def is_users_instance(value: Any) -> bool:
    """An object of a class of the user's (not a class, function or module)."""
    return _is_users_object(value)


#: ``class -> (what its classes hold, `_reaching_special_functions`)``: read
#: again only when a class of its MRO holds something else.
_SPECIAL_MEMO: dict[type, tuple[tuple, list[Any]]] = {}


def _reaching_special_functions(cls: type) -> list[Any]:
    """`special_functions` of *cls* whose text changes a global, calls a
    function by name or calls a method of a global object: the others
    change nothing outside their objects, whatever their globals hold."""
    try:
        snapshot = tuple(tuple(vars(k).values()) for k in cls.__mro__ if k is not object)
    except TypeError:
        snapshot = None
    kept = _SPECIAL_MEMO.get(cls)
    if kept is not None and snapshot is not None:
        try:
            same = len(kept[0]) == len(snapshot) and all(
                len(a) == len(b) and all(map(operator.is_, a, b)) for a, b in zip(kept[0], snapshot)
            )
        except Exception:  # noqa: BLE001 - a class attribute's comparison
            same = False
        if same:
            return kept[1]
    out = []
    for fn in special_functions(cls):
        facts = _call_facts(fn)
        source = _source_of(fn)
        if (
            source is None
            or facts is None
            or facts.global_methods
            or source_global_mutations(source)
            or source_called_names(source)
        ):
            out.append(fn)
    if snapshot is not None:
        if len(_SPECIAL_MEMO) > 512:
            _SPECIAL_MEMO.clear()
        _SPECIAL_MEMO[cls] = (snapshot, out)
    return out


def special_functions(cls: type) -> list[Any]:
    """The functions Python runs implicitly on an object of the user's class
    *cls*: its special methods (``__getitem__``, ``__exit__``, ``__add__``,
    ...) and its properties' getters, setters and deleters, from the
    classes of the user's in its MRO."""
    out: list[Any] = []
    for klass in cls.__mro__:
        if klass is object or not _is_users_class(klass):
            continue
        for name, member in list(vars(klass).items()):
            if isinstance(member, property):
                out.extend(fn for fn in (member.fget, member.fset, member.fdel) if fn is not None)
                continue
            if not (name.startswith("__") and name.endswith("__")) or name in _NOT_RUN_BY_A_STATEMENT:
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if isinstance(member, types.FunctionType):
                out.append(member)
    return out


class _Found:
    """What the values handed over lead to, collected over one statement."""

    def __init__(self, namespace: Mapping[str, Any], *, objects: bool = True) -> None:
        self.namespace = namespace
        #: Whether a change a method makes to its own object counts. The
        #: decorator passes False: it hashes the arguments again after a miss,
        #: and an object reached through them that changed stops the result
        #: from being stored.
        self.objects = objects
        self.sources: dict[str, None] = {}
        self.receivers: set[str] = set()
        self.unreadable: list[str] = []
        #: The functions whose sources are in `sources`.
        self.functions: list[types.FunctionType] = []
        #: Whether a callable found changes an object in place.
        self.changes = False
        self._seen: set[tuple[Any, ...]] = set()
        #: ``(object, method code)`` pairs `_held_by` has looked at.
        self._held_seen: set[tuple[int, Any]] = set()
        #: The code of the functions `through_globals` has read.
        self._globals_seen: set[Any] = set()
        #: ``(method code, class) -> (changes its object, reaches its
        #: class)``: a method met on each of many objects of one class is
        #: read once (`_method_effects`).
        self._method_effects: dict[tuple[Any, type], tuple[bool, bool]] = {}
        #: `_method_key` of the methods `_callable` read that neither change
        #: their object nor reach through it: the same method of another
        #: object of that class adds nothing.
        self._inert_methods: set[tuple[Any, type, str | None]] = set()
        #: ``(class, attribute, root)`` that `with_methods` found bound to
        #: such a method.
        self._inert_attributes: set[tuple[type, str, str | None]] = set()
        #: The classes `_special_methods` has read.
        self._special_seen: set[type] = set()
        #: The ids of the functions in `functions`, which it holds.
        self._function_ids: set[int] = set()
        #: ``id(object) -> names of the notebook bound to it``, built on the
        #: first `_changed`.
        self._names_of: dict[int, list[str]] | None = None
        #: ``(method names or None, root) -> ids`` of the containers and
        #: objects whose whole reach a finished walk has looked through for
        #: those names (None: for the callables it holds) under that root:
        #: each method of many objects that share what they hold (an order's
        #: customer, the customer's other orders) walks only what no earlier
        #: one did (`_reach`). No code runs while one statement is read, so
        #: what an object reaches stays the same.
        self._covered: dict[tuple[frozenset[str] | None, str | None], set[int]] = {}
        #: The objects in `_covered`, kept alive so that their ids stay theirs.
        self._covered_objects: list[Any] = []

    def add(self, value: Any, root: str | None, open_containers: bool) -> None:
        """Add what calling *value* changes. A builtin container, or an
        object of the user's class, is opened for the callables it holds
        only with *open_containers*: when the callee calls what it is handed
        there (`_callables_reached`)."""
        kind = type(value)
        if kind is types.MethodType or kind is types.BuiltinMethodType:
            # One of many objects' same method (``r.value() for r in rows``):
            # read once when what it does does not depend on its object.
            if _method_key(value, root) in self._inert_methods:
                return
        if open_containers and _openable(value):
            if (id(value), root, True) in self._seen:
                return
            self._seen.add((id(value), root, True))
            covered, opened = self._reach_key(None, root)
            items, complete = self._reach(value, covered, opened, objects=False)
            if not complete:
                label = repr(root) if root else "an argument"
                self.unreadable.append(
                    f"{label} is handed to a call that calls what it holds, and holds more than "
                    f"{_INSTANCE_LIMIT} objects of your classes to look through"
                )
            for item in items:
                self.add(item, root, False)
            if complete:
                self._cover(covered, opened)
            if type(value) in _OPENED or not callable(value):
                return
        elif type(value) in _OPENED:
            return
        # A bound method is made anew on each attribute read, and its id
        # reused once it is dropped: it is known by its object and function.
        key = (id(value.__self__), id(value.__func__)) if isinstance(value, types.MethodType) else id(value)
        if (key, root) in self._seen:
            return
        self._seen.add((key, root))
        self._callable(value, root)

    def with_methods(self, value: Any, root: str | None, names: frozenset[str], held: bool) -> None:
        """Add the methods and stored functions called *names* of *value*,
        when it is an object of the user's class, that a callee handed it
        calls (``model.predict(v)``); with *held*, of the objects of the
        user's it holds as well (``s.fn(v)`` for ``s`` in ``steps``)."""
        objects = [value] if _is_users_object(value) or (isinstance(value, type) and _is_users_class(value)) else []
        covered: set[int] | None = None
        opened: list[Any] = []
        complete = False
        if held and _openable(value):
            covered, opened = self._reach_key(names, root)
            items, complete = self._reach(value, covered, opened, objects=True)
            if not complete:
                label = repr(root) if root else "an argument"
                self.unreadable.append(
                    f"{label} is handed to a call that calls methods of what it holds, and holds more than "
                    f"{_INSTANCE_LIMIT} objects of your classes to look through"
                )
            objects.extend(_users_objects(items))
        self._methods_of(objects, root, names)
        if covered is not None and complete:
            self._cover(covered, opened)

    def _reach_key(self, names: frozenset[str] | None, root: str | None) -> tuple[set[int], list[Any]]:
        """The ids `_covered` holds for *names* under *root*, and an empty
        list for the walk about to start to note what it opens in."""
        return self._covered.setdefault((names, root), set()), []

    def _reach(self, value: Any, covered: set[int], opened: list[Any], objects: bool) -> tuple[list[Any], bool]:
        """`_callables_reached` of *value*, not opening again what *covered*
        says an earlier walk for the same question looked through: what is
        found there was added then. *opened* gets what this walk opens."""
        if id(value) in covered:
            return [], True
        return _callables_reached(value, objects=objects, covered=covered, opened_out=opened)

    def _cover(self, covered: set[int], opened: list[Any]) -> None:
        """Note that the whole reach of each of *opened*, the containers and
        objects a finished walk opened, has been looked through: called
        only after what the walk found has been added."""
        covered.update(map(id, opened))
        self._covered_objects.extend(opened)

    def _methods_of(self, objects: list[Any], root: str | None, names: frozenset[str]) -> None:
        """Add the methods and stored functions called *names* of each of
        *objects* (`with_methods`), a class at a time.

        ``o.total()`` runs what ``type(o).__mro__`` binds to ``total``,
        read from the class as it is now (a patch ``Order.total = f`` made
        after the objects were built is what is read), unless the object's
        own ``__dict__`` holds that name (``o.total = f`` on one order):
        such an object is looked at by itself. For the others the method is
        one function, bound to each, and `_bound_group` reads it once for
        all of them."""
        groups: dict[type, list[Any]] = {}
        for obj in objects:
            groups.setdefault(type(obj), []).append(obj)
        for kind, group in groups.items():
            if issubclass(kind, type):
                for obj in group:
                    for name in names:
                        self._method_of(obj, kind, name, root)
                continue
            dicts = list(map(_instance_dict, group))
            for name in names:
                rest = []
                for obj, own in zip(group, dicts):
                    if name not in own:
                        rest.append(obj)
                    elif not _plain_own(kind, name, own[name]):
                        self._method_of(obj, kind, name, root)
                    elif type(own[name]) is types.FunctionType:
                        self.add(own[name], root, False)  # what `_attribute` reads: the function itself
                    elif callable(own[name]):
                        self._method_of(obj, kind, name, root)
                if not rest or (kind, name, root) in self._inert_attributes:
                    continue
                sample = rest[0]
                member = _attribute(sample, name)
                if member is _MISSING or not callable(member):
                    continue
                if type(member) is types.MethodType and member.__self__ is sample:
                    self._bound_group(member.__func__, rest, root)
                    if (member.__func__, kind, root) in self._inert_methods:
                        self._inert_attributes.add((kind, name, root))
                elif getattr(member, "__self__", None) is sample:
                    # A builtin's method bound to each object (a subclass of
                    # ``list``'s ``append``): each one by itself.
                    for obj in rest:
                        self._method_of(obj, kind, name, root)
                else:
                    # The class's own callable, the same for every object.
                    self._method_of(sample, kind, name, root)

    def _method_of(self, obj: Any, kind: type, name: str, root: str | None) -> None:
        """Add the method or stored function called *name* of *obj*."""
        if (kind, name, root) in self._inert_attributes and name not in _instance_dict(obj):
            return  # another object's same method, read already
        member = _attribute(obj, name)
        if member is not _MISSING and callable(member):
            self.add(member, root, False)
            if type(member) is types.MethodType and _method_key(member, root) in self._inert_methods:
                self._inert_attributes.add((kind, name, root))

    def through_globals(self, fn: types.FunctionType) -> None:
        """What the function *fn* of the user's, and the ones of theirs it
        calls by name, run through the objects they read by a free name: a
        method of an object of the user's class (``bus.emit(x)``), and the
        handlers of a logger (``lg.info(...)``) that are."""
        pending = [fn]
        while pending:
            current = pending.pop()
            if current.__code__ in self._globals_seen:
                continue
            self._globals_seen.add(current.__code__)
            self._closed_over(current)
            facts = _call_facts(current)
            if facts is None:
                continue
            scope = current.__globals__
            root_of = (lambda name: name) if scope is self.namespace else (lambda name: None)
            for name, attr in facts.global_methods:
                owner = scope.get(name, _MISSING)
                if owner is _MISSING:
                    continue
                if _is_users_object(owner) or (isinstance(owner, type) and _is_users_class(owner)):
                    member = _attribute(owner, attr)
                    if member is not _MISSING and callable(member):
                        self.add(member, root_of(name), False)
                else:
                    for callback in _logging_callbacks(owner, attr):
                        self.add(callback, None, False)
            for name in facts.named_calls:
                target = scope.get(name)
                if isinstance(target, types.FunctionType) and self._users(target):
                    pending.append(target)

    def _closed_over(self, fn: types.FunctionType) -> None:
        """The callables of the user's the function *fn* closes over: a
        wrapper a decorator made without ``functools.wraps`` (``def
        call(*a): return fn(*a)``), a ``compose(f, g)`` closure. Calling *fn*
        calls them, as far as cash can tell."""
        for cell in fn.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue  # not bound yet
            if _is_users_callable(value) or (isinstance(value, types.FunctionType) and self._users(value)):
                self.add(value, None, False)

    def held_by_its_object(self, func: ast.Attribute, namespace: Mapping[str, Any]) -> None:
        """A method of the user's called by name (``bus.emit(3)``) that calls
        what its object holds (``cb(x)`` for ``cb`` in ``self.subs``) or
        another of its methods: what those change, the call changes."""
        found = _users_target(func, namespace)
        if found is None or not isinstance(found[0], types.MethodType):
            return
        owner, fn = found[0].__self__, found[0].__func__
        if isinstance(owner, (type, types.ModuleType)) or not isinstance(fn, types.FunctionType):
            return
        if self._users(fn):
            self._held_by(owner, fn, found[1])

    def _held_by(self, owner: Any, fn: types.FunctionType, root: str | None) -> None:
        """What the method *fn* of *owner* reaches through its first
        parameter (`_held_by_many`)."""
        self._held_by_many([owner], fn, root)

    def _held_by_many(self, owners: list[Any], fn: types.FunctionType, root: str | None) -> None:
        """What the method *fn* of each of *owners*, objects of one class,
        reaches through its first parameter: what they hold that it calls,
        and the methods of theirs (or of what they hold) it calls.

        Asked of all of them at once, not of each in turn: what *fn* looks
        for is the same for every object (its source says which attributes
        and parameters it calls), only what the objects hold differs, and
        that is gathered for all of them by one walk (`_add_open_many`,
        `_with_methods_many`). The union of what each object leads to is
        what the walk from all of them leads to, so nothing one of them held
        is missed."""
        code = fn.__code__
        owners = [owner for owner in owners if (id(owner), code) not in self._held_seen]
        if not owners:
            return
        self._held_seen.update((id(owner), code) for owner in owners)
        facts = _call_facts(fn)
        if facts is None or not code.co_argcount:
            return
        first = code.co_varnames[0]
        if first in _called_params(fn):
            self._add_open_many(owners, root)
        if facts.attributes and (first in facts.attribute_called or first in facts.held_attribute_called):
            self._with_methods_many(owners, root, facts.attributes, first in facts.held_attribute_called)
        if facts.super_methods:
            # ``super().name(...)`` resolves along the class's MRO: the same
            # function for every one of *owners*, bound to each.
            sample = owners[0]
            for member in _super_members(sample, fn, facts.super_methods):
                if type(member) is types.MethodType and member.__self__ is sample:
                    self._bound_group(member.__func__, owners, root)
                else:
                    self.add(member, root, False)

    def _bound_group(self, func: Any, owners: list[Any], root: str | None) -> None:
        """`add` of the function *func* bound to each of *owners*, objects
        of one class: what `add` checks first (a method read already, or
        found not to depend on its object), then `_bound_effects` once for
        the objects not seen yet."""
        kind = type(owners[0])
        if (func, kind, root) in self._inert_methods:
            return
        fid = id(func)
        owners = [owner for owner in owners if ((id(owner), fid), root) not in self._seen]
        if not owners:
            return
        self._seen.update(((id(owner), fid), root) for owner in owners)
        self._bound_effects(func, owners, root)

    def _bound_effects(self, func: Any, owners: list[Any], root: str | None) -> None:
        """What calling the method *func* bound to each of *owners* changes
        (`_callable` of a bound method). *owners* are of one class (or a
        single class or module): its source, and whether it changes its
        object or reaches through it, are read once for the class; the
        objects it changes are each noted, and what it reaches through them
        is asked of all of them at once (`_held_by_many`)."""
        owner = owners[0]
        label = repr(root) if root else "a function"
        cls = owner if isinstance(owner, type) else type(owner)
        inert = not isinstance(owner, types.ModuleType)
        for fn in wrapper_layers(func):
            if not isinstance(fn, types.FunctionType) or not self._users(fn):
                continue
            source = self._source(fn, label)
            if source is not None and not isinstance(owner, types.ModuleType):
                changes, reaches_class = self._method_effects_of(fn, cls, source)
                if changes:
                    inert = False
                    if self.objects or isinstance(owner, type):
                        for each in owners:
                            self._changed(each, root)
                    if not isinstance(owner, type) and reaches_class:
                        self._changed(cls, None)
                if not isinstance(owner, type) and _reaches_through_its_object(fn):
                    inert = False
                    self._held_by_many(owners, fn, root)
        if inert:
            self._inert_methods.add((func, cls, root))

    def _add_open_many(self, owners: list[Any], root: str | None) -> None:
        """`add` of each of *owners*, objects of the user's, opened for the
        callables they hold: one walk from all of them (`_reached_from`),
        allowed `_INSTANCE_LIMIT` objects for each, so it stops only where
        a walk of each would have too."""
        owners = [owner for owner in owners if (id(owner), root, True) not in self._seen]
        if not owners:
            return
        self._seen.update((id(owner), root, True) for owner in owners)
        covered, opened = self._reach_key(None, root)
        sources = [owner for owner in owners if id(owner) not in covered]
        items, complete = (
            _reached_from(sources, False, covered, opened, _INSTANCE_LIMIT * len(sources)) if sources else ([], True)
        )
        if not complete:
            label = repr(root) if root else "an argument"
            self.unreadable.append(
                f"{label} is handed to a call that calls what it holds, and holds more than "
                f"{_INSTANCE_LIMIT} objects of your classes to look through"
            )
        for item in items:
            self.add(item, root, False)
        if complete:
            self._cover(covered, opened)
        for owner in owners:
            if callable(owner) and (id(owner), root) not in self._seen:
                self._seen.add((id(owner), root))
                self._callable(owner, root)

    def _with_methods_many(self, owners: list[Any], root: str | None, names: frozenset[str], held: bool) -> None:
        """`with_methods` of each of *owners*, objects of the user's, at
        once: with *held*, one walk from all of them (as `_add_open_many`)."""
        objects = list(owners)
        complete = False
        covered: set[int] | None = None
        opened: list[Any] = []
        if held:
            covered, opened = self._reach_key(names, root)
            sources = [owner for owner in owners if id(owner) not in covered]
            items, complete = (
                _reached_from(sources, True, covered, opened, _INSTANCE_LIMIT * len(sources)) if sources else ([], True)
            )
            if not complete:
                label = repr(root) if root else "an argument"
                self.unreadable.append(
                    f"{label} is handed to a call that calls methods of what it holds, and holds more than "
                    f"{_INSTANCE_LIMIT} objects of your classes to look through"
                )
            objects.extend(_users_objects(items))
        self._methods_of(objects, root, names)
        if covered is not None and complete:
            self._cover(covered, opened)

    def called_by_name(self, func: ast.Name | ast.Attribute, namespace: Mapping[str, Any]) -> None:
        """A function of the user's called by name (``fire(3)``) runs what
        it reaches through the objects it reads (`through_globals`); a
        logger's method (``lg.info(...)``) runs the handlers of the user's
        it has."""
        target = _users_target(func, namespace)
        if target is not None:
            value = target[0]
            if isinstance(func, ast.Attribute):
                self.method_called(value)
            if isinstance(value, types.FunctionType) and self._users(value):
                self.through_globals(value)
            return
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = namespace.get(func.value.id, _MISSING)
            if owner is not _MISSING:
                for callback in _logging_callbacks(owner, func.attr):
                    self.add(callback, None, False)

    def wrapping_object_called(
        self, call: ast.Call, namespace: Mapping[str, Any], bound: Mapping[str, tuple[ast.expr, ...]]
    ) -> None:
        """A callable object that wraps a function, called by its own name:
        ``pz()`` with ``pz = functools.partial(record, 'Z')``, ``vz(xs)`` with
        ``vz = np.vectorize(record)``, ``mc(h)`` with ``mc =
        operator.methodcaller('rec', 'Z')`` (the method of that name of each
        object of the user's it is handed)."""
        target = _users_target(call.func, namespace)  # type: ignore[arg-type]
        if target is None:
            return
        value, root = target
        if isinstance(value, functools.partial):
            self.add(value, root, False)
        elif isinstance(value, operator.methodcaller):
            self._methods_named(value.__reduce__()[1][0], call, namespace, bound)
        elif isinstance(value, types.FunctionType) and isinstance(getattr(value, "registry", None), Mapping):
            # ``functools.singledispatch``: any implementation registered.
            for impl in list(value.registry.values()):
                if _is_users_callable(impl):
                    self.add(impl, root, False)
        else:
            pyfunc = getattr(value, "pyfunc", None) if type(value).__name__ == "vectorize" else None
            if pyfunc is not None and _is_users_callable(pyfunc):
                self.add(pyfunc, root, False)

    def methodcaller_called(
        self, call: ast.Call, namespace: Mapping[str, Any], bound: Mapping[str, tuple[ast.expr, ...]]
    ) -> None:
        """``operator.methodcaller('add', v)(rec)``: the method of that name
        of what it is handed."""
        maker = call.func
        if not (
            isinstance(maker, ast.Call)
            and isinstance(maker.func, (ast.Name, ast.Attribute))
            and maker.args
            and isinstance(maker.args[0], ast.Constant)
            and isinstance(maker.args[0].value, str)
        ):
            return
        target = _users_target(maker.func, namespace)
        if target is not None and target[0] is operator.methodcaller:
            self._methods_named(maker.args[0].value, call, namespace, bound)

    def _methods_named(
        self, name: str, call: ast.Call, namespace: Mapping[str, Any], bound: Mapping[str, tuple[ast.expr, ...]]
    ) -> None:
        for arg in call.args[:1]:
            for obj, obj_root in _resolve_argument(arg, namespace, bound).values:
                if _is_users_object(obj) or (isinstance(obj, type) and _is_users_class(obj)):
                    member = _attribute(obj, name)
                    if member is not _MISSING and callable(member):
                        self.add(member, obj_root, False)

    def run_implicitly(self, tree: ast.AST, namespace: Mapping[str, Any], scope: CallScope) -> None:
        """The methods of the user's that Python runs for the statement
        without a call in its text, as callees (`method_called`): a context
        manager's ``__enter__`` and ``__exit__`` (``with timer('fit'):``), a
        property's getter or setter (``bag.value``), and the special
        methods of an object of the user's class it names (``bag['k']``,
        ``len(bag)``, ``bag + 1``, ``f'{bag}'``, iterating ``bag``): which of
        them an expression runs is not worked out, so all of them count."""
        for node in nodes_in(tree, scope):
            if isinstance(node, ast.withitem):
                manager = node.context_expr
                if isinstance(manager, ast.Call) and isinstance(manager.func, (ast.Name, ast.Attribute)):
                    target = _users_target(manager.func, namespace)
                    if target is not None and isinstance(target[0], type) and _is_users_class(target[0]):
                        self._special_methods(target[0])
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                value = namespace.get(node.id, _MISSING)
                if value is not _MISSING and _is_users_object(value):
                    self._special_methods(type(value))
            elif isinstance(node, ast.Attribute):
                owner = _static_value(node.value, namespace)
                if owner is None:
                    continue
                if not _is_users_object(owner[0]):
                    continue  # a class's attribute: a property runs on objects only
                cls = type(owner[0])
                try:
                    member = inspect.getattr_static(cls, node.attr)
                except AttributeError:
                    member = _MISSING
                if isinstance(member, property):
                    for fn in (member.fget, member.fset, member.fdel):
                        self.method_called(fn)
                elif member is not _MISSING and _is_users_object(member):
                    # A descriptor of the user's class (``__get__``).
                    self._special_methods(type(member))
                member = _attribute(owner[0], node.attr)
                if member is not _MISSING and _is_users_object(member):
                    self._special_methods(type(member))

    def _special_methods(self, cls: type) -> None:
        """The special methods (``__getitem__``, ``__exit__``, ...) and the
        property getters the user's class *cls* defines, as callees."""
        if cls in self._special_seen:
            return
        self._special_seen.add(cls)
        for fn in _reaching_special_functions(cls):
            self.method_called(fn)

    def method_called(self, value: Any) -> None:
        """A method of the user's class called on its object or class
        (``store.process(r)``, ``Store.process(r)``, a static or class
        method): the call runs its body, as a call of a function by name
        does, so its source is a callee's (what it changes of the globals,
        and what it calls by name). A method whose source cannot be read is
        left as before: what Python runs for it is not the user's text."""
        if isinstance(value, types.MethodType):
            fn = value.__func__
        elif isinstance(value, types.FunctionType):
            fn = value
        else:
            return
        for layer in wrapper_layers(fn):
            if not isinstance(layer, types.FunctionType) or not self._users(layer):
                continue
            source = _source_of(layer)
            if source is None:
                continue
            self.sources.setdefault(source)
            if id(layer) not in self._function_ids:
                self._function_ids.add(id(layer))
                self.functions.append(layer)
            self.through_globals(layer)

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
            if self._names_of is None:
                self._names_of = {}
                for name, bound in list(self.namespace.items()):
                    self._names_of.setdefault(id(bound), []).append(name)
            self.receivers.update(self._names_of.get(id(obj), ()))

    def _method_effects_of(self, fn: types.FunctionType, cls: type, source: str) -> tuple[bool, bool]:
        """Whether the method *fn* of *cls* changes its object, and whether
        it reaches its class: read once per method and class, not once per
        object (``[r.value() for r in rows]``)."""
        key = (fn.__code__, cls)
        effects = self._method_effects.get(key)
        if effects is None:
            changes = _changes_its_first_argument(fn, cls, source)
            effects = (changes, changes and _reaches_its_class(source))
            self._method_effects[key] = effects
        return effects

    def _source(self, fn: types.FunctionType, label: str) -> str | None:
        source = _source_of(fn)
        if source is None:
            self.unreadable.append(f"{label} is handed to a call and its source cannot be read")
            return None
        self.sources.setdefault(source)
        if id(fn) not in self._function_ids:
            self._function_ids.add(id(fn))
            self.functions.append(fn)
        return source

    def _callable(self, value: Any, root: str | None) -> None:
        label = repr(root) if root else "a function"
        if isinstance(value, functools.partial):
            # ``partial(apply_, record)``: ``apply_`` runs ``record``.
            opened = _opened_by(value.func, None)
            self.add(value.func, root, False)
            for position, arg in enumerate(value.args):
                if type(arg) not in _ATOMS:
                    self.add(arg, root, opened.opens(position, None))
            for keyword, arg in value.keywords.items():
                if type(arg) not in _ATOMS:
                    self.add(arg, root, opened.opens(None, keyword))
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
                if source is not None:
                    self.through_globals(layer)
            return
        if isinstance(value, types.MethodType):
            self._bound_effects(value.__func__, [value.__self__], root)
            return
        if isinstance(value, (types.BuiltinMethodType, types.MethodWrapperType)):
            owner = getattr(value, "__self__", None)
            if owner is None or isinstance(owner, (types.ModuleType, type)):
                return
            if getattr(value, "__name__", "") in MUTATING_METHODS:
                self._changed(owner, root)
            elif isinstance(value, types.BuiltinMethodType):
                self._inert_methods.add(_method_key(value, root))
            return
        if isinstance(value, type):
            init = _class_member(value, "__init__")
            if _generated_dataclass_init(value, init):
                # ``extract(doc, InvoiceFields)``: the ``__init__`` that
                # ``@dataclass`` writes has no source, and only sets the
                # fields of the new object. What it runs of the user's is the
                # field factories and ``__post_init__``.
                for spec in dataclasses.fields(value):
                    if spec.default_factory is not dataclasses.MISSING:
                        self.add(spec.default_factory, root, False)
                init = _class_member(value, "__post_init__")
            if isinstance(init, types.FunctionType) and self._users(init):
                self._source(init, label)
            return
        if not callable(value) or isinstance(value, types.ModuleType):
            return
        call = _class_member(type(value), "__call__")
        if isinstance(call, types.FunctionType) and self._users(call):
            source = self._source(call, label)
            if source is not None and _changes_its_first_argument(call, type(value), source):
                if self.objects:
                    self._changed(value, root)
                if _reaches_its_class(source):
                    self._changed(type(value), None)

    def _users(self, fn: types.FunctionType) -> bool:
        return fn.__globals__ is self.namespace or _is_users(fn)


def _method_key(method: Any, root: str | None) -> tuple[Any, type, str | None]:
    """The same method of every object of one class: a bound method's
    function, a builtin method's
    name, with the class of the object it is bound to."""
    owner = method.__self__
    cls = owner if isinstance(owner, type) else type(owner)
    if type(method) is types.MethodType:
        return (method.__func__, cls, root)
    return (method.__name__, cls, root)


#: `_reaches_through_its_object` of each function asked about: ``(code,
#: id(globals)) -> answer``, holding no function (a reference cash keeps to
#: one would read as another holder to the share check). Cleared with
#: `_CLASS_ATTRIBUTES`: asked once per question, not once for each of the
#: many objects a method is bound to.
_THROUGH_ITS_OBJECT: dict[tuple[types.CodeType, int], bool] = {}


def _reaches_through_its_object(fn: types.FunctionType) -> bool:
    """Whether `_Found._held_by` finds anything for the method *fn* on some
    object: it calls its first parameter, or calls methods on it or on
    what it holds. What it finds depends on the object; whether it looks
    does not."""
    key = (fn.__code__, id(fn.__globals__))
    known = _THROUGH_ITS_OBJECT.get(key)
    if known is None:
        known = _THROUGH_ITS_OBJECT[key] = _looks_through_its_object(fn)
    return known


def _looks_through_its_object(fn: types.FunctionType) -> bool:
    facts = _call_facts(fn)
    code = fn.__code__
    if facts is None or not code.co_argcount:
        return False
    first = code.co_varnames[0]
    if first in _called_params(fn) or facts.super_methods:
        return True
    return bool(facts.attributes) and (first in facts.attribute_called or first in facts.held_attribute_called)


def _super_members(owner: Any, fn: types.FunctionType, names: frozenset[str]) -> list[Any]:
    """The methods *names* that ``super().name(...)`` in the method *fn* of
    *owner* reaches: the next class's in the MRO of *owner*'s class after
    the one defining *fn*, bound to *owner*."""
    if not names:
        return []
    mro = type(owner).__mro__
    start = next(
        (i for i, k in enumerate(mro) if any(getattr(v, "__code__", None) is fn.__code__ for v in vars(k).values())),
        None,
    )
    if start is None:
        return []
    out = []
    for name in names:
        for k in mro[start + 1 :]:
            member = vars(k).get(name, _MISSING)
            if member is _MISSING:
                continue
            if isinstance(member, staticmethod):
                out.append(member.__func__)
            elif isinstance(member, classmethod):
                out.append(types.MethodType(member.__func__, type(owner)))
            elif isinstance(member, types.FunctionType):
                out.append(types.MethodType(member, owner))
            break
    return out


def _generated_dataclass_init(cls: type, init: Any) -> bool:
    """Whether *init* is the ``__init__`` that ``@dataclass`` wrote for
    *cls*: compiled from a string, so it has no source to read."""
    return (
        isinstance(init, types.FunctionType)
        and dataclasses.is_dataclass(cls)
        and init.__code__.co_filename.startswith("<")
        and _source_of(init) is None
    )


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


def hands_a_state_changing_callable(
    fn: Any, args: tuple, kwargs: Mapping[str, Any], *, through_globals: bool = True, objects: bool = True
) -> bool:
    """Whether calling *fn* with *args* and *kwargs* hands it a user callable
    that changes state when called: a function that changes a global or what
    it closes over, a method or a callable object that changes its object,
    or one whose source cannot be read. A builtin container or an object of
    the user's class among the arguments is looked into when *fn* calls
    what it is handed there.

    ``run_steps(steps, data)`` with ``steps = [f]`` runs ``f``; served from
    the cache, ``f``'s writes would not happen.

    *through_globals*: also count what *fn* reaches through the objects it
    reads by a free name (``bus.emit()`` running what ``bus`` holds). The
    decorator passes False: what a cached function reaches through its own
    globals is its own purity analysis's to report (IMPURE-SCOPE-MUTATION,
    IMPURE-SIDE-EFFECTS), and only the arguments are new on each call.

    *objects*: count a method changing its own object (`_Found.objects`). The
    decorator passes False, as it watches the arguments for that itself."""
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()
    found: _Found | None = None
    opened: _Opened | None = None
    for position, keyword, value in chain(
        zip(range(len(args)), repeat(None), args), zip(repeat(None), kwargs.keys(), kwargs.values())
    ):
        if type(value) in _ATOMS:
            continue
        if _is_users_object(value):
            # ``with cm:``, ``b.value``, ``b[k]`` in the body run the special
            # methods and properties of the object's class.
            if found is None:
                found = _Found({}, objects=objects)
            found._special_methods(type(value))
        opens = False
        methods = 0
        if _openable(value):
            if opened is None:
                opened = _opened_by(fn, None)
            opens = opened.opens(position, keyword)
            methods = opened.calls_methods(position, keyword)
            if not opens and not methods and not callable(value):
                continue
        elif not callable(value):
            continue
        if found is None:
            found = _Found({}, objects=objects)
        if methods:
            found.with_methods(value, None, opened.attributes, methods == 2)  # type: ignore[union-attr]
        found.add(value, None, opens)
    registry = getattr(fn, "registry", None) if isinstance(fn, types.FunctionType) else None
    if isinstance(registry, Mapping):
        # ``functools.singledispatch``: any implementation registered runs.
        for impl in list(registry.values()):
            if _is_users_callable(impl):
                if found is None:
                    found = _Found({}, objects=objects)
                found.add(impl, None, False)
    own = _user_function_of(fn)[0] if through_globals else None
    if own is not None and (_reaches_through_globals(own) or _closes_over_a_callable(own)):
        # ``fire(3)`` running ``bus.emit``, which runs what ``bus`` holds.
        if found is None:
            found = _Found({}, objects=objects)
        found.through_globals(own)
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


def _closes_over_a_callable(fn: types.FunctionType) -> bool:
    """Whether *fn* closes over a callable of the user's (a wrapper made
    without ``functools.wraps``, a ``compose(f, g)`` closure), which
    `_Found.through_globals` follows. A ``functools.wraps`` wrapper's
    wrapped function is not one: the call cache reads each layer itself
    (`wrapper_layers`) and captures what it changes."""
    if not fn.__closure__:
        return False
    layers = {id(layer) for layer in wrapper_layers(fn)}
    for cell in fn.__closure__:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if id(value) not in layers and _is_users_callable(value):
            return True
    return False


def _reaches_through_globals(fn: types.FunctionType) -> bool:
    """Whether *fn*, or a function of the user's it calls by name, calls a
    method of an object it reads by a free name (`_CallFacts.global_methods`):
    a quick answer before `_Found.through_globals` looks at the objects."""
    pending, seen = [fn], set()
    while pending:
        current = pending.pop()
        if current.__code__ in seen:
            continue
        seen.add(current.__code__)
        facts = _call_facts(current)
        if facts is None:
            continue
        scope = current.__globals__
        for name, attr in facts.global_methods:
            owner = scope.get(name)
            if _is_users_object(owner) or (isinstance(owner, type) and _is_users_class(owner)):
                return True
            if attr in _LOGGING_CALLS and _logging_callbacks(owner, attr):
                return True
        for name in facts.named_calls:
            target = scope.get(name)
            if isinstance(target, types.FunctionType) and _is_users(target):
                pending.append(target)
    return False


def _global_source(namespace: Mapping[str, Any], name: str) -> str | None:
    value = namespace.get(name)
    if not isinstance(value, types.FunctionType):
        return None
    if _is_cashs(value):
        # A ``@cash.cache`` function: its body, not cash's wrapper around it,
        # whose bookkeeping is no state of the user's.
        wrapped = getattr(value, "__wrapped__", None)
        return _source_of(wrapped) if isinstance(wrapped, types.FunctionType) else None
    return _source_of(value)


def _is_cashs(fn: types.FunctionType) -> bool:
    """Whether *fn* is cash's own code (a wrapper ``@cash.cache`` made)."""
    name = fn.__globals__.get("__name__")
    return isinstance(name, str) and (name == "cash" or name.startswith("cash."))


class _Opened(NamedTuple):
    """Which arguments of a call the callee calls what is inside of."""

    positions: frozenset[int] = frozenset()
    keywords: frozenset[str] = frozenset()
    #: The position a ``*args`` the callee calls into starts at, if any.
    star_from: int | None = None
    #: Whether a ``**kwargs`` the callee calls into takes the other keywords.
    double_star: bool = False
    #: The arguments, by position and by keyword, the callee calls a method
    #: of (``model.predict(v)``), and those it calls a method of something
    #: they hold of (``s.fn(v)`` for ``s`` in ``steps``).
    attribute: tuple[frozenset[int], frozenset[str]] = (frozenset(), frozenset())
    held_attribute: tuple[frozenset[int], frozenset[str]] = (frozenset(), frozenset())
    #: The names of those methods.
    attributes: frozenset[str] = frozenset()

    def opens(self, position: int | None, keyword: str | None) -> bool:
        if keyword == DOUBLE_STAR:  # a ``**kw`` unpacking: any keyword
            return bool(self.keywords) or self.double_star or self.star_from is not None
        if keyword is not None:
            return keyword in self.keywords or self.double_star
        if position is None:  # after a ``*args`` unpacking: any position
            return bool(self.positions or self.keywords) or self.star_from is not None
        return position in self.positions or (self.star_from is not None and position >= self.star_from)

    def calls_methods(self, position: int | None, keyword: str | None) -> int:
        """0 when the callee calls no method of the argument, 1 when it calls
        one of the argument itself, 2 when of something it holds."""
        if not self.attributes:
            return 0
        for level, (positions, keywords) in ((2, self.held_attribute), (1, self.attribute)):
            if keyword is not None:
                if keyword in keywords or (keyword == DOUBLE_STAR and keywords):
                    return level
            elif position is None:
                if positions or keywords:
                    return level
            elif position in positions:
                return level
        return 0


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
    facts = _call_facts(fn)
    if not called and (facts is None or not facts.attributes):
        return _OPENS_NOTHING
    code = fn.__code__
    names = code.co_varnames[: code.co_argcount]

    def where(params: frozenset[str] | set[str]) -> tuple[frozenset[int], frozenset[str]]:
        positions = frozenset(i - offset for i, name in enumerate(names) if name in params and i >= offset)
        keywords = frozenset(n for n in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount] if n in params)
        return positions, keywords

    positions, keywords = where(called)
    extra = code.co_argcount + code.co_kwonlyargcount
    star_from = None
    if code.co_flags & inspect.CO_VARARGS:
        if code.co_varnames[extra] in called:
            star_from = max(0, code.co_argcount - offset)
        extra += 1
    double_star = bool(code.co_flags & inspect.CO_VARKEYWORDS) and code.co_varnames[extra] in called
    if facts is None or not facts.attributes:
        return _Opened(positions, keywords, star_from, double_star)
    return _Opened(
        positions,
        keywords,
        star_from,
        double_star,
        where(facts.attribute_called),
        where(facts.held_attribute_called),
        facts.attributes,
    )


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
    #: Parameters the body calls a method of, or a function stored on
    #: (``model.predict(v)``).
    attribute_called: frozenset[str] = frozenset()
    #: Parameters the body calls a method of something they hold of
    #: (``s.fn(v)`` for ``s`` in ``steps``, ``self.model.fit(x)``).
    held_attribute_called: frozenset[str] = frozenset()
    #: The names of those methods and stored functions.
    attributes: frozenset[str] = frozenset()
    #: ``(name, attribute)`` of each method the body calls on an object it
    #: reads by a free name (``bus.emit(x)``, ``lg.info(...)``).
    global_methods: frozenset[tuple[str, str]] = frozenset()
    #: The free names the body calls (``helper(x)``).
    named_calls: frozenset[str] = frozenset()
    #: The methods the body calls through ``super()`` (``super().run(r)``).
    super_methods: frozenset[str] = frozenset()


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


def _read_call_facts(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> _CallFacts:
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
    attribute_called: set[str] = set()
    held_attribute_called: set[str] = set()
    attributes: set[str] = set()
    global_methods: set[tuple[str, str]] = set()
    named_calls: set[str] = set()
    super_methods: set[str] = set()
    local = params | {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            called |= roots(call.func)
            if isinstance(call.func, ast.Name) and call.func.id not in local:
                named_calls.add(call.func.id)
        else:
            owner = call.func.value
            if isinstance(owner, ast.Name) and owner.id in params:
                attribute_called.add(owner.id)
                attributes.add(call.func.attr)
            elif reached := roots(owner):
                held_attribute_called.update(reached)
                attributes.add(call.func.attr)
            elif isinstance(owner, ast.Name) and owner.id not in local:
                global_methods.add((owner.id, call.func.attr))
            elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super":
                super_methods.add(call.func.attr)
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
    return _CallFacts(
        frozenset(called),
        tuple(handoffs),
        frozenset(attribute_called),
        frozenset(held_attribute_called),
        frozenset(attributes),
        frozenset(global_methods),
        frozenset(named_calls),
        frozenset(super_methods),
    )


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
                or keyword == DOUBLE_STAR
            ):
                called |= params
    return frozenset(called)


#: What a logger's method or the logging module's function called hands
#: a record to the handlers.
_LOGGING_CALLS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log", "handle"}
)


def _logging_callbacks(owner: Any, attr: str) -> list[Any]:
    """The methods of the user's that ``owner.attr(...)`` runs when *owner*
    is a logger (or the logging module) and *attr* logs: those of each
    handler, filter and formatter of the user's class along the logger's
    parents. Empty for anything else; the logging module is not imported
    for it."""
    if attr not in _LOGGING_CALLS:
        return []
    logging_module = sys.modules.get("logging")
    if logging_module is None:
        return []
    if owner is logging_module:
        logger: Any = logging_module.getLogger()
    elif isinstance(owner, logging_module.Logger):
        logger = owner
    else:
        return []
    found: list[Any] = []

    def add(obj: Any, names: tuple[str, ...]) -> None:
        if _is_users_object(obj):
            for name in names:
                member = _attribute(obj, name)
                if member is not _MISSING and callable(member):
                    found.append(member)
        elif callable(obj) and _is_users_callable(obj):
            found.append(obj)  # a filter given as a function

    while logger is not None:
        for flt in getattr(logger, "filters", ()):
            add(flt, ("filter",))
        for handler in getattr(logger, "handlers", ()):
            add(handler, ("emit", "handle", "filter", "format", "flush"))
            for flt in getattr(handler, "filters", ()):
                add(flt, ("filter",))
            add(getattr(handler, "formatter", None), ("format", "formatTime", "formatException"))
        logger = logger.parent if getattr(logger, "propagate", False) else None
    return found


def _class_member(cls: type, name: str) -> Any:
    try:
        return inspect.getattr_static(cls, name)
    except AttributeError:
        return None


def _callables_reached(
    value: Any, objects: bool = False, covered: set[int] | None = None, opened_out: list[Any] | None = None
) -> tuple[list[Any], bool]:
    """``(callables, complete)``: the callables *value* holds -- a builtin
    container's items, an object of the user's class's attributes (its
    ``__dict__`` and ``__slots__``, read without running code) -- and those
    the containers and objects of the user's among them hold, however deeply
    nested, each once. A level at a time, the types sorted at C speed.
    *complete* is False when more than `_INSTANCE_LIMIT` objects of the
    user's classes had to be opened, and the walk stopped. With *objects*,
    the objects of the user's classes it holds as well.

    *covered* are the ids of containers and objects whose whole reach the
    caller has already looked through for the same question: the walk does
    not open them again (`_Found._reach`). *opened_out*, when given, gets
    *value* and each container and object the walk opened."""
    if type(value) in (list, tuple) and set(map(type, value)) <= _ATOMS:
        return [], True  # a flat list of plain values: read at C speed
    if _plain_data.is_tree(value):
        return [], True  # lists and tuples of plain values only
    return _reached_from([value], objects, covered, opened_out, _INSTANCE_LIMIT)


def _reached_from(
    sources: list[Any], objects: bool, covered: set[int] | None, opened_out: list[Any] | None, limit: int
) -> tuple[list[Any], bool]:
    """`_callables_reached` of each of *sources* at once: what any of them
    reaches, each object opened once, the sources themselves left out of
    what is found. Incomplete past *limit* objects of the user's classes.

    One walk for many sources (the orders a method is called on) where a
    walk each opened what they share once per source. Asked with a *limit*
    of `_INSTANCE_LIMIT` for each source, it is never cut short where a walk
    of each would all have finished: together they reach no more than the
    sum of what each reaches."""
    found: list[Any] = []
    seen = set(map(id, sources))
    if opened_out is not None:
        opened_out.extend(sources)
    level = list(sources)
    users_objects = 0
    #: ``type -> whether the walk opens its objects``, asked once per type.
    opens: dict[type, bool] = {}
    while level:
        items = list(chain.from_iterable(map(_items, level)))
        if not items:
            break
        found.extend(filter(callable, items))
        opened = set()
        for kind in set(map(type, items)):
            known = opens.get(kind)
            if known is None:
                known = opens[kind] = kind in _OPENED or _users_instance_type(kind)
            if known:
                opened.add(kind)
        if not opened:
            break
        nested = [c for c in compress(items, map(opened.__contains__, map(type, items))) if id(c) not in seen]
        if covered:
            nested = [c for c in nested if id(c) not in covered]
        # Held twice in one level (an order's customer, met from each of
        # its orders): opened once.
        nested = list(dict(zip(map(id, nested), nested)).values())
        seen.update(map(id, nested))
        if opened_out is not None:
            opened_out.extend(nested)
        if objects:
            found.extend(c for c in nested if type(c) not in _OPENED and not callable(c))
        users_objects += sum(1 for c in nested if type(c) not in _OPENED)
        if users_objects > limit:
            return found, False
        level = nested
    return found, True


def _reached(value: Any) -> tuple[list[Any], bool]:
    """``(values, complete)``: what an item of *value* may be that calling,
    or calling a method of, runs code of the user's: the callables it holds
    and the objects of the user's classes (`_callables_reached`)."""
    return _callables_reached(value, objects=True)


#: `_reached` of the containers a statement iterates, for the several
#: questions asked about one statement before it runs: ``id -> (container,
#: round, answer)``. A round ends whenever code of the user's may have run
#: (`new_analysis_round`).
_REACHED_THIS_ROUND: dict[int, tuple[Any, int, tuple[list[Any], bool]]] = {}
_ROUND = [0]


def new_analysis_round() -> None:
    """Forget what `_reached_this_round` found: code of the user's is about
    to run, or has run, and may have changed what a container holds."""
    _ROUND[0] += 1
    _REACHED_THIS_ROUND.clear()
    _ANSWERED_THIS_ROUND.clear()
    _STEPS_THIS_ROUND.clear()
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()


def _reached_this_round(value: Any) -> tuple[list[Any], bool]:
    """`_reached` of *value*, read once per analysis round: ``[x.strip() for
    x in lines]`` is asked about several times before it runs, and each
    answer reads every line."""
    kept = _REACHED_THIS_ROUND.get(id(value))
    if kept is not None and kept[0] is value and kept[1] == _ROUND[0]:
        return kept[2]
    answer = _reached(value)
    if len(_REACHED_THIS_ROUND) > 256:
        _REACHED_THIS_ROUND.clear()
    _REACHED_THIS_ROUND[id(value)] = (value, _ROUND[0], answer)
    return answer


#: `_this_round`'s answers: ``key -> (round, the objects the key names
#: by id, answer)``. Kept only while `answers_shared` is open, in the round
#: it was opened in, and emptied when it closes: an answer holds the
#: user's objects, and a reference cash keeps would read as another holder
#: to the share check that follows a statement.
_ANSWERED_THIS_ROUND: dict[tuple[Any, ...], tuple[int, tuple[Any, ...], list[Any]]] = {}
#: The round `answers_shared` was opened in, or None outside it.
_SHARED_ROUND: list[int | None] = [None]


@contextlib.contextmanager
def answers_shared() -> Iterator[None]:
    """Let the analyses of one statement before it runs share what
    `_this_round` works out, until code runs (`new_analysis_round`) or the
    block ends. Nothing is kept past it."""
    outer = _SHARED_ROUND[0]
    _SHARED_ROUND[0] = _ROUND[0]
    try:
        yield
    finally:
        _SHARED_ROUND[0] = outer
        _ANSWERED_THIS_ROUND.clear()


def stop_sharing_answers() -> None:
    """End what `answers_shared` keeps, as the statement's code is about to
    run: an answer worked out before it is not one for what it does."""
    _SHARED_ROUND[0] = None
    _ANSWERED_THIS_ROUND.clear()


def _this_round(key: tuple[Any, ...], objects: tuple[Any, ...], work: Callable[[], list[Any]]) -> list[Any]:
    """What *work* answers, once per analysis round for *key* while
    `answers_shared` is open: no code of the user's runs within a round, so
    what the objects *key* names by id (*objects*, the very ones) hold is
    the same each time it is asked. The list answered is shared: callers
    only read it."""
    if _SHARED_ROUND[0] != _ROUND[0]:
        return work()
    kept = _ANSWERED_THIS_ROUND.get(key)
    if kept is not None and kept[0] == _ROUND[0] and all(map(operator.is_, kept[1], objects)):
        return kept[2]
    answer = work()
    if len(_ANSWERED_THIS_ROUND) > 256:
        _ANSWERED_THIS_ROUND.clear()
    _ANSWERED_THIS_ROUND[key] = (_ROUND[0], objects, answer)
    return answer


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
    if isinstance(container, _CONTAINERS):
        # A namedtuple, a subclass of the user's of a list or dict: its
        # items, read through the builtin's own slots so no ``__iter__`` of
        # theirs runs, and its attributes.
        if isinstance(container, dict):
            items: Iterable[Any] = chain(dict.keys(container), dict.values(container))
        else:
            base = next(kind for kind in _CONTAINERS if isinstance(container, kind))
            items = base.__iter__(container)  # type: ignore[attr-defined]
        return chain(items, _attributes_of(container))
    # `_attributes_of`, inlined: this runs once for each object a walk opens.
    try:
        own = object.__getattribute__(container, "__dict__")
    except (AttributeError, TypeError):
        own = None
    values = list(own.values()) if isinstance(own, dict) else []
    for member, cls in _slot_members(type(container)):
        try:
            values.append(member.__get__(container, cls))
        except AttributeError:
            pass  # an unset slot
    return values


#: The builtin containers whose subclasses' items `_items` reads.
_CONTAINERS = (dict, list, tuple, set, frozenset, collections.deque)


def users_object_attributes(value: Any) -> list[Any]:
    """The values of the instance attributes of *value* when it is an object
    of the user's class (its ``__dict__`` and ``__slots__``, read without
    running code); empty for anything else."""
    _CLASS_ATTRIBUTES.clear()
    _CHANGES_ITS_OBJECT.clear()
    _SLOT_MEMBERS.clear()
    _THROUGH_ITS_OBJECT.clear()
    return _attributes_of(value) if _is_users_object(value) else []


#: ``class -> its slot descriptors``, each with the class defining it
#: (`_attributes_of`). Read once by each question asked of this module, as
#: `_CLASS_ATTRIBUTES` is, and cleared with it: not once per object.
_SLOT_MEMBERS: dict[type, tuple[tuple[Any, type], ...]] = {}


def _slot_members(kind: type) -> tuple[tuple[Any, type], ...]:
    """The slot descriptors of the class *kind* and of the classes in its
    MRO, each with its class."""
    members = _SLOT_MEMBERS.get(kind)
    if members is None:
        found: list[tuple[Any, type]] = []
        for cls in kind.__mro__:
            slots = cls.__dict__.get("__slots__")
            if slots is None:
                continue
            for name in (slots,) if isinstance(slots, str) else slots:
                member = cls.__dict__.get(name)
                if type(member) is types.MemberDescriptorType:
                    found.append((member, cls))
        members = _SLOT_MEMBERS[kind] = tuple(found)
    return members


def _attributes_of(obj: Any) -> list[Any]:
    """The values of *obj*'s instance attributes: its ``__dict__`` and its
    ``__slots__``, read through the slot descriptors so no property or
    ``__getattr__`` runs."""
    values = list(_instance_dict(obj).values())
    for member, cls in _slot_members(type(obj)):
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
