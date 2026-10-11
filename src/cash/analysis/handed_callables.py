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
from ..tracking.function_tracker import is_local_module
from .ast_util import CallScope, calls_in
from .callee_effects import (
    parse_function_source,
    all_param_names,
    callee_global_mutations,
    iterated_sources,
    params_mutated_in_function,
    source_global_mutations,
)
from .mutations import MUTATING_METHODS, chain_is_pure

__all__ = [
    "HandedCallables",
    "handed_callable_values",
    "handed_callables",
    "handed_values",
    "hands_a_state_changing_callable",
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


_NONE = HandedCallables()

#: Containers opened for the callables they hold: their items are read at C
#: speed and no code of the user's runs.
_OPENED = frozenset({list, tuple, set, frozenset, dict, OrderedDict, defaultdict})


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
        if not call.args and not call.keywords and not isinstance(call.func, ast.Subscript):
            continue
        opened: _Opened | None = None
        for position, keyword, arg in _arguments(call):
            for value, root in handed_values(arg, namespace):
                if type(value) in _OPENED and opened is None:
                    opened = _opened_by(_callee_of(call.func, namespace), call.func)
                found.add(value, root, opened is not None and opened.opens(position, keyword))
        if isinstance(call.func, ast.Subscript):
            # ``ops['dbl'](v)``: a callable read out of a container is called.
            for value, root in handed_values(call.func, namespace):
                found.add(value, root, False)
    if not (found.sources or found.receivers or found.unreadable):
        return _NONE
    return HandedCallables(tuple(found.sources), frozenset(found.receivers), tuple(found.unreadable))


def handed_callable_values(call: ast.Call, namespace: Mapping[str, Any]) -> list[Any]:
    """The callables *call* is handed, as *namespace* holds them: its
    arguments (`handed_values`), and the ones inside a builtin container
    when the callee calls what it is handed there. Functions, methods,
    classes and callable objects, each once; a partial's function."""
    opened: _Opened | None = None
    out: dict[int, Any] = {}
    for position, keyword, arg in _arguments(call):
        for value, _root in handed_values(arg, namespace):
            if type(value) in _OPENED:
                if opened is None:
                    opened = _opened_by(_callee_of(call.func, namespace), call.func)
                items = _callables_in(value) if opened.opens(position, keyword) else []
            else:
                items = [value]
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


def handed_values(arg: ast.expr, namespace: Mapping[str, Any]) -> list[tuple[Any, str | None]]:
    """``(value, root)`` for each value the argument expression *arg* hands
    over that can be found without running code, *root* the name it was read
    from (None for a literal's element built in place)."""
    out: list[tuple[Any, str | None]] = []
    pending: list[ast.expr] = [arg]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Starred):
            pending.append(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            pending.extend(node.elts)
        elif isinstance(node, ast.Dict):
            pending.extend(v for v in node.values if v is not None)
            pending.extend(k for k in node.keys if k is not None)
        elif isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            found = _static_value(node, namespace)
            if found is not None:
                out.append(found)
    return out


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
    try:
        if isinstance(container, dict):
            # ``dict.get``: a defaultdict's factory does not run.
            return dict.get(container, index.value, _MISSING)
        if isinstance(container, (list, tuple)):
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
        """Add what calling *value* changes. A builtin container is opened
        for the callables it holds only with *open_containers*: when the
        callee calls what it is handed there."""
        if type(value) in _OPENED:
            if open_containers and (id(value), root) not in self._seen:
                self._seen.add((id(value), root))
                for item in _callables_in(value):
                    self.add(item, root, False)
            return
        if (id(value), root) in self._seen:
            return
        self._seen.add((id(value), root))
        self._callable(value, root)

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
            if not self._users(value):
                return
            source = self._source(value, label)
            if source is not None and value.__closure__:
                free = set(value.__code__.co_freevars)
                if free & source_global_mutations(source):
                    self._changed(value, root)
            return
        if isinstance(value, types.MethodType):
            owner, fn = value.__self__, value.__func__
            if not isinstance(fn, types.FunctionType) or not self._users(fn):
                return
            source = self._source(fn, label)
            if source is not None and not isinstance(owner, types.ModuleType):
                if _changes_its_first_argument(fn, owner if isinstance(owner, type) else type(owner), source):
                    self._changed(owner, root)
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
    code = fn.__code__
    if code in _SOURCES:
        return _SOURCES.get(code)
    try:
        source: str | None = inspect.getsource(fn)
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
    or one whose source cannot be read. A builtin container among the
    arguments is looked into when *fn* calls what it is handed there.

    ``run_steps(steps, data)`` with ``steps = [f]`` runs ``f``; served from
    the cache, ``f``'s writes would not happen."""
    found: _Found | None = None
    opened: _Opened | None = None
    for position, keyword, value in chain(
        zip(range(len(args)), repeat(None), args), zip(repeat(None), kwargs.keys(), kwargs.values())
    ):
        kind = type(value)
        if kind in _ATOMS or not (kind in _OPENED or callable(value)):
            continue
        if kind in _OPENED:
            if opened is None:
                opened = _opened_by(fn, None)
            if not opened.opens(position, keyword):
                continue
        if found is None:
            found = _Found({})
        found.add(value, None, True)
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
            if isinstance(expr, ast.Subscript):
                expr = expr.value
            elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
                expr = expr.func.value  # ``ops.get('dbl')(x)``
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


def _callables_in(container: Any) -> list[Any]:
    """The callables in *container* and in the builtin containers it holds,
    however deeply nested, each once. A level at a time, at C speed."""
    if _plain_data.is_tree(container):
        return []  # lists and tuples of plain values only
    found: list[Any] = []
    seen = {id(container)}
    level = [container]
    while level:
        items = list(chain.from_iterable(_items(c) for c in level))
        if not items:
            break
        found.extend(filter(callable, items))
        nested = [c for c in compress(items, map(_OPENED.__contains__, map(type, items))) if id(c) not in seen]
        seen.update(map(id, nested))
        level = nested
    return found


def _items(container: Any) -> Iterable[Any]:
    if isinstance(container, dict):
        return chain(dict.keys(container), dict.values(container))
    return container


def _changes_its_first_argument(fn: types.FunctionType, cls: type, source: str, seen: frozenset[int] = frozenset()) -> bool:
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
