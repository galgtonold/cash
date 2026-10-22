"""Which parameters a callable of the user's changes in place, read from its
code without running any of the user's.

A callable handed to a library call that runs it over items (``s.apply(f)``,
``sorted(xs, key=f)``) changes the objects a frame's cells hold when it
changes its parameter in place. The rule is the one cash keeps everywhere:
what it cannot read counts as changing. So a parameter (or a local name bound
to it, to a part of it or to one of its bound methods) handed to a call counts
as changed unless that call's callee is read whole and leaves it alone:

- a plain function of the user's, read in turn;
- a bound method, a partial, a static or class method, a callable object of
  the user's: the function it runs, read in turn;
- a function or object of a library: data, as everywhere else in cash.

A callee cash cannot name without running code -- a closure variable that is
not a function, an item of a container, a parameter, the result of a call,
an attribute of an object whose class answers attributes itself -- counts as
changing what it is handed. ``exec`` and ``eval`` reach every parameter.

Every value is read by exact type and from the dicts that hold it, so no
property, ``__getattr__``, ``__class__`` or dict-subclass method of the
user's runs.
"""

from __future__ import annotations

import ast
import builtins
import functools
import operator
import sys
import types
from collections.abc import Mapping
from typing import Any

from .callee_effects import (
    all_param_names,
    body_mutations,
    iterated_sources,
    parse_function_def,
    positional_param_names,
)
from .handed_callables import function_source
from .mutations import MUTATING_METHODS

__all__ = ["callable_changes_parameters", "lambda_changes_parameters"]

#: How deep calls are followed; past it a callee counts as changing.
_MAX_DEPTH = 6

#: Builtins that run text, reading the caller's variables.
_RUNS_TEXT = (exec, eval)

#: Builtins that change an object's attributes by a name known only when
#: they run: ``setattr(o, 'z', 1)``.
_BY_NAME = (setattr, delattr)

#: Dunders that change their object in place.
_CHANGING_DUNDERS = frozenset(
    {"__setitem__", "__delitem__", "__setattr__", "__delattr__", "__iadd__", "__isub__", "__imul__", "__ior__"}
)

#: ``operator`` functions that change their first argument.
_SETS_FIRST = tuple(
    getattr(operator, name) for name in ("setitem", "delitem", "iadd", "isub", "imul", "ior", "iand", "iconcat")
)

_MISSING = object()

#: What a name or an attribute names, as `_Reader` sorts it.
_UNKNOWN = "unknown"  # cannot be named without running code
_TRACKED = "tracked"  # a parameter, or a local bound to one
_LOCAL = "local"  # a local bound to something else


def _type_dict(cls: type) -> Mapping[str, Any]:
    return type.__dict__["__dict__"].__get__(cls)


def _mro(cls: type) -> tuple[type, ...]:
    return type.__dict__["__mro__"].__get__(cls)


def _module_dict(module: types.ModuleType) -> Mapping[str, Any] | None:
    found = object.__getattribute__(module, "__dict__")
    return found if type(found) is dict else None


def _dict_get(mapping: Any, key: str) -> Any:
    """``mapping[key]`` or `_MISSING`, through ``dict``'s own method: a
    dict subclass's ``get`` or ``__getitem__`` does not run."""
    if not isinstance(mapping, dict):
        return _MISSING
    return dict.get(mapping, key, _MISSING)


def _plain_class(value: Any) -> bool:
    """A class with no metaclass of its own: its attributes are read from
    the dicts of its MRO."""
    return type(value) is type


def _class_attribute(cls: type, name: str) -> Any:
    """*name* as the dicts of *cls*'s MRO hold it, or `_MISSING`."""
    for klass in _mro(cls):
        if klass is object:
            continue
        found = _type_dict(klass).get(name, _MISSING)
        if found is not _MISSING:
            return found
    return _MISSING


def _answers_attributes(cls: type) -> bool:
    """Whether *cls* (a plain class) computes attributes itself."""
    for klass in _mro(cls):
        if klass is object:
            continue
        held = _type_dict(klass)
        if "__getattr__" in held or "__getattribute__" in held:
            return True
    return False


def _is_users_function(fn: types.FunctionType) -> bool:
    name = _dict_get(fn.__globals__, "__name__")
    if name is _MISSING or name is None or name == "__main__":
        return True
    module = sys.modules.get(name) if type(name) is str else None
    if module is None:
        return False
    from ..tracking.function_tracker import is_local_module

    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def _is_users_class(cls: type) -> bool:
    if not _plain_class(cls):
        return False
    name = _type_dict(cls).get("__module__")
    if name == "__main__":
        return True
    module = sys.modules.get(name) if type(name) is str else None
    if module is not None and name != "builtins":
        from ..tracking.function_tracker import is_local_module

        try:
            return is_local_module(module)
        except (TypeError, AttributeError):
            return False
    return any(type(v) is types.FunctionType and _is_users_function(v) for v in _type_dict(cls).values())


def is_users(value: Any) -> bool:
    """Whether calling *value* runs code of the user's, asked by exact type."""
    kind = type(value)
    if kind is types.FunctionType:
        return _is_users_function(value)
    if kind is functools.partial:
        return is_users(value.func)
    if kind is types.MethodType or kind is staticmethod or kind is classmethod:
        return is_users(value.__func__)
    if kind is type:
        return _is_users_class(value)
    if kind in (types.BuiltinFunctionType, types.ModuleType, types.MethodWrapperType):
        return False
    return _is_users_class(kind)


class _Reader:
    """Reads functions and lambdas for the parameters they change, sharing
    what it has read across one question (`_seen`)."""

    def __init__(self) -> None:
        self._seen: dict[tuple[Any, Any], frozenset[str] | None] = {}

    # -- callables -------------------------------------------------------

    def callable_changes(self, value: Any, depth: int = 0) -> bool | None:
        """Whether calling *value* changes a parameter it is handed; None
        when what it runs cannot be read."""
        if depth > _MAX_DEPTH:
            return None
        kind = type(value)
        if kind is functools.partial:
            return self.callable_changes(value.func, depth + 1)
        if kind is staticmethod or kind is classmethod:
            return self.callable_changes(value.__func__, depth + 1)
        if kind is types.MethodType:
            changed = self.function_changes(value.__func__, depth + 1, bound=value.__self__)
            return None if changed is None else bool(changed)
        if kind is types.FunctionType:
            changed = self.function_changes(value, depth + 1)
            return None if changed is None else bool(changed)
        if kind is type:
            # Building an object runs ``__init__`` (and ``__new__``) with it.
            if not _is_users_class(value) or _answers_attributes(value):
                return None
            init = _class_attribute(value, "__init__")
            new = _class_attribute(value, "__new__")
            if new is not _MISSING and new is not object.__new__:
                return None
            if init is _MISSING or init is object.__init__:
                return False
            if type(init) is not types.FunctionType:
                return None
            changed = self.function_changes(init, depth + 1, bound=_UNBOUND_SELF)
            return None if changed is None else bool(changed)
        if not _is_users_class(kind) or _answers_attributes(kind):
            return None
        call = _class_attribute(kind, "__call__")
        if type(call) is not types.FunctionType:
            return None
        changed = self.function_changes(call, depth + 1, bound=value)
        return None if changed is None else bool(changed)

    def function_changes(self, fn: types.FunctionType, depth: int, bound: Any = _MISSING) -> frozenset[str] | None:
        """The parameters *fn* changes in place (the first left out when
        *bound* is given: that is the object it is bound to); None when its
        source cannot be read."""
        key = (fn.__code__, id(bound) if bound is not _MISSING else None)
        if key in self._seen:
            # A call back into a function being read: what it changes is
            # read where it was first met.
            return self._seen[key] or frozenset()
        self._seen[key] = frozenset()
        source = function_source(fn)
        fdef = parse_function_def(source) if source else None
        if fdef is None:
            self._seen[key] = None
            return None
        known: dict[str, Any] = {}
        positional = positional_param_names(fdef)
        if bound is not _MISSING and positional:
            known[positional[0]] = bound
        closure = {}
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    closure[name] = cell.cell_contents
                except ValueError:
                    closure[name] = _MISSING
        scope = _Scope(closure, fn.__globals__, _builtins_of(fn), known)
        params = frozenset(all_param_names(fdef))
        changed = self.body_changes(fdef, params, scope, depth)
        if bound is not _MISSING and positional:
            changed = changed - {positional[0]}
        self._seen[key] = frozenset(changed)
        return frozenset(changed)

    # -- bodies ----------------------------------------------------------

    def body_changes(
        self, fdef: ast.FunctionDef | ast.AsyncFunctionDef, params: frozenset[str], scope: _Scope, depth: int
    ) -> set[str]:
        """The *params* the body of *fdef* changes in place."""
        reach = _aliases(fdef, params)
        locals_ = _bound_names(fdef) - params

        def back(names) -> set[str]:
            out: set[str] = set()
            for name in names:
                out |= reach.get(name, set())
            return out

        changed = back(body_mutations(fdef))
        for node in ast.walk(fdef):
            if not isinstance(node, ast.Call):
                continue
            changed |= self._call_changes(node, reach, locals_, scope, depth)
            if changed >= params:
                break
        return changed & params

    def _call_changes(
        self, call: ast.Call, reach: dict[str, set[str]], locals_: set[str], scope: _Scope, depth: int
    ) -> set[str]:
        """The parameters the call *call* in a body may change."""
        passed: dict[int | str | None, set[str]] = {}
        for i, arg in enumerate(call.args):
            slot = None if isinstance(arg, ast.Starred) else i
            names = _reached(arg, reach)
            if names:
                passed.setdefault(slot, set()).update(names)
        for keyword in call.keywords:
            names = _reached(keyword.value, reach)
            if names:
                passed.setdefault(keyword.arg, set()).update(names)
        everything: set[str] = set().union(*passed.values()) if passed else set()
        func = call.func
        if isinstance(func, ast.Attribute):
            owners = _reached(func.value, reach)
            if owners:
                # A method called on a parameter or on a part of it.
                method = func.attr
                if method in MUTATING_METHODS or method in _CHANGING_DUNDERS:
                    return owners | everything
                if isinstance(func.value, ast.Name) and func.value.id in scope.known:
                    # ``self._set(d)``: the object it is bound to is known.
                    value = _attribute_of(scope.known[func.value.id], method)
                    return self._handed_to(value, passed, everything, depth)
                # A method of a value cash knows nothing of: a library's, as
                # everywhere in cash. A callable of the user's handed to it
                # with the parameter (``frame['o'].apply(upd)``) is read.
                return self._handed_callables(call, reach, scope, depth)
        elif isinstance(func, ast.Name) and func.id in reach:
            # ``put('z', 1)`` with ``put = d.__setitem__``; a parameter called.
            return reach[func.id] | everything
        elif not isinstance(func, ast.Name) and _reached(func, reach):
            # ``getattr(d, '__setitem__')('z', 1)``, ``d['f'](x)``.
            return _reached(func, reach) | everything
        value = scope.resolve(func, locals_)
        if value is _LIBRARY:
            return self._handed_callables(call, reach, scope, depth)
        if value is _UNKNOWN or value is _LOCAL:
            return everything
        if any(value is runs for runs in _RUNS_TEXT):
            return set().union(*reach.values())
        if any(value is by_name for by_name in _BY_NAME):
            return passed.get(0, set())
        if any(value is setter for setter in _SETS_FIRST):
            return passed.get(0, set()) | self._handed_callables(call, reach, scope, depth)
        if not is_users(value):
            return self._handed_callables(call, reach, scope, depth)
        return self._handed_to(value, passed, everything, depth) | self._handed_callables(call, reach, scope, depth)

    def _handed_to(self, value: Any, passed, everything: set[str], depth: int) -> set[str]:
        """What handing *passed* (``slot -> names``) to *value* changes."""
        if not everything or value is _LIBRARY:
            return set()
        if value is _MISSING or value is _UNKNOWN or value is _LOCAL:
            return everything
        if type(value) is types.FunctionType:
            changed = self.function_changes(value, depth + 1)
            if changed is None:
                return everything
            if not changed:
                return set()
            source = function_source(value)
            fdef = parse_function_def(source) if source else None
            if fdef is None:
                return everything
            ordered = positional_param_names(fdef)
            out: set[str] = set()
            for slot, names in passed.items():
                if slot is None or (type(slot) is int and slot >= len(ordered)):
                    out |= names  # through ``*args``: any parameter
                elif type(slot) is int and ordered[slot] in changed or slot in changed:
                    out |= names
            return out
        if not is_users(value):
            return set()
        verdict = self.callable_changes(value, depth + 1)
        return set() if verdict is False else everything

    def _handed_callables(self, call: ast.Call, reach, scope: _Scope, depth: int) -> set[str]:
        """A callable of the user's handed, with a parameter, to a call that
        may run it over that parameter's items (``frame['o'].apply(upd)``):
        the parameter changes when the callable changes what it is handed."""
        involved = set()
        for part in (call.func, *call.args, *(k.value for k in call.keywords)):
            involved |= _reached(part, reach)
        if not involved:
            return set()
        for arg in (*call.args, *(k.value for k in call.keywords)):
            target = arg.value if isinstance(arg, ast.Starred) else arg
            if isinstance(target, ast.Lambda):
                if self.lambda_changes(target, scope.namespace_view(), depth + 1) is not False:
                    return involved
                continue
            if _reached(target, reach):
                continue
            value = scope.resolve(target, set())
            if value is _UNKNOWN or value is _LOCAL or value is _MISSING or value is _LIBRARY:
                continue
            if is_users(value) and self.callable_changes(value, depth + 1) is not False:
                return involved
        return set()

    # -- lambdas ---------------------------------------------------------

    def lambda_changes(self, lam: ast.Lambda, namespace: Mapping[str, Any], depth: int = 0) -> bool | None:
        """Whether the lambda *lam*, its free names read from *namespace*,
        changes a parameter in place."""
        if depth > _MAX_DEPTH:
            return None
        fdef = ast.FunctionDef(name="<lambda>", args=lam.args, body=[ast.Return(lam.body)], decorator_list=[])
        if "type_params" in ast.FunctionDef._fields:
            fdef.type_params = []
        params = frozenset(all_param_names(fdef))
        if not params:
            return False
        scope = _Scope({}, namespace, vars(builtins), {})
        return bool(self.body_changes(fdef, params, scope, depth))


#: Stands for the object ``__init__`` builds: not known yet.
_UNBOUND_SELF = object()


def _builtins_of(fn: types.FunctionType) -> Mapping[str, Any]:
    found = fn.__builtins__ if hasattr(fn, "__builtins__") else None
    if type(found) is dict:
        return found
    return vars(builtins)


class _Scope:
    """Where a function's free names are read: its closure, its module's
    globals, the builtins; and *known*, parameters whose value is known
    (the object a method is bound to)."""

    def __init__(self, closure: Mapping[str, Any], globals_: Any, builtins_: Mapping[str, Any], known: dict) -> None:
        self.closure = closure
        self.globals = globals_
        self.builtins = builtins_
        self.known = known

    def namespace_view(self) -> Mapping[str, Any]:
        """The names a lambda in the body reads, as one mapping."""
        merged: dict[str, Any] = {}
        if isinstance(self.globals, dict):
            merged.update(dict.items(self.globals))
        merged.update(self.closure)
        return merged

    def name(self, name: str) -> Any:
        if name in self.closure:
            value = self.closure[name]
            return _UNKNOWN if value is _MISSING else value
        value = _dict_get(self.globals, name)
        if value is _MISSING:
            value = _dict_get(self.builtins, name)
        return _UNKNOWN if value is _MISSING else value

    def resolve(self, node: ast.expr, locals_: set[str]) -> Any:
        """The value *node* names, read without running code; `_LOCAL` for
        a local of the body, `_UNKNOWN` when it cannot be read."""
        if isinstance(node, ast.Name):
            if node.id in locals_:
                return _LOCAL
            return self.name(node.id)
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value, locals_)
            if owner is _LOCAL or owner is _UNKNOWN:
                return owner
            return _attribute_of(owner, node.attr)
        if isinstance(node, ast.Subscript):
            owner = self.resolve(node.value, locals_)
            if owner is _LOCAL or owner is _UNKNOWN:
                return owner
            key = node.slice
            if isinstance(key, ast.Constant):
                if type(owner) is dict:
                    found = dict.get(owner, key.value, _MISSING)
                    return _UNKNOWN if found is _MISSING else found
                if type(owner) in (list, tuple) and type(key.value) is int:
                    try:
                        return owner[key.value]
                    except IndexError:
                        return _UNKNOWN
            return _UNKNOWN
        return _UNKNOWN


def _attribute_of(owner: Any, name: str) -> Any:
    """``owner.name`` read from the dicts that hold it; `_UNKNOWN` when
    reading it may run code of the user's."""
    kind = type(owner)
    if kind is types.ModuleType:
        held = _module_dict(owner)
        found = _dict_get(held, name) if held is not None else _MISSING
        return _UNKNOWN if found is _MISSING else found
    if kind is type:
        if _answers_attributes(owner):
            return _UNKNOWN
        found = _class_attribute(owner, name)
        if found is _MISSING:
            return _UNKNOWN
        if type(found) is staticmethod:
            return found.__func__
        if type(found) is classmethod:
            return types.MethodType(found.__func__, owner)
        return found
    if owner is _UNBOUND_SELF:
        return _UNKNOWN
    if owner is _LIBRARY:
        return _LIBRARY
    if not _is_users_class(kind):
        # A library object's attribute: what it does is the library's. One
        # it only holds (``M.setz = setz`` on a ``SimpleNamespace``) is
        # whatever was put there.
        held = _instance_dict(owner, kind)
        if held is not None and name in held:
            return dict.__getitem__(held, name)
        return _LIBRARY
    if _answers_attributes(kind):
        return _UNKNOWN
    found = _class_attribute(kind, name)
    if found is not _MISSING and type(found) not in (types.FunctionType, staticmethod, classmethod):
        # A property or another descriptor computes the attribute.
        if hasattr(type(found), "__get__"):
            return _UNKNOWN
    try:
        own = object.__getattribute__(owner, "__dict__")
    except AttributeError:
        own = None
    if type(own) is dict and name in own:
        return dict.__getitem__(own, name)
    if found is _MISSING:
        return _UNKNOWN
    if type(found) is types.FunctionType:
        return types.MethodType(found, owner)
    if type(found) is staticmethod:
        return found.__func__
    if type(found) is classmethod:
        return types.MethodType(found.__func__, kind)
    return found


def _instance_dict(owner: Any, kind: type) -> dict | None:
    """*owner*'s own ``__dict__`` when its class reads attributes the plain
    way, so a value found there is what the attribute gives."""
    if (
        _class_attribute(kind, "__getattribute__") is not _MISSING
        or _class_attribute(kind, "__getattr__") is not _MISSING
    ):
        return None
    try:
        held = object.__getattribute__(owner, "__dict__")
    except (AttributeError, TypeError):
        return None
    return held if type(held) is dict else None


#: Stands for an attribute of a library's object: not the user's code.
_LIBRARY = object()


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _reached(node: ast.AST, reach: Mapping[str, set[str]]) -> set[str]:
    """The parameters the expression *node* hands over: a name bound to
    one, a part of one, ``getattr(d, ...)``, one in a container built here."""
    out: set[str] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Name):
            out |= reach.get(current.id, set())
        elif isinstance(current, (ast.Attribute, ast.Subscript, ast.Starred)):
            pending.append(current.value)
        elif isinstance(current, (ast.Tuple, ast.List, ast.Set)):
            pending.extend(current.elts)
        elif isinstance(current, ast.Dict):
            pending.extend(v for v in current.values if v is not None)
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Name)
            and current.func.id in {"getattr", "vars"}
        ):
            # ``getattr(d, '__setitem__')``, ``vars(o).update(...)``.
            pending.extend(current.args[:1])
        elif isinstance(current, (ast.IfExp, ast.BoolOp, ast.NamedExpr)):
            pending.extend(ast.iter_child_nodes(current))
    return out


def _aliases(fdef: ast.FunctionDef | ast.AsyncFunctionDef, params: frozenset[str]) -> dict[str, set[str]]:
    """Each name of the body that may hold a parameter, a part of one or a
    bound method of one, with the parameters it reaches: the parameters
    themselves, ``x = d``, ``put = d.__setitem__``, ``for v in d.values()``."""
    reach: dict[str, set[str]] = {p: {p} for p in params}
    looped = iterated_sources(fdef)
    pairs: list[tuple[list[str], ast.AST]] = []
    for node in ast.walk(fdef):
        if isinstance(node, ast.Assign):
            targets = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            pairs.append((targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            pairs.append(([n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)], node.value))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    pairs.append(
                        ([n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)], item.context_expr)
                    )
    grew = True
    while grew:
        grew = False
        for targets, value in pairs:
            names = _reached(value, reach) | _method_owner(value, reach)
            for target in targets:
                if names - reach.get(target, set()):
                    reach.setdefault(target, set()).update(names)
                    grew = True
        for target, sources in looped.items():
            names: set[str] = set()
            for source in sources:
                names |= reach.get(source, set())
            if names - reach.get(target, set()):
                reach.setdefault(target, set()).update(names)
                grew = True
    return reach


def _method_owner(value: ast.AST, reach: Mapping[str, set[str]]) -> set[str]:
    """``d.values()``, ``d.items()``: the items of a parameter."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        if value.func.attr in {"values", "items", "__iter__", "get", "setdefault", "pop"}:
            return _reached(value.func.value, reach)
    return set()


def _bound_names(fdef: ast.AST) -> set[str]:
    """Names the body binds: assigned, looped over, imported, defined."""
    out: set[str] = set()
    for node in ast.walk(fdef):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fdef:
            out.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return out


def callable_changes_parameters(value: Any) -> bool | None:
    """Whether calling the user's callable *value* changes a parameter it is
    handed in place; None when what it runs cannot be read whole."""
    return _Reader().callable_changes(value)


def lambda_changes_parameters(lam: ast.Lambda, namespace: Mapping[str, Any]) -> bool:
    """Whether *lam*, its free names read from *namespace*, changes a
    parameter in place; one that cannot be read whole counts as changing."""
    return _Reader().lambda_changes(lam, namespace) is not False
