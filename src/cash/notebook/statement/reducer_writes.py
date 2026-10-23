"""What a reducer of the user's that cash ran may have written.

Cash pickles a statement's values to digest and store them, and a pickle
runs the value's own ``__reduce__``, ``__reduce_ex__``, ``__getstate__``,
``__getnewargs__`` or ``__getnewargs_ex__`` (or a reducer of the user's
registered with ``copyreg``): the user's code, run by cash, between
statements. One that writes (``df.at[3, 'o'] = ...``) changes a variable
no statement named, and whatever read it before would be served what it
read then. `writers` reads each such reducer's body, and the functions of
the user's it calls, for a write; `reach` names the variables a writing
one's code can reach, which then move on (`StatementProcessor`).

A body that only reads -- ``return self.__dict__.copy()`` -- writes
nothing, and nothing moves. A call into a library is read as the callee
analysis reads it (`callee_effects`): it changes no global of the user's.
A function of the user's whose source cannot be read is taken to write.
"""

from __future__ import annotations

import ast
import copyreg
import inspect
import sys
import textwrap
import types
import weakref
from collections.abc import Mapping
from typing import Any

from cash._user_attrs import attribute_or, instance_of
from cash.analysis.ast_util import bytecode_global_refs
from cash.analysis.callee_effects import (
    free_vars_mutated_in_function,
    params_mutated_in_function,
    parse_function_source,
    source_global_rebinds,
)
from cash.install_paths import is_user_code_module
from cash.kept_state import user_state_runs, user_state_types_since
from cash.source_reading import getsourcelines

__all__ = ["Writer", "reach", "runs_now", "writers", "writes_into"]

#: The methods a pickle runs on an object, found on its class.
_REDUCERS = ("__reduce__", "__reduce_ex__", "__getstate__", "__getnewargs__", "__getnewargs_ex__")


class Writer:
    """A class whose reducer of the user's writes: its *name* for the
    warning, and the functions of the user's its pickle runs (*functions*,
    None when one of them could not be read)."""

    __slots__ = ("functions", "name", "written")

    def __init__(self, name: str, functions: list[Any] | None, written: frozenset[str] = frozenset()) -> None:
        self.name = name
        self.functions = functions
        #: The global names its code changes, as read from it.
        self.written = written


def runs_now() -> int:
    """The mark `writers` counts from: `kept_state.user_state_runs`."""
    return user_state_runs()


def writers(since: int) -> list[Writer]:
    """The classes whose reducer of the user's ran in a pickle of cash's
    since the mark *since* (`runs_now`) and writes. One that ran for a
    class cash could not name is a writer whose functions are unknown."""
    ran, unnamed = user_state_types_since(since)
    found = [writer for t in ran if (writer := _writer(t)) is not None]
    if unnamed:
        found.append(Writer("an object", None))
    return found


def writes_into(since: int, user_ns: Mapping[str, Any]) -> bool:
    """Has a reducer that writes, run since the mark *since*, reached a
    variable of *user_ns* (`reach`)? One whose code reaches none (a counter
    of its own module's) leaves nothing a statement reads changed, and a
    hit that skips it gives what plain Python gives, where it never runs."""
    found = writers(since)
    if not found:
        return False
    names = reach(found, user_ns)
    return names is None or bool(names)


def reach(found: list[Writer], user_ns: Mapping[str, Any]) -> set[str] | None:
    """The variables of *user_ns* the code of the writers *found* can reach:
    each global name their functions look up. None when a writer's
    functions are unknown, or one writes the object it pickles (which
    variables hold it is not known): every variable may have changed."""
    names: set[str] = set()
    for writer in found:
        if writer.functions is None:
            return None
        for function in writer.functions:
            names.update(chain[0] for chain in bytecode_global_refs(function))
        names.update(writer.written)
    return {
        name
        for name in names
        if name in user_ns
        and not name.startswith("_")
        # A module, class or function it only names (``time.sleep``) is
        # code it runs, not data it writes; one it writes is in `written`.
        and (name in _written(found) or type(user_ns[name]) not in _CODE_TYPES)
    }


_CODE_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)


def _written(found: list[Writer]) -> set[str]:
    return {name for writer in found for name in writer.written}


#: `_writer` by class: None for a class whose reducers write nothing.
_WRITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PURE = object()


def _writer(t: type) -> Writer | None:
    try:
        found = _WRITERS.get(t)
    except TypeError:
        return _decide(t)
    if found is None:
        found = _WRITERS[t] = _decide(t) or _PURE
    return None if found is _PURE else found


def _decide(t: type) -> Writer | None:
    functions = _reducers_of(t)
    name = attribute_or(t, "__qualname__", None) or attribute_or(t, "__name__", "an object")
    seen: dict[int, Any] = {}
    roots = {id(function) for function in functions}
    pending = list(functions)
    writes = False
    written: set[str] = set()
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen[id(function)] = function
        body = _body(function)
        if body is None or (id(function) in roots and _writes_itself(body)):
            return Writer(name, None)
        changed = free_vars_mutated_in_function(body)
        if changed or _rebinds(function):
            written |= changed
            writes = True
        elif id(function) not in roots and params_mutated_in_function(body):
            # A function of the user's the reducer hands something to, and
            # which changes what it is handed: what was handed may be a
            # variable's object.
            writes = True
        pending.extend(_users_callees(function, body, t))
    return Writer(name, list(seen.values()), frozenset(written)) if writes else None


def _reducers_of(t: type) -> list[Any]:
    """The functions of the user's a pickle of an instance of *t* runs: a
    reducer registered for it with ``copyreg``, else the reducing methods
    its classes define."""
    registered = copyreg.dispatch_table.get(t)
    if registered is not None:
        return [registered] if _users(registered) else []
    found = []
    for klass in t.__mro__:
        if klass is object:
            continue
        for name in _REDUCERS:
            method = _function(klass.__dict__.get(name))
            if method is not None and _users(method):
                found.append(method)
    return found


def _function(member: Any) -> Any:
    """The plain function behind a class member, or None."""
    if type(member) in (classmethod, staticmethod):
        member = member.__func__
    return member if type(member) is types.FunctionType else None


def _users(function: Any) -> bool:
    return is_user_code_module(sys.modules.get(attribute_or(function, "__module__", None) or ""))


def _body(function: Any) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """*function*'s definition, parsed; None when its source is not there."""
    try:
        lines, _first = getsourcelines(function)
    except (OSError, TypeError, IndentationError, SyntaxError):
        return None
    return parse_function_source(textwrap.dedent("".join(lines)))


def _writes_itself(body: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Does *body* change its first argument (the object being pickled, or
    the class) in place? Which variables hold that object is not known."""
    params = [*body.args.posonlyargs, *body.args.args]
    return bool(params) and params[0].arg in params_mutated_in_function(body)


def _rebinds(function: Any) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return True
    return bool(source_global_rebinds(source))


def _users_callees(function: Any, body: ast.AST, t: type) -> list[Any]:
    """The functions of the user's *body* calls by name, or as a method
    of its first argument (``self.parts()``), found in *function*'s globals
    and on *t*."""
    found = []
    g = attribute_or(function, "__globals__", {})
    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        callee = None
        if isinstance(node.func, ast.Name):
            callee = g.get(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in ("self", "cls"):
                callee = inspect.getattr_static(t, node.func.attr, None)
            else:
                holder = g.get(node.func.value.id)
                if instance_of(holder, type):
                    callee = inspect.getattr_static(holder, node.func.attr, None)
        if instance_of(callee, type):
            callee = _function(callee.__dict__.get("__init__"))
        callee = _function(callee)
        if callee is not None and _users(callee):
            found.append(callee)
    return found
