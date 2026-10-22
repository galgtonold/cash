"""Code of the user's that ran in a statement and changes something its key
and outputs leave out.

The key counts the functions a statement reaches along the paths cash can
read from its text: a name it calls, a method, a special method of an object
it names, an item it reads. Others run unseen: ``Vec.__add__`` on an object
in an object-dtype column (``s + 1``), a closure's cell, a classmethod, an
item two steps down (``d[k][0] + 1``). When such a function appends to a
global list (``calls.append(1)``), a served entry drops the append.

On Python 3.12+ every function of the user's that ran is recorded
(``trace_recorder.Unit.codes``). After a run, `unseen_writers` reads each
one's own text for what it changes outside its objects: a global it changes
or rebinds, a variable of a closure it changes, a method it calls on a
library object it reads by a free name (``calls.append``, ``cache.set``).
What it calls by name needs no reading: that ran too, and is read in turn.
A change the statement's outputs already hold is kept by a served entry.
Any other means the run is not stored (`UNSEEN_WRITES_REASON`), and a hit on
an entry whose recorded trace holds such a function is not served.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any

from ...analysis.callee_effects import free_vars_mutated_in_function
from ...analysis.mutations import KNOWN_PURE_METHODS
from ...exceptions import SOURCE_RETRIEVAL_ERRORS
from ...source_reading import getsource

__all__ = ["UNSEEN_WRITES_REASON", "own_writes", "unseen_writers"]

#: What the statement says when it is not stored for one.
UNSEEN_WRITES_REASON = (
    "{fn}() ran in this statement and changes {targets}, which the statement's own text does not reach; "
    "a stored result would not change {it}"
)

#: ``CASH_UNSEEN_LOG=<path>``: each function found, one JSON line, to count them.
_LOG_ENV = "CASH_UNSEEN_LOG"

_UNKNOWN = "?"


def unseen_writers(
    codes: Mapping[types.CodeType, tuple[dict, str]],
    statement_codes: Iterable[types.CodeType],
    outputs: Iterable[str],
    namespace: Mapping[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    """``(function, what it changes)`` for each function of *codes* (code ->
    its globals and module, as the trace records them) that changes
    something outside its objects the statement's *outputs* do not hold.
    The statement's own code (*statement_codes*, with the functions and
    comprehensions inside it) is the statement's analysis's, not looked at."""
    own: set[types.CodeType] = set()
    pending = list(statement_codes)
    while pending:
        code = pending.pop()
        if code in own:
            continue
        own.add(code)
        pending.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    kept = set(outputs)
    found: list[tuple[str, tuple[str, ...]]] = []
    for code, (globals_, _module) in list(codes.items()):
        if code in own or code.co_qualname == "<module>":
            continue
        writes = own_writes(code)
        if not writes:
            continue
        targets: list[str] = []
        for name in sorted(writes):
            if name == _UNKNOWN:
                targets.append("what its unreadable text changes")
            elif name in code.co_freevars:
                targets.append(f"{name!r} of the function it was made in")
            elif globals_ is namespace and name in kept:
                continue
            elif name in globals_:
                if _changes_by_method(code, name) and not _changeable(globals_[name], code, name):
                    continue
                targets.append(repr(name) if globals_ is namespace else f"{globals_.get('__name__')}.{name}")
        if targets:
            found.append((code.co_qualname, tuple(targets)))
    if found:
        _log(found)
    return found


@functools.lru_cache(maxsize=4096)
def own_writes(code: types.CodeType) -> frozenset[str]:
    """The free names the function of *code* changes by its own text: a
    global or closure variable it changes in place or rebinds, and an
    object it reads by a free name and calls a method of that is not known
    to leave it alone (``calls.append(1)``, ``cache.set(k, v)``). ``"?"``
    when its text cannot be read."""
    node = _function_node(code)
    if node is None:
        return frozenset((_UNKNOWN,))
    try:
        changed = set(free_vars_mutated_in_function(node))
    except RecursionError:
        return frozenset((_UNKNOWN,))
    changed |= _rebound(node)
    changed |= {name for name, _attr in _methods_on_free_names(node)}
    return frozenset(changed)


def _changes_by_method(code: types.CodeType, name: str) -> bool:
    """*name* is in `own_writes` only for a method called on it."""
    node = _function_node(code)
    if node is None:
        return False
    try:
        if name in free_vars_mutated_in_function(node) or name in _rebound(node):
            return False
    except RecursionError:
        return False
    return True


def _changeable(value: Any, code: types.CodeType, name: str) -> bool:
    """Whether a method called on *value* can change it or what it holds: not
    a module's function (``np.sum``), a class's (whose methods ran and are
    read in turn), an object of the user's class (the same), or a logger
    without a handler of the user's (`_logging_callbacks`)."""
    if isinstance(value, (types.ModuleType, type, types.BuiltinFunctionType, types.FunctionType)):
        return False
    from ...analysis.handed_callables import _is_users_object, _logging_callbacks

    if _is_users_object(value):
        return False
    # A random generator's draws are the statement's randomness, which cash
    # follows on its own (``hidden_generators``).
    if type(value).__module__.startswith(("numpy.random", "random")):
        return False
    node = _function_node(code)
    attrs = {attr for owner, attr in _methods_on_free_names(node)} if node is not None else set()
    logging_module = sys.modules.get("logging")
    if logging_module is not None and isinstance(value, getattr(logging_module, "Logger", ())):
        return any(_logging_callbacks(value, attr) for attr in attrs)
    return True


@functools.lru_cache(maxsize=4096)
def _function_node(code: types.CodeType) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | None:
    """The definition of *code*'s function, from its file, or None."""
    try:
        source = getsource(code)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    import textwrap

    text = textwrap.dedent(source)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A lambda in the middle of a line: the line alone may not parse.
        try:
            tree = ast.parse("(" + text.strip().rstrip(",").rstrip("\\") + ")")
        except SyntaxError:
            return None
    kinds = (ast.Lambda,) if code.co_name == "<lambda>" else (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, kinds) and (isinstance(node, ast.Lambda) or node.name == code.co_name):
            return node
    return None


def _locals_of(node: ast.AST) -> set[str]:
    args = node.args  # type: ignore[attr-defined]
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    declared: set[str] = set()
    for sub in _own_nodes(node):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            declared.update(sub.names)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, ast.alias):
            names.add((sub.asname or sub.name).split(".")[0])
    return names - declared


def _own_nodes(node: ast.AST) -> Iterable[ast.AST]:
    """The nodes of *node*'s body, not of functions or classes defined in it."""
    body = node.body if isinstance(node.body, list) else [node.body]  # type: ignore[attr-defined]
    pending = list(body)
    while pending:
        sub = pending.pop()
        yield sub
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(sub))


def _rebound(node: ast.AST) -> set[str]:
    """Names declared ``global`` or ``nonlocal`` and bound in *node*."""
    declared: set[str] = set()
    bound: set[str] = set()
    for sub in _own_nodes(node):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            declared.update(sub.names)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
    return declared & bound


def _methods_on_free_names(node: ast.AST) -> set[tuple[str, str]]:
    """``(name, method)`` of each method *node* calls on a free name, not
    one known to leave its object alone."""
    local = _locals_of(node)
    out: set[tuple[str, str]] = set()
    for sub in _own_nodes(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id not in local
            and sub.func.attr not in KNOWN_PURE_METHODS
        ):
            out.add((sub.func.value.id, sub.func.attr))
    return out


def _log(found: list[tuple[str, tuple[str, ...]]]) -> None:
    path = os.environ.get(_LOG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:  # @cash:assume-safe
            for fn, targets in found:
                fh.write(json.dumps({"fn": fn, "targets": list(targets)}) + "\n")
    except OSError:
        pass
