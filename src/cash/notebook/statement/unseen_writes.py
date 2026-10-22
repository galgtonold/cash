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
or rebinds (``calls.append(1)``), a variable of a closure it changes. A
method of a library object it calls (``porter.stem(w)``) is taken to leave
the object alone, as for a function the statement calls by name.
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
import types
from collections.abc import Iterable, Mapping
from typing import Any

from ...analysis.callee_effects import free_vars_mutated_in_function
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
    statement: str = "",
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
                targets.append(repr(name) if globals_ is namespace else f"{globals_.get('__name__')}.{name}")
        if targets:
            found.append((code.co_qualname, tuple(targets)))
    if found:
        _log(found, statement)
    return found


@functools.lru_cache(maxsize=4096)
def own_writes(code: types.CodeType) -> frozenset[str]:
    """The free names the function of *code* changes by its own text: a
    global or closure variable it changes in place (``calls.append(1)``,
    ``d[k] = v``) or rebinds, as the statement's own analysis reads a
    callee. ``"?"`` when its text cannot be read."""
    node = _function_node(code)
    if node is None:
        return frozenset((_UNKNOWN,))
    try:
        changed = set(free_vars_mutated_in_function(node))
    except RecursionError:
        return frozenset((_UNKNOWN,))
    changed |= _rebound(node)
    return frozenset(changed)


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


def _log(found: list[tuple[str, tuple[str, ...]]], statement: str) -> None:
    path = os.environ.get(_LOG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:  # @cash:assume-safe
            for fn, targets in found:
                row = {"fn": fn, "targets": list(targets), "stmt": statement[:200], "pid": os.getpid()}
                fh.write(json.dumps(row) + "\n")
    except OSError:
        pass
