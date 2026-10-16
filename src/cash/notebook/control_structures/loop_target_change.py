"""A ``for`` body that changes its loop variable before the work cash caches.

Each pass of a loop run iteration by iteration is keyed on the loop
variable's value when the ``for`` binds it. A body statement that changes
that value in place (``q.append(...)``, ``q[0] = ...``, ``q += ...``,
``q.attr = ...``) moves it after the key was taken, so a later statement of
the same pass that cash caches is looked up under the value as it was
bound: two passes that were equal when bound share one entry. Read from the
loop's text alone, once per loop.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable
from typing import NamedTuple

from ...analysis.cacheability import analyze_statement

__all__ = ["TargetChange", "target_changed_before_cached_work"]

_BUILTINS = frozenset(dir(builtins))


class TargetChange(NamedTuple):
    """Where a loop body changes its loop variable, and the cached work after."""

    name: str
    """The loop variable changed."""

    change_line: int
    """The line of the statement that changes it, as the loop's node numbers it."""

    work_line: int
    """The line of the first statement after it that cash caches per pass."""


def _bound_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _read_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _work_calls(stmt: ast.stmt) -> list[ast.Call]:
    """The calls in *stmt* that cash keys and stores per pass: every call but
    a builtin's (``len``, ``print``) and the statement's own outer method
    call on a name (``acc.append(...)``, which changes ``acc`` and runs
    every time)."""
    outer = stmt.value if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None
    found = []
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if node is outer and isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            continue
        if isinstance(func, ast.Name) and func.id in _BUILTINS:
            continue
        found.append(node)
    return found


def _caches_per_pass(stmt: ast.stmt, targets: set[str]) -> bool:
    """Does *stmt* (a simple statement) hold work cash caches per pass: a
    call of the user's or a library's, or a binding that reads a loop
    variable (``y = q[0]``)?"""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        return False
    if _work_calls(stmt):
        return True
    binds = isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and bool(_bound_names(stmt))
    return binds and bool(_read_names(stmt) & targets)


def _simple_statements(stmt: ast.stmt):
    """*stmt* itself, or the simple statements a compound one holds."""
    bodies = [getattr(stmt, f, None) for f in ("body", "orelse", "finalbody")]
    bodies += [h.body for h in getattr(stmt, "handlers", ())] + [c.body for c in getattr(stmt, "cases", ())]
    if not any(isinstance(b, list) for b in bodies) or isinstance(
        stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        yield stmt
        return
    for body in bodies:
        for child in body or ():
            yield from _simple_statements(child)


def target_changed_before_cached_work(
    node: ast.For,
    targets: set[str],
    skipped: Callable[[ast.stmt], bool] = lambda _stmt: False,
) -> TargetChange | None:
    """The first top-level statement of *node*'s body that changes one of
    the loop variables *targets* in place, with a later body statement cash
    caches per pass; None for an ordinary loop.

    A statement that rebinds the variable (``q = q + [1]``) is keyed on that
    new value and does not count. *skipped* names a statement cash does not
    cache (``# @cash:no-cache``).
    """
    if not targets:
        return None
    for i, stmt in enumerate(node.body):
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.Try)):
            continue
        try:
            mutated = set(analyze_statement(ast.unparse(stmt), None).top_level_mutated_vars)
        except (SyntaxError, ValueError, TypeError, RecursionError):
            continue
        changed = (mutated & targets) - (_bound_names(stmt) - _augmented(stmt))
        if not changed:
            continue
        for later in node.body[i + 1 :]:
            for simple in _simple_statements(later):
                if not skipped(simple) and _caches_per_pass(simple, targets):
                    return TargetChange(sorted(changed)[0], stmt.lineno, simple.lineno)
    return None


def _augmented(stmt: ast.stmt) -> set[str]:
    """The name ``q += ...`` changes, which on a list is a change in place."""
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return set()
