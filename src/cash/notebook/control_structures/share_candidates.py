"""Which names a control structure may have made share an object.

When a loop, branch or try block ends, the variables it made share an object
with another variable are linked (``update_lineage_after_execution``), so a
later change through one name moves the other. Asking the share check of
every name the structure changed costs a walk of what each one holds: a loop
over the first hundred of 200,000 records doing ``r['a'] += 1`` would walk all
of them, once per run, though it stored no reference anywhere.

An object comes to share another only when a reference to an existing object
is stored: a name is bound, an item or attribute is set to something that is
not a fresh value, or code runs that may store one (a call). So the names
asked about are:

* the names the structure binds (assignment targets, ``with ... as``,
  ``except ... as``, imports, ``def`` and ``class``);
* the variable at the root of an item or attribute set to a value that may be
  an existing object (``splits[name] = rows``, not ``r['a'] += 1``);
* when the structure calls anything but a builtin that stores nothing
  (``len``, ``print``, ``range``, ...), every name it changes, and every name
  the call reaches as its receiver or an argument.

A loop target over an iterable named by variables alone (``for r in rows``,
``for r in rows[:100]``, ``for k, v in d.items()``, ``for name, rows in
[('train', train_rows), ...]``) is an element of what those variables hold:
it is linked to them without a check (*static*), and through their own links
to whatever else holds the element.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping
from typing import Any

__all__ = ["share_candidates"]

#: Builtins whose call stores no reference to its arguments anywhere a
#: variable could reach: what they return is bound by an assignment, which is
#: a candidate of its own.
_STORE_NOTHING = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "callable",
        "chr",
        "divmod",
        "float",
        "format",
        "hasattr",
        "hash",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "max",
        "min",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "round",
        "str",
        "sum",
    }
)

#: Builtins that hand on the items of their arguments: a loop over them takes
#: the items those arguments hold.
_PASS_ITEMS = frozenset({"enumerate", "zip", "reversed", "sorted", "list", "tuple", "iter"})

#: Methods that hand on the items of their receiver.
_ITEM_METHODS = frozenset({"items", "values", "keys"})


def _is_builtin(name: str, user_ns: Mapping[str, Any]) -> bool:
    """Whether *name* still means the builtin of that name in *user_ns*."""
    builtin = getattr(builtins, name, None)
    return builtin is not None and user_ns.get(name, builtin) is builtin


def _root(node: ast.AST) -> str | None:
    """The variable at the root of ``a.b[c].d``, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _fresh(value: ast.AST | None) -> bool:
    """Whether *value* is always a new value holding no existing object: a
    constant, an f-string, a comparison, or arithmetic on such."""
    if value is None or isinstance(value, (ast.Constant, ast.JoinedStr, ast.Compare)):
        return True
    if isinstance(value, ast.UnaryOp):
        return _fresh(value.operand)
    if isinstance(value, ast.BinOp):
        return _fresh(value.left) and _fresh(value.right)
    if isinstance(value, ast.BoolOp):
        return all(_fresh(v) for v in value.values)
    return False


def _item_roots(node: ast.AST, user_ns: Mapping[str, Any]) -> set[str] | None:
    """The variables whose items a loop over *node* takes, when *node* names
    them alone; None when it computes something else (a call of the user's,
    a generator)."""
    if isinstance(node, ast.Constant):
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        root = _root(node)
        if root is None:
            return None
        if isinstance(node, ast.Subscript) and not _plain_index(node.slice):
            return None
        return {root}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        found: set[str] = set()
        for elt in node.elts:
            sub = _item_roots(elt, user_ns)
            if sub is None:
                return None
            found |= sub
        return found
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "range" and _is_builtin("range", user_ns):
            return set()
        if isinstance(func, ast.Name) and func.id in _PASS_ITEMS and _is_builtin(func.id, user_ns):
            args = node.args
        elif isinstance(func, ast.Attribute) and func.attr in _ITEM_METHODS and not node.args:
            args = [func.value]
        else:
            return None
        found = set()
        for arg in args:
            sub = _item_roots(arg, user_ns)
            if sub is None:
                return None
            found |= sub
        return found
    return None


def _plain_index(node: ast.AST) -> bool:
    """A subscript that takes items, not computes them: a constant, a name,
    a slice of such."""
    if isinstance(node, ast.Slice):
        return all(part is None or _plain_index(part) for part in (node.lower, node.upper, node.step))
    if isinstance(node, ast.Tuple):
        return all(_plain_index(elt) for elt in node.elts)
    return isinstance(node, (ast.Constant, ast.Name)) or (isinstance(node, ast.UnaryOp) and _plain_index(node.operand))


def _target_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def share_candidates(
    node: ast.AST, mutated: set[str], user_ns: Mapping[str, Any]
) -> tuple[set[str], dict[str, set[str]]]:
    """``(names, static)`` for the structure *node*, which changed *mutated*:
    the names whose objects the share check is asked about, and each loop
    target linked without a check to the variables whose items it takes."""
    names: set[str] = set()
    static: dict[str, set[str]] = {}
    calls = False
    # The calls of an iterable `_item_roots` took (``enumerate(rows)``,
    # ``d.items()``): they hand on items, and store nothing.
    taken: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.For, ast.AsyncFor)):
            targets = _target_names(sub.target)
            roots = _item_roots(sub.iter, user_ns)
            if roots is None:
                names |= targets
            else:
                taken.update(id(call) for call in ast.walk(sub.iter) if isinstance(call, ast.Call))
                for target in targets:
                    static.setdefault(target, set()).update(roots)
        elif isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            fresh = _fresh(sub.value)
            for target in targets:
                if isinstance(target, ast.Name) and not isinstance(sub, ast.AugAssign):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names |= _target_names(target)
                elif not fresh:
                    root = _root(target)
                    if root is not None:
                        names.add(root)
        elif isinstance(sub, ast.NamedExpr):
            names.add(sub.target.id)
        elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
            names |= _target_names(sub.optional_vars)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in sub.names)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, ast.Call) and id(sub) not in taken:
            func = sub.func
            if isinstance(func, ast.Name) and func.id in _STORE_NOTHING and _is_builtin(func.id, user_ns):
                continue
            calls = True
            for part in [func, *sub.args, *(kw.value for kw in sub.keywords)]:
                root = _root(part)
                if root is not None:
                    names.add(root)
    if calls:
        names |= mutated
    return names, static
