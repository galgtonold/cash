"""Where the names a function calls through are bound.

The helper walk resolves each call site's name in the function's namespace
(`build_namespace`, plus the imports written inside its body:
`local_import_map`), and records the module and attribute chain the call
site uses (`binding_path`), so the decorator can re-resolve that binding on
every call (`resolve_binding`) and see a helper rebound since the analysis
(`bindings_changed`).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import sys
import types
import weakref
from collections.abc import Callable
from typing import Any

from ..install_paths import in_own_package, top_package
from ..tracking.function_tracker import is_local_module
from .purity_report import PurityReport


def local_import_map(func_def: ast.AST, func: Any) -> dict[str, tuple[str, tuple[str, ...]]]:
    """``local name -> (module, attribute prefix)`` for imports in a function body.

    ``from helpmod import scale`` inside the body binds a LOCAL, so the helper
    walk, which resolves names in the module's globals, found nothing and an
    edit to ``scale`` was served stale -- in the common shape of an
    import moved into the function to break an import cycle. Each import runs
    on every call, so the binding it makes is ``helpmod.scale`` as the module
    holds it at call time: that is the path recorded for the per-call check.
    """
    package = None
    g = getattr(func, "__globals__", None)
    if isinstance(g, dict):
        package = g.get("__package__")
    if package is None:
        package = (getattr(func, "__module__", "") or "").rpartition(".")[0]
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = (alias.name, ())
                else:
                    top = alias.name.split(".")[0]
                    found[top] = (top, ())
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    module = importlib.util.resolve_name("." * node.level + (node.module or ""), package or None)
                except (ImportError, ValueError):
                    continue
            else:
                module = node.module or ""
            if not module:
                continue
            for alias in node.names:
                if alias.name != "*":
                    found[alias.asname or alias.name] = (module, (alias.name,))
    return found


def _module_is_user_code(module_name: str, root_module: str | None) -> bool:
    """Is *module_name* user code, decided WITHOUT importing it?

    The top-level package is checked first, with a spec lookup that imports
    nothing, so a library imported inside a function to defer its cost
    (``import torch``) is never imported early on its behalf.
    """
    if in_own_package(module_name, top_package(root_module)):
        return True
    top = top_package(module_name)
    if top is None:
        return False
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return False
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in ("built-in", "frozen"):
        return False
    return is_local_module(types.SimpleNamespace(__file__=origin))


def resolve_local_import(module_name: str, prefix: tuple[str, ...], root_module: str | None) -> Any:
    """The object a function-body import binds, importing a USER module if the
    body has not run yet. That import is the one the body is about to make;
    doing it now is what lets the first call's key see the helper. A library
    module is only read if it is already loaded."""
    module = sys.modules.get(module_name)
    if module is None:
        if not _module_is_user_code(module_name, root_module):
            return None
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - the body will raise it, not the analysis
            return None
    obj: Any = module
    for attr in prefix:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def callee_chain(node: ast.AST) -> tuple[str, ...] | None:
    """The name chain a call site uses: ``_sieve`` -> ``("_sieve",)``,
    ``mod.sub.f`` -> ``("mod", "sub", "f")``; None for anything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return tuple(reversed(parts))


def binding_path(caller: Any, chain: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]] | None:
    """``(module_name, chain)`` when *chain* starts at a name *caller* looks up
    in its module's globals, so ``sys.modules[module_name]`` + the chain finds
    what the call site finds. None for a closure cell or a namespace that is
    not a registered module (``exec``, a class body)."""
    if not chain:
        return None
    code = getattr(caller, "__code__", None)
    if code is not None and chain[0] in (getattr(code, "co_freevars", ()) or ()):
        return None
    module_name = getattr(caller, "__module__", None)
    module = sys.modules.get(module_name or "")
    if module is None or getattr(module, "__dict__", None) is not getattr(caller, "__globals__", None):
        return None
    return module_name, chain


def resolve_callee_chain(namespace: dict[str, Any], chain: tuple[str, ...]) -> Any:
    """What *chain* names in *namespace*, through modules only; None if nothing."""
    obj = namespace.get(chain[0])
    for part in chain[1:]:
        if not isinstance(obj, types.ModuleType):
            return None
        obj = getattr(obj, part, None)
    return obj


def held_ref(obj: Any) -> Callable[[], Any]:
    """A weak reference where the object allows one, a strong one otherwise."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


UNRESOLVED = object()


def resolve_binding(module_name: str, chain: tuple[str, ...]) -> Any:
    """What ``sys.modules[module_name]`` + *chain* holds now, or ``UNRESOLVED``."""
    obj: Any = sys.modules.get(module_name)
    if obj is None:
        return UNRESOLVED
    for attr in chain:
        obj = getattr(obj, attr, UNRESOLVED)
        if obj is UNRESOLVED:
            return UNRESOLVED
    return obj


def bindings_changed(report: PurityReport) -> bool:
    """Does any call-site binding the report followed hold a different object now?

    A module that has left ``sys.modules`` proves nothing either way and is
    skipped. Identity, not equality: a re-created function with the same
    code is still a different object, whose globals may differ.
    """
    for module_name, chain, ref in report.helper_bindings:
        live = resolve_binding(module_name, chain)
        if live is UNRESOLVED:
            continue
        if live is not ref():
            return True
    return False


def called_names_in_tree(tree: ast.AST) -> list[str]:
    """Bare names called anywhere in *tree*, including inside lambdas.

    Used for a CLASS body, where the interesting call can sit inside a
    default-factory lambda: ``field(default_factory=lambda: B(0))``. Only
    ``ast.Name`` callees -- an attribute call (``mod.f()``) is resolved by
    the ordinary callee machinery, not here.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def _class_namespaces(cls: type) -> list[dict[str, Any]]:
    """Every globals dict in which *cls*'s body names might resolve.

    Returns a LIST, and callers must try all of them, because no single one
    is reliably right:

    * The defining module is the obvious candidate, but a class built by
      ``exec`` into a namespace that never reaches ``sys.modules`` (notebook
      cell, REPL) has none.
    * A member's ``__globals__`` covers that case -- but picking the FIRST
      member with one is wrong. A dataclass's GENERATED methods carry the
      ``dataclasses`` machinery's globals, not the user's, and whether such
      a method comes first in ``vars(cls)`` varies by Python version. On
      3.13 it does, so ``B`` in ``field(default_factory=lambda: B(10))``
      resolved against the wrong namespace and the class was never folded;
      on 3.14 it happened to work. CI caught it on all three 3.13 runners.

    Ordering is best-first (module, then member globals), but correctness
    does not depend on it -- the caller searches until a name resolves.
    """
    namespaces: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(candidate: Any) -> None:
        if isinstance(candidate, dict) and id(candidate) not in seen:
            seen.add(id(candidate))
            namespaces.append(candidate)

    module = sys.modules.get(getattr(cls, "__module__", "") or "")
    add(getattr(module, "__dict__", None))

    members: list[Any] = list(vars(cls).values())
    fields_map = getattr(cls, "__dataclass_fields__", None)
    if isinstance(fields_map, dict):
        for fld in fields_map.values():
            factory = getattr(fld, "default_factory", None)
            if factory is not None and factory is not dataclasses.MISSING:
                # Put field factories FIRST among members: a factory lambda is
                # written in the user's module, while a generated __init__ is
                # not, so it is the more likely place for the name to resolve.
                members.insert(0, factory)
    for member in members:
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        add(getattr(member, "__globals__", None))
    return namespaces


def resolve_in_class_namespaces(cls: type, name: str) -> Any:
    """First binding of *name* across *cls*'s candidate namespaces, else None."""
    for namespace in _class_namespaces(cls):
        if name in namespace:
            return namespace[name]
    return None


def build_namespace(func: Callable[..., Any]) -> dict[str, Any]:
    """Return a merged ``__globals__`` + closure-cell namespace for *func*.

    Lets :func:`~cash.analysis.ast_util.resolve_callee` see helpers defined as closures
    (nested function definitions) - not just module-level names.
    Without this, a ``@cash.cache``d function inside another
    function couldn't recurse into its sibling helpers, and any
    impurity those helpers contained would be missed.

    Closure cells are pulled from ``func.__code__.co_freevars`` paired
    with ``func.__closure__``. An empty cell (rare - happens when a
    closure variable is never assigned) is silently skipped.

    The returned dict is a shallow copy of ``__globals__`` with
    closure entries layered on top - same-name closure variables
    shadow globals, matching Python's normal scoping.
    """
    ns = dict(getattr(func, "__globals__", None) or {})
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None) or ()
    if code is not None and closure:
        freevars = getattr(code, "co_freevars", ()) or ()
        for name, cell in zip(freevars, closure):
            try:
                ns[name] = cell.cell_contents
            except ValueError:
                # Cell exists but has no value yet (forward reference
                # in mutually-recursive closures). Skip.
                continue
    return ns
