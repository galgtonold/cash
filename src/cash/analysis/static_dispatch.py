"""Lookups by a constant string, spelled as the plain call they make.

``sys.modules["helper"].g(x)``, ``globals()["g"](x)`` and
``operator.attrgetter("g")(helper)`` all call ``helper.g``; the helper walk
follows only the plain spelling, so these are rewritten into it before the
walk reads the body, and a lookup whose name is computed is reported.
"""

from __future__ import annotations

import ast
import builtins
import operator
import sys
from typing import Any

from .ast_util import resolve_callee
from .purity_report import ISSUE_DYNAMIC_PATTERN, ISSUE_UNTRACKABLE_DEP, PurityIssue

#: Prefix of the name a ``sys.modules["pkg.mod"]`` lookup is spelled as
#: (`spell_static_dispatch`): ``<module pkg.mod>``, bound in the namespace.
MODULE_NAME_PREFIX = "<module "


def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


_DYNAMIC_MODULE = "%s with a non-constant name - dynamic import; the module's code is not tracked"


_DYNAMIC_GETTER = "operator.%s(name)(obj) with non-constant name - dynamic dispatch"


def _is_identifier(node: ast.AST) -> bool:
    key = _const_str(node)
    return key is not None and key.isidentifier()


def namespace_spelled(base: ast.AST) -> str | None:
    """How *base* is spelled when it is a runtime namespace: ``globals()``,
    ``vars(x)``, ``x.__dict__``."""
    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and not base.keywords:
        if base.func.id == "globals" and not base.args:
            return "globals()"
        if base.func.id == "vars" and len(base.args) == 1:
            return "vars(...)"
    if isinstance(base, ast.Attribute) and base.attr == "__dict__":
        return "obj.__dict__"
    return None


def computed_namespace_lookup(node: ast.AST) -> str | None:
    """How *node* is spelled for a report when it looks a name up in a
    runtime namespace by a string that is not a constant identifier --
    ``globals()[k]``, ``vars(m)[k]``, ``m.__dict__[k]``, ``globals().get(k)``
    -- else None. What it finds is known only when the code runs."""
    if isinstance(node, ast.Subscript) and not _is_identifier(node.slice):
        spelled = namespace_spelled(node.value)
        return f"{spelled}[...]" if spelled else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and not _is_identifier(node.args[0])
    ):
        spelled = namespace_spelled(node.func.value)
        return f"{spelled}.get(...)" if spelled else None
    return None


class _StaticDispatchSpeller(ast.NodeTransformer):
    """Rewrites the lookups that name a module or a function by a string into
    the plain spelling the helper walk follows (`spell_static_dispatch`)."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace
        #: ``(line, kind, description)`` for each lookup whose name is computed
        #: at run time, which no rewrite can follow.
        self.untracked: list[tuple[int, str, str]] = []

    def _is(self, node: ast.AST, target: Any) -> bool:
        """Does *node* (a name or a module attribute) name *target*?"""
        if isinstance(node, ast.Name) and node.id not in self._namespace:
            return getattr(builtins, node.id, None) is target
        return resolve_callee(node, self._namespace) is target

    def _is_sys_modules(self, node: ast.AST) -> bool:
        return self._is(node, sys.modules)

    def _module_name(self, key: str, like: ast.AST) -> ast.AST | None:
        module = sys.modules.get(key)
        if module is None:
            return None
        name = f"{MODULE_NAME_PREFIX}{key}>"
        self._namespace[name] = module
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), like)

    def _namespace_lookup(self, node: ast.Subscript) -> tuple[ast.AST | None, str] | None:
        """``(owner or None, what)`` when *node* looks a name up in a runtime
        namespace: ``globals()[k]`` (owner None: the module), ``vars(x)[k]``,
        ``x.__dict__[k]``."""
        base = node.value
        if isinstance(base, ast.Call) and not base.keywords:
            if self._is(base.func, globals) and not base.args:
                return None, "globals()[...]"
            if self._is(base.func, vars) and len(base.args) == 1:
                return base.args[0], "vars(...)[...]"
        if isinstance(base, ast.Attribute) and base.attr == "__dict__":
            return base.value, "obj.__dict__[...]"
        return None

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load):
            return node
        key = _const_str(node.slice)
        if self._is_sys_modules(node.value):
            if key is None:
                self.untracked.append((node.lineno, ISSUE_UNTRACKABLE_DEP, _DYNAMIC_MODULE % "sys.modules[...]"))
                return node
            return self._module_name(key, node) or node
        lookup = self._namespace_lookup(node)
        if lookup is None or key is None or not key.isidentifier():
            return node
        owner, _ = lookup
        if owner is None:
            return ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node)
        return ast.copy_location(ast.Attribute(value=owner, attr=key, ctx=ast.Load()), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        # sys.modules.get("pkg.mod")
        if isinstance(func, ast.Attribute) and func.attr == "get" and self._is_sys_modules(func.value) and node.args:
            key = _const_str(node.args[0])
            if key is None:
                self.untracked.append((node.lineno, ISSUE_UNTRACKABLE_DEP, _DYNAMIC_MODULE % "sys.modules.get(...)"))
                return node
            return self._module_name(key, node) or node
        # attrgetter("g")(helper) -> helper.g; methodcaller("g", x)(helper) -> helper.g(x)
        if isinstance(func, ast.Call) and len(node.args) == 1 and not node.keywords and func.args:
            name = _const_str(func.args[0])
            if self._is(func.func, operator.attrgetter) and len(func.args) == 1 and not func.keywords:
                if name is None or not all(p.isidentifier() for p in name.split(".")):
                    self.untracked.append((node.lineno, ISSUE_UNTRACKABLE_DEP, _DYNAMIC_GETTER % "attrgetter"))
                    return node
                spelled: ast.AST = node.args[0]
                for part in name.split("."):
                    spelled = ast.copy_location(ast.Attribute(value=spelled, attr=part, ctx=ast.Load()), node)
                return spelled
            if self._is(func.func, operator.methodcaller):
                if name is None or not name.isidentifier():
                    self.untracked.append((node.lineno, ISSUE_UNTRACKABLE_DEP, _DYNAMIC_GETTER % "methodcaller"))
                    return node
                method = ast.copy_location(ast.Attribute(value=node.args[0], attr=name, ctx=ast.Load()), node)
                return ast.copy_location(ast.Call(func=method, args=func.args[1:], keywords=func.keywords), node)
        # globals()[k](x), globals().get(k)(x), vars(m)[k](x): the function
        # called is named by a run-time string, so neither its code nor what
        # it reads is in the key -- an edit to what it reads served the old
        # result. Reported as `getattr(obj, name)(x)` is: the call runs uncached.
        lookup = computed_namespace_lookup(func)
        if lookup is not None:
            self.untracked.append(
                (node.lineno, ISSUE_UNTRACKABLE_DEP, f"{lookup}(...) with a non-constant name - dynamic dispatch")
            )
            return node
        # globals()[k].g(x), globals().get(k).g(x), vars(m)[k].g(x): an
        # object looked up through a runtime namespace by a computed name,
        # then called through it.
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        lookup = computed_namespace_lookup(root) if root is not func else None
        if lookup is not None:
            self.untracked.append(
                (
                    node.lineno,
                    ISSUE_DYNAMIC_PATTERN,
                    f"calls through {lookup} with a computed name - that lookup does not reach "
                    f"the cache key, so editing the callable it names will not invalidate; "
                    f"name it with depends_on=[...]",
                )
            )
        return node


def spell_static_dispatch(func_def: ast.AST, namespace: dict[str, Any], qualname: str) -> list[PurityIssue]:
    """Rewrite, in place, lookups by a CONSTANT string into what they name, and
    report the ones whose name is computed at run time.

    ``sys.modules["helper"].g(x)``, ``globals()["helper"].g(x)``,
    ``vars(helper)["g"](x)``, ``helper.__dict__["g"](x)``,
    ``operator.attrgetter("g")(helper)(x)`` and
    ``operator.methodcaller("g", x)(helper)`` all call ``helper.g``, and none
    was followed: an edit to ``g`` served the old result, with no word, while
    ``getattr(helper, name)`` and ``importlib.import_module`` warned. The
    constant spellings become ``helper.g`` for the walk (a module found in
    ``sys.modules`` is bound in *namespace* under a ``<module name>`` name);
    a computed name is reported like the dispatch it is.
    """
    speller = _StaticDispatchSpeller(namespace)
    speller.visit(func_def)
    return [
        PurityIssue(kind=kind, description=what, where=qualname, line=line) for line, kind, what in speller.untracked
    ]
