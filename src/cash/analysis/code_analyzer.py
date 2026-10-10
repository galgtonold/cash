"""AST-based code analysis for detecting statement inputs and outputs.

The :class:`CodeAnalyzer` walks Python AST trees to identify which
variables a statement reads (inputs) and writes (outputs).  This
analysis drives cache key computation and dependency tracking.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import logging
import re
import sys
import textwrap
import types
from collections import ChainMap
from collections.abc import Callable, Mapping
from typing import Any

from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..code_digest import unwrap_partials
from ..effects import Action, classify_call
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import getsource
from .ast_util import bytecode_global_refs, copy_tree, parse_cached
from .callee_effects import callee_global_mutations
from .file_effects import NOTEBOOK_POLICY, SCANNED_KINDS
from .handed_callables import handed_callables
from .helper_code import is_user_code
from .namespace_effects import capturable_globals, notebook_global_rebinds

__all__ = [
    "CodeAnalyzer",
    "calls_ipython",
    "clean_cell_source",
    "magic_python",
    "python_magic_argument",
    "split_magic_argument",
    "expr_has_trailing_semicolon",
    "parse_cell_source",
    "splitlines_like_the_parser",
    "statement_code",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level AST visitors
# ---------------------------------------------------------------------------


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def _own_bindings(scope: ast.AST) -> set[str]:
    """The names the function or lambda *scope* binds itself: its parameters
    and every name its own body assigns, imports, defines or catches, minus
    those it declares ``global``.

    Nested functions, lambdas and classes are not entered (their own names
    are theirs), nor are comprehension targets; a ``:=`` inside a
    comprehension binds here, as Python scopes it. A name this misses is
    resolved as a global, which can only add an edge, never lose one.
    """
    args = scope.args
    bound = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if a is not None}
    declared_global: set[str] = set()
    stack: list[ast.AST] = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            continue  # its decorators and defaults may hold a `:=`; missing one only adds an edge
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        if isinstance(node, ast.comprehension):
            stack.extend((node.iter, *node.ifs))  # its target is the comprehension's own
            continue
        stack.extend(ast.iter_child_nodes(node))
    return bound - declared_global


def _comprehension_targets(node: ast.AST) -> set[str]:
    """Names the comprehension *node* binds in its own scope (its targets)."""
    return {n.id for gen in node.generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)}


def _dotted_chain(node: ast.expr) -> list[str] | None:
    """``["a", "b", "c"]`` for ``a.b.c``; None when the root is not a name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    parts.reverse()
    return parts


class _CallVisitor(ast.NodeVisitor):
    """Collect all function-call names from an AST for find_called_functions.

    ``referenced`` is every name and ``a.b`` chain READ anywhere, called or
    not: ``map(inner, xs)``, ``pool.map(inner, xs)``, ``delayed(inner)(x)``,
    ``for fn in [inner]``, ``def f(fn=inner)``.

    A chain whose first name the function binds itself -- a parameter, an
    assignment, a loop or ``with`` target, a comprehension variable, a free
    variable from an enclosing function (*outer_locals*) -- is a local, not
    the global of that name, and is left out: ``def f(df): df.values`` must
    never look at a module-level ``df``.
    """

    def __init__(self, outer_locals: frozenset[str] | set[str] = frozenset()) -> None:
        self.names_to_resolve: list[str] = []
        self.referenced: list[str] = []
        self._scopes: list[set[str]] = [set(outer_locals)] if outer_locals else []

    def _is_local(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _scoped(self, bound: set[str], nodes: list[ast.AST]) -> None:
        self._scopes.append(bound)
        try:
            for node in nodes:
                self.visit(node)
        finally:
            self._scopes.pop()

    def _visit_function(self, node: ast.AST) -> None:
        # Decorators, defaults and annotations run in the enclosing scope.
        for deco in getattr(node, "decorator_list", ()):
            self.visit(deco)
        args = node.args
        for default in (*args.defaults, *(d for d in args.kw_defaults if d is not None)):
            self.visit(default)
        if not isinstance(node, ast.Lambda):
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
                if a is not None and a.annotation is not None:
                    self.visit(a.annotation)
            if node.returns is not None:
                self.visit(node.returns)
        body = node.body if isinstance(node.body, list) else [node.body]
        self._scoped(_own_bindings(node), body)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    visit_Lambda = _visit_function

    def _visit_comprehension(self, node: ast.AST) -> None:
        # The first iterable is evaluated in the enclosing scope.
        first, *rest = node.generators
        self.visit(first.iter)
        inner: list[ast.AST] = [*first.ifs]
        for gen in rest:
            inner.extend((gen.iter, *gen.ifs))
        if isinstance(node, ast.DictComp):
            inner.extend((node.key, node.value))
        else:
            inner.append(node.elt)
        self._scoped(_comprehension_targets(node), inner)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension
    visit_DictComp = _visit_comprehension

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_local(node.id):
            self.referenced.append(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            chain = _dotted_chain(node)
            if chain is not None and not self._is_local(chain[0]):
                self.referenced.append(".".join(chain))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _dotted_chain(node.func)
        if chain is not None and not self._is_local(chain[0]):
            self.names_to_resolve.append(".".join(chain))
        self.generic_visit(node)


def _exec_literal(node: ast.Call) -> ast.Module | None:
    """The code ``exec("...")`` runs in the namespace it is called from, parsed:
    a single string-literal argument, no namespaces of its own. None for any
    other call, or text that does not parse."""
    if not (
        isinstance(node.func, ast.Name)
        and node.func.id == "exec"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    try:
        return ast.parse(node.args[0].value)
    except (SyntaxError, ValueError):
        return None


class _FlowVisitor(ast.NodeVisitor):
    """Track variable reads (inputs) and writes (outputs) across scopes.

    Lifted from the inline class inside ``analyze_code_block`` so that its
    methods each have their own complexity budget.
    """

    def __init__(self) -> None:
        # Stack of sets of defined variables. scopes[0] is the cell level.
        self.scopes: list[set[str]] = [set()]
        # Parallel stack: True where the scope at the same index is a
        # comprehension scope. Used for PEP 572 walrus scoping -- a ``:=``
        # target binds in the nearest enclosing NON-comprehension scope.
        self.comp_scopes: list[bool] = [False]
        self.outputs: set[str] = set()
        self.real_inputs: set[str] = set()
        self.defined_in_block: set[str] = set()
        self.modified_objects: set[str] = set()

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def is_defined(self, name: str) -> bool:
        """Check if name is defined in any active scope."""
        return any(name in scope for scope in reversed(self.scopes))

    def define_variable(self, name: str) -> None:
        self.scopes[-1].add(name)
        if len(self.scopes) == 1:
            self.outputs.add(name)
            self.defined_in_block.add(name)

    @staticmethod
    def _add_args_to_scope(scope: set, args: ast.arguments) -> None:
        """Register all argument names from an ast.arguments node into scope."""
        for arg in args.args:
            scope.add(arg.arg)
        for arg in args.posonlyargs:
            scope.add(arg.arg)
        for arg in args.kwonlyargs:
            scope.add(arg.arg)
        if args.vararg:
            scope.add(args.vararg.arg)
        if args.kwarg:
            scope.add(args.kwarg.arg)

    # ------------------------------------------------------------------
    # visit_* methods
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.define_variable(node.id)
        elif isinstance(node.ctx, ast.Load) and not self.is_defined(node.id):
            self.real_inputs.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # 1. Decorators (in current scope)
        for decorator in node.decorator_list:
            self.visit(decorator)

        # 2. Defaults (in current scope)
        if node.args.defaults:
            for default in node.args.defaults:
                self.visit(default)
        if node.args.kw_defaults:
            for default in node.args.kw_defaults:
                if default:
                    self.visit(default)

        # 3. Define function name in CURRENT scope
        self.define_variable(node.name)

        # 4. Enter new scope, add arguments, visit body, exit scope
        self.scopes.append(set())
        self.comp_scopes.append(False)
        self._add_args_to_scope(self.scopes[-1], node.args)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()
        self.comp_scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.define_variable(node.name)
        self.scopes.append(set())
        self.comp_scopes.append(False)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()
        self.comp_scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if node.args.defaults:
            for default in node.args.defaults:
                self.visit(default)
        if node.args.kw_defaults:
            for default in node.args.kw_defaults:
                if default:
                    self.visit(default)
        self.scopes.append(set())
        self.comp_scopes.append(False)
        self._add_args_to_scope(self.scopes[-1], node.args)
        self.visit(node.body)
        self.scopes.pop()
        self.comp_scopes.pop()

    # --- Comprehension scoping (Python 3: iteration vars are local) ---

    def _visit_comprehension(self, node: ast.expr, value_nodes: list[ast.expr]) -> None:
        """Handle ListComp / SetComp / GeneratorExp / DictComp."""
        self.scopes.append(set())
        self.comp_scopes.append(True)
        for generator in node.generators:  # type: ignore[attr-defined]
            self.visit(generator.iter)
            self._define_comp_target(generator.target)
            for if_clause in generator.ifs:
                self.visit(if_clause)
        for vnode in value_nodes:
            self.visit(vnode)
        self.scopes.pop()
        self.comp_scopes.pop()

    def _define_comp_target(self, target: ast.expr) -> None:
        """Define comprehension target variable(s) in current scope only."""
        if isinstance(target, ast.Name):
            self.scopes[-1].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_comp_target(elt)
        elif isinstance(target, ast.Starred):
            self._define_comp_target(target.value)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store):
            parent_name = self._extract_base_name(node.value)
            if parent_name:
                if not self.is_defined(parent_name):
                    self.real_inputs.add(parent_name)
                if len(self.scopes) == 1:
                    self.outputs.add(parent_name)
                    self.modified_objects.add(parent_name)
        elif isinstance(node.ctx, ast.Load):
            parent_name = self._extract_base_name(node.value)
            if parent_name and not self.is_defined(parent_name):
                self.real_inputs.add(parent_name)
        self.visit(node.slice)
        self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            parent_name = self._extract_base_name(node.value)
            if parent_name:
                if not self.is_defined(parent_name):
                    self.real_inputs.add(parent_name)
                if len(self.scopes) == 1:
                    self.outputs.add(parent_name)
                    self.modified_objects.add(parent_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # Walrus ``target := value``. Visit the value FIRST so a self-referential
        # read (``n := n + 1``) registers ``n`` as an input before the binding
        # hides it (mirrors visit_Assign). PEP 572 also binds the target in the
        # nearest enclosing NON-comprehension scope, so a walrus inside a
        # comprehension still defines a cell-level output.
        self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            self._define_walrus_target(target.id)
        else:  # defensive: PEP 572 only permits a bare Name target
            self.visit(target)

    def _define_walrus_target(self, name: str) -> None:
        idx = len(self.scopes) - 1
        while idx > 0 and self.comp_scopes[idx]:
            idx -= 1
        self.scopes[idx].add(name)
        if idx == 0:
            self.outputs.add(name)
            self.defined_in_block.add(name)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            if not self.is_defined(node.target.id):
                self.real_inputs.add(node.target.id)
            self.define_variable(node.target.id)
        elif isinstance(node.target, (ast.Subscript, ast.Attribute)):
            parent_name = self._extract_base_name(node.target)
            if parent_name:
                if not self.is_defined(parent_name):
                    self.real_inputs.add(parent_name)
                if len(self.scopes) == 1:
                    self.outputs.add(parent_name)
                    self.modified_objects.add(parent_name)
            # What selects the element is read too: `too_high` in
            # `sales.loc[too_high, ['price']] /= 100`. Unread, it was no input,
            # so a restart rebuilt the statement without its producer --
            # UpstreamStateError, `name 'too_high' is not defined`
            # -- and an edit to the mask did not re-key the statement.
            target = node.target
            while isinstance(target, (ast.Subscript, ast.Attribute)):
                if isinstance(target, ast.Subscript):
                    self.visit(target.slice)
                target = target.value
        if node.value:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        executed = _exec_literal(node) if len(self.scopes) == 1 else None
        if executed is not None:
            # ``exec('w = base * 2')`` runs its text in the cell's namespace:
            # what it reads and binds is the statement's, as if written inline.
            for stmt in executed.body:
                self.visit(stmt)
        if isinstance(node.func, ast.Attribute):
            for keyword in node.keywords:
                if keyword.arg == "inplace" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    parent_name = self._extract_base_name(node.func.value)
                    if parent_name:
                        if not self.is_defined(parent_name):
                            self.real_inputs.add(parent_name)
                        if len(self.scopes) == 1:
                            self.outputs.add(parent_name)
                            self.modified_objects.add(parent_name)
                    break
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.define_variable(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self.define_variable(name)

    def _extract_base_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            return self._extract_base_name(node.value)
        return None


class _ForbiddenVisitor(ast.NodeVisitor):
    """Find the calls whose kind the notebook refuses and that are judged by
    what their names are bound to (``SCANNED_KINDS``): the clock, the person
    at the keyboard. Imports written in the statement itself are followed.
    """

    def __init__(self, user_ns: Mapping[str, Any]) -> None:
        self.user_ns = user_ns
        self.local_ns: dict[str, Any] = {}
        self.namespace = ChainMap(self.local_ns, user_ns)  # type: ignore[arg-type]
        self.found_reasons: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        reason = _forbidden_call(node, self.namespace)
        if reason is not None:
            self.found_reasons.append(reason)
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:
        """A function's BODY runs when it is called, not where it is defined.

        Every ``def`` whose body called ``time.time()`` got
        a "NOT CACHED: def f(...) - time.time" row -- "a def is never
        something I wanted cached; the row reads as if cash refuses to cache
        my function". What does run here is the decorators, the default
        arguments and the annotations, so those are still visited. A CLASS
        body runs at definition time and is left alone.
        """
        for deco in node.decorator_list:
            self.visit(deco)
        args = node.args
        for default in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            self.visit(default)
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
            + [a for a in (args.vararg, args.kwarg) if a is not None]
        ):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Its body runs when it is called, like a function's."""
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_name = alias.name.split(".")[0]
            module = _loaded(base_name)
            if module is not None:
                self.local_ns[alias.asname or base_name] = (
                    sys.modules.get(alias.name, module) if alias.asname else module
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _loaded(node.module or "") if not node.level else None
        if module is None:
            return
        for alias in node.names:
            obj = getattr(module, alias.name, None)
            if obj is not None:
                self.local_ns[alias.asname or alias.name] = obj


#: Standard-library modules whose calls the scan knows, imported on demand when
#: the statement imports one that nothing has loaded yet. Any other module is
#: followed only once it is loaded: the scan never imports a library.
_STDLIB_ROOTS: frozenset[str] = frozenset({"time", "datetime", "uuid", "os", "getpass", "secrets"})


def _loaded(name: str) -> Any:
    module = sys.modules.get(name)
    if module is None and name in _STDLIB_ROOTS:
        module = importlib.import_module(name)
    return module


def _forbidden_call(node: ast.Call, namespace: Mapping[str, Any]) -> str | None:
    """The reason *node* makes a statement uncacheable, or None.

    Only a call whose first name is bound (in the namespace, by an import in
    the statement, or as a builtin) counts: an unbound name is not a call to
    anything yet.
    """
    root = node.func
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or not (root.id in namespace or hasattr(builtins, root.id)):
        return None
    effect = classify_call(node, namespace)
    if effect is None or effect.kind not in SCANNED_KINDS or NOTEBOOK_POLICY[effect.kind] is not Action.REFUSE:
        return None
    return ".".join(effect.name.split(".")[-2:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _callee_name(obj: Any, known_by_id: Mapping[int, str]) -> str | None:
    """The ``module.qualname`` a called or referenced *obj* is registered under.

    A known function is found by identity -- itself or what its wrappers
    wrap -- so the edge gets the registry's own name, however that was
    spelled. Otherwise the script's ``__main__`` is named the way the
    registry names it (`resolve_main_module`, read from the innermost
    function's globals: a wrapper's globals are its decorator's module).
    Run as ``python pipeline.py``, a cached ``inner`` is registered as
    ``pipeline.inner`` but its ``__module__`` says ``__main__``, and a
    caller of it recorded no edge: editing ``inner`` never reached it.
    """
    inner = obj
    walked: set[int] = set()
    while id(inner) not in walked:  # every layer, however many; a cycle ends
        walked.add(id(inner))
        name = known_by_id.get(id(inner))
        if name is not None:
            return name
        nxt = getattr(inner, "__wrapped__", None)
        if nxt is None:
            break
        inner = nxt
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(qualname, str):
        return None
    module = getattr(obj, "__module__", None) or "__unknown__"
    if module in MAIN_MODULE_NAMES:
        module = resolve_main_module(inner)
    return f"{module}.{qualname}"


def static_attribute(obj: Any, name: str) -> Any:
    """``obj.name`` as analysis may read it: without running user code.

    Looked up with `inspect.getattr_static`, so no property, ``__getattr__``
    or other descriptor of a user's value runs: ``df.values`` on a
    module-level table would build an array of the whole table, and a lazy
    loader would load. A ``staticmethod`` or ``classmethod`` gives the
    function it holds and a slot its value; any other descriptor comes back
    as itself (a ``property`` object names no function). A module's own
    ``__getattr__`` (PEP 562, how packages import submodules lazily) still
    answers a name its namespace lacks: that is how the call itself would
    reach it. Raises AttributeError when nothing is found.
    """
    try:
        value = inspect.getattr_static(obj, name, _MISSING)
        if value is _MISSING and isinstance(obj, types.ModuleType) and "__getattr__" in vars(obj):
            value = getattr(obj, name)
    except Exception as exc:  # noqa: BLE001 - a probe of arbitrary objects
        raise AttributeError(name) from exc
    if value is _MISSING:
        raise AttributeError(name)
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    if isinstance(value, types.MemberDescriptorType) and not isinstance(obj, type):
        return value.__get__(obj, type(obj))  # a __slots__ slot: a plain read
    return value


_MISSING = object()

#: How many property getters and ``__getattr__`` bodies one call chain is
#: followed through (`_chain_targets`) before the rest counts as unresolved.
_DYNAMIC_HOPS = 8


def _dynamic_hook(obj: Any, root_module: str | None) -> Callable | None:
    """The user-written ``__getattr__`` (or ``__getattribute__``) that answers
    a name *obj*'s class does not hold, or None. A module's own
    ``__getattr__`` is not one: `static_attribute` already asks it."""
    if isinstance(obj, types.ModuleType):
        return None
    cls = obj if isinstance(obj, type) else type(obj)
    owner = type(cls) if isinstance(obj, type) else cls
    for hook_name in ("__getattr__", "__getattribute__"):
        hook = inspect.getattr_static(owner, hook_name, None)
        if isinstance(hook, (staticmethod, classmethod)):
            hook = hook.__func__
        if isinstance(hook, types.FunctionType) and is_user_code(hook, root_module):
            return hook
    return None


def _body_targets(fn: Callable) -> list[Any]:
    """What the names and ``a.b`` chains *fn*'s body reads hold, resolved
    statically through its globals, its closure, modules and classes, the
    way `CodeAnalyzer._referenced_function` resolves a read name: nothing
    of the user's runs. A property getter's ``return impl.inner`` gives
    ``impl.inner``; a ``__getattr__``'s ``getattr(impl, name)`` gives
    ``impl``."""
    try:
        tree = ast.parse(textwrap.dedent(getsource(fn)))
    except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError):
        return []
    code = getattr(fn, "__code__", None)
    visitor = _CallVisitor()
    visitor.visit(tree)
    namespace: dict[str, Any] = dict(getattr(fn, "__globals__", None) or {})
    for name, cell in zip(getattr(code, "co_freevars", ()) or (), getattr(fn, "__closure__", None) or ()):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:
            continue
    found: list[Any] = []
    for chain in dict.fromkeys([*visitor.referenced, *visitor.names_to_resolve]):
        parts = chain.split(".")
        if parts[0] not in namespace:
            continue
        obj = namespace[parts[0]]
        try:
            for part in parts[1:]:
                if not isinstance(obj, (types.ModuleType, type)):
                    obj = _MISSING
                    break
                obj = static_attribute(obj, part)
        except AttributeError:
            continue
        if obj is not _MISSING:
            found.append(obj)
    return found


def _chain_targets(obj: Any, rest: list[str], root_module: str | None, hops: int = 0) -> tuple[list[Any], bool]:
    """What ``obj.<rest>`` can reach, read without running the user's code,
    and whether a property or ``__getattr__`` of the user's code was in the
    way.

    A plain hop is `static_attribute`. A hop that gives a property of the
    user's code, or that only the user's ``__getattr__`` can answer, is
    followed through the getter's body instead (`_body_targets`): what it
    reads stands for what it returns, so ``api.inner`` with ``inner`` a
    property returning ``impl.inner`` reaches ``impl.inner``. That is an
    over-approximation (every function the getter reads counts), never a
    run of the getter. The second value is True when such a hop was met.
    """
    for i, part in enumerate(rest):
        try:
            value = static_attribute(obj, part)
        except AttributeError:
            hook = _dynamic_hook(obj, root_module)
            if hook is None:
                return [], False
            return _through_body(hook, rest[i:], root_module, hops), True
        if isinstance(value, (property, functools.cached_property)):
            fget = value.fget if isinstance(value, property) else value.func
            if not (isinstance(fget, types.FunctionType) and is_user_code(fget, root_module)):
                return [], False
            return _through_body(fget, rest[i + 1 :], root_module, hops), True
        obj = value
    return [obj], False


def _through_body(fn: Callable, rest: list[str], root_module: str | None, hops: int) -> list[Any]:
    """`_chain_targets` continued from each object *fn*'s body reads."""
    if hops >= _DYNAMIC_HOPS:
        return []
    reached: list[Any] = []
    for target in _body_targets(fn):
        if not rest:
            reached.append(target)
            continue
        more, _ = _chain_targets(target, rest, root_module, hops + 1)
        reached.extend(more)
    return reached


class CodeAnalyzer:
    """Analyzes function code to determine dependencies and compute hashes."""

    @staticmethod
    def find_called_functions(
        func: Callable, known_functions: dict[str, Callable] | None = None, *, include_references: bool = False
    ) -> set[str]:
        """
        Parse the function AST and find calls to other functions.
        Resolves names using the function's globals to handle imports and aliases.
        Returns a set of function names (qualnames).

        With ``include_references`` (and ``known_functions`` to filter by),
        a known function the body only REFERENCES counts too: handed to
        ``map``, a pool, ``joblib.delayed``, kept in a list or a default. A
        ``functools.partial`` over one resolves to it, called or referenced.
        ``sum(map(inner, [n]))`` with ``inner`` cached kept its old
        result after ``inner``'s helper changed -- only a CALL made an edge.
        """

        return CodeAnalyzer.find_called_functions_and_gaps(
            func, known_functions, include_references=include_references
        )[0]

    @staticmethod
    def find_called_functions_and_gaps(
        func: Callable, known_functions: dict[str, Callable] | None = None, *, include_references: bool = False
    ) -> tuple[set[str], list[str]]:
        """`find_called_functions`, and the call chains in *func* that go
        through a property or ``__getattr__`` of the user's code to
        something cash cannot tell is keyed: nothing it can name, or a
        plain function of the user's (the key follows a cached one only).

        ``api.inner(x)`` where ``inner`` is a property returning
        ``impl.inner``, or ``api``'s class forwards names with
        ``__getattr__``: the chain is followed through the getter's body.
        When that finds no cached function, every cached function named
        like the attribute (``inner``) counts instead, so editing any of
        them invalidates; when none is named so either, the chain is a gap.
        """
        code = getattr(func, "__code__", None)
        visitor = _CallVisitor(frozenset(getattr(code, "co_freevars", ()) or ()))
        try:
            source = textwrap.dedent(getsource(func))
            tree = ast.parse(source)
        except SOURCE_RETRIEVAL_ERRORS:
            # No source (`python - <<EOF`, `python -c`, `exec`): the names
            # its bytecode looks up stand in for the calls, so a cached
            # function it calls is still an edge.
            visitor.names_to_resolve = [".".join(chain) for chain in bytecode_global_refs(func)]
        else:
            visitor.visit(tree)

        # An opaque callable (builtin / C-extension / ufunc / partial) may have
        # source available via ``__wrapped__`` yet lack ``__globals__``; without
        # it, names can't be resolved, so skip dependency analysis.
        globals_dict = getattr(func, "__globals__", None)
        if globals_dict is None:
            return set(), []
        resolved_qualnames: set[str] = set()
        gaps: list[str] = []
        known_by_id = {id(f): n for n, f in known_functions.items()} if known_functions else {}
        root_module = getattr(func, "__module__", None)

        for name in visitor.names_to_resolve:
            parts = name.split(".")
            obj = globals_dict.get(parts[0])

            if obj is None and hasattr(builtins, parts[0]):
                obj = getattr(builtins, parts[0])

            # Existence check, not truthiness: a resolved name may be bound to
            # an object whose __bool__ is ambiguous/raises (DataFrame, ndarray),
            # and `if obj:` would crash analysis of any function that references
            # such a value in its globals.
            if obj is not None:
                targets, dynamic = _chain_targets(obj, parts[1:], root_module)
                found = False
                for target in targets:
                    try:
                        target = unwrap_partials(target)
                        fqn = _callee_name(target, known_by_id)
                    except AttributeError:
                        continue  # Expected: some callables lack __qualname__
                    if fqn is not None and (known_functions is None or fqn in known_functions):
                        resolved_qualnames.add(fqn)
                        found = True
                if dynamic and not found:
                    # Through a getter cash could not see into, or to a plain
                    # helper the key does not follow that way.
                    same_named = [
                        n for n, f in (known_functions or {}).items() if getattr(f, "__name__", None) == parts[-1]
                    ]
                    resolved_qualnames.update(same_named)
                    if not same_named and (
                        not targets
                        or any(isinstance(t, types.FunctionType) and is_user_code(t, root_module) for t in targets)
                    ):
                        gaps.append(name)

        if include_references and known_functions is not None:
            called = set(visitor.names_to_resolve)
            for name in dict.fromkeys(visitor.referenced):
                if name in called:
                    continue
                fqn = CodeAnalyzer._referenced_function(name, globals_dict, known_by_id)
                if fqn is not None and fqn in known_functions:
                    resolved_qualnames.add(fqn)

        return resolved_qualnames, gaps

    @staticmethod
    def _referenced_function(
        name: str, globals_dict: dict[str, Any], known_by_id: Mapping[int, str] | None = None
    ) -> str | None:
        """``module.qualname`` of the function a READ name holds, or None.

        Resolved through modules and classes only: an instance's attribute can
        be a property, and analysis must not run user code to find a name.
        """
        parts = name.split(".")
        obj = globals_dict.get(parts[0])
        try:
            for part in parts[1:]:
                if not isinstance(obj, (types.ModuleType, type)):
                    return None
                obj = inspect.getattr_static(obj, part, None)
            obj = unwrap_partials(obj)
            if not callable(obj) or isinstance(obj, type):
                return None
            return _callee_name(obj, known_by_id or {})
        except Exception:  # noqa: BLE001 - a probe of arbitrary globals
            return None

    @staticmethod
    def _logical_line_start_flags(code: str) -> list[bool]:
        """Return one bool per physical line: True if the line *begins* a new
        logical (top-level) Python line.

        A physical line does NOT begin a logical line when it continues the
        previous one — i.e. it sits inside an open ``()``/``[]``/``{}`` group,
        inside a triple-quoted string, or follows a ``\\`` line continuation.
        This lets :meth:`strip_magics` distinguish a real cell/line magic
        (``%time`` / ``!ls`` at the *start* of a statement) from a ``%`` or
        ``!`` that merely begins a *continuation* line of a multi-line
        statement (``print("...%.4f"\\n      % (a, b))``) — the latter is
        ordinary Python and must not be deleted.
        """
        lines = code.split("\n")
        flags: list[bool] = []
        depth = 0  # open bracket/paren/brace nesting (outside strings)
        in_str: str | None = None  # open triple-quote delimiter, or None
        prev_backslash = False
        in_magic = False  # the line before was a magic that ends in a backslash
        for line in lines:
            is_start = depth == 0 and in_str is None and not prev_backslash
            flags.append(is_start)
            # A dropped magic is a self-contained logical line, but for the
            # lines it continues with a trailing backslash (``!pip install a \\``):
            # do not let its characters perturb the scanner state.
            if (is_start and _is_magic_line(line)) or (in_magic and not is_start):
                prev_backslash = in_magic = line.rstrip().endswith("\\")
                continue
            in_magic = False
            i, n = 0, len(line)
            backslash = False
            while i < n:
                c = line[i]
                if in_str is not None:  # inside a triple string
                    if c == "\\":
                        i += 2
                        continue
                    if line.startswith(in_str, i):
                        in_str = None
                        i += 3
                        continue
                    i += 1
                    continue
                if c == "#":  # comment to end of line
                    break
                if c == "\\" and i == n - 1:  # explicit continuation
                    backslash = True
                    i += 1
                    continue
                if c in "([{":
                    depth += 1
                    i += 1
                    continue
                if c in ")]}":
                    depth = max(0, depth - 1)
                    i += 1
                    continue
                if c in ('"', "'"):
                    if line.startswith(c * 3, i):  # triple-quoted string
                        delim = c * 3
                        j, closed = i + 3, False
                        while j < n:
                            if line[j] == "\\":
                                j += 2
                                continue
                            if line.startswith(delim, j):
                                j += 3
                                closed = True
                                break
                            j += 1
                        if closed:
                            i = j
                            continue
                        in_str = delim  # spills onto next line
                        i = n
                        continue
                    j = i + 1  # single-line string
                    while j < n:
                        if line[j] == "\\":
                            j += 2
                            continue
                        if line[j] == c:
                            j += 1
                            break
                        j += 1
                    i = j
                    continue
                i += 1
            prev_backslash = backslash
        return flags

    @staticmethod
    def strip_magics(code: str) -> str:
        """Remove Jupyter magics from code.

        A cell magic keeps its body only when cash runs the body as Python in
        the user's namespace (``%%time``, ``%%capture``); IPython runs any
        other cell magic's body (``%%writefile``, ``%%script``, ``%%timeit``,
        ``%%debug``) in a way cash cannot see, so the cell strips to nothing
        (:func:`_cell_magic_body`).

        Only lines that *begin a logical line* and start with ``%`` or ``!``,
        or assign one (``files = !ls``, ``t = %time f()``), are treated as
        magics. The name such a line binds comes from IPython, not from code
        cash can read, so it has no producer here, as a name bound by a
        statement cash cannot see. A ``%`` (modulo / ``%``-format) or ``!`` that
        opens a continuation line of a multi-line statement is real Python and
        is preserved — otherwise a valid statement such as::

            print("Asian call = %.4f\\n"
                  "European   = %.4f"
                  % (a, b))

        would be mangled into unparseable code, making the simulator raise a
        *fictional* SyntaxError that silently disables cache restore.
        """
        # Fast path: a cell that already parses contains no magics to strip
        # (a statement-leading ``%``/``!`` is never valid top-level Python), so
        # return it untouched and avoid any line surgery on multi-line code.
        # Await-tolerant so an async cell takes this path too.
        try:
            CodeAnalyzer.parse_cell(code)
            return code
        except SyntaxError:
            pass
        code = _cell_magic_body(code)
        starts = CodeAnalyzer._logical_line_start_flags(code)
        out: list[str] = []
        in_magic = False  # dropping a magic's backslash-continued lines
        for line, is_start in zip(code.split("\n"), starts):
            if in_magic and not is_start:
                continue
            in_magic = is_start and _is_magic_line(line)
            if in_magic:
                indent = line[: len(line) - len(line.lstrip())]
                if indent:
                    # An indented magic is the leading (often sole) statement of
                    # a block — ``if colab:``, ``for``, ``def`` … Deleting it
                    # outright can leave an EMPTY suite (``if colab:`` with
                    # nothing under it), a *fictional* SyntaxError that makes cash
                    # treat a perfectly good cell as unparseable and stop
                    # dependency-tracking it. Neutralise it in place with
                    # ``pass`` so the block keeps a body (a magic contributes no
                    # variable dependencies, so this loses nothing for analysis).
                    out.append(indent + "pass")
                # A top-level magic is dropped entirely: removing a module-level
                # statement never empties a block.
                continue
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def parse_cell(code: str) -> ast.Module:
        """Parse a notebook cell (or statement) to an AST, tolerating a
        top-level ``await``.

        Plain ``ast.parse`` raises ``SyntaxError`` on a module-level ``await``
        (it needs ``PyCF_ALLOW_TOP_LEVEL_AWAIT``), so a legitimate top-level-await
        cell used to be seen as a syntax error and silently skipped.
        This mirrors the flag IPython itself uses to compile async cells. A
        genuine syntax error still raises, so callers' error handling is
        unchanged.
        """
        return compile(
            code,
            "<cash-cell>",
            "exec",
            ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )

    @staticmethod
    def analyze_code_block(
        code: str,
        tree: ast.Module | None = None,
        resolve_source: Callable[[str], str | None] | None = None,
        user_ns: dict[str, Any] | None = None,
    ) -> tuple[set[str], set[str]]:
        """
        Analyze a block of code to find input (read) and output (written) variables.

        Args:
            code: The source code to analyze.
            tree: Optional pre-parsed AST to avoid redundant parsing.
            resolve_source: Optional ``name -> source`` for called functions.
                When supplied, a global that a CALLEE mutates in place is
                declared as an OUTPUT of the calling statement, so
                the global gets a producer and a lineage that advances.

                An output, and deliberately NOT also an input, even though the
                inline spelling ``state['n'] += 1`` does read and write. Adding
                the input edge makes sibling statements invalidate each other::

                    r1 = increment()   # writes counter, and (wrongly) reads it
                    r2 = increment()   # writes counter -> bumps its lineage
                                       # -> r1's input changed -> r1 re-executes

                Measured on that cell: the callee ran three times, ``counter``
                reached 3, and ``r1``/``r2`` disagreed about which came first.
                The re-run is scheduled by the upstream checker, not by the
                statement processor, so it does not show up as a re-executed
                statement -- only as a second trip through the call unit.

                An earlier attempt propagated the mutation into
                ``all_mutated_vars`` alone. That is enough to make the
                idempotent-rerun reset FIRE, and not enough to give it anything
                to reconstruct from: no producer recorded for the global. The
                reset then rewound to the seed, and a second cell sharing the
                global was served the first's values. The output edge is what
                fixes that; the input edge only breaks siblings.

        Returns:
            A tuple of (input_vars, output_vars).
        """
        if tree is None:
            clean_code = CodeAnalyzer.strip_magics(code)
            tree = CodeAnalyzer.parse_cell(clean_code)  # tolerate top-level await
        if "run_line_magic" in code:
            # What a `%time` line's Python reads and binds, in a loop body too.
            tree = magic_python(tree)

        visitor = _FlowVisitor()
        visitor.visit(tree)
        inputs, outputs = visitor.real_inputs, visitor.outputs
        if resolve_source is not None:
            # Raises when a callee cannot be analysed; the caller decides what
            # a statement it cannot see through means (``statement_effects``
            # runs it uncached).
            if user_ns is not None:
                handed = handed_callables(tree, user_ns)
                extra = callee_global_mutations(tree, resolve_source, extra_sources=handed.sources) | handed.receivers
                extra = capturable_globals(extra, user_ns) | notebook_global_rebinds(tree, resolve_source, user_ns)
            else:
                extra = callee_global_mutations(tree, resolve_source)
            if extra:
                outputs = outputs | set(extra)
        if user_ns is not None and outputs:
            # A statement that neither imports nor assigns a name cannot have
            # produced a MODULE. `sc.pp.calculate_qc_metrics(adata, inplace=True)`
            # read as mutating its receiver, which is rooted at `sc`: the badge
            # said "Produced sc" and every such line bumped the module's lineage,
            # so every statement reading `sc` missed. What the
            # call really changes, `adata`, is observed at runtime instead. Module
            # SETTINGS (`plt.rcParams.update(...)`) are routed separately.
            bound = {
                n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
            }
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bound.update((a.asname or a.name).split(".")[0] for a in node.names)
            outputs = {o for o in outputs if o in bound or not isinstance(user_ns.get(o), types.ModuleType)}
        return inputs, outputs

    @staticmethod
    def reassigned_names(code: str, tree: ast.Module | None = None) -> set[str]:
        """Return cell-level names that are PURELY reassigned with a fresh value.

        A name qualifies when it is bound via a plain ``name = ...`` (``Name``
        store) target — rebinding it to a new object — AND it is never mutated
        in place anywhere in the same cell. In-place mutation of an existing
        object (``name[k] = ...``, ``name.attr = ...``, ``name.method(inplace=
        True)``) binds the *same* object and is tracked as ``modified_objects``;
        a name that is both reassigned and mutated in the cell is therefore
        excluded.

        This isolates the self-referential *reassignment* chain
        (``df = df.sort(); ...; df = df.rename()``) — where a non-idempotent
        re-run on a stale value is unsafe and the input version must be restored
        — from in-place *mutation* cells (``df['c'] = ...``), which own a
        separate mutation-lineage restoration path and re-run safely.
        """
        if tree is None:
            clean_code = CodeAnalyzer.strip_magics(code)
            try:
                tree = ast.parse(clean_code)
            except SyntaxError:
                return set()
        visitor = _FlowVisitor()
        visitor.visit(tree)
        return set(visitor.defined_in_block) - set(visitor.modified_objects)

    @staticmethod
    def top_level_assigned_names(code: str, tree: ast.Module | None = None) -> set[str]:
        """Return names bound by an UNCONDITIONAL top-level assignment.

        Only ``Name`` targets of ``Assign``/``AnnAssign``/``AugAssign`` nodes
        that appear directly in ``tree.body`` qualify — i.e. assignments that
        run every time the statement executes. Assignments nested inside
        ``if``/``for``/``while``/``try``/``with`` are *conditional* (they may or
        may not run depending on a branch/loop/exception) and are excluded.

        This distinguishes a guaranteed initializer (``x = 'default'`` at the
        top level) from a conditional rebind (``if flag: x = 'overridden'``),
        which is what the conditional-producer-init scheduling needs.
        """
        if tree is None:
            clean_code = CodeAnalyzer.strip_magics(code)
            try:
                tree = ast.parse(clean_code)
            except SyntaxError:
                return set()
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for tgt in targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
        return names

    @staticmethod
    def scan_for_forbidden_functions(code: str, user_ns: dict[str, Any], tree: ast.Module | None = None) -> list[str]:
        """
        Scan code for calls to functions that should never be cached (e.g., time.time()).

        Args:
            code: The code to analyze
            user_ns: The current user namespace (to resolve existing variables)
            tree: Optional pre-parsed AST to avoid redundant parsing.

        Returns:
            List of detected forbidden function names (e.g. ['time.time'])
        """
        if tree is None:
            try:
                tree = ast.parse(CodeAnalyzer.strip_magics(code))
            except SyntaxError:
                return []

        visitor = _ForbiddenVisitor(user_ns)
        visitor.visit(tree)
        return list(set(visitor.found_reasons))

    @staticmethod
    def scan_function_bodies_for_forbidden_functions(code: str, user_ns: dict[str, Any]) -> list[str]:
        """:meth:`scan_for_forbidden_functions` of *code* as the functions it
        defines run when CALLED: their bodies too, nested functions and
        lambdas included.

        The statement scan skips a ``def``'s body on purpose -- defining a
        function reads no clock. A caller asking whether calling the
        functions *code* defines is a function of their inputs needs the
        bodies: ``def stamp(i): return time.time()`` reads the clock on every
        call. Imports in a body are followed as in a statement.
        """
        try:
            tree = ast.parse(CodeAnalyzer.strip_magics(code))
        except SyntaxError:
            return []
        visitor = _ForbiddenVisitor(user_ns)
        visitor.visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for statement in node.body:
                    visitor.visit(statement)
            elif isinstance(node, ast.Lambda):
                visitor.visit(node.body)
        return list(set(visitor.found_reasons))


#: Cell magics whose body runs as Python in the user's namespace through cash:
#: ``%%capture`` hands its body to ``run_cell``, and cash runs a ``%%time`` or
#: ``%%prun`` body itself, under the magic. Any other cell magic writes its body
#: to a file, hands it to another program, runs it in a scope of its own
#: (``%%timeit``) or under a debugger (``%%debug``): IPython runs it, and cash
#: reads none of it as Python.
_PYTHON_BODY_CELL_MAGICS = frozenset({"time", "capture", "prun"})

#: What IPython's input transform turns a magic or a shell command into: a
#: call of one of these on ``get_ipython()``.
_IPYTHON_RUNNERS = frozenset({"run_line_magic", "run_cell_magic", "getoutput", "system"})


def calls_ipython(tree: ast.AST) -> bool:
    """Whether *tree* runs a magic or a shell command, written as IPython's
    transform writes one (``get_ipython().run_line_magic('time', 'x = f()')``).

    What such a call reads and binds is IPython's business, not code cash
    can analyse.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        shell = node.func.value
        if (
            node.func.attr in _IPYTHON_RUNNERS
            and isinstance(shell, ast.Call)
            and isinstance(shell.func, ast.Name)
            and shell.func.id == "get_ipython"
        ):
            return True
    return False


#: Line magics whose argument is a Python statement run in the user's namespace.
PYTHON_ARG_MAGICS = frozenset({"time", "timeit", "prun"})


def python_magic_argument(call: ast.AST) -> str | None:
    """The argument of *call* when it is a ``%time``/``%timeit``/``%prun``
    line magic as IPython's transform writes it, else None."""
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_line_magic"
        and len(call.args) == 2
        and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in call.args)
        and call.args[0].value in PYTHON_ARG_MAGICS
    ):
        return call.args[1].value
    return None


def split_magic_argument(arg: str) -> tuple[list[str], ast.Module] | None:
    """``(options, statement)``: the words of *arg* before the Python
    statement it runs, and that statement; None when no tail is Python."""
    words = arg.split(" ")
    for start in range(len(words)):
        try:
            return words[:start], CodeAnalyzer.parse_cell(" ".join(words[start:]).strip())
        except SyntaxError:
            continue
    return None


def magic_python(tree: ast.Module) -> ast.Module:
    """*tree* with the Python statement of each ``%time``, ``%timeit`` and
    ``%prun`` line read in after the line: ``for i in r: %time
    acc.append(i * k)`` reads ``acc`` and ``k`` and changes ``acc``, as the
    plain loop does. *tree* itself is left as it is; without such a line it
    is what comes back."""
    if not any(python_magic_argument(node) is not None for node in ast.walk(tree)):
        return tree
    return _MagicPythonSplicer().visit(copy_tree(tree))


class _MagicPythonSplicer(ast.NodeTransformer):
    """See :func:`magic_python`."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                spliced: list[ast.stmt] = []
                for stmt in value:
                    spliced.append(self.visit(stmt))
                    if not isinstance(stmt, _COMPOUND_STATEMENTS):
                        spliced.extend(_magic_statements(stmt))
                setattr(node, field, spliced)
            elif isinstance(value, ast.AST):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
        return node


_COMPOUND_STATEMENTS = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def _magic_statements(stmt: ast.stmt) -> list[ast.stmt]:
    """The Python statements the line magics in the simple statement *stmt* run."""
    out: list[ast.stmt] = []
    for node in ast.walk(stmt):
        arg = python_magic_argument(node)
        split = None if arg is None else split_magic_argument(arg)
        if split is not None and not calls_ipython(split[1]):
            out.extend(split[1].body)
    return out


#: A line that assigns the result of a magic or a shell command, as IPython's
#: ``MagicAssign`` and ``SystemAssign`` transforms read it.
_TARGET = r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*|\[[^\]\n]*\])*"
_MAGIC_ASSIGN = re.compile(rf"\s*\(?{_TARGET}(\s*,\s*{_TARGET})*\s*,?\s*\)?\s*=\s*[%!]")

#: A help request, as IPython's ``help_end`` and ``EscapedCommand`` transforms
#: read it: ``obj?``, ``obj??``, ``?obj``, ``??obj`` (a wildcard ``np.*load*?`` too).
_HELP = re.compile(r"\s*(\?{1,2}\s*[\w.*%]+|[\w.*%]+\?{1,2})\s*(#.*)?$")


def _is_magic_line(line: str) -> bool:
    """Whether *line*, beginning a logical line, is IPython syntax rather than Python."""
    return (
        line.strip().startswith(("%", "!", "?"))
        or _MAGIC_ASSIGN.match(line) is not None
        or _HELP.match(line) is not None
    )


def _cell_magic_body(code: str) -> str:
    """*code* without its cell magic: the body when IPython runs it as Python
    in the user's namespace, else nothing. *code* unchanged when it is not a
    cell magic (IPython reads ``%%`` as one only on the cell's first line)."""
    lines = code.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or not lines[first].startswith("%%"):
        return code
    name = lines[first][2:].split(maxsplit=1)[0] if lines[first][2:].strip() else ""
    if name not in _PYTHON_BODY_CELL_MAGICS:
        return ""
    return "\n".join(lines[first + 1 :])


@functools.lru_cache(maxsize=1024)
def clean_cell_source(cell_code: str) -> str:
    """*cell_code* as the upstream simulation reads it: ``\r\n`` normalised,
    magics stripped (:meth:`CodeAnalyzer.strip_magics`)."""
    return CodeAnalyzer.strip_magics(cell_code.replace("\r\n", "\n"))


def parse_cell_source(cell_code: str) -> ast.Module | None:
    """:func:`clean_cell_source` parsed, or None when it does not parse.

    Both steps are memoised, so the steps of one upstream check share a single
    parse of each cell: read the tree, never change it.
    """
    return parse_cached(clean_cell_source(cell_code))


# Characters ``str.splitlines()`` treats as line breaks that the CPython
# parser does not -- the parser (and therefore ``node.lineno`` /
# ``node.end_lineno``) recognizes only "\r\n", "\r" and "\n". Built with
# ``chr()`` rather than escape literals so the exact code points stay
# unambiguous on the page: vertical tab, form feed, FILE/GROUP/RECORD
# SEPARATOR (U+001C-U+001E), NEL (U+0085), LINE SEPARATOR (U+2028) and
# PARAGRAPH SEPARATOR (U+2029). See ``splitlines_like_the_parser`` below.
_PARSER_INCOMPATIBLE_LINEBREAKS = "".join(chr(c) for c in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029))
_LINEBREAK_MASK = str.maketrans(_PARSER_INCOMPATIBLE_LINEBREAKS, " " * len(_PARSER_INCOMPATIBLE_LINEBREAKS))


def splitlines_like_the_parser(raw_cell: str) -> list[str]:
    """Split ``raw_cell`` into lines using the parser's line-ending rules,
    not ``str.splitlines()``'s.

    ``str.splitlines(keepends=True)`` breaks on more characters than the
    CPython tokenizer does -- see ``_PARSER_INCOMPATIBLE_LINEBREAKS`` above
    -- so indexing its result by ``node.lineno`` desyncs the moment any of
    those appear anywhere earlier in the cell (observed: a vertical tab
    inside one string literal silently truncated that statement's display;
    a form feed at the end of one line corrupted the display of the NEXT
    statement). Fixed here by masking those characters to a plain space --
    one-for-one, so every position keeps its original index -- before
    calling ``str.splitlines()``, then slicing the boundaries it finds back
    out of the ORIGINAL (unmasked) ``raw_cell``. The returned lines contain
    the real original characters verbatim; only the *decision of where a
    line ends* used the masked copy.

    Both ``str.translate`` and ``str.splitlines`` are single C-level passes
    over the whole string, so this stays cheap enough for the fast path
    of ``statement_source.statement_source`` to keep its measured win over
    always calling ``ast.get_source_segment`` (see that docstring for the
    numbers).
    """
    masked = raw_cell.translate(_LINEBREAK_MASK)
    lines = []
    pos = 0
    for masked_line in masked.splitlines(keepends=True):
        length = len(masked_line)
        lines.append(raw_cell[pos : pos + length])
        pos += length
    return lines


def expr_has_trailing_semicolon(raw_cell: str, node: ast.stmt) -> bool:
    """True if expression statement *node* is followed by a ``;`` in the raw
    source (IPython display suppression). ``ast.unparse`` discards it, so we
    recover it from the original cell text.

    Both coordinates must be read the way the PARSER wrote them, or this
    silently answers ``False`` and echoes a repr the user suppressed:

    * the line index needs the parser's line-break rules, not
      ``str.splitlines()``'s wider set -- see
      ``splitlines_like_the_parser``. One vertical tab or form feed
      anywhere earlier in the cell shifts every later index.
    * ``end_col_offset`` is a UTF-8 *byte* offset, not a character index,
      so the line is sliced as bytes. Reading it as characters slid the
      slice past the ``;`` whenever anything non-ASCII sat earlier on the
      same line (``df[df.city == "Zürich"];``).

    Lines here keep their endings, so the remainder of the cell appends
    verbatim rather than being rebuilt with ``"\\n".join``.
    """
    if not isinstance(node, ast.Expr):
        return False
    end_line = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if end_line is None or end_col is None:
        return False
    lines = splitlines_like_the_parser(raw_cell)
    if end_line > len(lines):
        return False
    rest = lines[end_line - 1].encode()[end_col:].decode()
    rest += "".join(lines[end_line:])
    return rest.lstrip().startswith(";")


def statement_code(node: ast.stmt, raw_cell: str | None = None) -> str:
    """The text a top-level statement is keyed and run as: ``ast.unparse``,
    plus the ``;`` that followed an expression in *raw_cell*, the text *node*
    was parsed from.

    IPython reads that ``;`` as "show no repr", and ``ast.unparse`` drops it,
    so it is put back: a cached re-run would show a repr the user suppressed,
    and the upstream simulation, keying the statement without it, would
    disagree with the runtime about its key. The runtime and the simulation
    both key a statement through here. Raises what ``ast.unparse`` raises.
    """
    code = ast.unparse(node)
    if raw_cell is not None and expr_has_trailing_semicolon(raw_cell, node):
        code += ";"
    return code
