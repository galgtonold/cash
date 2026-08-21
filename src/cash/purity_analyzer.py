"""Decorator-side purity analyzer.

Walks the body of a ``@cash.cache``-decorated function plus any
*module-bounded* helpers it calls, and reports:

* **Known-impure calls** - ``requests.post``, ``os.system``,
  ``logging.info``, file-write methods (``to_csv``, ``write``, ...).
* **Explicit dynamism** - ``eval``/``exec``/``compile``,
  ``getattr(obj, name)(...)`` where *name* is not a constant,
  calling a *parameter* directly (``def f(cb): cb(x)``).
* **Discarded-return calls** - ``f(x)`` as a statement (return
  value thrown away). Strong syntactic signal of "I'm calling this
  for the side effect". Skipped for callees in
  :data:`KNOWN_PURE_BUILTINS` (where discarding is just dead code).
* **Scope mutations** - ``global``/``nonlocal``, assignment to
  ``Attribute`` or ``Subscript`` targets, augmented-assign to same.

As a side benefit, the same walk captures source hashes of every
analyzed user-code helper. Those hashes become part of the cache
key, so editing a helper invalidates the parent cache automatically
on the next run.

**Boundary rule** - recursion follows callees that are *user code*:
the callable's source file is outside stdlib / site-packages
(``is_local_module``) **or** the callable shares the cached
function's top-level package. Everything else is treated as opaque
and optimistic (not flagged). Users who want a library call flagged
add ``cash.mark_stateful(library_func)``.

The analyzer is pure: no side effects of its own, no warnings
emitted from here. The decorator layer turns the report into
warnings / exceptions / cache-key components.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import logging
import sys
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import SOURCE_RETRIEVAL_ERRORS
from .notebook.cacheability import (
    PANDAS_INPLACE_METHODS,
    _get_base_name,
    _get_call_module,
    _get_call_name,
    _is_open_write_mode,
)
from .notebook.function_tracker import is_local_module
from .notebook.purity import (
    _IMPURE_FUNCTION_CALLS,
    _IMPURE_MODULE_CALLS,
    _WRITE_METHODS,
    KNOWN_PURE_BUILTINS,
    is_pure,
    is_stateful,
)
from .source_norm import bytecode_identity, normalize_source_for_hash

logger = logging.getLogger(__name__)

__all__ = [
    "PurityAnalyzer",
    "PurityReport",
    "PurityIssue",
    "ISSUE_IMPURE_CALL",
    "ISSUE_DYNAMIC_PATTERN",
    "ISSUE_UNTRACKABLE_DEP",
    "ISSUE_DISCARDED_CALL",
    "ISSUE_SCOPE_MUTATION",
    "ISSUE_MUTABLE_GLOBAL",
]

ISSUE_IMPURE_CALL = "impure_call"
ISSUE_DYNAMIC_PATTERN = "dynamic_pattern"
# Patterns where a dependency is resolved from a runtime value, so cash cannot
# see an edit to it and a cached result can go silently stale: eval/exec/compile,
# getattr(obj, name)() dynamic dispatch, importlib.import_module. Caching
# correctness cannot be guaranteed, so these RAISE by default (opt in with
# assume_safe=True). Distinct from ISSUE_DYNAMIC_PATTERN, which stays advisory.
ISSUE_UNTRACKABLE_DEP = "untrackable_dep"
ISSUE_DISCARDED_CALL = "discarded_call"
ISSUE_SCOPE_MUTATION = "scope_mutation"
ISSUE_MUTABLE_GLOBAL = "mutable_global"


@dataclass(frozen=True)
class PurityIssue:
    """A single issue surfaced by :class:`PurityAnalyzer`.

    Attributes:
        kind: One of ``impure_call``, ``dynamic_pattern``,
            ``discarded_call``, ``scope_mutation``.
        description: Human-readable summary (e.g. ``"requests.post()"``,
            ``"global X"``, ``"discards return of helper(...)``).
        where: Qualified name + line of the function containing the
            issue. For helpers, this is the helper's qualname so the
            user can fix the source of the problem, not just the
            outermost decorator.
        line: Source line of the issue in its containing function.
    """

    kind: str
    description: str
    where: str
    line: int = 0


@dataclass(frozen=True)
class PurityReport:
    """Result of analyzing a callable + its module-bounded helpers.

    Attributes:
        issues: All findings, in stable order (kind, line).
        helper_source_hashes: ``qualname -> digest`` for every user-code
            helper actually walked, captured at analysis time. Used as the
            fallback when per-call re-resolution fails (helper was
            deleted/renamed since analysis). The digest is over NORMALIZED
            source, so a comment or reformat in a helper does not
            invalidate its callers; for a helper with no readable source
            it is ``bytecode_identity``, matching what the per-call rehash
            computes for the same object.
        helper_resolution_paths: ``qualname -> (module_name, attr_chain)``
            for every walked helper. Used by the decorator to
            re-resolve the helper from ``sys.modules`` on each
            call and re-hash its source, so in-process
            redefinitions (notebook cells, REPL) invalidate the
            parent's cache key. ``attr_chain`` is the
            dot-separated ``__qualname__`` of the helper inside
            its module (e.g. ``("Klass", "method")`` for a method).
        opaque_callees: Qualified names of callees we encountered
            but couldn't read source for (C extensions, missing source,
            partial application). Treated as pure by default; strict
            mode promotes their presence to an issue.

            Opaque for PURITY only. One that still has a ``__code__``
            also gets an entry in ``helper_source_hashes`` and
            ``helper_resolution_paths``, digested from its compiled form,
            so editing it invalidates its callers. It used to be dropped
            from both, which made every edit to such a helper invisible
            to the cache key -- a silently stale result, not a recompute.
    """

    issues: tuple[PurityIssue, ...] = ()
    helper_source_hashes: dict[str, str] = field(default_factory=dict)
    helper_resolution_paths: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    opaque_callees: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """True when no issues were flagged."""
        return not self.issues

    def format(self) -> str:
        """Human-readable multi-line summary."""
        if self.is_clean:
            return "(no purity issues detected)"
        by_where: dict[str, list[PurityIssue]] = {}
        for issue in self.issues:
            by_where.setdefault(issue.where, []).append(issue)
        lines = []
        for where, issues in by_where.items():
            lines.append(f"  in {where}:")
            for i in issues:
                lines.append(f"    line {i.line}: [{i.kind}] {i.description}")
        return "\n".join(lines)


# Names of constructors/methods that return a *freshly-allocated* mutable
# object. A local bound only to one of these (or to a mutable literal /
# comprehension) cannot alias caller-visible state, so mutating it in place
# is pure - see ``_compute_local_owned``.
_FRESH_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({
    "list", "dict", "set", "bytearray",
    "defaultdict", "OrderedDict", "Counter", "deque",
})
_FRESH_CONSTRUCTOR_ATTRS: frozenset[str] = frozenset({
    # numpy fresh-array factories
    "zeros", "empty", "ones", "full", "array", "asarray",
    "zeros_like", "empty_like", "ones_like", "full_like",
    "arange", "linspace",
    # pandas
    "DataFrame", "Series",
    # generic "make me a fresh copy"
    "copy", "deepcopy", "fromkeys",
})

# Mutable-literal AST nodes (a fresh container by construction).
_FRESH_LITERAL_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)


def _is_fresh_alloc(node: ast.AST | None) -> bool:
    """True when *node* evaluates to a brand-new mutable object.

    Conservative: a bare name, attribute, subscript, or any expression we
    don't recognise as a fresh allocation returns False (so the binding is
    treated as a possible alias and mutations to it stay flagged).
    """
    if node is None:
        return False
    if isinstance(node, _FRESH_LITERAL_NODES):
        return True
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in _FRESH_CONSTRUCTOR_NAMES:
            return True
        if isinstance(f, ast.Attribute) and f.attr in _FRESH_CONSTRUCTOR_ATTRS:
            return True
    return False


def _iter_scope_statements(func_def: ast.AST):
    """Yield nodes in *func_def*'s own scope, NOT descending into nested
    function/lambda bodies (separate scopes with their own locals)."""
    for child in ast.iter_child_nodes(func_def):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _iter_scope_statements(child)


def _compute_local_owned(func_def: ast.AST, param_names: frozenset[str]) -> frozenset[str]:
    """Return names that are *fresh locals* of *func_def*: assigned only to
    freshly-allocated objects, never declared ``global``/``nonlocal``, and not
    parameters. Mutating such a name in place (``x[i] = ...``, ``x.append(...)``)
    is pure - it can't reach caller-visible state, and returning it just hands
    the caller a new object. This is the escape analysis that distinguishes a
    local accumulator from a mutation of shared state.
    """
    escaped: set[str] = set()          # names declared global/nonlocal
    bound: dict[str, bool] = {}        # name -> every binding so far is fresh

    for node in _iter_scope_statements(func_def):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            escaped.update(node.names)
        elif isinstance(node, ast.Assign):
            fresh = _is_fresh_alloc(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    bound[tgt.id] = bound.get(tgt.id, True) and fresh
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                fresh = _is_fresh_alloc(node.value)
                bound[node.target.id] = bound.get(node.target.id, True) and fresh

    return frozenset(
        name for name, all_fresh in bound.items()
        if all_fresh and name not in param_names and name not in escaped
    )


class _PurityVisitor(ast.NodeVisitor):
    """Single-function-body visitor that collects :class:`PurityIssue`s.

    Does NOT recurse into callees - the analyzer handles that. Just
    flags everything visible in this one body and records called
    names so the analyzer can resolve and recurse.
    """

    __slots__ = (
        "issues", "called_callable_nodes", "_param_names",
        "_qualname", "_line_offset", "_local_owned", "read_names",
        "_assign_kinds", "_name_call_nodes",
    )

    def __init__(self, qualname: str, param_names: frozenset[str],
                 line_offset: int = 0,
                 local_owned: frozenset[str] = frozenset()) -> None:
        self.issues: list[PurityIssue] = []
        self.called_callable_nodes: list[ast.AST] = []
        # Bare names read (Load context) in this body - used to detect reads of
        # mutable module globals.
        self.read_names: set[str] = set()
        # For each simple ``name = ...`` target, the kinds of RHS it was ever
        # assigned ({"dynamic"} / {"other"} / both). A name assigned ONLY from a
        # dynamic source (getattr(obj,name), eval, importlib) and then CALLED is
        # untrackable dispatch obscured by a local; requiring "dynamic only"
        # keeps a name later reassigned to a safe value from false-positiving.
        self._assign_kinds: dict[str, set[str]] = {}
        # Calls whose function is a bare Name, checked against the taint set in
        # :meth:`finalize_taint` after the whole body is walked.
        self._name_call_nodes: list[ast.Call] = []
        self._param_names = param_names
        self._qualname = qualname
        # When source comes from inspect.getsource on a method, line
        # numbers in the parsed AST are 1-based relative to the
        # dedented source, not the original file. We just report them
        # as-is - the qualname tells the user where to look.
        self._line_offset = line_offset
        # Fresh locals: in-place mutation of these is pure (escape analysis).
        self._local_owned = local_owned

    # --- impure / dynamic / called-name detection on Call nodes ---

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.read_names.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self._name_call_nodes.append(node)
        self._record_call(node)
        self.generic_visit(node)

    @staticmethod
    def _is_dynamic_source(value: ast.AST) -> bool:
        """True when *value* resolves a callable from a runtime value.

        ``eval``/``exec``/``compile`` bound by name; ``getattr(obj, name)`` with
        a non-constant name; ``importlib.import_module(...)`` / ``__import__``.
        Assigning one of these to a local and calling it is untrackable dispatch.
        """
        if isinstance(value, ast.Name) and value.id in {"eval", "exec", "compile"}:
            return True
        if isinstance(value, ast.Call):
            f = value.func
            if (isinstance(f, ast.Name) and f.id == "getattr" and len(value.args) >= 2
                    and not (isinstance(value.args[1], ast.Constant)
                             and isinstance(value.args[1].value, str))):
                return True
            if isinstance(f, ast.Attribute) and f.attr == "import_module":
                return True
            if isinstance(f, ast.Name) and f.id == "__import__":
                return True
        return False

    def finalize_taint(self) -> None:
        """Flag calling a local that was bound ONLY from a dynamic source.

        Run once after the whole body is walked (assignments may follow or
        precede the call in source order). A name is untrackable only if every
        assignment to it was dynamic -- so ``f = getattr(o,n); f()`` and
        ``ev = eval; ev(x)`` flag, but ``f = getattr(...); f = helper; f()``
        does not (it was rebound to a tracked value).
        """
        tainted = {n for n, kinds in self._assign_kinds.items() if kinds == {"dynamic"}}
        for node in self._name_call_nodes:
            name = node.func.id  # type: ignore[attr-defined]
            if name in tainted:
                self.issues.append(PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=(
                        f"calls {name!r}, which was bound from a runtime value "
                        "(dynamic dispatch through a local)"
                    ),
                    where=self._qualname,
                    line=getattr(node, "lineno", 0),
                ))

    def _record_call(self, node: ast.Call) -> None:
        func_node = node.func
        line = getattr(node, "lineno", 0)

        # Explicit dynamism: eval / exec / compile by bare name.
        if isinstance(func_node, ast.Name) and func_node.id in {"eval", "exec", "compile"}:
            self.issues.append(PurityIssue(
                kind=ISSUE_UNTRACKABLE_DEP,
                description=f"{func_node.id}(...) - explicit dynamic execution",
                where=self._qualname,
                line=line,
            ))
            return

        # getattr(obj, name)(...) where name is not a constant string.
        if (
            isinstance(func_node, ast.Call)
            and isinstance(func_node.func, ast.Name)
            and func_node.func.id == "getattr"
            and len(func_node.args) >= 2
            and not (isinstance(func_node.args[1], ast.Constant) and isinstance(func_node.args[1].value, str))
        ):
            self.issues.append(PurityIssue(
                kind=ISSUE_UNTRACKABLE_DEP,
                description="getattr(obj, name)(...) with non-constant name - dynamic dispatch",
                where=self._qualname,
                line=line,
            ))
            return

        # Dynamic import: importlib.import_module(...) / __import__(...). The
        # imported module's members are resolved from a runtime value, so an
        # edit to that module is invisible to the cache key.
        _is_import_module = (
            isinstance(func_node, ast.Attribute) and func_node.attr == "import_module"
        )
        _is_dunder_import = isinstance(func_node, ast.Name) and func_node.id == "__import__"
        if _is_import_module or _is_dunder_import:
            self.issues.append(PurityIssue(
                kind=ISSUE_UNTRACKABLE_DEP,
                description=(
                    f"{'importlib.import_module' if _is_import_module else '__import__'}"
                    "(...) - dynamic import; the imported module's code is not tracked"
                ),
                where=self._qualname,
                line=line,
            ))
            return

        # Calling a parameter: def f(cb): cb(x).
        if isinstance(func_node, ast.Name) and func_node.id in self._param_names:
            self.issues.append(PurityIssue(
                kind=ISSUE_DYNAMIC_PATTERN,
                description=f"calls parameter {func_node.id!r} as a function",
                where=self._qualname,
                line=line,
            ))
            return

        # Known-impure module-qualified calls (requests.post, os.system, ...).
        func_name = _get_call_name(func_node)
        module_name = _get_call_module(func_node)
        if func_name:
            dotted = f"{module_name}.{func_name}" if module_name else func_name
            if dotted in _IMPURE_MODULE_CALLS or func_name in _IMPURE_FUNCTION_CALLS:
                # Special case: open() in read mode is not impure.
                if func_name == "open" and not module_name and not _is_open_write_mode(node):
                    pass  # open(path) for read - track via file-deps, not as impurity
                else:
                    self.issues.append(PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"{dotted}() - known I/O / side-effecting",
                        where=self._qualname,
                        line=line,
                    ))
                    return

            # Method calls in _WRITE_METHODS (to_csv, write, savefig, ...).
            # Skipped when the receiver is a fresh local (e.g. ``lines.append``
            # where ``lines = []``): mutating a local accumulator is pure.
            if (
                isinstance(func_node, ast.Attribute)
                and func_node.attr in _WRITE_METHODS
                and not self._receiver_is_local_owned(func_node.value)
            ):
                base = _get_base_name(func_node.value)
                base_str = f"{base}." if base else ""
                self.issues.append(PurityIssue(
                    kind=ISSUE_IMPURE_CALL,
                    description=f"{base_str}{func_node.attr}() - write method",
                    where=self._qualname,
                    line=line,
                ))
                return

            # Pandas inplace=True kwarg - mutates the receiver.
            if (
                isinstance(func_node, ast.Attribute)
                and func_node.attr in PANDAS_INPLACE_METHODS
                and not self._receiver_is_local_owned(func_node.value)
            ):
                for kw in node.keywords:
                    if (
                        kw.arg == "inplace"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        base = _get_base_name(func_node.value)
                        base_str = f"{base}." if base else ""
                        self.issues.append(PurityIssue(
                            kind=ISSUE_IMPURE_CALL,
                            description=f"{base_str}{func_node.attr}(inplace=True) - in-place mutation",
                            where=self._qualname,
                            line=line,
                        ))
                        return

        # Not flagged as anything - record for recursion attempt.
        self.called_callable_nodes.append(node)

    # --- discarded-call detection on Expr statements ---

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call):
            call = node.value
            func_node = call.func
            # Plain bare-name call: foo(x)
            if isinstance(func_node, ast.Name):
                name = func_node.id
                if name not in KNOWN_PURE_BUILTINS and name not in {"eval", "exec", "compile"}:
                    self.issues.append(PurityIssue(
                        kind=ISSUE_DISCARDED_CALL,
                        description=f"discards return of {name}(...)",
                        where=self._qualname,
                        line=getattr(node, "lineno", 0),
                    ))
            elif isinstance(func_node, ast.Attribute):
                method = func_node.attr
                # If the method is already flagged elsewhere (write-methods,
                # impure module calls), it's recorded by _record_call. The
                # discarded-return flag here adds nothing useful - skip to
                # avoid double-counting. Skip known-pure idioms too.
                if (
                    method not in _WRITE_METHODS
                    and method not in PANDAS_INPLACE_METHODS
                ):
                    base = _get_base_name(func_node.value)
                    base_str = f"{base}." if base else ""
                    self.issues.append(PurityIssue(
                        kind=ISSUE_DISCARDED_CALL,
                        description=f"discards return of {base_str}{method}(...)",
                        where=self._qualname,
                        line=getattr(node, "lineno", 0),
                    ))
        self.generic_visit(node)

    # --- scope mutation detection ---

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        for name in node.names:
            self.issues.append(PurityIssue(
                kind=ISSUE_SCOPE_MUTATION,
                description=f"global {name} - reads/writes module-level state",
                where=self._qualname,
                line=getattr(node, "lineno", 0),
            ))
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        for name in node.names:
            self.issues.append(PurityIssue(
                kind=ISSUE_SCOPE_MUTATION,
                description=f"nonlocal {name} - writes enclosing-scope state",
                where=self._qualname,
                line=getattr(node, "lineno", 0),
            ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._maybe_flag_mutation_target(target, node.lineno)
        # Record the RHS kind for a simple ``name = ...`` so a dynamically-bound
        # local that is later called can be flagged (see finalize_taint).
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            kind = "dynamic" if self._is_dynamic_source(node.value) else "other"
            self._assign_kinds.setdefault(node.targets[0].id, set()).add(kind)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._maybe_flag_mutation_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._maybe_flag_mutation_target(target, node.lineno)
        self.generic_visit(node)

    def _receiver_is_local_owned(self, value: ast.AST) -> bool:
        """True when *value* is a bare name that is a fresh local of this
        function (escape analysis) - mutating it in place is pure."""
        return isinstance(value, ast.Name) and value.id in self._local_owned

    def _maybe_flag_mutation_target(self, target: ast.AST, line: int) -> None:
        # Mutating a fresh local (``pos[i] = ...`` where ``pos = np.zeros(n)``)
        # is pure: the object can't reach caller-visible state and returning it
        # just hands the caller a new object. Skip those.
        if isinstance(target, (ast.Attribute, ast.Subscript)) and \
                self._receiver_is_local_owned(target.value):
            return
        if isinstance(target, ast.Attribute):
            base = _get_base_name(target.value)
            base_str = f"{base}." if base else ""
            self.issues.append(PurityIssue(
                kind=ISSUE_SCOPE_MUTATION,
                description=f"{base_str}{target.attr} = ... - attribute mutation",
                where=self._qualname,
                line=line,
            ))
        elif isinstance(target, ast.Subscript):
            base = _get_base_name(target.value)
            base_str = f"{base}[...]" if base else "[...]"
            self.issues.append(PurityIssue(
                kind=ISSUE_SCOPE_MUTATION,
                description=f"{base_str} = ... - subscript mutation",
                where=self._qualname,
                line=line,
            ))


def _is_user_code(callee: Any, root_module: str | None) -> bool:
    """Decide whether to recurse into *callee* during purity analysis.

    The boundary rule from the design discussion: user code is
    anything that (a) shares the cached function's top-level package
    OR (b) lives outside stdlib/site-packages.

    Args:
        callee: Resolved callable from the cached function's globals.
        root_module: ``__module__`` of the cached function. Used for
            the same-top-level-package shortcut.

    Returns:
        True when the analyzer should attempt to read source and
        recurse into *callee*. False for library code we trust
        unless explicitly marked stateful.
    """
    module = inspect.getmodule(callee)
    if module is None:
        return False

    # Top-level package shortcut - handles common case fast and
    # works for editable installs (both pieces share the same
    # top-level package by construction).
    callee_mod = getattr(module, "__name__", "") or ""
    if root_module:
        root_top = root_module.split(".", 1)[0]
        callee_top = callee_mod.split(".", 1)[0]
        if root_top and callee_top and root_top == callee_top:
            return True

    # Fall back to the file-path-based check used by the notebook
    # subsystem. Catches editable installs that DON'T share the
    # cached function's package (e.g. user's project depends on a
    # locally-developed sibling lib also installed `-e`).
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def _resolve_callee(node: ast.AST, namespace: dict[str, Any]) -> Any | None:
    """Resolve a ``Call`` node's function expression to a callable.

    Handles bare names (``foo``) and dotted attribute chains
    (``mod.sub.func``) by walking *namespace* - a merged view of
    the caller's ``__globals__`` and any closure cells produced by
    :func:`_build_namespace`. Returns ``None`` for any expression
    we can't reduce statically (subscript, higher-order calls,
    conditionals).
    """
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        parts: list[str] = [node.attr]
        cur: ast.AST = node.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        obj: Any = namespace.get(cur.id)
        if obj is None:
            return None
        for attr in reversed(parts):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    return None


def _called_names_in_tree(tree: ast.AST) -> list[str]:
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


def _resolve_in_class_namespaces(cls: type, name: str) -> Any:
    """First binding of *name* across *cls*'s candidate namespaces, else None."""
    for namespace in _class_namespaces(cls):
        if name in namespace:
            return namespace[name]
    return None


def _build_namespace(func: Callable[..., Any]) -> dict[str, Any]:
    """Return a merged ``__globals__`` + closure-cell namespace for *func*.

    Lets :func:`_resolve_callee` see helpers defined as closures
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


class PurityAnalyzer:
    """Walks a callable's body + its module-bounded helpers and
    returns a :class:`PurityReport`.

    Results are cached by the analyzed callable's source hash.
    Multiple :class:`Cash` instances share a single process-wide
    analyzer via :func:`get_analyzer`.
    """

    _MAX_DEPTH = 6  # Defensive depth cap. Real call hierarchies
    # rarely exceed 3-4 levels before hitting library code; the cap
    # bounds pathological cases (recursive helpers that resolve
    # through different code paths).

    _CACHE_SIZE_LIMIT = 500

    def __init__(self) -> None:
        self._cache: dict[str, PurityReport] = {}
        self._cache_lock = threading.Lock()

    def analyze(self, func: Callable[..., Any]) -> PurityReport:
        """Return a :class:`PurityReport` for *func*.

        Idempotent and cached by source hash. Explicit ``@pure``
        annotation short-circuits to an empty report; ``@stateful``
        short-circuits to a report flagging the function itself.
        """
        # Explicit annotations short-circuit (the user has spoken).
        if is_pure(func):
            return PurityReport()
        if is_stateful(func):
            qualname = _qualname_of(func)
            return PurityReport(
                issues=(PurityIssue(
                    kind=ISSUE_IMPURE_CALL,
                    description="explicitly marked @stateful",
                    where=qualname,
                    line=0,
                ),),
            )

        source_hash = _try_source_hash(func)
        if source_hash is not None:
            with self._cache_lock:
                cached = self._cache.get(source_hash)
                if cached is not None:
                    return cached

        report = self._analyze_uncached(func)

        if source_hash is not None:
            with self._cache_lock:
                if len(self._cache) >= self._CACHE_SIZE_LIMIT:
                    # Drop the oldest entry; insertion-order dict.
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                self._cache[source_hash] = report
        return report

    def _analyze_uncached(self, root_func: Callable[..., Any]) -> PurityReport:
        root_module = getattr(root_func, "__module__", None)

        all_issues: list[PurityIssue] = []
        helper_hashes: dict[str, str] = {}
        helper_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        opaque: list[str] = []
        visited: set[str] = set()

        def _record_resolution_path(func: Callable[..., Any], qualname: str) -> None:
            """Note where to re-resolve *func* from ``sys.modules`` per call.

            Lets the decorator pick up in-process redefinitions (notebook
            cells, REPL). The attr_chain is the helper's ``__qualname__``
            split on '.' so methods like ``Klass.method`` resolve. The root
            function is skipped -- it is not a "helper" and the decorator
            holds its own reference.
            """
            helper_module = getattr(func, "__module__", None)
            helper_inner_qualname = getattr(func, "__qualname__", None)
            if (
                func is not root_func
                and helper_module
                and helper_inner_qualname
                and "<locals>" not in helper_inner_qualname
            ):
                helper_paths[qualname] = (
                    helper_module,
                    tuple(helper_inner_qualname.split(".")),
                )

        # The third element is HASH_ONLY: fold this callable's code into the
        # cache key, but do not analyze it for purity. Set for classes reached
        # from another class's body -- they are followed for correctness, not
        # audited, and analyzing them reports every ``self.x = x`` in an
        # ordinary __init__ as a scope mutation.
        def _queue_class_refs(cls: Any, tree: ast.AST, depth: int) -> None:
            """Queue user-code objects a CLASS body constructs, hash-only."""
            if not isinstance(cls, type) or depth >= self._MAX_DEPTH:
                return
            for called in _called_names_in_tree(tree):
                target = _resolve_in_class_namespaces(cls, called)
                if target is None or target is cls or not callable(target):
                    continue
                if getattr(target, "_cash_cached", False):
                    continue
                if is_pure(target) or is_stateful(target):
                    continue
                if not _is_user_code(target, root_module):
                    continue
                stack.append((target, depth + 1, True))

        stack: list[tuple[Callable[..., Any], int, bool]] = [(root_func, 0, False)]
        while stack:
            func, depth, hash_only = stack.pop()
            qualname = _qualname_of(func)
            if qualname in visited:
                continue
            visited.add(qualname)

            # Read source. Failure -> opaque leaf for PURITY: we cannot see
            # what it does, so we decline to judge it.
            #
            # It must NOT become invisible to the CACHE KEY as well, which is
            # what dropping it outright used to do. A helper whose source
            # cannot be read contributed nothing to its callers' state hash,
            # so ANY edit to it went unnoticed -- not merely a constant.
            # Measured with an exec-defined helper under a filename absent
            # from linecache: replacing its entire body still served the
            # stale result. Fall back to the compiled identity, which is
            # exactly what ``Cash._hash_callable_source`` recomputes live for
            # the same object, so the snapshot and the per-call value agree
            # instead of disagreeing forever.
            try:
                src = inspect.getsource(func)
            except SOURCE_RETRIEVAL_ERRORS:
                opaque.append(qualname)
                digest = bytecode_identity(func)
                if digest is not None:
                    helper_hashes[qualname] = digest
                    _record_resolution_path(func, qualname)
                continue
            src = textwrap.dedent(src)

            # Hash the NORMALIZED source for cache-key invalidation of
            # helpers. Root function's hash is captured separately by the
            # decorator via _hash_callable_source - we record all walked
            # callables here so the decorator can fold them into the state
            # hash uniformly. Normalizing means a comment or reformat in a
            # helper no longer invalidates its callers, which was the more
            # surprising half of the old behaviour: users expect editing a
            # function to recompute it, not editing something it calls.
            helper_hashes[qualname] = hashlib.sha256(
                normalize_source_for_hash(src).encode("utf-8")
            ).hexdigest()

            _record_resolution_path(func, qualname)

            try:
                tree = ast.parse(src)
            except SyntaxError:
                opaque.append(qualname)
                continue

            if hash_only:
                # Followed for the cache key, not audited: reporting every
                # ``self.x = x`` in an ordinary __init__ as a scope mutation
                # would bury the real findings. Keep walking what IT builds,
                # so the closure stays transitive.
                opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            func_def = _find_first_function_def(tree)
            if func_def is None:
                # A CLASS, not a function: a ClassDef has no FunctionDef at
                # the top, so there is no body to analyze for purity. Bailing
                # here also ended the WALK, which lost the code the class
                # reaches. A dataclass field like
                # ``field(default_factory=lambda: B(0))`` constructs B, so
                # editing B changes what every instance holds -- measured, the
                # cache returned ``A(value=B(value=10))`` where a fresh call
                # produced ``A(value=B(value=1000))``. A wrong answer, not a
                # stale one.
                #
                # Queue what the class body CALLS. Names only, from actual
                # Call nodes: annotations are deliberately not consulted,
                # since ``value: B`` never runs and following it would
                # invalidate on a type hint.
                opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            # Drop the decorator expressions: ``inspect.getsource`` includes the
            # ``@c.cache(...)`` / ``@get_cash().cache`` lines, and analyzing them
            # as if they were body statements walks into the decorator factory's
            # source (cash's own internals), flagging its mutations as the
            # user's (finding #9). We only want to analyze the function body.
            func_def.decorator_list = []

            param_names = frozenset(
                arg.arg for arg in (
                    func_def.args.args
                    + func_def.args.posonlyargs
                    + func_def.args.kwonlyargs
                )
            )
            if func_def.args.vararg:
                param_names = param_names | {func_def.args.vararg.arg}
            if func_def.args.kwarg:
                param_names = param_names | {func_def.args.kwarg.arg}

            local_owned = _compute_local_owned(func_def, param_names)
            visitor = _PurityVisitor(
                qualname=qualname, param_names=param_names, local_owned=local_owned,
            )
            visitor.visit(func_def)
            visitor.finalize_taint()
            all_issues.extend(visitor.issues)

            # Flag reads of module globals that are reassigned/mutated somewhere
            # in the module - a silent staleness footgun (the cached result won't
            # change when the global does). Constants (never written) are not
            # flagged, so this stays quiet on dispatch tables / lookup maps.
            self._flag_mutable_global_reads(
                func, func_def, qualname, visitor.read_names, all_issues,
            )

            if depth >= self._MAX_DEPTH:
                continue

            # Resolve callees and queue user-code helpers. The merged
            # namespace includes closure cells so nested-function
            # helpers (defined inside another function) are visible
            # for recursion.
            namespace = _build_namespace(func)

            def _queue_helper(callee: Any, line: int) -> None:
                if callee is None or not callable(callee):
                    return
                # A call to another @cash.cache-decorated function is a
                # dependency-graph edge, not a helper to walk: its own source
                # hash and purity are tracked as a separate node. Recursing
                # would read cash's wrapper machinery (which ``functools.wraps``
                # makes look like same-package user code) and flag cash's own
                # internal mutations as the user's (finding #9).
                if getattr(callee, "_cash_cached", False):
                    return
                if is_pure(callee):
                    return
                if is_stateful(callee):
                    all_issues.append(PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"calls @stateful {_qualname_of(callee)}()",
                        where=qualname,  # noqa: B023 - loop var, called within iteration
                        line=line,
                    ))
                    return
                if not _is_user_code(callee, root_module):
                    return
                stack.append((callee, depth + 1, False))  # noqa: B023 - same

            for call_node in visitor.called_callable_nodes:
                _queue_helper(
                    _resolve_callee(call_node.func, namespace),
                    getattr(call_node, "lineno", 0),
                )

            # A helper referenced by NAME but reached through a value -- not in
            # call position -- was never walked, so an edit to it silently
            # served a stale result: ``fn = helper; fn(x)``, ``_apply(helper,
            # x)``, storing a function in a local then calling it. Any read name
            # that resolves to a user FUNCTION is a source dependency; walk it
            # too. Restricted to functions/methods so modules,
            # classes, and arbitrary attribute reads are not folded in, and the
            # existing user-code / purity / cached-node filters still apply. The
            # direction is safe: at worst it over-invalidates (a referenced-but-
            # unused function recomputes), never serves stale.
            for _name in visitor.read_names:
                _val = namespace.get(_name)
                if inspect.isfunction(_val) or inspect.ismethod(_val):
                    _queue_helper(_val, 0)

        # Stable order: by where (insertion) then line then kind.
        all_issues_sorted = tuple(sorted(
            all_issues,
            key=lambda i: (i.where, i.line, i.kind, i.description),
        ))
        return PurityReport(
            issues=all_issues_sorted,
            helper_source_hashes=helper_hashes,
            helper_resolution_paths=helper_paths,
            opaque_callees=tuple(sorted(set(opaque))),
        )

    def _flag_mutable_global_reads(
        self,
        func: Callable[..., Any],
        func_def: ast.AST,
        qualname: str,
        read_names: set[str],
        all_issues: list[PurityIssue],
    ) -> None:
        """Append an issue for each module global *func* reads that is
        reassigned/mutated elsewhere in its module - the result would go stale
        when that global changes. Reads of never-written globals (constants,
        dispatch tables) are not flagged."""
        if not read_names:
            return
        module = inspect.getmodule(func)
        if module is None:
            return
        modified = _module_modified_globals(module)
        if not modified:
            return
        module_ns = getattr(func, "__globals__", None) or {}
        locals_ = _function_locals(func_def)
        freevars = set(getattr(getattr(func, "__code__", None), "co_freevars", ()) or ())
        own_name = getattr(func, "__name__", None)
        candidates = (
            read_names & modified
        ) - locals_ - freevars - _PY_BUILTIN_NAMES
        for name in sorted(candidates):
            if name == own_name or name not in module_ns:
                continue
            all_issues.append(PurityIssue(
                kind=ISSUE_MUTABLE_GLOBAL,
                description=(
                    f"reads module global {name!r} that is reassigned or mutated "
                    f"elsewhere - cached results won't reflect changes to it; pass "
                    f"it as an argument or declare it via depends_on"
                ),
                where=qualname,
                line=0,
            ))


_global_analyzer: PurityAnalyzer | None = None
_global_analyzer_lock = threading.Lock()


def get_analyzer() -> PurityAnalyzer:
    """Return the process-wide :class:`PurityAnalyzer` singleton.

    Multiple :class:`Cash` instances share one analyzer cache so
    redundant AST walks are avoided across instances.
    """
    global _global_analyzer
    if _global_analyzer is not None:
        return _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = PurityAnalyzer()
        return _global_analyzer


import builtins as _builtins  # noqa: E402

_PY_BUILTIN_NAMES = frozenset(dir(_builtins))
_MODULE_MOD_GLOBALS_CACHE: dict[str, frozenset[str]] = {}


def _target_root_name(node: ast.AST) -> str | None:
    """Root ``Name`` id of an Attribute/Subscript chain (``a.b[c]`` -> ``a``)."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _collect_bound_names(target: ast.AST, out: set[str]) -> None:
    """Names bound by an assignment target (``Name`` / nested tuple/list)."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
        for el in ast.iter_child_nodes(target):
            _collect_bound_names(el, out)


def _function_locals(func_node: ast.AST) -> frozenset[str]:
    """Names local to a function scope: parameters plus names it binds, minus
    any declared ``global``/``nonlocal``. Does not descend into nested scopes."""
    args = getattr(func_node, "args", None)
    locs: set[str] = set()
    decl: set[str] = set()
    if args is not None:
        for a in (args.posonlyargs + args.args + args.kwonlyargs):
            locs.add(a.arg)
        if args.vararg:
            locs.add(args.vararg.arg)
        if args.kwarg:
            locs.add(args.kwarg.arg)
    body = getattr(func_node, "body", [])
    # A lambda's body is a single expression, not a statement list.
    stack = list(body) if isinstance(body, list) else [body]
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue  # separate scope
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            decl.update(n.names)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                _collect_bound_names(t, locs)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if isinstance(n.target, ast.Name):
                locs.add(n.target.id)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _collect_bound_names(n.target, locs)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars:
                    _collect_bound_names(item.optional_vars, locs)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                locs.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.iter_child_nodes(n):
            stack.append(child)
    return frozenset(locs - decl)


class _GlobalMutationScanner(ast.NodeVisitor):
    """Collects module-global names that are reassigned or mutated *inside a
    function body* (i.e. reachable at runtime), scope-aware so a function's
    local that merely shares a name with a global is not mistaken for a
    mutation of that global.

    Mutations at module top level are ignored on purpose: they run once at
    import, before any cached function is called, so the global is effectively
    constant during runtime and reading it is safe. This avoids flagging
    registries/config dicts that are populated at import and then never change.
    (A mutation inside a function that only ever runs at import - e.g. a
    decorator body - is still flagged conservatively, since we cannot prove
    statically that the function never runs at call time. A false flag is a
    harmless warning; a missed one would be a silent stale cache.)"""

    def __init__(self) -> None:
        self.modified: set[str] = set()
        self._locals_stack: list[frozenset[str]] = []  # enclosing function locals

    @property
    def _in_function(self) -> bool:
        return bool(self._locals_stack)

    def _is_global(self, name: str) -> bool:
        # Inside a function, a name is the global only if it isn't shadowed by
        # a local there. (Only consulted when _in_function is True.)
        return all(name not in loc for loc in self._locals_stack)

    def visit_FunctionDef(self, node):  # noqa: N802
        self._locals_stack.append(_function_locals(node))
        self.generic_visit(node)
        self._locals_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # noqa: N802
        self._locals_stack.append(_function_locals(node))
        self.generic_visit(node)
        self._locals_stack.pop()

    def visit_Global(self, node):  # noqa: N802
        # An explicit `global G` inside a function is intent to rebind the
        # module global at runtime. (`global` at module scope is a no-op.)
        if self._in_function:
            self.modified.update(node.names)
        self.generic_visit(node)

    def visit_AugAssign(self, node):  # noqa: N802
        base = _target_root_name(node.target)
        if self._in_function and base and self._is_global(base):
            self.modified.add(base)
        self.generic_visit(node)

    def visit_Assign(self, node):  # noqa: N802
        for t in node.targets:
            if isinstance(t, (ast.Subscript, ast.Attribute)):
                base = _target_root_name(t)
                if self._in_function and base and self._is_global(base):
                    self.modified.add(base)
        self.generic_visit(node)

    def visit_Delete(self, node):  # noqa: N802
        for t in node.targets:
            base = t.id if isinstance(t, ast.Name) else _target_root_name(t)
            if self._in_function and base and self._is_global(base):
                self.modified.add(base)
        self.generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        f = node.func
        if (self._in_function and isinstance(f, ast.Attribute)
                and f.attr in _WRITE_METHODS
                and isinstance(f.value, ast.Name) and self._is_global(f.value.id)):
            self.modified.add(f.value.id)
        self.generic_visit(node)


def _module_modified_globals(module: Any) -> frozenset[str]:
    """Module-global names that are reassigned/mutated somewhere in *module*.

    Cached per module name. Returns an empty set when the source can't be read
    (so we never flag on incomplete information)."""
    name = getattr(module, "__name__", None)
    if not name:
        return frozenset()
    cached = _MODULE_MOD_GLOBALS_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    except SOURCE_RETRIEVAL_ERRORS:
        _MODULE_MOD_GLOBALS_CACHE[name] = frozenset()
        return frozenset()
    try:
        scanner = _GlobalMutationScanner()
        scanner.visit(tree)
        result = frozenset(scanner.modified)
    except Exception:  # noqa: BLE001 - analysis must never break caching
        logger.debug("global-mutation scan failed for module %s", name)
        result = frozenset()
    _MODULE_MOD_GLOBALS_CACHE[name] = result
    return result


def _qualname_of(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<callable>")
    return f"{module}.{qualname}"


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    try:
        src = inspect.getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.sha256(
        normalize_source_for_hash(src).encode("utf-8")
    ).hexdigest()


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
