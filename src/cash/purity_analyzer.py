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
import hashlib
import inspect
import logging
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .notebook.cacheability import (
    PANDAS_INPLACE_METHODS,
    _get_base_name,
    _get_call_module,
    _get_call_name,
    _is_open_write_mode,
)
from .notebook.function_tracker import is_local_module
from .notebook.purity import (
    KNOWN_PURE_BUILTINS,
    _IMPURE_FUNCTION_CALLS,
    _IMPURE_MODULE_CALLS,
    _WRITE_METHODS,
    is_pure,
    is_stateful,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PurityAnalyzer",
    "PurityReport",
    "PurityIssue",
    "ISSUE_IMPURE_CALL",
    "ISSUE_DYNAMIC_PATTERN",
    "ISSUE_DISCARDED_CALL",
    "ISSUE_SCOPE_MUTATION",
]

ISSUE_IMPURE_CALL = "impure_call"
ISSUE_DYNAMIC_PATTERN = "dynamic_pattern"
ISSUE_DISCARDED_CALL = "discarded_call"
ISSUE_SCOPE_MUTATION = "scope_mutation"


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
        helper_source_hashes: ``qualname -> sha256(source)`` for
            every user-code helper actually walked, captured at
            analysis time. Used as the fallback when per-call
            re-resolution fails (helper was deleted/renamed since
            analysis).
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
        "_qualname", "_line_offset", "_local_owned",
    )

    def __init__(self, qualname: str, param_names: frozenset[str],
                 line_offset: int = 0,
                 local_owned: frozenset[str] = frozenset()) -> None:
        self.issues: list[PurityIssue] = []
        self.called_callable_nodes: list[ast.AST] = []
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

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._record_call(node)
        self.generic_visit(node)

    def _record_call(self, node: ast.Call) -> None:
        func_node = node.func
        line = getattr(node, "lineno", 0)

        # Explicit dynamism: eval / exec / compile by bare name.
        if isinstance(func_node, ast.Name) and func_node.id in {"eval", "exec", "compile"}:
            self.issues.append(PurityIssue(
                kind=ISSUE_DYNAMIC_PATTERN,
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
                kind=ISSUE_DYNAMIC_PATTERN,
                description="getattr(obj, name)(...) with non-constant name - dynamic dispatch",
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

        stack: list[tuple[Callable[..., Any], int]] = [(root_func, 0)]
        while stack:
            func, depth = stack.pop()
            qualname = _qualname_of(func)
            if qualname in visited:
                continue
            visited.add(qualname)

            # Read source. Failure -> opaque leaf.
            try:
                src = inspect.getsource(func)
            except (OSError, TypeError):
                opaque.append(qualname)
                continue
            src = textwrap.dedent(src)

            # Hash the (dedented) source for cache-key invalidation
            # of helpers. Root function's hash is captured separately
            # by the decorator via _hash_callable_source - we record
            # all walked callables here so the decorator can fold
            # them into the state hash uniformly.
            helper_hashes[qualname] = hashlib.sha256(src.encode("utf-8")).hexdigest()

            # Capture a resolution hint so the decorator can re-resolve
            # this helper from sys.modules per call and pick up
            # in-process redefinitions (notebook cells, REPL). The
            # attr_chain is the helper's __qualname__ split on '.' so
            # methods like ``Klass.method`` resolve correctly. Skip the
            # root function itself (it's not a "helper" - the decorator
            # has its own reference).
            helper_module = getattr(func, "__module__", None)
            helper_inner_qualname = getattr(func, "__qualname__", None)
            if (
                func is not root_func
                and helper_module
                and helper_inner_qualname
                and "<locals>" not in helper_inner_qualname
            ):
                attr_chain = tuple(helper_inner_qualname.split("."))
                helper_paths[qualname] = (helper_module, attr_chain)

            try:
                tree = ast.parse(src)
            except SyntaxError:
                opaque.append(qualname)
                continue

            func_def = _find_first_function_def(tree)
            if func_def is None:
                opaque.append(qualname)
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
            all_issues.extend(visitor.issues)

            if depth >= self._MAX_DEPTH:
                continue

            # Resolve callees and queue user-code helpers. The merged
            # namespace includes closure cells so nested-function
            # helpers (defined inside another function) are visible
            # for recursion.
            namespace = _build_namespace(func)
            for call_node in visitor.called_callable_nodes:
                callee = _resolve_callee(call_node.func, namespace)
                if callee is None or not callable(callee):
                    continue
                # A call to another @cash.cache-decorated function is a
                # dependency-graph edge, not a helper to walk: its own source
                # hash and purity are tracked as a separate node. Recursing
                # would read cash's wrapper machinery (which ``functools.wraps``
                # makes look like same-package user code) and flag cash's own
                # internal mutations as the user's (finding #9).
                if getattr(callee, "_cash_cached", False):
                    continue
                if is_pure(callee):
                    continue
                if is_stateful(callee):
                    callee_qual = _qualname_of(callee)
                    all_issues.append(PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"calls @stateful {callee_qual}()",
                        where=qualname,
                        line=getattr(call_node, "lineno", 0),
                    ))
                    continue
                if not _is_user_code(callee, root_module):
                    continue
                stack.append((callee, depth + 1))

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


def _qualname_of(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<callable>")
    return f"{module}.{qualname}"


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
