from __future__ import annotations

"""AST-based code analysis for detecting statement inputs and outputs.

The :class:`CodeAnalyzer` walks Python AST trees to identify which
variables a statement reads (inputs) and writes (outputs).  This
analysis drives cache key computation and dependency tracking.
"""

import ast
import builtins
import datetime
import hashlib
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_norm import normalize_source_for_hash

__all__ = ["CodeAnalyzer"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level AST visitors
# ---------------------------------------------------------------------------

class _CallVisitor(ast.NodeVisitor):
    """Collect all function-call names from an AST for find_called_functions."""

    def __init__(self) -> None:
        self.names_to_resolve: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self.names_to_resolve.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            curr: ast.expr = node.func
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
                self.names_to_resolve.append(".".join(reversed(parts)))
        self.generic_visit(node)


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

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.define_variable(node.id)
        elif isinstance(node.ctx, ast.Load) and not self.is_defined(node.id):
            self.real_inputs.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
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

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
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

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension(node, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._visit_comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension(node, [node.key, node.value])

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
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

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            parent_name = self._extract_base_name(node.value)
            if parent_name:
                if not self.is_defined(parent_name):
                    self.real_inputs.add(parent_name)
                if len(self.scopes) == 1:
                    self.outputs.add(parent_name)
                    self.modified_objects.add(parent_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
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

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
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
        if node.value:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute):
            for keyword in node.keywords:
                if (keyword.arg == 'inplace'
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True):
                    parent_name = self._extract_base_name(node.func.value)
                    if parent_name:
                        if not self.is_defined(parent_name):
                            self.real_inputs.add(parent_name)
                        if len(self.scopes) == 1:
                            self.outputs.add(parent_name)
                            self.modified_objects.add(parent_name)
                    break
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.define_variable(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
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
    """Detect calls to forbidden (non-deterministic) functions.

    Lifted from the inline class inside ``scan_for_forbidden_functions``.
    """

    def __init__(self, forbidden_objs: dict, user_ns: dict[str, Any]) -> None:
        self.forbidden_objs = forbidden_objs
        self.user_ns = user_ns
        self.local_ns: dict = {}
        self.found_reasons: list[str] = []

    def _resolve_name(self, name: str) -> Any:
        if name in self.local_ns:
            return self.local_ns[name]
        if name in self.user_ns:
            return self.user_ns[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        return None

    def _resolve_node_value(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        if isinstance(node, ast.Attribute):
            parent = self._resolve_node_value(node.value)
            if parent is not None and hasattr(parent, node.attr):
                return getattr(parent, node.attr)
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func_obj = self._resolve_node_value(node.func)
        if func_obj in self.forbidden_objs:
            self.found_reasons.append(self.forbidden_objs[func_obj])
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            base_name = alias.name.split('.')[0]
            if base_name in ('time', 'datetime', 'uuid'):
                try:
                    mod = __import__(base_name)
                    store_name = alias.asname or base_name
                    if not alias.asname and base_name != alias.name:
                        self.local_ns[base_name] = mod
                    else:
                        self.local_ns[store_name] = mod
                except ImportError:
                    pass  # Expected: module may not be importable in analysis context

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module and any(node.module.startswith(m) for m in ('time', 'datetime', 'uuid')):
            try:
                mod = __import__(node.module, fromlist=[n.name for n in node.names])
                for alias in node.names:
                    obj = getattr(mod, alias.name, None)
                    if obj:
                        self.local_ns[alias.asname or alias.name] = obj
            except ImportError:
                pass  # Expected: module may not be importable in analysis context


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """Analyzes function code to determine dependencies and compute hashes."""

    @staticmethod
    def opaque_identity(func: Callable) -> str:
        """Return a stable identity string for an OPAQUE callable.

        Builtins, C-extension functions, ufuncs, ``functools.partial`` objects,
        and other callables without retrievable source/``__code__`` cannot be
        source-hashed or AST-analyzed. We key them on a best-effort stable
        identity — ``module.qualname`` — so caching still works and does not
        crash. Falls back to ``__name__`` then ``repr()`` when qualname is
        absent.
        """
        module = getattr(func, '__module__', None) or '?'
        qualname = (
            getattr(func, '__qualname__', None)
            or getattr(func, '__name__', None)
            or repr(func)
        )
        return f"{module}.{qualname}"

    @staticmethod
    def get_source_hash(func: Callable) -> str:
        """Compute SHA256 hash of the function's source code.

        Falls back to hashing the function's bytecode (``__code__``) when
        ``inspect.getsource()`` fails — for example, functions defined inside
        IPython cells intercepted by ``%cash_on``.

        Last resort: for OPAQUE callables (builtins, C extensions, ufuncs,
        ``functools.partial``) that have neither retrievable source nor a usable
        ``__code__``, hash a stable identity string instead of raising, so they
        can be cached without crashing.
        """
        try:
            source = inspect.getsource(func)
            normalized = normalize_source_for_hash(source)
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        except SOURCE_RETRIEVAL_ERRORS:
            pass  # Expected: source unavailable for builtins/C extensions, or a
                  # co_filename that doesn't tokenize as Python; fall through to
                  # the bytecode hash.

        code_obj = getattr(func, '__code__', None)
        if code_obj is None and hasattr(func, '__wrapped__'):
            code_obj = getattr(func.__wrapped__, '__code__', None)
        if code_obj is not None:
            try:
                parts = [
                    code_obj.co_code,
                    str(code_obj.co_consts).encode('utf-8'),
                    str(code_obj.co_names).encode('utf-8'),
                    str(code_obj.co_varnames).encode('utf-8'),
                ]
                return hashlib.sha256(b''.join(parts)).hexdigest()
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug("[ANALYSIS] Failed to compute bytecode hash for function: %s", exc)

        # Opaque callable (builtin / C-extension / ufunc / partial): key on a
        # stable identity rather than crashing.
        identity = CodeAnalyzer.opaque_identity(func)
        return hashlib.sha256(f"__cash_opaque__:{identity}".encode('utf-8')).hexdigest()

    @staticmethod
    def find_called_functions(func: Callable, known_functions: dict[str, Callable] | None = None) -> set[str]:
        """
        Parse the function AST and find calls to other functions.
        Resolves names using the function's globals to handle imports and aliases.
        Returns a set of function names (qualnames).
        """
        import textwrap
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
        except SOURCE_RETRIEVAL_ERRORS:
            return set()

        visitor = _CallVisitor()
        visitor.visit(tree)

        # An opaque callable (builtin / C-extension / ufunc / partial) may have
        # source available via ``__wrapped__`` yet lack ``__globals__``; without
        # it, names can't be resolved, so skip dependency analysis.
        globals_dict = getattr(func, '__globals__', None)
        if globals_dict is None:
            return set()
        resolved_qualnames: set[str] = set()

        for name in visitor.names_to_resolve:
            parts = name.split('.')
            obj = globals_dict.get(parts[0])

            if obj is None and hasattr(builtins, parts[0]):
                obj = getattr(builtins, parts[0])

            # Existence check, not truthiness: a resolved name may be bound to
            # an object whose __bool__ is ambiguous/raises (DataFrame, ndarray),
            # and `if obj:` would crash analysis of any function that references
            # such a value in its globals.
            if obj is not None:
                try:
                    for part in parts[1:]:
                        obj = getattr(obj, part)
                    if hasattr(obj, '__qualname__'):
                        module = getattr(obj, '__module__', None) or '__unknown__'
                        fqn = f"{module}.{obj.__qualname__}"
                        if known_functions is None or fqn in known_functions:
                            resolved_qualnames.add(fqn)
                except AttributeError:
                    pass  # Expected: some callables lack __qualname__

        return resolved_qualnames

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
        lines = code.split('\n')
        flags: list[bool] = []
        depth = 0            # open bracket/paren/brace nesting (outside strings)
        in_str: str | None = None   # open triple-quote delimiter, or None
        prev_backslash = False
        for line in lines:
            is_start = (depth == 0 and in_str is None and not prev_backslash)
            flags.append(is_start)
            # A dropped magic is a self-contained logical line; do not let its
            # characters perturb the scanner state for following lines.
            if is_start and line.strip().startswith(('%', '!')):
                prev_backslash = False
                continue
            i, n = 0, len(line)
            backslash = False
            while i < n:
                c = line[i]
                if in_str is not None:                 # inside a triple string
                    if c == '\\':
                        i += 2
                        continue
                    if line.startswith(in_str, i):
                        in_str = None
                        i += 3
                        continue
                    i += 1
                    continue
                if c == '#':                           # comment to end of line
                    break
                if c == '\\' and i == n - 1:           # explicit continuation
                    backslash = True
                    i += 1
                    continue
                if c in '([{':
                    depth += 1
                    i += 1
                    continue
                if c in ')]}':
                    depth = max(0, depth - 1)
                    i += 1
                    continue
                if c in ('"', "'"):
                    if line.startswith(c * 3, i):      # triple-quoted string
                        delim = c * 3
                        j, closed = i + 3, False
                        while j < n:
                            if line[j] == '\\':
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
                        in_str = delim                 # spills onto next line
                        i = n
                        continue
                    j = i + 1                          # single-line string
                    while j < n:
                        if line[j] == '\\':
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

        Only lines that *begin a logical line* and start with ``%`` or ``!``
        are treated as magics. A ``%`` (modulo / ``%``-format) or ``!`` that
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
            CodeAnalyzer._parse_cell(code)
            return code
        except SyntaxError:
            pass
        starts = CodeAnalyzer._logical_line_start_flags(code)
        out: list[str] = []
        for line, is_start in zip(code.split('\n'), starts):
            if is_start and line.strip().startswith(('%', '!')):
                indent = line[:len(line) - len(line.lstrip())]
                if indent:
                    # An indented magic is the leading (often sole) statement of
                    # a block — ``if colab:``, ``for``, ``def`` … Deleting it
                    # outright can leave an EMPTY suite (``if colab:`` with
                    # nothing under it), a *fictional* SyntaxError that makes cash
                    # treat a perfectly good cell as unparseable and stop
                    # dependency-tracking it. Neutralise it in place with
                    # ``pass`` so the block keeps a body (a magic contributes no
                    # variable dependencies, so this loses nothing for analysis).
                    out.append(indent + 'pass')
                # A top-level magic is dropped entirely: removing a module-level
                # statement never empties a block.
                continue
            out.append(line)
        return '\n'.join(out)

    @staticmethod
    def _parse_cell(code: str) -> ast.Module:
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
            code, '<cash-cell>', 'exec',
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
                declared as an OUTPUT of the calling statement (CAS-265), so
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
            tree = CodeAnalyzer._parse_cell(clean_code)  # tolerate top-level await

        visitor = _FlowVisitor()
        visitor.visit(tree)
        inputs, outputs = visitor.real_inputs, visitor.outputs
        if resolve_source is not None:
            try:
                from .cacheability import callee_mutated_globals_for_tree
                extra = callee_mutated_globals_for_tree(
                    tree, resolve_source, user_ns)
            except Exception:  # noqa: BLE001 - analysis must never break a cell
                extra = frozenset()
            if extra:
                outputs = outputs | set(extra)
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
        forbidden_objs: dict = {}

        for name in ['time', 'monotonic', 'perf_counter', 'process_time', 'time_ns', 'localtime', 'gmtime']:
            if hasattr(time, name):
                forbidden_objs[getattr(time, name)] = f"time.{name}"

        forbidden_objs[datetime.datetime.now] = "datetime.now"
        forbidden_objs[datetime.datetime.utcnow] = "datetime.utcnow"
        forbidden_objs[datetime.date.today] = "date.today"

        for name in ['uuid1', 'uuid4']:
            if hasattr(uuid, name):
                forbidden_objs[getattr(uuid, name)] = f"uuid.{name}"

        if tree is None:
            try:
                tree = ast.parse(CodeAnalyzer.strip_magics(code))
            except SyntaxError:
                return []

        visitor = _ForbiddenVisitor(forbidden_objs, user_ns)
        visitor.visit(tree)
        return list(set(visitor.found_reasons))
