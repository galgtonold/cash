from __future__ import annotations

"""Pure-AST cacheability analysis for notebook statements.

**Boundary rule (baked here by design):** this module is pure-AST.
It accepts ``(code, tree)`` and returns a :class:`StatementAnalysis`.
The moment something needs runtime state — ``user_ns``, value
introspection, purity-registry lookup, annotation parsing — it belongs
in ``cacheability_decision.py``, not here.  That sibling module owns
the merge of AST findings with runtime context; see ``CONTEXT.md``
entry *Cacheability decision*.

Folds ``mutation_detector.py`` and ``side_effects.py`` into one module.
Both AST-visitor classes and their supporting dataclasses live here;
their names are re-exported for any existing import sites that reference
them directly.
"""

import ast
import textwrap
from dataclasses import dataclass

__all__ = [
    # Primary API
    "StatementAnalysis",
    "analyze_statement",
    "standalone_method_mutation_receivers",
    "standalone_method_call_receivers",
    "selfref_inplace_write_vars",
    "params_mutated_in_function",
    "standalone_call_arg_targets",
    "function_arg_mutations",
    "function_global_mutations",
    "stateful_self_functions",
    "stateful_closure_vars",
    "partial_arg_mutations",
    "mutating_partials",
    "reduce_free_mutations",
    "alias_mutation_sources",
    "aliased_sources",
    "crossref_reassigned_vars",
    "subscript_view_bindings",
    # Re-exported dataclasses (moved from mutation_detector / side_effects)
    "MutationInfo",
    "SideEffectInfo",
    # Constants re-exported for external consumers
    "MUTATING_METHODS",
    "PANDAS_INPLACE_METHODS",
    "KNOWN_PURE_METHODS",
]

# ---------------------------------------------------------------------------
# Mutation detection — moved from mutation_detector.py
# ---------------------------------------------------------------------------

# Methods that mutate their receiver object in-place
MUTATING_METHODS = {
    # list methods
    'append', 'extend', 'insert', 'pop', 'remove', 'sort', 'reverse', 'clear',
    # dict methods
    'update', 'popitem', 'setdefault',
    # set methods
    'add', 'discard', 'intersection_update', 'difference_update', 'symmetric_difference_update',
}

# Pandas methods that accept inplace=True
PANDAS_INPLACE_METHODS = {
    'fillna', 'dropna', 'drop', 'rename', 'reset_index', 'set_index',
    'sort_values', 'sort_index', 'replace', 'clip', 'where', 'mask',
    'drop_duplicates', 'eval', 'query', 'astype',
}

# Read-only inspection / display methods that never mutate their receiver.
# Deliberately conservative: this set only lets the runtime *skip* the
# before/after content observation (a perf optimisation that matters for large
# objects like DataFrames, where hashing twice per standalone call is costly).
# A name here must be unambiguously non-mutating — being wrong means a real
# mutation goes undetected. Anything not listed falls through to observation.
KNOWN_PURE_METHODS = frozenset({
    # pandas / numpy inspection & summary (return a new object, never mutate)
    'head', 'tail', 'describe', 'info', 'sample', 'value_counts', 'nunique',
    'unique', 'corr', 'cov', 'memory_usage', 'count', 'isna', 'isnull',
    'notna', 'notnull', 'nlargest', 'nsmallest', 'idxmax', 'idxmin',
    # display / plotting
    'plot', 'hist', 'boxplot', 'show',
})


@dataclass
class MutationInfo:
    """Information about a detected mutation."""

    variable: str
    method: str
    kind: str  # 'method_call', 'inplace_kwarg', 'augmented_assign', 'subscript_assign'
    line: int = 0


def _extract_base_name(node: ast.AST) -> str | None:
    """Extract the root variable name from a potentially nested AST node.

    Handles chained method calls like ``groups.setdefault(key, []).append(val)``
    by walking through Call nodes to reach the underlying variable.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return _extract_base_name(node.value)
    if isinstance(node, ast.Call):
        # For chained calls like obj.method1().method2(), walk through the Call
        # to find the root variable.
        if isinstance(node.func, ast.Attribute):
            return _extract_base_name(node.func.value)
        if isinstance(node.func, ast.Name):
            return node.func.id
    return None


def _iter_store_targets(target: ast.expr):
    """Yield the leaf store targets of an assignment target, flattening tuple/list
    unpacking and starred elements.

    ``df['a']`` -> the subscript itself; ``df['a'], df['b']`` -> both subscripts;
    ``a, *rest = ...`` -> the ``Name`` and the starred ``Name`` (callers ignore
    plain ``Name`` targets). Nested tuples (``(a, (b, c))``) are recursed into.
    """
    if isinstance(target, ast.Starred):
        yield from _iter_store_targets(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_store_targets(elt)
    else:
        yield target


class _MutationVisitor(ast.NodeVisitor):
    """AST visitor that collects :class:`MutationInfo` entries."""

    def __init__(self) -> None:
        self.mutations: list[MutationInfo] = []

    def _record_method_mutation(self, call: ast.Call, lineno: int) -> None:
        """Record a KNOWN-mutating method call on a named receiver.

        A method in ``MUTATING_METHODS`` (``append``/``pop``/``update``/...) or a
        pandas ``inplace=True`` call mutates its receiver regardless of where it
        appears — as a bare statement, a captured result (``r = lst.pop()``), a
        comprehension element (``[base.append(x) for ..]``), or an f-string
        placeholder (``f"{lst.append(x)}"``). Called from :meth:`visit_Call` so
        every call site is covered, not just top-level ``Expr`` statements
        (CAS-67).
        """
        if not isinstance(call.func, ast.Attribute):
            return
        method_name = call.func.attr
        base = _extract_base_name(call.func.value)
        if not base:
            return
        if method_name in MUTATING_METHODS:
            self.mutations.append(MutationInfo(
                variable=base, method=method_name,
                kind='method_call', line=lineno,
            ))
        elif method_name in PANDAS_INPLACE_METHODS:
            for kw in call.keywords:
                if (kw.arg == 'inplace'
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    self.mutations.append(MutationInfo(
                        variable=base, method=method_name,
                        kind='inplace_kwarg', line=lineno,
                    ))
                    break

    def visit_Call(self, node: ast.Call) -> None:
        """Detect KNOWN-mutating method calls (anywhere) and the numpy ufunc
        ``out=`` kwarg, which writes its target array in place: ``np.add(a, 10,
        out=a)`` mutates ``a`` (the out target), not the ``np`` receiver. Fires on
        any Call (result captured or not). For multi-output ufuncs ``out`` is a
        tuple: ``out=(q, r)``."""
        self._record_method_mutation(node, node.lineno)
        for kw in node.keywords:
            if kw.arg != 'out':
                continue
            targets = (
                kw.value.elts
                if isinstance(kw.value, (ast.Tuple, ast.List))
                else [kw.value]
            )
            for tgt in targets:
                base = _extract_base_name(tgt)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method='out=',
                        kind='out_kwarg', line=node.lineno,
                    ))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Detect augmented assignments like x += 1, arr *= 2."""
        base = _extract_base_name(node.target)
        if base:
            op_name = type(node.op).__name__
            self.mutations.append(MutationInfo(
                variable=base, method=f'__i{op_name.lower()}__',
                kind='augmented_assign', line=node.lineno,
            ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect subscript/attribute assignments like d[key] = val, obj.attr = val,
        including those nested in a tuple/list target (df['a'], df['b'] = ...)."""
        for target in node.targets:
            for store in _iter_store_targets(target):
                if isinstance(store, ast.Subscript):
                    base = _extract_base_name(store.value)
                    if base:
                        self.mutations.append(MutationInfo(
                            variable=base, method='__setitem__',
                            kind='subscript_assign', line=node.lineno,
                        ))
                elif isinstance(store, ast.Attribute):
                    base = _extract_base_name(store.value)
                    if base:
                        self.mutations.append(MutationInfo(
                            variable=base, method=f'__setattr__({store.attr})',
                            kind='attribute_assign', line=node.lineno,
                        ))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Detect del d[key], del lst[0]."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method='__delitem__',
                        kind='subscript_delete', line=node.lineno,
                    ))
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Side-effect detection — moved from side_effects.py
# ---------------------------------------------------------------------------

@dataclass
class SideEffectInfo:
    """Information about a detected side effect."""

    kind: str  # 'file_write', 'network', 'system', 'global_state'
    description: str
    line: int = 0


# Function calls known to produce I/O side effects.
# Format: (module_or_empty_string, function_name) -> side_effect_kind
_IO_SIDE_EFFECT_FUNCTIONS: dict[tuple[str, str], str] = {
    # File writing
    ('', 'open'): 'file_write',  # open() with write modes detected separately
    # os module
    ('os', 'remove'): 'file_write',
    ('os', 'unlink'): 'file_write',
    ('os', 'rmdir'): 'file_write',
    ('os', 'mkdir'): 'file_write',
    ('os', 'makedirs'): 'file_write',
    ('os', 'rename'): 'file_write',
    ('os', 'replace'): 'file_write',
    ('os', 'symlink'): 'file_write',
    ('os', 'system'): 'system',
    # shutil
    ('shutil', 'copy'): 'file_write',
    ('shutil', 'copy2'): 'file_write',
    ('shutil', 'copytree'): 'file_write',
    ('shutil', 'rmtree'): 'file_write',
    ('shutil', 'move'): 'file_write',
    # subprocess
    ('subprocess', 'run'): 'system',
    ('subprocess', 'call'): 'system',
    ('subprocess', 'Popen'): 'system',
    ('subprocess', 'check_call'): 'system',
    ('subprocess', 'check_output'): 'system',
    # pandas write operations
    ('', 'to_csv'): 'file_write',
    ('', 'to_excel'): 'file_write',
    ('', 'to_parquet'): 'file_write',
    ('', 'to_json'): 'file_write',
    ('', 'to_pickle'): 'file_write',
    ('', 'to_hdf'): 'file_write',
    ('', 'to_feather'): 'file_write',
    ('', 'to_sql'): 'file_write',
    # json/pickle/csv module
    ('json', 'dump'): 'file_write',
    ('pickle', 'dump'): 'file_write',
    ('csv', 'writer'): 'file_write',
    # requests/urllib
    ('requests', 'post'): 'network',
    ('requests', 'put'): 'network',
    ('requests', 'delete'): 'network',
    ('requests', 'patch'): 'network',
}

# Method names that indicate writing (when called on any object)
_WRITE_METHODS: frozenset[str] = frozenset({
    'to_csv', 'to_excel', 'to_parquet', 'to_json', 'to_pickle',
    'to_hdf', 'to_feather', 'to_sql', 'to_stata', 'to_latex',
    'to_html', 'to_clipboard', 'to_gbq', 'to_markdown',
    'savefig',   # matplotlib
    'save',      # numpy, PIL, torch
    'write',     # file objects
    'writelines',
})

# File open modes that indicate writing
_WRITE_MODES: frozenset[str] = frozenset({'w', 'wb', 'a', 'ab', 'w+', 'wb+', 'a+', 'ab+', 'x', 'xb'})


def _get_call_name(func_node: ast.AST) -> str | None:
    """Extract the function name from a call's func node."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None


def _get_call_module(func_node: ast.AST) -> str | None:
    """Extract the module/object prefix from a call's func node."""
    if isinstance(func_node, ast.Attribute):
        if isinstance(func_node.value, ast.Name):
            return func_node.value.id
        if isinstance(func_node.value, ast.Attribute):
            # e.g., os.path.join -> module = 'os.path'
            parts: list[str] = []
            node: ast.AST = func_node.value
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return '.'.join(reversed(parts))
    return None


def _get_base_name(node: ast.AST) -> str | None:
    """Extract a human-readable name for the object a method is called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_base_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        base = _get_base_name(node.value)
        return f"{base}[...]" if base else None
    return None


def _is_open_write_mode(call_node: ast.Call) -> bool:
    """Return True if an open() call uses a write mode."""
    # Check positional arg (2nd argument is mode)
    if len(call_node.args) >= 2:
        mode_arg = call_node.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            return mode_arg.value in _WRITE_MODES or any(c in mode_arg.value for c in 'wax')
    # Check keyword argument mode=...
    for kw in call_node.keywords:
        if kw.arg == 'mode' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value in _WRITE_MODES or any(c in kw.value.value for c in 'wax')
    return False


class _SideEffectVisitor(ast.NodeVisitor):
    """Collects side-effect call sites from an AST."""

    def __init__(self) -> None:
        self.effects: list[SideEffectInfo] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Detect function/method calls with side effects."""
        func_name = _get_call_name(node.func)
        module_name = _get_call_module(node.func)

        if func_name:
            key = (module_name or '', func_name)
            if key in _IO_SIDE_EFFECT_FUNCTIONS:
                kind = _IO_SIDE_EFFECT_FUNCTIONS[key]
                if func_name == 'open' and not module_name:
                    if _is_open_write_mode(node):
                        self.effects.append(SideEffectInfo(
                            kind='file_write',
                            description="open() with write mode",
                            line=getattr(node, 'lineno', 0),
                        ))
                else:
                    self.effects.append(SideEffectInfo(
                        kind=kind,
                        description=f"{module_name + '.' if module_name else ''}{func_name}()",
                        line=getattr(node, 'lineno', 0),
                    ))

            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in _WRITE_METHODS:
                    base = _get_base_name(node.func.value)
                    self.effects.append(SideEffectInfo(
                        kind='file_write',
                        description=f"{base + '.' if base else ''}{method}()",
                        line=getattr(node, 'lineno', 0),
                    ))

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatementAnalysis:
    """Pure-AST findings for a single notebook statement.

    All fields use immutable types because the dataclass is frozen.
    Callers that need mutable sets must copy on use.

    ``top_level_mutated_vars`` — variables mutated at the top level only
    (not inside class/function bodies).  Used for the pre-execution skip
    decision.

    ``all_mutated_vars`` — every variable mutated anywhere in the code.
    Used post-execution to update ``vars_with_mutation_lineage``.

    ``side_effects`` — I/O and system calls that make caching unsound.

    ``called_names`` — bare-name function-call targets (``ast.Call`` nodes
    whose ``func`` is an ``ast.Name``).  Caller resolves each against
    ``user_ns`` via ``_check_callable_stateful``.
    """

    top_level_mutated_vars: frozenset[str]
    all_mutated_vars: frozenset[str]
    side_effects: tuple[SideEffectInfo, ...]
    called_names: frozenset[str]

    def skip_reasons(self, outputs: set[str]) -> list[str]:
        """Render structured findings as human-readable skip reasons.

        Used to populate ``metrics['uncacheable_reasons']``.

        Args:
            outputs: Variable names that are *outputs* of this statement.
                     Mutations on outputs are expected and do not block
                     caching (the output itself gets a fresh lineage).
        """
        reasons: list[str] = []
        pure_mutations = self.top_level_mutated_vars - outputs
        if pure_mutations:
            reasons.append(f"In-place mutation on: {', '.join(sorted(pure_mutations))}")
        for e in self.side_effects:
            reasons.append(f"Side effect: {e.description} ({e.kind})")
        return reasons


def _expr_call_inplace_true(call: ast.Call) -> bool:
    """Return True if *call* passes ``inplace=True`` as a keyword."""
    for kw in call.keywords:
        if (
            kw.arg == 'inplace'
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
        ):
            return True
    return False


def _out_kwarg_target_bases(call: ast.Call) -> list[str]:
    """Base names written in place by a numpy-style ``out=`` kwarg.

    ``np.add(a, 1, out=a)`` mutates ``a``; multi-output ufuncs take a tuple
    (``out=(q, r)``); the target may be a slice (``out=arr[1:]`` -> ``arr``).
    The mutated object is the out target, NOT the call's own receiver (``np``).
    """
    bases: list[str] = []
    for kw in call.keywords:
        if kw.arg != 'out':
            continue
        targets = (
            kw.value.elts
            if isinstance(kw.value, (ast.Tuple, ast.List))
            else [kw.value]
        )
        for tgt in targets:
            base = _extract_base_name(tgt)
            if base:
                bases.append(base)
    return bases


def _selfref_target_base(target: ast.expr) -> str | None:
    """Base name of a subscript/attribute store target (``df['a']``/``df.iloc[i,j]``
    /``obj.attr`` -> ``df``/``df``/``obj``); ``None`` for a plain ``Name`` store."""
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return _extract_base_name(target)
    return None


def _rhs_reads_target(rhs: ast.expr, target: ast.expr) -> bool:
    """True if *rhs* reads the exact same subscript/attribute expression as *target*
    (e.g. ``df['a']`` appears in the RHS of ``df['a'] = df['a'] * 2``)."""
    try:
        tgt = ast.unparse(target)
    except Exception:  # noqa: BLE001 — unparse can fail on exotic nodes
        return False
    for sub in ast.walk(rhs):
        if isinstance(sub, (ast.Subscript, ast.Attribute)):
            try:
                if ast.unparse(sub) == tgt:
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


# Accessor attributes that index by POSITION (no recoverable column name).
_POSITIONAL_ACCESSORS = frozenset({'iloc', 'iat'})
# Accessor attributes that index by LABEL; the column is the last slice element.
_LABEL_ACCESSORS = frozenset({'loc', 'at'})


def _key_literals(col: ast.expr) -> frozenset[str] | None:
    """String column literal(s) in a selector, or ``None`` if not all string
    literals (a slice, a variable, an int, a boolean mask, etc.)."""
    if isinstance(col, ast.Constant) and isinstance(col.value, str):
        return frozenset({col.value})
    if isinstance(col, (ast.List, ast.Tuple)):
        keys: set[str] = set()
        for elt in col.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                keys.add(elt.value)
            else:
                return None
        return frozenset(keys)
    return None


def _subscript_column_keys(node: ast.expr) -> frozenset[str] | None:
    """String column key(s) a DataFrame subscript touches, or ``None`` if unknown.

    ``df['a']`` -> ``{'a'}``; ``df[['a','b']]`` -> ``{'a','b'}``;
    ``df.loc[mask, 'a']`` -> ``{'a'}``; ``df.loc[mask, ['a','b']]`` -> ``{'a','b'}``.
    ``None`` (unknown) for positional access (``df.iloc[i, 0]``), a variable/slice
    column, or whole-row selection (``df.loc[mask]``) — callers must then make no
    assumption about which column is touched.
    """
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if isinstance(value, ast.Attribute):
        if value.attr in _POSITIONAL_ACCESSORS:
            return None
        if value.attr in _LABEL_ACCESSORS:
            # df.loc[row, col]: the column is the last tuple element.
            # df.loc[row] (no column axis) -> whole-row -> unknown.
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 2:
                return _key_literals(node.slice.elts[-1])
            return None
        # df.<other-accessor>[...] — not a recognised column selector.
        return None
    # Plain df[...] subscript: the slice itself is the column selector.
    return _key_literals(node.slice)


def _rhs_reads_same_column(rhs: ast.expr, target: ast.expr, base: str) -> bool:
    """True if *rhs* reads a same-``base`` subscript whose column key(s) overlap
    the *target*'s written column key(s).

    Catches a masked self-write spelled differently from its target —
    ``df.loc[mask, 'a'] = df['a'] * 2`` writes and reads column ``'a'`` though the
    two subscripts are not textually identical (so :func:`_rhs_reads_target`
    misses it). A write to a DIFFERENT column read from another
    (``df.loc[mask, 'b'] = df['a']*2``) has disjoint keys and is NOT flagged,
    preserving the CAS-42 derived-column cache.
    """
    written = _subscript_column_keys(target)
    if not written:  # unknown/positional target -> defer to the exact-match path
        return False
    for sub in ast.walk(rhs):
        if not isinstance(sub, ast.Subscript) or _extract_base_name(sub) != base:
            continue
        read = _subscript_column_keys(sub)
        if read and (read & written):
            return True
    return False


# Statement scopes whose bodies run only later (when called/instantiated), so a
# mutation inside them is NOT a module-level write of the current cell.
_DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _module_level_stmts(body: list[ast.stmt]):
    """Yield every statement that executes when *body* runs at module level,
    descending into control-flow bodies (if/for/while/with/try) but NOT into
    deferred scopes (def/async def/class). A column transform guarded by an
    ``if`` or run in a ``for`` loop still mutates the frame when the cell runs."""
    for node in body:
        if isinstance(node, _DEFERRED_SCOPES):
            continue
        yield node
        for field in ('body', 'orelse', 'finalbody'):
            nested = getattr(node, field, None)
            if nested:
                yield from _module_level_stmts(nested)
        for handler in getattr(node, 'handlers', []):  # try/except handler bodies
            yield from _module_level_stmts(handler.body)


def _target_key_grows_receiver(target: ast.expr, base: str) -> bool:
    """True if a subscript target's KEY is a size-dependent index of *base* — the
    row-append idiom ``df.loc[len(df)] = ..`` / ``df.loc[df.shape[0]] = ..``.

    Such a write ADDS a row at a position derived from the receiver's own size, so
    it is non-idempotent (re-running grows the frame again) and the receiver must
    reset. Scoped to ``len(base)`` / ``base.shape`` / ``base.size`` in the key, so
    a masked write whose key merely reads the frame (``df.loc[df['a'] > 0, 'b'] =
    5``, idempotent) is NOT flagged (CAS-74)."""
    if not isinstance(target, ast.Subscript):
        return False
    for sub in ast.walk(target.slice):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id == 'len'
                and any(_extract_base_name(a) == base for a in sub.args)):
            return True
        if (isinstance(sub, ast.Attribute) and sub.attr in ('shape', 'size')
                and _extract_base_name(sub.value) == base):
            return True
    return False


def _selfref_write_base(target: ast.expr, rhs: ast.expr) -> str | None:
    """Base var if ``target = rhs`` is a self-referential in-place subscript/attr
    write — the target is also read in *rhs*, by exact text or column-key overlap,
    or the target's key grows the receiver (``df.loc[len(df)] = ..``)."""
    base = _selfref_target_base(target)
    if base and (
        _rhs_reads_target(rhs, target)
        or _rhs_reads_same_column(rhs, target, base)
        or _target_key_grows_receiver(target, base)
    ):
        return base
    return None


def selfref_inplace_write_vars(tree: ast.Module | None) -> frozenset[str]:
    """Base vars mutated by a NON-IDEMPOTENT in-place subscript/attribute write at
    the top level — re-running the statement re-applies the mutation.

    Covers, with the receiver restored to its cell-entry base on re-run:

    * self-referential writes whose RHS reads the target — ``df['a'] = df['a']*2``,
      ``df['a'] += 1``, ``df.iloc[i, j] += x``, ``df['a'] = df['a'].fillna(0)``,
      ``obj.attr = obj.attr + 1`` (CAS-54);
    * MASKED writes whose RHS reads the same column spelled differently
      (``df.loc[mask, 'a'] = df['a']*2``) — matched by column-key overlap, not
      exact text (CAS-55, see :func:`_rhs_reads_same_column`);
    * tuple/list unpacking that reads & writes overlapping columns
      (``df['a'], df['b'] = df['b'], df['a']`` — a column swap) (CAS-56);
    * ``del`` of a subscript/attribute (``del df['b']``, ``del obj.cache``) — a
      second ``del`` raises, so the receiver must reset (CAS-56);
    * any of the above nested in an if/for/while/with body
      (``if cond: df['a'] = df['a']*2``) — scanned via :func:`_module_level_stmts`
      (CAS-57; the reset itself uses the live value's lineage, which survives the
      simulator's control-structure collapse).

    Such writes are NON-IDEMPOTENT, so on an isolated cell re-run the lineage-
    carrying receiver (DataFrame/Series/custom object) must be restored first —
    otherwise the value accumulates (``df['a']*2`` doubles again) or the re-run
    errors. The caller routes these vars through the same stale-value reset used
    for method receivers (see CAS-54).

    Deliberately EXCLUDES writes to a NEW target read from OTHER keys
    (``df['b'] = df['a'] + 1``, ``df['VolAdj'] = df.groupby('Ticker')['Close']…``,
    ``df['c'], df['d'] = df['a'], df['b']``): those are idempotent on re-run and
    keep their per-statement cache, preserving the CAS-42 design. Augmented
    assignment (``+=``) is always self-referential. Scans module-level statements
    including those nested in if/for/while/with bodies (CAS-57) but NOT inside
    def/class scopes (their bodies run only when called).
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in _module_level_stmts(tree.body):
        if isinstance(node, ast.AugAssign):
            base = _selfref_target_base(node.target)
            if base:
                out.add(base)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                # Tuple/list unpacking: test each element against the whole RHS, so
                # a swap (df['a'], df['b'] = df['b'], df['a']) flags df while new
                # columns (df['c'], df['d'] = df['a'], df['b']) stay excluded.
                elts = (
                    target.elts
                    if isinstance(target, (ast.Tuple, ast.List))
                    else [target]
                )
                for elt in elts:
                    if isinstance(elt, ast.Starred):
                        elt = elt.value
                    base = _selfref_write_base(elt, node.value)
                    if base:
                        out.add(base)
        elif isinstance(node, ast.Delete):
            # del df['b'] / del obj.cache removes in place and is non-idempotent.
            for target in node.targets:
                base = _selfref_target_base(target)
                if base:
                    out.add(base)
    return frozenset(out)


def _positional_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Ordered positional parameter names (posonly + normal), excluding *args."""
    return [a.arg for a in (*func.args.posonlyargs, *func.args.args)]


def _all_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every parameter name (posonly + normal + kwonly + *args + **kwargs)."""
    params = {
        a.arg
        for a in (
            *func.args.posonlyargs,
            *func.args.args,
            *func.args.kwonlyargs,
        )
    }
    if func.args.vararg:
        params.add(func.args.vararg.arg)
    if func.args.kwarg:
        params.add(func.args.kwarg.arg)
    return params


def _params_mutated_via_nested_calls(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    params: set[str],
    resolve_source,
    seen: frozenset[str],
) -> set[str]:
    """Params of *func* mutated only by being passed to another resolvable call.

    For each ``Name`` call in the body, resolve the callee, recursively find which
    of ITS params it mutates, and map those back to any of *func*'s params passed
    at the matching position / keyword.
    """
    out: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        callee_name = node.func.id
        if callee_name in seen:
            continue  # recursion guard (mutual / self recursion)
        callee = _resolve_function_def(callee_name, resolve_source)
        if callee is None:
            continue
        callee_muts = params_mutated_in_function(
            callee, resolve_source, seen | {callee_name}
        )
        if not callee_muts:
            continue
        pos_params = _positional_param_names(callee)
        for i, arg in enumerate(node.args):
            if (isinstance(arg, ast.Name) and arg.id in params
                    and i < len(pos_params) and pos_params[i] in callee_muts):
                out.add(arg.id)
        for kw in node.keywords:
            if (kw.arg and isinstance(kw.value, ast.Name)
                    and kw.value.id in params and kw.arg in callee_muts):
                out.add(kw.value.id)
    return out


def _resolve_function_def(name, resolve_source):
    """Parse *name*'s source via *resolve_source* into a FunctionDef, or None."""
    if resolve_source is None:
        return None
    source = resolve_source(name)
    if not source:
        return None
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return None
    if parsed.body and isinstance(
        parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return parsed.body[0]
    return None


def params_mutated_in_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    resolve_source=None,
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Parameter names a function body mutates IN PLACE.

    A parameter counts as mutated when the body performs an in-place mutation on
    it — subscript/attribute assignment, augmented assignment, a mutating method
    call, ``out=`` kwarg, or ``del`` — i.e. the same signals as
    :attr:`StatementAnalysis.all_mutated_vars`. Plain reassignment (``x = ...``)
    rebinds a local and does NOT mutate the caller's object, so it does not count.

    Used (with :func:`function_arg_mutations`) to attribute an argument mutation
    back to the caller's variable: ``def f(x): x.append(1)`` plus ``f(data)``
    means ``data`` is mutated in place, so it must reset on isolated re-run
    (CAS-58).

    When *resolve_source* is given (a ``name -> source`` lookup), the analysis is
    interprocedural: a parameter mutated only via a further resolvable call
    (``def outer(y): inner(y)`` where ``inner`` mutates its arg) is also detected
    (CAS-61). *seen* guards against mutual / self recursion. Without
    *resolve_source* the analysis is one level deep (the original CAS-58
    behaviour).
    """
    params = _all_param_names(func)
    if not params:
        return frozenset()
    visitor = _MutationVisitor()
    for stmt in func.body:
        visitor.visit(stmt)
    mutated = {m.variable for m in visitor.mutations} & params
    if resolve_source is not None:
        mutated |= _params_mutated_via_nested_calls(
            func, params, resolve_source, seen
        )
    return frozenset(mutated)


def standalone_call_arg_targets(
    tree: ast.Module | None,
) -> frozenset[tuple[str, tuple[str | None, ...], tuple[tuple[str, str], ...]]]:
    """Top-level bare-``Expr`` calls to a NAME, with their variable arguments.

    Returns ``(func_name, positional, keywords)`` per call:

    * ``positional`` — a tuple with the variable name for each positional argument
      that is a bare ``Name``, or ``None`` for anything else (literal, expression,
      ``*args``) since only a tracked variable can be a reset target.
    * ``keywords`` — ``(param_name, arg_var)`` pairs for keyword arguments whose
      value is a bare ``Name``.

    Only bare-``Expr`` calls (result discarded) are returned: a call made purely
    for effect is the mutation pattern, whereas a pure call captures its result
    (``r = f(x)``). Method calls (``obj.m(x)``) are handled by the method-receiver
    path and excluded here.
    """
    if tree is None:
        return frozenset()
    out: set[tuple[str, tuple[str | None, ...], tuple[tuple[str, str], ...]]] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name):
            continue
        positional = tuple(
            a.id if isinstance(a, ast.Name) else None for a in call.args
        )
        keywords = tuple(
            (kw.arg, kw.value.id)
            for kw in call.keywords
            if kw.arg is not None and isinstance(kw.value, ast.Name)
        )
        out.add((call.func.id, positional, keywords))
    return frozenset(out)


def function_arg_mutations(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Caller variables mutated in place by being passed to a user-defined
    function that mutates the corresponding parameter (CAS-58).

    *resolve_source* maps a function name to its source string (or ``None`` if it
    is not a resolvable user-defined function — a builtin, C function, lambda, or
    unknown name). For each top-level bare-``Expr`` call the function body is
    parsed, its mutated parameters are found via :func:`params_mutated_in_function`
    (interprocedurally — a param mutated only through a further resolvable call is
    detected too, CAS-61), and each is mapped back to the call's positional /
    keyword argument variable.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for func_name, positional, keywords in standalone_call_arg_targets(tree):
        fdef = _resolve_function_def(func_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(
            fdef, resolve_source, frozenset({func_name})
        )
        if not mutated_params:
            continue
        pos_params = _positional_param_names(fdef)
        for i, arg_var in enumerate(positional):
            if arg_var and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(arg_var)
        for param, arg_var in keywords:
            if param in mutated_params:
                out.add(arg_var)
    return frozenset(out)


def _free_vars_mutated_in_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Module-global / free variables a function body mutates in place (CAS-68).

    A name mutated in place (``items.append``, ``store[k]=``, ``g += 1`` under a
    ``global`` declaration) that is neither a parameter nor a plain local
    assignment is a free variable resolved from the enclosing / module scope —
    calling the function mutates that global. Parameter mutations are CAS-58's
    job and are excluded; a name rebound locally (``acc = []`` then
    ``acc.append``) refers to the local and is excluded, UNLESS declared
    ``global`` / ``nonlocal``.
    """
    params = _all_param_names(func)
    global_decls: set[str] = set()
    local_assigned: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            global_decls.update(node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        local_assigned.add(leaf.id)
    visitor = _MutationVisitor()
    for stmt in func.body:
        visitor.visit(stmt)
    mutated = {m.variable for m in visitor.mutations}
    local_assigned -= global_decls
    return frozenset(mutated - params - local_assigned)


def function_global_mutations(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Module globals mutated in place by a called function (CAS-68 part A).

    For each top-level bare-``Expr`` call ``f()`` whose source resolves, find the
    free / global variables ``f`` mutates in place and return them. The checker
    marks these for reset (adds them to the current cell's inputs and mutation
    set) so their producers restore the cell-entry base on an isolated re-run,
    instead of the hidden global accumulating (``g = 0; def bump(): global g;
    g += 1`` + ``bump()`` doubling).
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for func_name, _positional, _keywords in standalone_call_arg_targets(tree):
        fdef = _resolve_function_def(func_name, resolve_source)
        if fdef is None:
            continue
        out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


_MUTABLE_LITERAL_CALLS = frozenset({'list', 'dict', 'set'})


def _is_mutable_default(node: ast.expr) -> bool:
    """True if *node* is a mutable literal default (``[]``, ``{}``, ``set()``)."""
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _MUTABLE_LITERAL_CALLS and not node.args)


def _function_mutates_own_object(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if calling *func* mutates state carried on the function OBJECT itself
    (CAS-68 part B), which persists across calls and must be reset by re-running
    the ``def``:

    * a **mutable default argument** the body mutates in place
      (``def collect(x, acc=[]): acc.append(x)``);
    * an assignment to a **function attribute**
      (``def tick(): tick.count = getattr(tick, 'count', 0) + 1``).
    """
    fname = func.name
    for node in ast.walk(func):
        target = node.target if isinstance(node, ast.AugAssign) else None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == fname):
                    return True
        elif (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == fname):
            return True

    mutated = params_mutated_in_function(func)
    if mutated:
        pos = list(func.args.posonlyargs) + list(func.args.args)
        defs = func.args.defaults
        if defs:
            for param, default in zip(pos[-len(defs):], defs):
                if param.arg in mutated and _is_mutable_default(default):
                    return True
        for kw, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
            if default is not None and kw.arg in mutated and _is_mutable_default(default):
                return True
    return False


def _called_function_names(tree: ast.Module) -> frozenset[str]:
    """Names called as ``name(...)`` anywhere in the cell (bare OR captured)."""
    return frozenset(
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    )


def stateful_self_functions(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Called functions that carry mutable state on their own object (CAS-68 B).

    For each function called in the cell (captured or bare) whose source
    resolves, return its name if calling it mutates state on the function object
    (a mutated mutable default arg, or a function-attribute assignment). The
    checker force-resets these so their ``def`` re-runs and recreates fresh state
    on an isolated re-run, instead of the default / attribute accumulating.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in _called_function_names(tree):
        fdef = _resolve_function_def(name, resolve_source)
        if fdef is not None and _function_mutates_own_object(fdef):
            out.add(name)
    return frozenset(out)


def subscript_view_bindings(tree: ast.Module | None) -> dict[str, str]:
    """Map ``{alias: base}`` for ``alias = base[...]`` subscript bindings (CAS-74).

    A numpy ``base[slice]`` is a VIEW that shares memory with ``base``, so mutating
    the alias (``v += 1``, ``v[i] = x``) mutates ``base`` in place. This is
    pure-AST (base must be a ``Name``); the caller gates on ``base`` actually being
    an ndarray at runtime (a list slice is a COPY, not a view). Scans module-level
    statements incl. control bodies, excludes def/class scopes.
    """
    if tree is None:
        return {}
    out: dict[str, str] = {}
    for node in _module_level_stmts(tree.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)):
            base = _extract_base_name(node.value.value)
            if base:
                out[node.targets[0].id] = base
    return out


def partial_arg_mutations(tree: ast.Module | None, resolve_partial, resolve_source) -> frozenset[str]:
    """Vars mutated through a called ``functools.partial`` binding (CAS-72).

    *resolve_partial* maps a name to ``(target_func_name, [bound_arg_vars])`` for a
    ``p = partial(f, x, y)`` binding, or ``None``. For each partial called in the
    cell, the target ``f``'s in-place param mutations are mapped to the bound
    positional args (``partial(push, shared)`` + ``p('a')`` mutates ``shared``
    because ``push`` mutates its first param), and ``f``'s free/global mutations
    are attributed too (``partial(tick, 1)`` where ``tick`` mutates ``counter``).
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in _called_function_names(tree):
        binding = resolve_partial(name)
        if binding is None:
            continue
        f_name, bound_args = binding
        fdef = _resolve_function_def(f_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(fdef)
        pos_params = _positional_param_names(fdef)
        for i, arg in enumerate(bound_args):
            if arg and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(arg)
        out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


def mutating_partials(tree: ast.Module | None, resolve_partial, resolve_source) -> frozenset[str]:
    """Partial vars called in the cell that bind a MUTATED positional arg (CAS-72).

    ``p = partial(push, shared)`` captures the ``shared`` LIST OBJECT, so resetting
    the ``shared`` name to a fresh list does not help — ``p`` still appends to the
    old object. These partials must be re-created (their ``def`` re-run) so they
    re-bind to the reset arg. A partial whose target mutates only a FREE/global var
    is NOT included: a global is resolved dynamically, so resetting it suffices.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in _called_function_names(tree):
        binding = resolve_partial(name)
        if binding is None:
            continue
        f_name, bound_args = binding
        fdef = _resolve_function_def(f_name, resolve_source)
        if fdef is None:
            continue
        mutated_params = params_mutated_in_function(fdef)
        pos_params = _positional_param_names(fdef)
        for i, arg in enumerate(bound_args):
            if arg and i < len(pos_params) and pos_params[i] in mutated_params:
                out.add(name)
                break
    return frozenset(out)


def reduce_free_mutations(tree: ast.Module | None, resolve_source) -> frozenset[str]:
    """Free/global vars mutated by a function passed to ``functools.reduce`` (CAS-72).

    ``reduce(combine, [1, 2, 3], 0)`` invokes ``combine`` once per element; if
    ``combine`` mutates a free variable (``log.append(b)``) that mutation happens,
    so the free var must reset on isolated re-run.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_reduce = ((isinstance(fn, ast.Name) and fn.id == 'reduce')
                     or (isinstance(fn, ast.Attribute) and fn.attr == 'reduce'))
        if is_reduce and node.args and isinstance(node.args[0], ast.Name):
            fdef = _resolve_function_def(node.args[0].id, resolve_source)
            if fdef is not None:
                out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


def _factory_body_scope(factory: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names bound in the factory's OWN scope: its params + names assigned at its
    direct body level (the locals a returned closure captures)."""
    scope = _all_param_names(factory)
    for stmt in factory.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        scope.add(leaf.id)
        elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and isinstance(stmt.target, ast.Name):
            scope.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.add(stmt.name)
    return scope


def _factory_returns_stateful_closure(factory: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if *factory* defines an inner function that mutates a variable
    captured from the factory's own scope (a ``nonlocal`` cell or a factory-local
    container). Such a closure carries state that persists across calls of the
    returned function and is only reset by re-running the factory (CAS-68 B).
    """
    factory_scope = _factory_body_scope(factory)
    for node in ast.walk(factory):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node is not factory):
            if _free_vars_mutated_in_function(node) & factory_scope:
                return True
    return False


def stateful_closure_vars(tree: ast.Module | None, resolve_var_factory) -> frozenset[str]:
    """Closure variables called in the cell that carry mutable captured state
    (CAS-68 B). *resolve_var_factory* maps a name to the ``FunctionDef`` of the
    factory it was assigned from (``c = make_counter()`` → ``make_counter``'s
    def), or ``None``. A name is flagged when its factory returns a closure that
    mutates factory-local state, so the checker force-resets it and its producer
    (``c = make_counter()``) re-runs to recreate the fresh closure.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for name in _called_function_names(tree):
        factory = resolve_var_factory(name)
        if factory is not None and _factory_returns_stateful_closure(factory):
            out.add(name)
    return frozenset(out)


def crossref_reassigned_vars(tree: ast.Module | None) -> frozenset[str]:
    """Names reassigned from a permutation of their own prior values (CAS-65).

    The swap / rotate / temp-swap family: a variable that is **read** (its
    pre-cell value) and then **reassigned** in the same cell, but not via a plain
    single-target self-accumulation (``x = x + 1``, which the lineage-base reset
    already handles and whose input capture preserves the base). Two shapes, both
    lineage-invisible on isolated re-run — the swapped output's content equals its
    recorded output hash, so no lineage / content signal detects the staleness:

    * **Tuple / list unpack** whose LHS name also appears in the RHS —
      ``a, b = b, a``, ``a, b, c = c, a, b``.
    * **Read-before-write across statements** — a name read in an earlier
      statement and reassigned (``Name`` store) in a later one: the temp-swap
      ``tmp = a; a = b; b = tmp`` (flags ``a`` and ``b``, not ``tmp``).

    A single-statement self-reference (``x = x + 1``, ``total = total + k``) is NOT
    flagged: its read and write are in the same statement, and its cell-entry base
    lineage is preserved, so the existing no-lineage lineage-base reset covers it.
    """
    if tree is None:
        return frozenset()
    flagged: set[str] = set()
    stmts = list(_module_level_stmts(tree.body))

    # (A) element-wise tuple/list swap: ``(t_i) = (v_i)`` where a target name is
    # reused elsewhere in the RHS and is NOT assigned from itself at its own
    # position. Excludes ``df, meta = process(df)`` (RHS is not a literal tuple —
    # ``df``'s new value derives from itself, handled by the reassign-reset path).
    for node in stmts:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(tgt.elts) == len(node.value.elts)):
                continue
            rhs_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for te, ve in zip(tgt.elts, node.value.elts):
                if (isinstance(te, ast.Name) and te.id in rhs_names
                        and not (isinstance(ve, ast.Name) and ve.id == te.id)):
                    flagged.add(te.id)

    # (B) a name READ in an earlier statement and later REASSIGNED from a value
    # that does NOT read the name itself (temp-swap ``tmp = a; a = b; b = tmp``).
    # The self-referential-reassignment exclusion (``df['x']=..; df=df.sort()``)
    # keeps a sequential mutate-then-transform out of the set — its reassignment
    # reads the var and is handled by the existing reassign-reset machinery.
    read_before: set[str] = set()
    for node in stmts:
        if isinstance(node, ast.Assign):
            rhs_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for tgt in node.targets:
                for t in _iter_store_targets(tgt):
                    if (isinstance(t, ast.Name) and t.id in read_before
                            and t.id not in rhs_names):
                        flagged.add(t.id)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                read_before.add(n.id)

    return frozenset(flagged)


def _cell_alias_map(tree: ast.Module) -> dict[str, str]:
    """Map each alias name in the cell to its direct source name (shared object).

    Recognises every binding form that makes the target share the RHS object,
    not just ``y = x``:

    * simple / chained ``Name`` assignment — ``y = x``, ``a = b = x`` (each
      target aliases x);
    * 1:1 tuple / list unpack of a literal — ``(y,) = (x,)``, ``a, b = c, d``
      (element-wise, only ``Name``-to-``Name`` pairs);
    * walrus binding — ``(y := x).append(..)``.

    Bindings inside control-flow bodies (if / for / while / with / try) are
    scanned too — an alias formed in a loop body still shares the object
    (CAS-61). Deferred scopes (def / class) are not descended into. Only a bare
    ``Name`` RHS counts as aliasing; ``y = x.copy()`` / ``y = x[:]`` are copies
    and excluded. Self-binds (``x = x``) are skipped. A ternary
    (``y = x if c else z``) is intentionally not handled here (flow-sensitive,
    two possible sources) — tracked separately.
    """
    alias_map: dict[str, str] = {}

    def _bind(target: ast.AST, value: ast.AST) -> None:
        if (isinstance(target, ast.Name) and isinstance(value, ast.Name)
                and target.id != value.id):
            alias_map[target.id] = value.id

    for node in _module_level_stmts(tree.body):
        # Walrus binding in this statement's own expressions. (Nested control
        # bodies are visited as their own yielded statements, so restrict the
        # walk to NamedExprs that are not themselves inside a deferred scope.)
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr):
                _bind(sub.target, sub.value)
        if not isinstance(node, ast.Assign):
            continue
        # Simple or chained ``Name`` assignment: ``y = x`` / ``a = b = x``.
        if isinstance(node.value, ast.Name):
            for tgt in node.targets:
                _bind(tgt, node.value)
        # 1:1 literal unpack: ``(y,) = (x,)`` / ``a, b = c, d`` (Name pairs only).
        for tgt in node.targets:
            if (isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(tgt.elts) == len(node.value.elts)):
                for te, ve in zip(tgt.elts, node.value.elts):
                    _bind(te, ve)
    return alias_map


def _resolve_alias_root(name: str, alias_map: dict[str, str]) -> str:
    """Follow the (transitive) alias chain from *name* to its root source."""
    cur, seen = name, {name}
    while cur in alias_map and alias_map[cur] not in seen:
        cur = alias_map[cur]
        seen.add(cur)
    return cur


def aliased_sources(tree: ast.Module | None, names) -> frozenset[str]:
    """Root source names that *names* alias via a top-level ``Name = Name`` bind.

    For each name that resolves through the alias map to a different root, return
    that root. Used to extend an existing mutation set (selfref writes, method
    receivers) from an in-cell alias back to the upstream holder it shares an
    object with, so the holder resets on an isolated re-run (CAS-60). Names that
    are not aliases contribute nothing.
    """
    if tree is None or not names:
        return frozenset()
    alias_map = _cell_alias_map(tree)
    if not alias_map:
        return frozenset()
    out: set[str] = set()
    for name in names:
        root = _resolve_alias_root(name, alias_map)
        if root != name:
            out.add(root)
    return frozenset(out)


def alias_mutation_sources(tree: ast.Module | None) -> frozenset[str]:
    """Upstream variables whose object is mutated in place through an alias (CAS-60).

    A bare ``Name = Name`` binding (``y = x``) makes ``y`` share ``x``'s object,
    so a later in-place mutation through ``y`` (``y.append(..)``, ``y[0] += 1``)
    also mutates ``x``. The mutation analysis attributes the change to the alias
    ``y`` — which is created in the cell and has no producer to restore from —
    so the upstream holder ``x`` is never marked for reset and the mutation
    accumulates on an isolated re-run. This resolves each mutated name back
    through the (transitive) alias map and returns the root source names, which
    the checker unions into ``current_cell_mutated`` so the source resets.

    Scope: top-level (``tree.body``) ``Name = Name`` aliases only; the RHS must be
    a bare ``Name`` (``y = x.copy()`` / ``y = x[:]`` are copies, not aliases, and
    are correctly excluded). Flow-insensitive — an alias re-bound before the
    mutation still maps back, but resetting an un-mutated source to its identical
    base is a correctness-safe no-op. A mutated name that is not an alias maps to
    nothing and is left to the existing in-place-mutation reset.
    """
    if tree is None:
        return frozenset()
    alias_map = _cell_alias_map(tree)
    if not alias_map:
        return frozenset()
    try:
        mutated = set(analyze_statement(ast.unparse(tree), None).all_mutated_vars)
    except (SyntaxError, ValueError, TypeError):
        return frozenset()
    return aliased_sources(tree, mutated)


def standalone_method_mutation_receivers(tree: ast.Module | None) -> frozenset[str]:
    """Base variables mutated by a *top-level bare-``Expr``* method call.

    Returns the receiver variable for statements like ``lst.append(x)``,
    ``box.add(1)``, ``box.items.append(1)`` or ``df.dropna(inplace=True)`` —
    a method known to mutate its receiver (``MUTATING_METHODS``) or a pandas
    method invoked with ``inplace=True``.

    These receivers carry no Store target, so ``CodeAnalyzer._FlowVisitor``
    never surfaces them as *outputs* and their lineage is left frozen — a
    cached downstream consumer then serves a stale value after the mutation is
    edited.  Both the runtime (``StatementProcessor.process_statement``) and the
    upstream simulation (``VirtualLineage._update_virtual_lineage``) union this
    set into the statement's outputs so the receiver gets a fresh, source-based
    lineage identically on both sides.

    Scope is deliberately narrow — only ``tree.body`` (top-level) bare ``Expr``
    statements:

    * Mutations inside loops/functions are handled elsewhere (the
      loop-mutation lineage path), so they are excluded to avoid double-bump.
    * A captured result (``r = lst.append(x)``) is an ``Assign``, not a bare
      ``Expr``, and is excluded.
    * Pure standalone calls (``df.head()``) use a method outside the known
      mutating sets and are excluded — so they are never over-invalidated.
    """
    if tree is None:
        return frozenset()
    receivers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        # numpy ``out=`` writes its target in place (works regardless of how the
        # call's own receiver is spelled), so bump the out target's lineage.
        receivers.update(_out_kwarg_target_bases(call))
        if not isinstance(call.func, ast.Attribute):
            continue
        method_name = call.func.attr
        base = _extract_base_name(call.func.value)
        if not base:
            continue
        if method_name in MUTATING_METHODS or (
            method_name in PANDAS_INPLACE_METHODS and _expr_call_inplace_true(call)
        ):
            receivers.add(base)
    return frozenset(receivers)


def standalone_method_call_receivers(tree: ast.Module | None) -> frozenset[tuple[str, str]]:
    """``(receiver_base, method_name)`` for every top-level bare-``Expr`` method call.

    The *broad* candidate set for the precise method-mutation extension: unlike
    :func:`standalone_method_mutation_receivers` it does not filter by method
    name — it returns ``df.head()`` and ``bus.on(fn)`` alike. The runtime then
    classifies each candidate (known-mutating / known-pure / observe-by-content)
    to decide whether the receiver actually mutated; the simulation reads that
    recorded verdict. Scope matches the narrow helper: only ``tree.body``
    (top-level) bare ``Expr`` statements, so loop/function bodies and captured
    results (``r = df.head()``) are excluded.
    """
    if tree is None:
        return frozenset()
    calls: set[tuple[str, str]] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        # numpy ``out=`` target is a candidate receiver (method label ``out=``);
        # it is tier-1 (known-mutating) so the runtime/sim route it directly.
        for out_base in _out_kwarg_target_bases(call):
            calls.add((out_base, 'out='))
        if not isinstance(call.func, ast.Attribute):
            continue
        base = _extract_base_name(call.func.value)
        if base:
            calls.add((base, call.func.attr))
    return frozenset(calls)


def analyze_statement(code: str, tree: ast.Module | None) -> StatementAnalysis:
    """Return a :class:`StatementAnalysis` for *code* using pure-AST analysis.

    No runtime state, no ``user_ns`` access, no annotation parsing.

    Args:
        code: Python source code of the statement.
        tree: Optional pre-parsed AST.  When ``None`` the code is parsed
              here; a :class:`SyntaxError` produces an empty analysis
              rather than raising.
    """
    if tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return StatementAnalysis(
                top_level_mutated_vars=frozenset(),
                all_mutated_vars=frozenset(),
                side_effects=(),
                called_names=frozenset(),
            )

    # --- All mutations (full tree walk) ---
    full_visitor = _MutationVisitor()
    full_visitor.visit(tree)
    all_mutated = frozenset(m.variable for m in full_visitor.mutations)

    # --- Top-level mutations (skip function/class bodies) ---
    top_level_visitor = _MutationVisitor()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        top_level_visitor.visit(node)
    top_level_mutated = frozenset(m.variable for m in top_level_visitor.mutations)

    # --- Side effects ---
    se_visitor = _SideEffectVisitor()
    se_visitor.visit(tree)

    # --- Called bare names (for stateful-call check in caller) ---
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)

    return StatementAnalysis(
        top_level_mutated_vars=top_level_mutated,
        all_mutated_vars=all_mutated,
        side_effects=tuple(se_visitor.effects),
        called_names=frozenset(called),
    )
