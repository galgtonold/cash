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
from dataclasses import dataclass

__all__ = [
    # Primary API
    "StatementAnalysis",
    "analyze_statement",
    "standalone_method_mutation_receivers",
    "standalone_method_call_receivers",
    "selfref_inplace_write_vars",
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


class _MutationVisitor(ast.NodeVisitor):
    """AST visitor that collects :class:`MutationInfo` entries."""

    def __init__(self) -> None:
        self.mutations: list[MutationInfo] = []

    def visit_Expr(self, node: ast.Expr) -> None:
        """Detect standalone method calls like lst.append(x)."""
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                method_name = call.func.attr
                base = _extract_base_name(call.func.value)

                if base and method_name in MUTATING_METHODS:
                    self.mutations.append(MutationInfo(
                        variable=base, method=method_name,
                        kind='method_call', line=node.lineno,
                    ))

                if base and method_name in PANDAS_INPLACE_METHODS:
                    for kw in call.keywords:
                        if (
                            kw.arg == 'inplace'
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            self.mutations.append(MutationInfo(
                                variable=base, method=method_name,
                                kind='inplace_kwarg', line=node.lineno,
                            ))
                            break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect the numpy ufunc ``out=`` kwarg, which writes its target array
        in place: ``np.add(a, 10, out=a)`` mutates ``a`` (the out target), not
        the ``np`` receiver. Fires on any Call (result captured or not). For
        multi-output ufuncs ``out`` is a tuple: ``out=(q, r)``."""
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
        """Detect subscript/attribute assignments like d[key] = val, obj.attr = val."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method='__setitem__',
                        kind='subscript_assign', line=node.lineno,
                    ))
            elif isinstance(target, ast.Attribute):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method=f'__setattr__({target.attr})',
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


def selfref_inplace_write_vars(tree: ast.Module | None) -> frozenset[str]:
    """Base vars mutated by a SELF-REFERENTIAL in-place subscript/attribute write
    at the top level — the written target is also *read* in the same statement.

    Covers ``df['a'] = df['a'] * 2``, ``df['a'] += 1``, ``df.iloc[i, j] += x``,
    ``df.iloc[i, j] = df.iloc[i, j] + 1``, ``df['a'] = df['a'].fillna(0)``,
    ``obj.attr = obj.attr + 1``.

    Such writes are NON-IDEMPOTENT: re-running them applies the operation again,
    so on an isolated cell re-run the lineage-carrying receiver (DataFrame/Series/
    custom object) must be restored to its cell-entry base first — otherwise the
    value accumulates (``df['a']*2`` doubles again). The caller routes these vars
    through the same stale-value reset used for method receivers (see CAS-54).

    Deliberately EXCLUDES writes to a NEW target read from OTHER keys
    (``df['b'] = df['a'] + 1``, ``df['VolAdj'] = df.groupby('Ticker')['Close']…``):
    those are idempotent on re-run and keep their per-statement cache, preserving
    the CAS-42 design. Augmented assignment (``+=``) is always self-referential.
    Only ``tree.body`` (top-level); loop/function bodies are handled elsewhere.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AugAssign):
            base = _selfref_target_base(node.target)
            if base:
                out.add(base)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                base = _selfref_target_base(target)
                if base and _rhs_reads_target(node.value, target):
                    out.add(base)
    return frozenset(out)


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
