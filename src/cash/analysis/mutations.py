"""Which variables a statement changes in place.

Pure AST: the method tables (``MUTATING_METHODS`` and its kin), the mutation
visitor, the self-referential and cross-referencing writes, the receivers of a
bare method call, and the accumulator-loop shapes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .file_effects import _SideEffectVisitor

__all__ = [
    "MUTATING_METHODS",
    "ACCUMULATOR_METHODS",
    "PANDAS_INPLACE_METHODS",
    "KNOWN_PURE_METHODS",
    "RECEIVER_READONLY_WRITE_METHODS",
    "MutationInfo",
    "selfref_inplace_write_vars",
    "subscript_view_bindings",
    "crossref_reassigned_vars",
    "consumed_input_names",
    "standalone_method_mutation_receivers",
    "standalone_method_call_receivers",
    "standalone_method_call_inner_methods",
    "MODULE_SETTING_FUNCTIONS",
    "module_setting_receivers",
    "chain_is_pure",
    "top_level_call_argument_bases",
    "is_pandas_plot_call",
    "assigned_method_call_receivers",
    "selfref_reassignment_targets",
    "accumulator_loop_body_shape",
    "cacheable_accumulator_loop",
]


# ---------------------------------------------------------------------------
# Mutation detection — moved from mutation_detector.py
# ---------------------------------------------------------------------------

# Methods that mutate their receiver object in-place
MUTATING_METHODS = {
    # list methods
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "sort",
    "reverse",
    "clear",
    # dict methods
    "update",
    "popitem",
    "setdefault",
    # set methods
    "add",
    "discard",
    "intersection_update",
    "difference_update",
    "symmetric_difference_update",
}


# The subset of MUTATING_METHODS that GROW a collection element-by-element — the
# classic accumulator loop (``out = []`` then ``for e in it: out.append(f(e))``).
# Only these earn the "rewrite as a comprehension" guidance hint in
# ``StatementAnalysis.skip_reasons``: an accumulator loop has a byte-identical
# comprehension form (``out = [f(e) for e in it]``) that assigns its result and
# therefore caches. Other in-place mutations (``pop``/``sort``/``df['x'] = …``)
# have no such rewrite and must NOT get the hint (part b).
ACCUMULATOR_METHODS = frozenset({"append", "extend", "add", "update"})


# Pandas methods that accept inplace=True
PANDAS_INPLACE_METHODS = {
    "fillna",
    "dropna",
    "drop",
    "rename",
    "reset_index",
    "set_index",
    "sort_values",
    "sort_index",
    "replace",
    "clip",
    "where",
    "mask",
    "drop_duplicates",
    "eval",
    "query",
    "astype",
}


# Read-only inspection / display methods that never mutate their receiver.
# Deliberately conservative: this set only lets the runtime *skip* the
# before/after content observation (a perf optimisation that matters for large
# objects like DataFrames, where hashing twice per standalone call is costly).
# A name here must be unambiguously non-mutating — being wrong means a real
# mutation goes undetected. Anything not listed falls through to observation.
#
# A DataFrame/Series/ndarray cannot be observed (its content hash is a sample),
# so an unlisted method on one is ASSUMED to mutate it and bumps its lineage:
# the last line of a cell showing a frame -- ``comparison.round(4)``,
# ``feat_demo.describe().round(3)`` -- was badged an in-place
# mutation, and editing it re-ran everything built from the frame below it.
# A chain is pure when nothing inside it is known to mutate
# (:func:`chain_is_pure`): ``df.sort_values('x').head()`` leaves ``df``
# alone, ``df.pop('b').round(2)`` does not.
KNOWN_PURE_METHODS = frozenset(
    {
        # pandas / numpy inspection & summary (return a new object, never mutate)
        "head",
        "tail",
        "describe",
        "info",
        "sample",
        "value_counts",
        "nunique",
        "unique",
        "corr",
        "cov",
        "memory_usage",
        "count",
        "isna",
        "isnull",
        "notna",
        "notnull",
        "nlargest",
        "nsmallest",
        "idxmax",
        "idxmin",
        # pandas / numpy arithmetic summaries (an ``out=`` target is tier-1 on its own)
        "round",
        "abs",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "quantile",
        "groupby",
        "agg",
        "aggregate",
        "pivot_table",
        "copy",
        # display / plotting
        "plot",
        "hist",
        "boxplot",
        "show",
    }
)


# pandas ``to_*`` writers: they READ the DataFrame/Series and write it out to a
# file / external sink.  They do NOT mutate the receiver, so they must never bump
# its lineage — the receiver-mutation classifier would otherwise assume-mutate a
# DataFrame receiver (``compute_hash`` samples large frames, so it cannot prove
# purity) and make ``df.to_csv(path)`` a spurious *producer* of ``df``.  That
# spurious edge makes upstream reconstruction re-schedule the write as if to
# rebuild ``df``, re-firing a NON-IDEMPOTENT append (``df.to_csv(log, mode='a')``)
# and corrupting the file.
#
# This governs ONLY the receiver-mutation question.  The file-WRITE side effect
# these calls carry is tracked SEPARATELY (``cash.effects.METHOD_VERBS`` /
# ``statement_writes_files`` / the re-execution planner's writer scheduling), so
# a genuinely-edited writer still re-runs — it just no longer masquerades as a
# mutation of the frame it read.
#
# Deliberately EXCLUDES ``savefig``: its receiver is an identity-coupled
# matplotlib Figure (never cached, always re-derived as a unit), and savefig
# OVERWRITES its PNG (idempotent), so treating it as a Figure mutation carries
# none of the non-idempotent harm — and doing so is load-bearing for the
# carrier-coherence path, which relies on the savefig→fig edge to
# re-derive the chart when the plotted data is edited.  The identity-coupled
# receiver check routes ``fig.savefig(...)`` (like every other Figure/Axes method)
# to the mutation path.  ``save`` / ``write`` / ``writelines`` are likewise
# excluded: they collide with methods on other receiver types (a custom
# ``obj.save()`` may well mutate), and those receivers stay on the observe/assume
# path where a real mutation is still caught.
RECEIVER_READONLY_WRITE_METHODS = frozenset(
    {
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_json",
        "to_feather",
        "to_excel",
        "to_hdf",
        "to_stata",
        "to_sql",
        "to_gbq",
        "to_clipboard",
        "to_html",
        "to_markdown",
        "to_latex",
    }
)


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


def _extract_receiver_base_name(node: ast.AST) -> str | None:
    """Root variable of a METHOD-CALL RECEIVER, or ``None`` if it has no variable.

    Differs from :func:`_extract_base_name` in exactly one case: a receiver that
    is a constructor/factory call spelled as a bare name — ``open(p, 'a')`` in
    ``open(p, 'a').write(x)``, or ``Path(p)`` in ``Path(p).write_text(x)``.
    :func:`_extract_base_name` walks the Call and returns the CALLEE (``open``),
    but the callee is not the receiver: the call builds a NEW object that no
    variable is bound to, so there is no receiver lineage to bump. Booking that
    as a mutation of ``open`` made the writer statement re-execute during
    upstream reconstruction, and because the write is a ``mode='a'`` append,
    re-execution DUPLICATED the line on disk.

    A chained call on a real variable — the documented
    ``groups.setdefault(k, []).append(v)`` intent — still resolves to ``groups``:
    it descends through the Attribute branch, which is retained.

    Used only by the method-mutation receiver helpers, so the broader
    :func:`_extract_base_name` behaviour its other callers rely on is unchanged.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return _extract_receiver_base_name(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _extract_receiver_base_name(node.func.value)
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
        every call site is covered, not just top-level ``Expr`` statements.
        """
        if not isinstance(call.func, ast.Attribute):
            return
        method_name = call.func.attr
        base = _extract_receiver_base_name(call.func.value)
        if not base:
            return
        if method_name in MUTATING_METHODS:
            self.mutations.append(
                MutationInfo(
                    variable=base,
                    method=method_name,
                    kind="method_call",
                    line=lineno,
                )
            )
        elif method_name in PANDAS_INPLACE_METHODS:
            for kw in call.keywords:
                if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.mutations.append(
                        MutationInfo(
                            variable=base,
                            method=method_name,
                            kind="inplace_kwarg",
                            line=lineno,
                        )
                    )
                    break

    def visit_Call(self, node: ast.Call) -> None:
        """Detect KNOWN-mutating method calls (anywhere) and the numpy ufunc
        ``out=`` kwarg, which writes its target array in place: ``np.add(a, 10,
        out=a)`` mutates ``a`` (the out target), not the ``np`` receiver. Fires on
        any Call (result captured or not). For multi-output ufuncs ``out`` is a
        tuple: ``out=(q, r)``."""
        self._record_method_mutation(node, node.lineno)
        for kw in node.keywords:
            if kw.arg != "out":
                continue
            targets = kw.value.elts if isinstance(kw.value, (ast.Tuple, ast.List)) else [kw.value]
            for tgt in targets:
                base = _extract_base_name(tgt)
                if base:
                    self.mutations.append(
                        MutationInfo(
                            variable=base,
                            method="out=",
                            kind="out_kwarg",
                            line=node.lineno,
                        )
                    )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Detect augmented assignments like x += 1, arr *= 2."""
        base = _extract_base_name(node.target)
        if base:
            op_name = type(node.op).__name__
            self.mutations.append(
                MutationInfo(
                    variable=base,
                    method=f"__i{op_name.lower()}__",
                    kind="augmented_assign",
                    line=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect subscript/attribute assignments like d[key] = val, obj.attr = val,
        including those nested in a tuple/list target (df['a'], df['b'] = ...)."""
        for target in node.targets:
            for store in _iter_store_targets(target):
                if isinstance(store, ast.Subscript):
                    base = _extract_base_name(store.value)
                    if base:
                        self.mutations.append(
                            MutationInfo(
                                variable=base,
                                method="__setitem__",
                                kind="subscript_assign",
                                line=node.lineno,
                            )
                        )
                elif isinstance(store, ast.Attribute):
                    base = _extract_base_name(store.value)
                    if base:
                        self.mutations.append(
                            MutationInfo(
                                variable=base,
                                method=f"__setattr__({store.attr})",
                                kind="attribute_assign",
                                line=node.lineno,
                            )
                        )
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Detect del d[key], del lst[0]."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(
                        MutationInfo(
                            variable=base,
                            method="__delitem__",
                            kind="subscript_delete",
                            line=node.lineno,
                        )
                    )
        self.generic_visit(node)


def _expr_call_inplace_true(call: ast.Call) -> bool:
    """Return True if *call* passes ``inplace=True`` as a keyword."""
    for kw in call.keywords:
        if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
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
        if kw.arg != "out":
            continue
        targets = kw.value.elts if isinstance(kw.value, (ast.Tuple, ast.List)) else [kw.value]
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


#: What ``ast.unparse`` raises on a node it cannot render: a hand-built node
#: missing a field, or one nested past the recursion limit.
_UNPARSE_ERRORS = (AttributeError, TypeError, ValueError, RecursionError)


def _rhs_reads_target(rhs: ast.expr, target: ast.expr) -> bool:
    """True if *rhs* reads the exact same subscript/attribute expression as *target*
    (e.g. ``df['a']`` appears in the RHS of ``df['a'] = df['a'] * 2``)."""
    try:
        tgt = ast.unparse(target)
    except _UNPARSE_ERRORS:
        return False
    for sub in ast.walk(rhs):
        if isinstance(sub, (ast.Subscript, ast.Attribute)):
            try:
                if ast.unparse(sub) == tgt:
                    return True
            except _UNPARSE_ERRORS:
                continue
    return False


# Accessor attributes that index by POSITION (no recoverable column name).
_POSITIONAL_ACCESSORS = frozenset({"iloc", "iat"})


# Accessor attributes that index by LABEL; the column is the last slice element.
_LABEL_ACCESSORS = frozenset({"loc", "at"})


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
    preserving the derived-column cache.
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
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(node, field, None)
            if nested:
                yield from _module_level_stmts(nested)
        for handler in getattr(node, "handlers", []):  # try/except handler bodies
            yield from _module_level_stmts(handler.body)


def _target_key_grows_receiver(target: ast.expr, base: str) -> bool:
    """True if a subscript target's KEY is a size-dependent index of *base* — the
    row-append idiom ``df.loc[len(df)] = ..`` / ``df.loc[df.shape[0]] = ..``.

    Such a write ADDS a row at a position derived from the receiver's own size, so
    it is non-idempotent (re-running grows the frame again) and the receiver must
    reset. Scoped to ``len(base)`` / ``base.shape`` / ``base.size`` in the key, so
    a masked write whose key merely reads the frame (``df.loc[df['a'] > 0, 'b'] =
    5``, idempotent) is NOT flagged."""
    if not isinstance(target, ast.Subscript):
        return False
    for sub in ast.walk(target.slice):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "len"
            and any(_extract_base_name(a) == base for a in sub.args)
        ):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in ("shape", "size") and _extract_base_name(sub.value) == base:
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
      ``obj.attr = obj.attr + 1``;
    * MASKED writes whose RHS reads the same column spelled differently
      (``df.loc[mask, 'a'] = df['a']*2``) — matched by column-key overlap, not
      exact text (see :func:`_rhs_reads_same_column`);
    * tuple/list unpacking that reads & writes overlapping columns
      (``df['a'], df['b'] = df['b'], df['a']`` — a column swap);
    * ``del`` of a subscript/attribute (``del df['b']``, ``del obj.cache``) — a
      second ``del`` raises, so the receiver must reset;
    * any of the above nested in an if/for/while/with body
      (``if cond: df['a'] = df['a']*2``) — scanned via :func:`_module_level_stmts`
      (the reset itself uses the live value's lineage, which survives the
      simulator's control-structure collapse).

    Such writes are NON-IDEMPOTENT, so on an isolated cell re-run the lineage-
    carrying receiver (DataFrame/Series/custom object) must be restored first —
    otherwise the value accumulates (``df['a']*2`` doubles again) or the re-run
    errors. The caller routes these vars through the same stale-value reset used
    for method receivers.

    Deliberately EXCLUDES writes to a NEW target read from OTHER keys
    (``df['b'] = df['a'] + 1``, ``df['VolAdj'] = df.groupby('Ticker')['Close']…``,
    ``df['c'], df['d'] = df['a'], df['b']``): those are idempotent on re-run and
    keep their per-statement cache, preserving the design. Augmented
    assignment (``+=``) is always self-referential. Scans module-level statements
    including those nested in if/for/while/with bodies but NOT inside
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
                elts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
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


def subscript_view_bindings(tree: ast.Module | None) -> dict[str, str]:
    """Map ``{alias: base}`` for ``alias = base[...]`` subscript bindings.

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
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
        ):
            base = _extract_base_name(node.value.value)
            if base:
                out[node.targets[0].id] = base
    return out


def crossref_reassigned_vars(tree: ast.Module | None) -> frozenset[str]:
    """Names reassigned from a permutation of their own prior values.

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
            if not (
                isinstance(tgt, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(tgt.elts) == len(node.value.elts)
            ):
                continue
            rhs_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for te, ve in zip(tgt.elts, node.value.elts):
                if (
                    isinstance(te, ast.Name)
                    and te.id in rhs_names
                    and not (isinstance(ve, ast.Name) and ve.id == te.id)
                ):
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
                    if isinstance(t, ast.Name) and t.id in read_before and t.id not in rhs_names:
                        flagged.add(t.id)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                read_before.add(n.id)

    return frozenset(flagged)


# ---------------------------------------------------------------------------
# Consumption detection (consumable / producer-re-execution engine)
# ---------------------------------------------------------------------------

# Builtins that merely *inspect* a name without advancing it. Everything else
# that receives the name as an argument is assumed to consume it — see
# ``consumed_input_names``.
_NON_CONSUMING_FUNCS = frozenset(
    {
        "type",
        "id",
        "repr",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "callable",
        "hash",
        "dir",
        "vars",
        "print",
        "len",
        "format",
    }
)


# Methods that report on a consumable without drawing from it. Being wrong here
# means a genuine consumption goes undetected (the producer is not re-run and
# the stale-value bug survives), so the set stays small and unambiguous.
_NON_CONSUMING_METHODS = frozenset(
    {
        # queue.Queue / SimpleQueue introspection
        "qsize",
        "empty",
        "full",
        "task_done",
        "join",
        # file-handle introspection (``seek``/``tell`` do not read bytes; ``seek``
        # in particular REWINDS, but the divergence probe compares positions and a
        # rewound handle legitimately reads from the new offset)
        "tell",
        "fileno",
        "seekable",
        "readable",
        "writable",
        "flush",
        "isatty",
    }
)


def consumed_input_names(tree: ast.Module | None) -> frozenset[str]:
    """Names this code actually consumes (drains / advances), not merely reads.

    Used to scope the consumable producer-re-execution channel to inputs the
    re-run cell really draws from: ``for x in g``, ``list(g)`` / ``sum(g)`` /
    ``next(g)``, comprehensions, ``q.get()``, ``fh.read()``. Reads that leave
    the object where it stands — ``q.qsize()``, ``q.empty()``, ``type(g)`` —
    must NOT count, or a cell that merely inspects a consumable would re-run its
    producer for nothing.

    **This is a cost / side-effect guard, not a correctness guard.** Re-running
    a producer matches ``run_all`` semantics either way, so an over-broad answer
    only wastes work while an over-narrow one would let the bug through. The
    analysis is therefore deliberately conservative in the *consuming*
    direction: a name is treated as consumed unless every one of its
    occurrences sits in a recognised non-consuming position. That makes an
    opaque ``foo(g)`` count as consumption (correct for ``foo = list``, merely
    redundant for a ``foo`` that only inspects).
    """
    if tree is None:
        return frozenset()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _occurrence_consumes(name_node: ast.Name) -> bool:
        parent = parents.get(name_node)
        if parent is None:
            return True
        # ``type(g)`` / ``print(g)`` — inspected, not drawn from.
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id in _NON_CONSUMING_FUNCS
            and name_node is not parent.func
        ):
            return False
        if isinstance(parent, ast.Attribute):
            grand = parents.get(parent)
            # ``q.qsize()`` / ``fh.tell()`` — receiver of a reporting method.
            if isinstance(grand, ast.Call) and grand.func is parent and parent.attr in _NON_CONSUMING_METHODS:
                return False
            # A bare attribute read (``g.gi_frame``, ``q.maxsize``) never draws.
            if not isinstance(grand, ast.Call):
                return False
            # ``q.get()`` / ``fh.read()`` / ``g.send(..)`` — consuming method.
            return True
        return True

    consumed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if _occurrence_consumes(node):
                consumed.add(node.id)
    return frozenset(consumed)


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
        base = _extract_receiver_base_name(call.func.value)
        if not base:
            continue
        if method_name in MUTATING_METHODS or (method_name in PANDAS_INPLACE_METHODS and _expr_call_inplace_true(call)):
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
            calls.add((out_base, "out="))
        if not isinstance(call.func, ast.Attribute):
            continue
        base = _extract_receiver_base_name(call.func.value)
        if base:
            method = call.func.attr
            # ``df.plot.bar(...)``: label it by the accessor, so the classifiers
            # can tell pandas' plotting from a method named ``bar``.
            if isinstance(call.func.value, ast.Attribute) and call.func.value.attr == "plot":
                method = f"plot.{method}"
            calls.add((base, method))
    return frozenset(calls)


def standalone_method_call_inner_methods(
    tree: ast.Module | None,
) -> dict[tuple[str, str], frozenset[str]]:
    """The methods called INSIDE each receiver of
    :func:`standalone_method_call_receivers`, keyed like its pairs.

    ``feat_demo.describe().round(3)`` is the pair ``('feat_demo', 'round')``;
    this says ``describe`` ran on ``feat_demo`` on the way (see
    :func:`chain_is_pure`). Two statements with the same pair merge their
    inner methods, the conservative way.
    """
    if tree is None:
        return {}
    inner: dict[tuple[str, str], frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not isinstance(func, ast.Attribute):
            continue
        base = _extract_receiver_base_name(func.value)
        if not base:
            continue
        methods: set[str] = set()
        receiver = func.value
        while True:
            if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
                methods.add(receiver.func.attr)
                receiver = receiver.func.value
            elif isinstance(receiver, (ast.Attribute, ast.Subscript)):
                receiver = receiver.value
            else:
                break
        method = func.attr
        if isinstance(func.value, ast.Attribute) and func.value.attr == "plot":
            method = f"plot.{method}"
        inner[(base, method)] = inner.get((base, method), frozenset()) | methods
    return inner


#: Module functions that change a setting the module keeps: ``pd.set_option``,
#: ``plt.style.use``, ``np.seterr``, ``warnings.filterwarnings``. Matched with
#: :func:`module_setting_receivers`, which also takes ``set`` and any ``set_*``.
MODULE_SETTING_FUNCTIONS = frozenset(
    {
        "use",
        "rc",
        "seterr",
        "filterwarnings",
        "simplefilter",
        "resetwarnings",
        "basicConfig",
        "reset_option",
    }
)


def module_setting_receivers(tree: ast.Module | None) -> frozenset[str]:
    """Names a top-level bare call changes a module's setting through.

    ``plt.rcParams.update({...})``, ``sys.path.append(p)``: a mutating method on
    something a name holds -- which, when the name is a module, is that
    module's state. ``pd.set_option(...)``, ``plt.style.use(...)``: a function
    that sets. Returned by NAME; only a caller with the namespace can tell the
    name is a module, and only then does this apply (``np.append(a, 1)`` is
    neither: a function on the module itself, returning a new array).

    Such a statement binds nothing, so rebuilding variables after a restart
    never reached it: charts came out in matplotlib's default style
    because ``plt.rcParams.update`` in the setup cell was not replayed.
    Counted as a change to the module, it is replayed with the import.
    """
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not isinstance(func, ast.Attribute):
            continue
        base = _extract_receiver_base_name(func.value)
        if not base:
            continue
        method = func.attr
        on_attribute = isinstance(func.value, ast.Attribute)
        if (
            (on_attribute and method in MUTATING_METHODS)
            or method == "set"
            or method.startswith("set_")
            or method in MODULE_SETTING_FUNCTIONS
        ):
            names.add(base)
    return frozenset(names)


def chain_is_pure(method: str, inner: frozenset[str]) -> bool:
    """Whether a call labelled *method*, with *inner* called on the way, is
    known not to mutate its receiver.

    The last method must be known pure, and nothing inside the chain known
    to mutate: ``df.pop('b').round(2)`` is labelled ``round`` and still
    removes a column. An unlisted inner method is let through, as it always
    was -- ``df.sort_values('x').head()`` -- because a method that mutates in
    place almost always returns ``None`` and cannot be chained; asking that
    every inner method be listed made ``vs_plan.sort_values(...).head()``
    an assumed mutation, and a restart rebuilt the frame from 1,312 files.

    A known-pure method inside the chain counts too: it returns a new object,
    and what follows acts on that. ``dwells[m].groupby('hour').size()`` was a
    mutation of ``dwells`` because ``size`` is not listed,
    and a repair re-ran it and everything built from ``dwells``.
    """
    if inner & MUTATING_METHODS:
        return False
    return method in KNOWN_PURE_METHODS or bool(inner & KNOWN_PURE_METHODS)


def _argument_root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def top_level_call_argument_bases(tree: ast.Module | None) -> frozenset[str]:
    """Names handed as arguments to the top-level call of each statement.

    ``tot.plot(ax=axes[0])`` -> ``{'axes'}``; ``bars = df.plot.bar(ax=ax)`` ->
    ``{'ax'}``. The runtime and the simulation route such an argument as
    MUTATED when it is a live Axes/Figure: a plotting call draws on the axes
    it is given. Keyed on the receiver alone, ``imp.plot.barh(..., ax=ax)``
    looked like a pure call on ``imp``, was served from cache during a replay,
    and the re-created figure was saved blank.
    """
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Call):
            continue
        for arg in [*value.args, *(kw.value for kw in value.keywords)]:
            root = _argument_root(arg)
            if root:
                names.add(root)
    return frozenset(names)


_PANDAS_PLOT_METHODS = frozenset({"plot", "hist", "boxplot"})


def is_pandas_plot_call(method: str, receiver: object) -> bool:
    """``df.plot(...)``, ``df.plot.bar(...)``, ``df.hist()``, ``df.boxplot()``
    on a pandas object: it draws on an Axes and leaves the data alone.

    A DataFrame cannot be content-observed (its hash samples), so any unknown
    method on one was ASSUMED to mutate it. That bumped ``data``'s lineage for
    ``data.groupby('region')['churn'].mean().plot.bar(ax=ax)``, and every cell
    reading ``data`` above it then re-ran its producers with nothing changed.
    The Axes it draws on is what
    changes, and the carrier-history pass follows it through ``ax=``.
    Shared by the runtime and the simulation, which must decide identically.
    """
    if not (method in _PANDAS_PLOT_METHODS or method.startswith("plot.")):
        return False
    return (type(receiver).__module__ or "").startswith("pandas")


def assigned_method_call_receivers(tree: ast.Module | None) -> frozenset[tuple[str, str]]:
    """``(receiver_base, method_name)`` for method calls on the RHS of a top-level assignment.

    The captured-return companion to :func:`standalone_method_call_receivers`.
    That helper only sees a bare-``Expr`` method call, so it misses the far more
    common form that BINDS the return value —
    ``counts, bins, patches = ax.hist(data)`` (an ``ast.Assign``) or
    ``h: BarContainer = ax.hist(data)`` (an ``ast.AnnAssign``). Such a statement
    BOTH draws on ``ax`` (the bars — a live-Axes mutation that cannot be replayed
    from the cached tuple) AND binds a value; caching the tuple while skipping the
    draw is the exact incoherence killed here, one statement-shape further out.

    The whole RHS value is walked, so a draw nested in a larger expression
    (``h = [ax.hist(d) for d in data]``) is caught too. The runtime and the
    simulation route ONLY the identity-coupled (live Axes/Figure) receivers from
    this set — keyed on the RECEIVER, exactly as the bare-``Expr`` path is — so a
    genuine pure capture on an ordinary receiver (``m = df.mean()``) is never
    routed and still caches. Scope is ``tree.body`` (top-level) assignments only,
    matching the bare-``Expr`` helper.
    """
    if tree is None:
        return frozenset()
    calls: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value: ast.AST | None = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value  # None for a bare annotation (``x: int``)
        else:
            continue
        if value is None:
            continue
        for sub in ast.walk(value):
            if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
                continue
            base = _extract_receiver_base_name(sub.func.value)
            if base:
                calls.add((base, sub.func.attr))
    return frozenset(calls)


def selfref_reassignment_targets(node: ast.AST) -> frozenset[str]:
    """Names a single statement *reassigns to itself* — the accumulator shape.

    A self-referential reassignment accumulator rebinds a name from an
    expression that reads that same name, so its value depends on its own prior
    value across loop iterations (``total = total + b``, ``total = f(total)``,
    or the augmented ``total += b``).  Unlike an in-place mutation
    (``results.append(x)``) this leaves no ``Store``-on-a-container trace, so
    :attr:`StatementAnalysis.all_mutated_vars` never surfaces it — which is why
    such accumulators were wrongly excluded from the loop-trust set and
    re-executed (re-draining one-shot iterables) on every downstream read.


    Detected shapes (single leaf statement only):

    * ``ast.AugAssign`` whose target is a bare ``Name`` (``total += b``).
    * ``ast.Assign`` to a *single* bare ``Name`` target where that same name
      appears as a ``Load`` anywhere in the RHS (``total = total + b``,
      ``total = f(total)``).

    A plain, non-self-referential rebinding (``x = g(i)``) is deliberately
    excluded: trusting it would under-invalidate (the loop could legitimately
    produce a different ``x`` when an upstream input changes, yet a trusted
    ``x`` would be served stale).  Tuple / multi-target / attribute / subscript
    targets are excluded for the same reason.

    Both the runtime loop-mutation collector
    (``control_structures.helpers.find_potentially_mutated_variables``) and the
    simulation collector (``VirtualLineage._find_loop_mutated_vars``) call this
    on each leaf body statement, so the two classify identically (unified-key
    rule).
    """
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
        return frozenset({node.target.id})
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        target_name = node.targets[0].id
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Name) and sub.id == target_name and isinstance(sub.ctx, ast.Load):
                return frozenset({target_name})
    return frozenset()


# ---------------------------------------------------------------------------
# Accumulator-loop shape detection
# ---------------------------------------------------------------------------
#
# History: this used to be consulted directly by a dispatcher in
# ``control_structures/processor.py`` that routed a matching loop through the
# statement cache as one unit BEFORE the cost-based
# ``_should_execute_loop_as_single_unit`` check ever ran -- so every
# accumulator loop, however cheap, skipped per-iteration decomposition and
# interception. Deleting that dispatch was too broad:
# a follow-up review (measured on a 150-iteration, 4.6s-body loop)
# found that above the cost check's own single-unit threshold (>50
# iterations, >1s estimated overhead), NEITHER mechanism caches anymore --
# decomposition never runs (the cost check chose single-unit), and the
# chosen single-unit branch is refused outright by the statement cache's
# in-place-mutation detector, because nothing was suppressing that refusal.
# ``force_outputs`` (passed by ``ForLoopHandler`` at its single-unit branch,
# ``for_handler.py``) is what suppresses it -- so this detector is consulted
# again, but now from INSIDE the cost path, purely to compute that
# ``force_outputs`` set. It is no longer a dispatch decision of its own: the
# cost check alone decides single-unit vs. decompose in both directions.

# Accumulator method -> the empty-seed kind(s) that legitimately seed it.
# ``append``/``extend`` grow a list; ``add`` grows a set; ``update`` grows a
# dict OR a set (both define ``update``), so it accepts either seed.
_ACCUMULATOR_SEED_KINDS: dict[str, frozenset[str]] = {
    "append": frozenset({"list"}),
    "extend": frozenset({"list"}),
    "add": frozenset({"set"}),
    "update": frozenset({"dict", "set"}),
}


def _empty_seed_kind(node: ast.expr) -> str | None:
    """Container kind a FRESH-EMPTY seed expression *node* produces, or ``None``.

    Recognises only the empty seed forms: ``[]`` / ``list()`` -> ``'list'``,
    ``{}`` / ``dict()`` -> ``'dict'``, ``set()`` -> ``'set'``. A non-empty
    literal (``[0]``, ``{1}``, ``{'k': 1}``) or any computed expression returns
    ``None`` so a pre-seeded accumulator is never matched.
    """
    if isinstance(node, ast.List) and not node.elts:
        return "list"
    if isinstance(node, ast.Dict) and not node.keys:
        return "dict"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args and not node.keywords:
        return {"list": "list", "dict": "dict", "set": "set"}.get(node.func.id)
    return None


def _simple_loop_target_names(target: ast.expr) -> list[str] | None:
    """Loop-target names when *target* is a bare ``Name`` or a (possibly nested)
    tuple/list of bare ``Name``s, else ``None``.

    Returns ``None`` the moment a leaf is not a bare ``Name`` — a starred target
    (``for a, *rest in ...``), a subscript, or an attribute — because
    capturing/restoring the leaked loop variable(s) on a cache hit requires
    enumerating EVERY name the loop binds. Bailing keeps the namespace on a hit
    byte-identical to running the real loop.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            sub = _simple_loop_target_names(elt)
            if sub is None:
                return None
            names.extend(sub)
        return names
    return None


def _expr_has_side_effects_or_foreign_mutation(expr: ast.expr, acc: str) -> bool:
    """True if the append-argument *expr* writes files or mutates any variable
    other than the accumulator *acc*.

    Reuses the module's own file-write scanner (:class:`_SideEffectVisitor`) and
    mutation scanner (:class:`_MutationVisitor`) so the accumulator fast path
    refuses exactly the inline effects the per-statement pipeline would refuse —
    ``acc.append(f.write(x))``, ``acc.append(other.pop())``. Effects hidden
    inside a called function's body are not visible here; those are caught by the
    pipeline's ``@stateful`` / forbidden-function scan (the loop still routes
    through :func:`decide_cacheability`), matching the semantics of the
    byte-identical comprehension form.
    """
    se = _SideEffectVisitor()
    se.visit(expr)
    if se.effects:
        return True
    mv = _MutationVisitor()
    mv.visit(expr)
    return any(m.variable != acc for m in mv.mutations)


def accumulator_loop_body_shape(
    for_node: ast.For,
) -> tuple[str, tuple[str, ...]] | None:
    """Detect the accumulator shape from the BODY alone, ignoring any seed.

    Returns ``(acc, loop_vars)`` or ``None``. This is
    :func:`cacheable_accumulator_loop` minus its requirement (2), the
    fresh-empty-seed check on the immediately-preceding sibling.

    **Why dropping (2) is safe here and not there.** Requirement (2) exists
    because caching a loop that appends to an ALREADY-POPULATED accumulator
    would drop or double its prefix -- the entry cannot know what the
    accumulator held going in. The one caller is a split loop's tail
    (``for_handler``), whose accumulator is populated by its own head, which
    re-runs deterministically immediately before it on every pass. The prefix
    is therefore reproduced rather than assumed, and the tail's key includes
    the accumulator's post-head lineage -- so a head that produced anything
    different misses instead of restoring over it.

    Do NOT reach for this anywhere the prefix is not re-derived that way; use
    :func:`cacheable_accumulator_loop`, which refuses the case outright.

    Deliberately a sibling rather than a generalisation of that function: its
    exact shape gate is depended on by a large body of integration tests, and
    this duplication is cheaper than the risk of changing its behaviour.
    """
    if for_node.orelse or len(for_node.body) != 1:
        return None
    stmt = for_node.body[0]
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)):
        return None
    acc = call.func.value.id
    if call.func.attr not in _ACCUMULATOR_SEED_KINDS:
        return None
    loop_vars = _simple_loop_target_names(for_node.target)
    if not loop_vars:
        return None
    for child in ast.walk(stmt):
        if isinstance(child, (ast.Break, ast.Continue)):
            return None
    for arg in (*call.args, *(kw.value for kw in call.keywords)):
        if _expr_has_side_effects_or_foreign_mutation(arg, acc):
            return None
    return acc, tuple(loop_vars)


def cacheable_accumulator_loop(
    for_node: ast.For,
    prev_node: ast.stmt | None,
) -> tuple[str, tuple[str, ...], ast.expr, ast.Call] | None:
    """Detect the NARROW cacheable accumulator-loop shape.

    Returns ``(acc, loop_vars, iterable_node, expr_call)`` when *for_node* is a
    pure accumulator loop seeded by its immediately-preceding sibling
    *prev_node*, else ``None``. A pure accumulator loop (``out = []`` then
    ``for e in it: out.append(f(e))``) is byte-identical to a comprehension yet
    is refused caching today because the ``append`` reads as an in-place
    mutation; matching this shape lets the caller compute the ``force_outputs``
    that make the whole loop cacheable as one unit, capturing BOTH the
    accumulator and the leaked loop variable as outputs.

    ALL of the following are required; anything else returns ``None`` so the
    caller falls back to today's per-iteration behaviour:

    1. ``for_node.body`` is EXACTLY one bare ``Expr(Call(Attribute(Name(acc),
       meth, ...)))`` with ``meth in {'append','extend','add','update'}`` and no
       ``for``/``else`` clause.
    2. *prev_node* is ``acc = <fresh empty seed>`` whose seed kind matches the
       method (``[]``/``list()`` for append/extend, ``set()`` for add,
       ``{}``/``dict()``/``set()`` for update). A non-empty or prior-cell seed is
       rejected — caching a partial accumulator would drop or double its prefix.
    3. No ``break``/``continue`` (guaranteed by (1); re-checked defensively).
    4. The append argument(s) write no files and mutate no variable but *acc*.
    5. The loop target is a bare ``Name`` or a tuple/list of bare ``Name``s.
    """
    # (2) The preceding sibling must be ``acc = <empty seed>`` — a single bare
    # Name target bound to a fresh-empty container.
    if not (
        isinstance(prev_node, ast.Assign) and len(prev_node.targets) == 1 and isinstance(prev_node.targets[0], ast.Name)
    ):
        return None
    acc = prev_node.targets[0].id
    seed_kind = _empty_seed_kind(prev_node.value)
    if seed_kind is None:
        return None

    # (1) Body is exactly one bare-Expr accumulator-method call on ``acc``, with
    # no for/else clause.
    if for_node.orelse or len(for_node.body) != 1:
        return None
    stmt = for_node.body[0]
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    if not (
        isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == acc
    ):
        return None
    meth = call.func.attr
    seeds_for_method = _ACCUMULATOR_SEED_KINDS.get(meth)
    if seeds_for_method is None or seed_kind not in seeds_for_method:
        return None

    # (5) Loop target is a simple Name / tuple of Names.
    loop_vars = _simple_loop_target_names(for_node.target)
    if not loop_vars:
        return None

    # (3) No break/continue (the one-Expr body cannot contain them, but a nested
    # comprehension/lambda could technically parse; re-check to be safe).
    for child in ast.walk(stmt):
        if isinstance(child, (ast.Break, ast.Continue)):
            return None

    # (4) The append argument(s) must be side-effect-free and mutate nothing but
    # ``acc`` — a file write or a foreign mutation makes the loop uncacheable.
    for arg in (*call.args, *(kw.value for kw in call.keywords)):
        if _expr_has_side_effects_or_foreign_mutation(arg, acc):
            return None

    return acc, tuple(loop_vars), for_node.iter, call
