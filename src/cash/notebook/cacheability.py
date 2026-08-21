from __future__ import annotations

"""Pure-AST cacheability analysis for notebook statements.

**Boundary rule (baked here by design):** this module is pure-AST.
It accepts ``(code, tree)`` and returns a :class:`StatementAnalysis`.
The moment something needs runtime state — ``user_ns``, value
introspection, purity-registry lookup, annotation parsing — it belongs
in ``cacheability_decision.py``, not here.  That sibling module owns
the merge of AST findings with runtime context.

Folds ``mutation_detector.py`` and ``side_effects.py`` into one module.
Both AST-visitor classes and their supporting dataclasses live here;
their names are re-exported for any existing import sites that reference
them directly.
"""

import ast
import os
import textwrap
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    # Primary API
    "StatementAnalysis",
    "analyze_statement",
    "statement_writes_files",
    "statement_written_paths",
    "statement_read_paths",
    "statement_saves_current_pyplot_figure",
    "standalone_method_mutation_receivers",
    "standalone_method_call_receivers",
    "assigned_method_call_receivers",
    "selfref_reassignment_targets",
    "cacheable_accumulator_loop",
    "selfref_inplace_write_vars",
    "params_mutated_in_function",
    "standalone_call_arg_targets",
    "function_arg_mutations",
    "function_global_mutations",
    "called_function_global_mutations",
    "callee_mutated_globals_for_tree",
    "callee_source_global_mutations",
    "stateful_self_functions",
    "stateful_closure_vars",
    "partial_arg_mutations",
    "mutating_partials",
    "reduce_free_mutations",
    "object_protocol_mutations",
    "ObjectProtocolResets",
    "alias_mutation_sources",
    "aliased_sources",
    "bare_alias_targets",
    "reference_alias_targets",
    "crossref_reassigned_vars",
    "consumed_input_names",
    "subscript_view_bindings",
    # Re-exported dataclasses (moved from mutation_detector / side_effects)
    "MutationInfo",
    "SideEffectInfo",
    # Constants re-exported for external consumers
    "MUTATING_METHODS",
    "PANDAS_INPLACE_METHODS",
    "KNOWN_PURE_METHODS",
    "RECEIVER_READONLY_WRITE_METHODS",
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

# The subset of MUTATING_METHODS that GROW a collection element-by-element — the
# classic accumulator loop (``out = []`` then ``for e in it: out.append(f(e))``).
# Only these earn the "rewrite as a comprehension" guidance hint in
# ``StatementAnalysis.skip_reasons``: an accumulator loop has a byte-identical
# comprehension form (``out = [f(e) for e in it]``) that assigns its result and
# therefore caches. Other in-place mutations (``pop``/``sort``/``df['x'] = …``)
# have no such rewrite and must NOT get the hint (part b).
ACCUMULATOR_METHODS = frozenset({'append', 'extend', 'add', 'update'})

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
# these calls carry is tracked SEPARATELY (``_WRITE_METHODS`` /
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
RECEIVER_READONLY_WRITE_METHODS = frozenset({
    'to_csv', 'to_parquet', 'to_pickle', 'to_json', 'to_feather',
    'to_excel', 'to_hdf', 'to_stata', 'to_sql', 'to_gbq',
    'to_clipboard', 'to_html', 'to_markdown', 'to_latex',
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

# matplotlib.pyplot module aliases. EVERY module-level ``plt.*`` call operates on
# pyplot's PROCESS-GLOBAL current figure — drawing (``plt.plot``, ``plt.hist``,
# ``plt.imshow``), styling (``plt.title``, ``plt.legend``, ``plt.grid``), or
# displaying (``plt.show``) — state cash does not track. Its only "input" is the
# module, so with a stable key such a call would cache and then, on a later run,
# either REPLAY a stale figure (``plt.show``) or SKIP the draw/style (losing that
# content from a freshly re-drawn figure — visible under ``%cash_persist`` or for
# an expensive draw above the cost floor). So any pyplot module-level call is a
# display side-effect: always re-execute, never cache. ``plt.savefig`` is the one
# exception — it is a file write (handled via ``_WRITE_METHODS`` below), not a
# display.
_PYPLOT_MODULE_ALIASES: frozenset[str] = frozenset({'plt', 'pyplot', 'matplotlib.pyplot'})

# pyplot calls that CREATE or FETCH a Figure/Axes rather than draw on / style the
# current one. They return identity-coupled objects already refused (and
# explained: "Identity-coupled figure") by the live-alias / figure-identity
# path, so leave them to it rather than relabel them a generic display effect.
_PYPLOT_FIGURE_ACCESSORS: frozenset[str] = frozenset({
    'figure', 'subplots', 'subplot', 'subplot_mosaic', 'subplot2grid',
    'axes', 'gca', 'gcf', 'get_current_fig_manager',
})

# Method names that indicate writing (when called on any object)
_WRITE_METHODS: frozenset[str] = frozenset({
    'to_csv', 'to_excel', 'to_parquet', 'to_json', 'to_pickle',
    'to_hdf', 'to_feather', 'to_sql', 'to_stata', 'to_latex',
    'to_html', 'to_clipboard', 'to_gbq', 'to_markdown',
    'savefig',   # matplotlib
    'save',      # numpy, PIL, torch
    'write',     # file objects
    'writelines',
    # pathlib.Path writes. Only the unambiguous names: generic
    # Path mutators like `rename`/`replace`/`touch` collide with common
    # methods on other types (str.replace!) and would over-flag.
    'write_text',
    'write_bytes',
})

# File open modes that indicate writing
_WRITE_MODES: frozenset[str] = frozenset({'w', 'wb', 'a', 'ab', 'w+', 'wb+', 'a+', 'ab+', 'x', 'xb'})

# Cheap textual pre-filter for statement_writes_files: superset of the names
# in the write-detection tables above, checked before any AST work.
_WRITE_TEXT_MARKERS: tuple[str, ...] = (
    'open(', 'write', 'to_', 'save', 'dump', 'os.', 'shutil.',
)


def statement_writes_files(code: str, tree: 'ast.Module | None' = None) -> bool:
    """True when *code* contains a file-WRITE side effect.

    Used by the upstream simulation to give file-writing statements a trace
    entry and by the re-execution planner to schedule stale writers — file
    writes have no variable edge, so lineage alone never re-runs them.
    Cheap: a textual marker pre-filter runs before the AST analysis.
    """
    if not any(m in code for m in _WRITE_TEXT_MARKERS):
        return False
    try:
        analysis = analyze_statement(code, tree)
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(e.kind == 'file_write' for e in analysis.side_effects)


def _is_append_mode_call(call: ast.Call) -> bool:
    """True when *call* carries a statically-visible APPEND mode string.

    Covers ``open(p, 'a')`` (mode is positional arg 1) and any writer taking a
    ``mode=`` keyword (``open(p, mode='a')``, ``df.to_csv(p, mode='a')``). A
    non-literal mode (``open(p, m)``) is NOT provable and returns False.
    """
    if isinstance(call.func, ast.Name) and call.func.id == 'open' and len(call.args) >= 2:
        mode_arg = call.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            if 'a' in mode_arg.value:
                return True
    for kw in call.keywords:
        if (kw.arg == 'mode'
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
                and 'a' in kw.value.value):
            return True
    return False


# Write calls that REPLACE their target wholesale, so re-running one lands the
# same bytes. ``to_hdf`` is deliberately ABSENT: pandas defaults it to
# ``mode='a'``, making it accumulating despite its truncating siblings.
_REPLACING_WRITE_METHODS: frozenset[str] = frozenset({
    'to_csv', 'to_excel', 'to_parquet', 'to_json', 'to_pickle', 'to_feather',
    'to_stata', 'to_latex', 'to_html', 'to_markdown',
    'savefig',      # matplotlib truncates the PNG
    'save',         # numpy / PIL / torch all truncate
    'write_text',   # pathlib truncates
    'write_bytes',
})

# Module-level writers that land the same result when repeated. Everything else
# in _IO_SIDE_EFFECT_FUNCTIONS (remove/rename/move/mkdir/rmtree...) is NOT
# repeatable -- a second run raises or acts on a target that is already gone.
_REPLACING_IO_FUNCTIONS: frozenset[tuple[str, str]] = frozenset({
    ('shutil', 'copy'), ('shutil', 'copy2'),
})

REPEATABILITY_REPLACING = 'replacing'
REPEATABILITY_ACCUMULATING = 'accumulating'
REPEATABILITY_UNKNOWN = 'unknown'


# Module writers that take an already-open FILE HANDLE rather than a path, so
# their repeatability is decided by whatever opened it -- never by the call
# itself. Maps (module, func) -> positional index of the handle argument.
_HANDLE_WRITE_FUNCTIONS: dict[tuple[str, str], int] = {
    ('json', 'dump'): 1,
    ('pickle', 'dump'): 1,
}


def _defers_to_open(node: ast.expr | None, local_handles: frozenset[str]) -> bool:
    """True when *node* is a handle whose ``open()`` is in this same statement.

    The ``open()`` carries the mode and is walked separately, so the write on
    the handle must contribute NO verdict of its own -- otherwise its UNKNOWN
    outranks the ``open()``'s provable one and a plain truncating write
    (``with open(p, 'wb') as f: pickle.dump(obj, f)``) is misread as unsafe to
    repeat.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
        return True
    return isinstance(node, ast.Name) and node.id in local_handles


def _open_mode_node(call: ast.Call) -> ast.expr | None:
    """The mode argument of an ``open()`` call, positional or keyword."""
    if len(call.args) >= 2:
        return call.args[1]
    for kw in call.keywords:
        if kw.arg == 'mode':
            return kw.value
    return None


def _locally_opened_handles(tree: ast.AST) -> set[str]:
    """Names bound to an ``open()`` handle WITHIN this statement.

    ``with open(p, 'w') as f: f.write(x)`` and ``f = open(p, 'w'); f.write(x)``
    both carry their mode on the ``open()`` call, which is walked separately.
    Without this, the ``f.write`` would contribute an UNKNOWN that outranks the
    ``open()``'s provable verdict and mislabels a plain truncating write.
    """
    handles: set[str] = set()

    def _is_open(node) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'open')

    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if _is_open(item.context_expr) and isinstance(item.optional_vars, ast.Name):
                    handles.add(item.optional_vars.id)
        elif isinstance(node, ast.Assign) and _is_open(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    handles.add(target.id)
    return handles


def _call_repeatability(call: ast.Call, local_handles: frozenset[str] = frozenset()) -> str | None:
    """Repeatability of one call node, or ``None`` if it is not a file write."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == 'open':
        mode = _open_mode_node(call)
        if mode is None:
            return None  # no mode argument -> defaults to 'r', a read
        if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
            # A computed mode proves nothing. `_is_open_write_mode` reports
            # False here, which the write-DETECTION path reads as "not a
            # write" -- but a `.write` on this handle still makes the statement
            # a writer, and the mode could be 'a' at runtime.
            return REPEATABILITY_UNKNOWN
        if not _is_open_write_mode(call):
            return None  # provably a read mode
        return (REPEATABILITY_ACCUMULATING if 'a' in mode.value
                else REPEATABILITY_REPLACING)
    if isinstance(func, ast.Attribute):
        method = func.attr
        if method not in _WRITE_METHODS:
            return None
        if _is_append_mode_call(call):
            return REPEATABILITY_ACCUMULATING
        if method in _REPLACING_WRITE_METHODS:
            return REPEATABILITY_REPLACING
        # ``f.write(...)`` / ``f.writelines(...)``: the mode lives on whatever
        # opened the handle, never on the write itself. If that ``open()`` is in
        # this same statement, defer -- its own node is walked separately and
        # carries the provable verdict. A handle bound in an earlier cell is
        # genuinely unresolvable from here.
        if _defers_to_open(func.value, local_handles):
            return None
        return REPEATABILITY_UNKNOWN
    if isinstance(func, ast.Name):
        return None
    return None


def statement_write_repeatability(code: str, tree: 'ast.Module | None' = None) -> str:
    """How safe is it to re-run *code*'s file writes?

    The question the write-detection helpers above do not answer:
    :data:`_WRITE_MODES` pools ``'a'`` with ``'w'``, so every consumer learns
    only "this writes a file", never "repeating this write duplicates data".

    Returns the WORST verdict across every write in the statement:

    * ``'accumulating'`` -- provably appends; re-firing duplicates the payload
    * ``'replacing'``    -- provably truncates; re-firing lands the same bytes
    * ``'unknown'``      -- cannot tell (variable mode, a handle opened
      elsewhere, ``os.rename``/``shutil.move``, ...)

    A statement with no recognised write is ``'replacing'``: there is nothing to
    repeat, so it never constrains the planner.
    """
    if tree is None:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return REPEATABILITY_UNKNOWN
    local_handles = frozenset(_locally_opened_handles(tree))
    verdicts: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        verdict = _call_repeatability(node, local_handles)
        if verdict is not None:
            verdicts.add(verdict)
        # Module-level writers (os.remove, shutil.move, ...) are recognised
        # through the side-effect table rather than the call shapes above.
        name = _get_call_name(node.func)
        module = _get_call_module(node.func)
        if not name or not module:
            continue
        key = (module, name)
        if _IO_SIDE_EFFECT_FUNCTIONS.get(key) != 'file_write':
            continue
        if key in _REPLACING_IO_FUNCTIONS:
            continue
        handle_idx = _HANDLE_WRITE_FUNCTIONS.get(key)
        if handle_idx is not None:
            handle = node.args[handle_idx] if len(node.args) > handle_idx else None
            if _defers_to_open(handle, local_handles):
                continue  # the open() decides; it is walked separately
        verdicts.add(REPEATABILITY_UNKNOWN)
    if REPEATABILITY_ACCUMULATING in verdicts:
        return REPEATABILITY_ACCUMULATING
    if REPEATABILITY_UNKNOWN in verdicts:
        return REPEATABILITY_UNKNOWN
    return REPEATABILITY_REPLACING


# Write-call method forms whose FIRST positional argument (or a common path
# keyword) names the output FILE. Deliberately excludes ``to_sql`` / ``to_gbq``
# / ``to_clipboard`` (no filesystem path), the file-handle methods
# ``write`` / ``writelines`` (the path lives on the ``open()`` that made the
# handle), and ``json``/``pickle`` ``dump`` (path on the nested ``open()``);
# those are recovered from the ``open()`` call in the same statement instead.
_PATH_ARG0_WRITE_METHODS: frozenset[str] = frozenset({
    'to_csv', 'to_parquet', 'to_pickle', 'to_json', 'to_feather',
    'to_excel', 'to_hdf', 'to_stata', 'savefig',
})

# Keyword names that carry the output path across the recognised write calls.
_PATH_KWARG_NAMES: frozenset[str] = frozenset({
    'path', 'path_or_buf', 'fname', 'excel_writer', 'file',
})


def _resolve_literal_path(node: ast.AST, namespace: dict[str, Any] | None) -> str | None:
    """Resolve a call argument to an output-path string, or ``None``.

    Only a string literal, or a simple ``Name`` bound to a ``str`` /
    ``os.PathLike`` in *namespace*, is resolvable. Anything computed — an
    f-string, a ``str.format``, an ``os.path.join``, an attribute — returns
    ``None`` so the caller stays conservative and never skips a writer whose
    target it cannot pin down. *namespace* is an injected argument (kept out of
    the module's global reach), so this function stays a pure function of its
    inputs.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and namespace is not None:
        val = namespace.get(node.id)
        if isinstance(val, str):
            return val
        if isinstance(val, os.PathLike):
            try:
                return os.fspath(val)
            except TypeError:
                return None
    return None


def _is_path_constructor(func: ast.AST) -> bool:
    """True for a ``Path(...)`` / ``pathlib.Path(...)`` constructor call func."""
    if isinstance(func, ast.Name):
        return func.id == 'Path'
    if isinstance(func, ast.Attribute):
        return func.attr == 'Path'
    return False


def _call_path_argument(
    call: ast.Call,
    index: int,
    namespace: dict[str, Any] | None,
    kwarg_names: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve the path from *call*'s positional arg *index* or a path keyword."""
    if len(call.args) > index and not isinstance(call.args[index], ast.Starred):
        return _resolve_literal_path(call.args[index], namespace)
    for kw in call.keywords:
        if kw.arg and kw.arg in kwarg_names:
            return _resolve_literal_path(kw.value, namespace)
    return None


def _write_call_path(
    call: ast.Call, namespace: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Return ``(resolved_path_or_None, is_path_bearing)`` for one call node.

    ``is_path_bearing`` marks the recognised writer forms whose own arguments
    name the output file. When it is True but the path is ``None`` the path was
    present but not statically resolvable — the caller treats that as "cannot
    verify" and stays conservative.
    """
    func = call.func
    # open(PATH, 'w'|'a'|...) — only a write mode counts.
    if isinstance(func, ast.Name) and func.id == 'open':
        if _is_open_write_mode(call):
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
        return None, False
    if isinstance(func, ast.Attribute):
        method = func.attr
        # Path(PATH).write_text(...) / Path(PATH).write_bytes(...)
        if method in ('write_text', 'write_bytes'):
            recv = func.value
            if isinstance(recv, ast.Call) and _is_path_constructor(recv.func):
                return _call_path_argument(recv, 0, namespace), True
            return None, True  # receiver path not inline -> unresolvable
        # np.save(PATH, arr) — the path is arg0, but ONLY for a numpy receiver;
        # torch.save(obj, PATH) puts the path second and PIL ``img.save(PATH)``
        # is ambiguous, so a non-numpy ``save`` stays conservative.
        if method == 'save':
            base = _get_base_name(func.value)
            if base in ('np', 'numpy'):
                return _call_path_argument(call, 0, namespace), True
            return None, True
        if method in _PATH_ARG0_WRITE_METHODS:
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    return None, False


def statement_written_paths(
    code: str,
    tree: 'ast.Module | None' = None,
    namespace: dict[str, Any] | None = None,
) -> set[str] | None:
    """Resolvable output path(s) a file-writing statement writes, or ``None``.

    Extracts the literal / resolvable output path for the common write forms:
    ``df.to_csv(PATH)`` and its ``to_parquet`` / ``to_pickle`` / ``to_json`` /
    ``to_feather`` / ``to_excel`` / ``to_hdf`` siblings, ``savefig(PATH)``,
    ``np.save(PATH, ...)``, ``open(PATH, 'w'|'wb'|'a'|...)``,
    ``Path(PATH).write_text/write_bytes(...)``, and the nested-handle forms
    ``json.dump(obj, open(PATH, ...))`` / ``pickle.dump(obj, open(PATH, ...))``
    (the path comes from the ``open()``).

    Returns the set of resolved paths only when EVERY path-bearing write call in
    the statement resolves to a string literal (or a simple ``Name`` bound to a
    ``str`` / ``os.PathLike`` in *namespace*). Returns ``None`` the moment a path
    is not statically resolvable (f-string, computed expression, unknown name),
    or no path-bearing write call is recognised — so the caller falls through to
    its conservative re-fire behaviour rather than skip a writer whose effect it
    cannot verify. Failure-tolerant: any extraction ambiguity yields ``None``.
    """
    if not any(m in code for m in _WRITE_TEXT_MARKERS):
        return None
    if tree is None:
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
    paths: set[str] = set()
    saw_path_bearing = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_path_bearing = _write_call_path(node, namespace)
        if not is_path_bearing:
            continue
        saw_path_bearing = True
        if resolved is None:
            return None  # a write target we could not pin down -> conservative
        paths.add(resolved)
    if not saw_path_bearing or not paths:
        return None
    return paths


# Cheap textual pre-filter for statement_read_paths.
_READ_TEXT_MARKERS: tuple[str, ...] = ('open(', 'read', 'load')


def _read_call_path(
    call: ast.Call, namespace: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Return ``(resolved_path_or_None, is_path_bearing)`` for one READ call node.

    Mirror of :func:`_write_call_path` for the recognised file-READ forms:
    ``open(PATH)`` in a non-write mode, ``*.read_csv(PATH)`` / any ``.read_<x>``
    reader, ``np.load(PATH)`` / ``joblib.load(PATH)``, the nested-handle
    ``pickle.load(open(PATH))`` / ``json.load(open(PATH))``, and
    ``Path(PATH).read_text/read_bytes()``. ``is_path_bearing`` marks a recognised
    reader whose args name an input file; a ``None`` path there means the target
    was present but not statically resolvable (caller stays conservative).
    """
    func = call.func
    # open(PATH) / open(PATH, 'r'|'rb'|...) -- only a NON-write mode counts.
    if isinstance(func, ast.Name) and func.id == 'open':
        if _is_open_write_mode(call):
            return None, False
        return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    if isinstance(func, ast.Attribute):
        method = func.attr
        # Path(PATH).read_text(...) / Path(PATH).read_bytes(...)
        if method in ('read_text', 'read_bytes'):
            recv = func.value
            if isinstance(recv, ast.Call) and _is_path_constructor(recv.func):
                return _call_path_argument(recv, 0, namespace), True
            return None, True
        # np.load / numpy.load / joblib.load(PATH); pickle/json.load(open(PATH)).
        if method == 'load':
            base = _get_base_name(func.value)
            if call.args and isinstance(call.args[0], ast.Call):
                inner = call.args[0]
                if isinstance(inner.func, ast.Name) and inner.func.id == 'open':
                    return _call_path_argument(inner, 0, namespace, _PATH_KWARG_NAMES), True
            if base in ('np', 'numpy', 'joblib'):
                return _call_path_argument(call, 0, namespace), True
            return None, False
        # pandas / polars readers: any ``.read_<fmt>(PATH)`` takes the path arg0.
        if method.startswith('read_'):
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    return None, False


def statement_read_paths(
    code: str,
    tree: 'ast.Module | None' = None,
    namespace: dict[str, Any] | None = None,
) -> set[str] | None:
    """Resolvable input path(s) a statement READS, or ``None`` when uncertain.

    Companion to :func:`statement_written_paths`, used by the re-execution
    planner to scope file-writer re-firing to writers whose output a downstream
    consumer actually reads. Returns the set of statically
    resolved read paths, an **empty set** when the statement has no path-bearing
    read at all, or ``None`` the moment a recognised reader's path is NOT
    statically resolvable (f-string / computed) -- so the caller treats the read
    set as unknown and never suppresses a writer it cannot prove is unread.
    """
    if not any(m in code for m in _READ_TEXT_MARKERS):
        return set()
    if tree is None:
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_path_bearing = _read_call_path(node, namespace)
        if not is_path_bearing:
            continue
        if resolved is None:
            return None  # a read target we could not pin down -> unknown
        paths.add(resolved)
    return paths


def _receiver_is_pyplot_module(recv: ast.AST, namespace: dict[str, Any] | None) -> bool:
    """True when *recv* is the ``matplotlib.pyplot`` MODULE, not a Figure/Axes.

    ``plt.savefig(...)`` (receiver is the module) saves pyplot's process-global
    current figure; ``fig.savefig(...)`` (receiver is a Figure) is receiver-bound
    and defended elsewhere. The only reliable separator is what the receiver name
    resolves to at runtime, so *namespace* is consulted when available.
    """
    # ``matplotlib.pyplot.savefig(...)`` -- an attribute chain ending in .pyplot.
    if isinstance(recv, ast.Attribute):
        return recv.attr == 'pyplot'
    if isinstance(recv, ast.Name):
        if namespace is not None and recv.id in namespace:
            mod = namespace[recv.id]
            # A Figure/Axes has no ``__name__``; the pyplot module's is exact.
            return getattr(mod, '__name__', '') == 'matplotlib.pyplot'
        # Namespace unavailable / name not bound: accept the conventional alias
        # as a conservative fallback (everyone imports pyplot as ``plt``).
        return recv.id in ('plt', 'pyplot')
    return False


def statement_saves_current_pyplot_figure(
    code: str,
    namespace: dict[str, Any] | None = None,
) -> bool:
    """True when *code* saves pyplot's CURRENT figure via a module-level call.

    ``plt.savefig(path)`` writes whatever figure pyplot's process-global ``Gcf``
    registry holds. Its only variable input is the MODULE ``plt`` -- there is no
    value-level edge from the statement to the figure it saves. So when the
    re-execution planner schedules such a write while the ``plt.subplots()`` /
    ``plt.figure()`` that registered the current figure is NOT scheduled,
    re-running the write makes ``plt.gcf()`` invent a blank default figure and
    flush it over the user's chart -- a silent on-disk wrong answer.

    This isolates that undefended module-level form so the planner can refuse it.
    The receiver-bound ``fig.savefig(path)`` is NOT flagged: it is defended by the
    carrier-history pass (its input ``fig`` is a tracked carrier). Detection is
    namespace-aware where possible and falls back to the conventional ``plt`` /
    ``pyplot`` alias. Failure-tolerant: any parse/analysis error returns False.
    """
    if 'savefig' not in code:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'savefig':
            if _receiver_is_pyplot_module(func.value, namespace):
                return True
    return False


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
            elif (module_name in _PYPLOT_MODULE_ALIASES
                  and func_name not in _WRITE_METHODS
                  and func_name not in _PYPLOT_FIGURE_ACCESSORS):
                # A pyplot module-level call (draw/style/show) mutating the global
                # figure — uncacheable. savefig is a file_write (below); figure
                # creation/access is left to the identity-coupling path.
                self.effects.append(SideEffectInfo(
                    kind='display',
                    description=f"{module_name}.{func_name}()",
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

    ``accumulator_mutated_vars`` — the subset of ``top_level_mutated_vars``
    grown by an accumulator method (``append``/``extend``/``add``/``update``;
    see :data:`ACCUMULATOR_METHODS`).  Purely advisory: it gates the
    comprehension guidance hint in :meth:`skip_reasons` and never changes a
    caching decision.

    ``alias_targets`` — names bound by a pure pointer copy of a bare ``Name``
    (``b = a``); see :func:`bare_alias_targets`.  Unlike the advisory field
    above this DOES block caching: restoring such a binding hands back a copy
    where Python guarantees identity.
    """

    top_level_mutated_vars: frozenset[str]
    all_mutated_vars: frozenset[str]
    side_effects: tuple[SideEffectInfo, ...]
    called_names: frozenset[str]
    accumulator_mutated_vars: frozenset[str] = frozenset()
    alias_targets: frozenset[str] = frozenset()

    def skip_reasons(self, outputs: set[str]) -> list[str]:
        """Render structured findings as human-readable skip reasons.

        Used to populate ``metrics['uncacheable_reasons']``.

        Args:
            outputs: Variable names that are *outputs* of this statement.
                     Mutations on outputs are expected and do not block
                     caching (the output itself gets a fresh lineage).
        """
        reasons: list[str] = []
        # An alias bind (``b = a``) is free to execute and MUST NOT be restored:
        # a hit rebinds the target to a deserialised copy, silently breaking the
        # ``b is a`` identity Python guarantees. Reported first — it is
        # a property of the statement's shape, not of its effects.
        if self.alias_targets:
            names = ', '.join(sorted(self.alias_targets))
            reasons.append(
                f"Alias assignment: {names} names the same object as the "
                "right-hand side; restoring a copy would break identity "
                "(and rebinding costs nothing to re-run)"
            )
        pure_mutations = self.top_level_mutated_vars - outputs
        if pure_mutations:
            reasons.append(f"In-place mutation on: {', '.join(sorted(pure_mutations))}")
            # Guidance only (part b): when the blocking mutation is an
            # accumulator (``out.append(f(e))`` in a loop), point the user at the
            # byte-identical comprehension form, which assigns its result and so
            # caches. Scoped to accumulator methods — a ``df['x'] = …`` subscript
            # store has no comprehension rewrite and gets no hint. Advisory: the
            # statement stays uncacheable (the reason above already fired).
            if pure_mutations & self.accumulator_mutated_vars:
                reasons.append(
                    "tip: assign the result to cache it — e.g. "
                    "`out = [f(e) for e in it]` instead of a for-append loop"
                )
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
    5``, idempotent) is NOT flagged."""
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
    means ``data`` is mutated in place, so it must reset on isolated re-run.

    When *resolve_source* is given (a ``name -> source`` lookup), the analysis is
    interprocedural: a parameter mutated only via a further resolvable call
    (``def outer(y): inner(y)`` where ``inner`` mutates its arg) is also detected
   . *seen* guards against mutual / self recursion. Without
    *resolve_source* the analysis is one level deep (the original
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
    function that mutates the corresponding parameter.

    *resolve_source* maps a function name to its source string (or ``None`` if it
    is not a resolvable user-defined function — a builtin, C function, lambda, or
    unknown name). For each top-level bare-``Expr`` call the function body is
    parsed, its mutated parameters are found via :func:`params_mutated_in_function`
    (interprocedurally — a param mutated only through a further resolvable call is
    detected too), and each is mapped back to the call's positional /
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
    """Module-global / free variables a function body mutates in place.

    A name mutated in place (``items.append``, ``store[k]=``, ``g += 1`` under a
    ``global`` declaration) that is neither a parameter nor a plain local
    assignment is a free variable resolved from the enclosing / module scope —
    calling the function mutates that global. Parameter mutations are a
    separate job and are excluded; a name rebound locally (``acc = []`` then
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
    """Module globals mutated in place by a called function (part A).

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


def called_function_global_mutations(
    tree, resolve_source, include_control_bodies: bool = False,
) -> frozenset[str]:
    """Module globals mutated in place by ANY function called in *tree*
    (CAS-260) — the capture-and-restore watch list.

    Sibling of :func:`function_global_mutations`, and the difference is the
    whole point: that one walks :func:`standalone_call_arg_targets`, i.e. only
    a top-level bare-``Expr`` call (``bump()``), because a *reset* target only
    makes sense for a call made for its effect. Capture-and-restore has to
    cover the spellings where the call's VALUE is used as well::

        x = compute(y)              # statement path
        out.append(compute(y))      # call path (CAS-243)

    Both are ``ast.Call`` nodes anywhere in the statement, which is exactly
    what :func:`_called_function_names` already collects, so this is that
    walk plus the same per-callee :func:`_free_vars_mutated_in_function`
    analysis. A rule that fired for one spelling and not the other is the
    CAS-145 defect this project has already paid for.

    **Why the caller must still filter.** This is pure AST: it names what the
    callee's source mutates, and says nothing about whether that name is a
    live, capturable notebook variable. A name that is a module, or absent
    from the namespace entirely (the callee's own module-level global, not the
    notebook's), must be dropped by the caller before it reaches an output
    set — capturing it would either serialise a module or invent a variable.

    **Calls inside a loop or branch body are deliberately NOT included.** A
    control structure is one unit to the upstream simulation and to the
    accumulator machinery, so its body's writes are the loop's to own, not any
    single body statement's. Three narrower placements were measured on::

        for t in [1, 2, 3]:
            of.append(loop_f(t))       # loop_f appends to a global LOOP_F

        per-iteration capture      LOOP_F == [2]           one iteration's
                                                           ABSOLUTE post-state,
                                                           restored over the
                                                           accumulation
        per-iteration skip-cache   LOOP_F == [2]           the planner replays
                                                           only the last writer
        per-call restore           LOOP_F == [1, 2, 2, 3]  restores interleaved
                                                           with the iterations
                                                           that genuinely ran

    All three are WORSE than the pre-existing behaviour, where the write is
    merely skipped and the value stays where the last real execution left it.
    Notably the third survives making the body statement always re-execute,
    which was the hypothesis under which the exclusion was briefly lifted.
    Owning this at the loop level is CAS-265; until then, one rule -- the loop
    owns its body -- applied at every site that consults this.

    Not interprocedural, matching :func:`function_global_mutations`: a global
    mutated only by a helper the callee calls is not detected. That is a
    fail-open gap (the write is silently skipped on a hit, exactly as today),
    not a new failure mode, and it keeps this identical to the analysis the
    checker has already shipped.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    # The checker asks with ``include_control_bodies=True``: it acts per CELL,
    # and its idempotent-rerun reset must cover everything the cell writes,
    # including through a loop. The statement path asks without it -- it acts
    # per STATEMENT, and a body statement claiming the whole accumulator was
    # measured wrong (CAS-265).
    names = (_called_function_names(tree) if include_control_bodies
             else _cell_level_called_function_names(tree))
    for name in names:
        fdef = _resolve_function_def(name, resolve_source)
        if fdef is not None:
            out |= _free_vars_mutated_in_function(fdef)
    return frozenset(out)


def callee_source_global_mutations(source: str) -> frozenset[str]:
    """Globals a single function's own SOURCE mutates in place.

    The same per-callee analysis :func:`called_function_global_mutations`
    applies, entered one level lower: given the text of one ``def``, name the
    free variables its body writes. Exists so the call unit -- which holds a
    live function object and resolves its source through ``inspect``, not
    through a name in the user namespace -- can share this analysis rather than
    reach past the module boundary for a private helper. The two paths agreeing
    on what counts as a callee's write is the point (CAS-145: a rule that fires
    for one spelling and not another is a defect this project has paid for).

    Returns an empty set for anything that is not a single function
    definition, and never raises.
    """
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, RecursionError):
        return frozenset()
    node = parsed.body[0] if parsed.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset()
    try:
        return _free_vars_mutated_in_function(node)
    except (ValueError, RecursionError):
        return frozenset()


#: Statement nodes whose bodies a control-structure handler owns, not the
#: statement path. Mirrors what ``statement/processor.py`` marks with
#: ``# __iteration_context__:`` / ``# control_context:``.
_CONTROL_STATEMENTS = (
    ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
)


def _cell_level_called_function_names(tree) -> frozenset[str]:
    """``name(...)`` callees reachable WITHOUT entering a control structure.

    :func:`_called_function_names` walks everything; this stops at a ``for`` /
    ``while`` / ``if`` / ``with`` / ``try``, so a call in the body of one is not
    reported. The control structure's own header expressions ARE walked --
    ``for t in gen(): ...`` reads ``gen()`` once, outside any iteration, so it
    belongs to the cell.
    """
    out: set[str] = set()

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _CONTROL_STATEMENTS):
                # The header (`iter`, `test`, `items`) still belongs to the
                # cell; only the bodies are the control structure's.
                for field, value in ast.iter_fields(child):
                    if field in ('body', 'orelse', 'finalbody', 'handlers'):
                        continue
                    for sub in (value if isinstance(value, list) else [value]):
                        if isinstance(sub, ast.AST):
                            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                                out.add(sub.func.id)
                            walk(sub)
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                out.add(child.func.id)
            walk(child)

    walk(tree)
    return frozenset(out)


_MUTABLE_LITERAL_CALLS = frozenset({'list', 'dict', 'set'})


def _is_mutable_default(node: ast.expr) -> bool:
    """True if *node* is a mutable literal default (``[]``, ``{}``, ``set()``)."""
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _MUTABLE_LITERAL_CALLS and not node.args)


_MEMOIZER_DECORATORS = frozenset({'lru_cache', 'cache'})


def _is_memoizer_decorator(dec: ast.expr) -> bool:
    """True for a ``functools`` memoizer decorator — ``@lru_cache`` / ``@cache``,
    bare or called (``@lru_cache(maxsize=None)``), plain or dotted
    (``@functools.lru_cache``)."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Name):
        return node.id in _MEMOIZER_DECORATORS
    if isinstance(node, ast.Attribute):
        return node.attr in _MEMOIZER_DECORATORS
    return False


def _function_mutates_own_object(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if calling *func* mutates state carried on the function OBJECT itself
    (part B), which persists across calls and must be reset by re-running
    the ``def``:

    * a **mutable default argument** the body mutates in place
      (``def collect(x, acc=[]): acc.append(x)``);
    * an assignment to a **function attribute**
      (``def tick(): tick.count = getattr(tick, 'count', 0) + 1``);
    * a **functools memoizer** (``@lru_cache`` / ``@cache``) — the cache persists
      across calls, so a re-run reuses memoised results and its body's
      side effects (a free-var append) no longer fire; re-running the ``def``
      recreates an empty cache.
    """
    if any(_is_memoizer_decorator(d) for d in func.decorator_list):
        return True
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
    """Called functions that carry mutable state on their own object (B).

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
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)):
            base = _extract_base_name(node.value.value)
            if base:
                out[node.targets[0].id] = base
    return out


def partial_arg_mutations(tree: ast.Module | None, resolve_partial, resolve_source) -> frozenset[str]:
    """Vars mutated through a called ``functools.partial`` binding.

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
    """Partial vars called in the cell that bind a MUTATED positional arg.

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
    """Free/global vars mutated by a function passed to ``functools.reduce``.

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
    returned function and is only reset by re-running the factory (B).
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
    (B). *resolve_var_factory* maps a name to the ``FunctionDef`` of the
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


# ---------------------------------------------------------------------------
# Object-protocol hidden state
#
# A ``with`` statement, a custom-dunder operation (``s[k]=v`` / ``del s[k]`` /
# ``v=s[k]`` / ``a(x)``), a constructor, a decorated call, or an instance /
# class method invokes a user-defined method whose body mutates hidden state.
# The mutation is invisible to the cell text, so on an isolated re-run it
# accumulates. This generalises the earlier case (hidden state via a called function) to
# the object protocol: resolve the receiver's class (or the wrapper / context
# manager), analyse the invoked method body, and attribute the mutation to one
# of three reset channels:
#
#   * **free_vars**   — a module/free variable the method mutates (``log``);
# reset like the A path (add to the cell's mutated set + inputs so the
#     producer's cell-entry base is restored).
#   * **receivers**   — the receiver INSTANCE whose ``self`` attribute the
#     method mutates in place (``cm.n += 1``); reset like a method receiver.
#   * **class_defs**  — the CLASS whose class variable the method mutates
#     (``Reg.registry.append`` / ``cls.log.append``); reset by re-running the
#     class ``def`` so the class-level container is recreated fresh.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectProtocolResets:
    """Reset targets attributed to hidden object-protocol mutations, one set per
    reset channel (see the module-section comment above)."""

    free_vars: frozenset[str]
    receivers: frozenset[str]
    class_defs: frozenset[str]
    # Free vars mutated by a base ``__init_subclass__`` hook fired during class
    # creation. Kept apart from ``free_vars`` because — unlike an
    # ordinary free-var mutation, whose upstream occurrences the simulator
    # sees as top-level statements — this mutation is hidden behind class creation
    # in EVERY subclass cell, so the simulator's content-base cross-cell guard is
    # blind to it. The caller applies the cross-cell suppression before
    # routing these to the (otherwise self-protecting) free-var reset.
    init_subclass_free_vars: frozenset[str] = frozenset()


def _first_param_name(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The name of *func*'s first positional parameter (``self`` / ``cls``), or
    ``None`` for a zero-arg function."""
    positional = list(func.args.posonlyargs) + list(func.args.args)
    return positional[0].arg if positional else None


def _has_named_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """True if *func* carries a ``@name`` decorator (bare or called)."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def _resolve_class_def(name, resolve_class_source) -> ast.ClassDef | None:
    """Parse *name*'s source via *resolve_class_source* into a ClassDef, or None."""
    if not name or resolve_class_source is None:
        return None
    source = resolve_class_source(name)
    if not source:
        return None
    try:
        parsed = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return None
    if parsed.body and isinstance(parsed.body[0], ast.ClassDef):
        return parsed.body[0]
    return None


def _class_bases(classdef: ast.ClassDef) -> list[str]:
    """The bare-``Name`` base-class names of *classdef* (in declaration order).
    Non-``Name`` bases (``Generic[T]``, ``a.B``) are skipped — only notebook
    classes resolvable by name are followed."""
    return [b.id for b in classdef.bases if isinstance(b, ast.Name)]


def _iter_class_hierarchy(
    classdef: ast.ClassDef | None, resolve_class_source, _seen: set[str] | None = None
):
    """Yield *classdef* and its resolvable base classes, depth-first.

    Follows each ``Name`` base via *resolve_class_source*, so an inherited method
    / class variable is seen. Cycle-guarded by class name. When
    *resolve_class_source* is ``None`` (unit-test callers), only *classdef* is
    yielded — the original own-class-only behaviour."""
    if classdef is None:
        return
    if _seen is None:
        _seen = set()
    if classdef.name in _seen:
        return
    _seen.add(classdef.name)
    yield classdef
    if resolve_class_source is None:
        return
    for base_name in _class_bases(classdef):
        base = _resolve_class_def(base_name, resolve_class_source)
        if base is not None:
            yield from _iter_class_hierarchy(base, resolve_class_source, _seen)


def _class_method(
    classdef: ast.ClassDef, method_name: str, resolve_class_source=None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The method named *method_name* on *classdef* or an inherited base (the
    first match walking the hierarchy)."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == method_name):
                return node
    return None


def _own_class_level_attr_names(classdef: ast.ClassDef) -> frozenset[str]:
    """Class variables assigned directly in *classdef*'s body (not inherited)."""
    out: set[str] = set()
    for node in classdef.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        out.add(leaf.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return frozenset(out)


def _class_level_attr_names(classdef: ast.ClassDef, resolve_class_source=None) -> frozenset[str]:
    """Class variables of *classdef* including those inherited from base classes
    (``registry = []``, ``count = 0``, ``ClassVar[...] = []``). These live on the
    owning class object and are shared by every instance, so mutating one
    accumulates until that class ``def`` re-runs."""
    out: set[str] = set()
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        out |= _own_class_level_attr_names(cls)
    return frozenset(out)


def _instance_attr_names(classdef: ast.ClassDef, resolve_class_source=None) -> frozenset[str]:
    """Attribute names assigned as ``self.<attr> = ...`` in any method of
    *classdef* or an inherited base — the per-instance attributes. A method
    mutating one of these mutates the receiver instance (reset the receiver), not
    the shared class."""
    out: set[str] = set()
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for meth in cls.body:
            if not isinstance(meth, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            recv = _first_param_name(meth)
            if recv is None:
                continue
            for node in ast.walk(meth):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    for leaf in _iter_store_targets(tgt):
                        if (isinstance(leaf, ast.Attribute)
                                and isinstance(leaf.value, ast.Name)
                                and leaf.value.id == recv):
                            out.add(leaf.attr)
    return frozenset(out)


def _class_var_owner(attr: str, classdef: ast.ClassDef, resolve_class_source) -> str | None:
    """The name of the class in *classdef*'s hierarchy that DEFINES class variable
    *attr* at its own body level — the class whose ``def`` must re-run to reset it
    (``Base.registry`` is owned by ``Base`` even when mutated via ``Sub``).
    The nearest defining class wins; ``None`` if no class defines it."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        if attr in _own_class_level_attr_names(cls):
            return cls.name
    return None


def _property_accessor(
    classdef: ast.ClassDef, attr: str, kind: str, resolve_class_source=None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``@property`` getter (``kind='getter'``) or ``@<attr>.setter`` setter
    (``kind='setter'``) named *attr* in *classdef*'s hierarchy, or None.

    A property defines two methods both named *attr*: the getter carries
    ``@property`` and the setter ``@<attr>.setter``. ``_class_method`` would return
    whichever comes first, so the accessors are matched by their decorator."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == attr):
                continue
            if kind == 'getter' and _has_named_decorator(node, 'property'):
                return node
            if kind == 'setter':
                for dec in node.decorator_list:
                    if (isinstance(dec, ast.Attribute) and dec.attr == 'setter'
                            and isinstance(dec.value, ast.Name) and dec.value.id == attr):
                        return node
    return None


def _descriptor_class(
    classdef: ast.ClassDef, attr: str, resolve_class_source
) -> ast.ClassDef | None:
    """The ClassDef of the data descriptor bound to class attribute *attr*
    (``field = Tracked()`` → ``Tracked``'s ClassDef), or None. Accessing
    ``obj.field`` / assigning ``obj.field = v`` dispatches to that class's
    ``__get__`` / ``__set__``."""
    for cls in _iter_class_hierarchy(classdef, resolve_class_source):
        for node in cls.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)):
                return _resolve_class_def(node.value.func.id, resolve_class_source)
    return None


def _walk_executable(node: ast.AST):
    """Yield *node* and every descendant that executes when it runs, WITHOUT
    descending into deferred scopes (def/async def/class). Used to scan a cell or
    a method body for the operations that actually run, skipping nested
    definitions."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFERRED_SCOPES):
            continue
        yield from _walk_executable(child)


def _iter_method_body_nodes(method: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield every executable node in *method*'s body (skipping nested scopes)."""
    for stmt in method.body:
        yield from _walk_executable(stmt)


def _first_attr_after_root(chain: ast.expr) -> str | None:
    """For a receiver chain like ``self.data`` / ``cls.log`` / ``self.cache[k]``
    return the attribute name attached directly to the root ``Name``
    (``data`` / ``log`` / ``cache``), or ``None`` if the root is bare."""
    node: ast.AST = chain
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node.value, ast.Name):
            return node.attr if isinstance(node, ast.Attribute) else None
        node = node.value
    return None


def _iter_inplace_mutation_chains(method: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield the receiver-chain expr of every IN-PLACE mutation in *method*'s
    body: a known-mutating method call, an augmented assignment, a subscript
    assignment / delete, or a pandas ``inplace=True`` call. Plain attribute
    rebinds (``self.x = value`` — construction / idempotent re-set) are excluded
    so a constructor is not flagged."""
    for node in _iter_method_body_nodes(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in MUTATING_METHODS:
                yield node.func.value
            elif attr in PANDAS_INPLACE_METHODS:
                for kw in node.keywords:
                    if (kw.arg == 'inplace' and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True):
                        yield node.func.value
                        break
        elif isinstance(node, ast.AugAssign):
            yield node.target
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Subscript):
                        yield leaf.value
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    yield tgt.value


def _super_called_methods(method: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Method names invoked via ``super().<name>(...)`` in *method*'s body — the
    inherited implementations whose hidden mutations must also be attributed
    (``super.__init__`` running ``Base.__init__``)."""
    out: set[str] = set()
    for node in _iter_method_body_nodes(method):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == 'super'):
            out.add(node.func.attr)
    return frozenset(out)


def _classify_method_mutations(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    recv_class_name: str,
    cdef: ast.ClassDef,
    resolve_class_source=None,
    _seen: set[tuple[str, str]] | None = None,
    _current: ast.ClassDef | None = None,
) -> tuple[bool, frozenset[str], frozenset[str]]:
    """Classify the hidden mutations *method* performs into the three reset
    channels: ``(mutates_self_instance, class_reset_targets, free_vars)``.

    * ``self.<attr>`` where ``<attr>`` is a per-instance attribute → instance
      mutation (reset the receiver);
    * ``cls.<attr>`` (classmethod), ``BaseOrOwnClass.<attr>``, or ``self.<attr>``
      where ``<attr>`` is a class-level variable → class-variable mutation. The
      reset target is the class that OWNS the variable (a base class when the var
      is inherited), so ``class_reset_targets`` is a SET of class names;
    * a module/free variable (neither a parameter nor a local) → free-var
      mutation (reset via the A path).

    Inheritance is followed: *cdef* + *resolve_class_source* give hierarchy-aware
    attribute sets and owner lookup, and ``super().<m>()`` calls recurse into the
    base implementation. *cdef* is fixed across the recursion (attr/owner context
    is the receiver's full hierarchy); *_current* is the class whose method is
    being classified, used to resolve ``super()`` against ITS bases. The recursion
    is guarded by ``(current_class, method)`` — a stable key, since
    :func:`_resolve_class_def` re-parses source and yields fresh node ids each
    call. A ``@staticmethod`` has no receiver binding.
    """
    if _current is None:
        _current = cdef
    if _seen is None:
        _seen = set()
    key = (_current.name, method.name)
    if key in _seen:
        return False, frozenset(), frozenset()
    _seen.add(key)

    class_attrs = _class_level_attr_names(cdef, resolve_class_source)
    instance_attrs = _instance_attr_names(cdef, resolve_class_source)
    recv = _first_param_name(method)
    is_classmethod = _has_named_decorator(method, 'classmethod')
    is_staticmethod = _has_named_decorator(method, 'staticmethod')
    params = _all_param_names(method)
    global_decls: set[str] = set()
    local_assigned: set[str] = set()
    for node in _iter_method_body_nodes(method):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            global_decls.update(node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Name):
                        local_assigned.add(leaf.id)
    local_assigned -= global_decls

    mutates_self = False
    class_targets: set[str] = set()
    free: set[str] = set()

    def _add_class_var(attr):
        owner = _class_var_owner(attr, cdef, resolve_class_source) if attr else None
        class_targets.add(owner or recv_class_name)

    for chain in _iter_inplace_mutation_chains(method):
        root = _extract_base_name(chain)
        if root is None:
            continue
        if not is_staticmethod and recv is not None and root == recv:
            first_attr = _first_attr_after_root(chain)
            if is_classmethod:
                _add_class_var(first_attr)
            elif first_attr is not None and first_attr in instance_attrs:
                mutates_self = True
            elif first_attr is not None and first_attr in class_attrs:
                _add_class_var(first_attr)
            else:
                # A bare ``self`` mutation or an attribute assigned nowhere we can
                # see — attribute it to the instance (the conservative reset).
                mutates_self = True
        elif _resolve_class_def(root, resolve_class_source) is not None:
            # ``ClassName.<attr>`` — the class (own or a base) owns the var.
            class_targets.add(root)
        elif root in params or root in local_assigned:
            # A non-receiver parameter (arg-mutation path) or a local.
            continue
        else:
            free.add(root)

    # ``super().<m>()`` runs the inherited implementation — attribute its hidden
    # mutations too (the first resolvable base of the CURRENT class defining <m>
    # wins, MRO-ish).
    for m_name in _super_called_methods(method):
        for base_name in _class_bases(_current):
            base_cdef = _resolve_class_def(base_name, resolve_class_source)
            base_m = (_class_method(base_cdef, m_name, resolve_class_source)
                      if base_cdef is not None else None)
            if base_m is not None:
                s2, c2, f2 = _classify_method_mutations(
                    base_m, recv_class_name, cdef, resolve_class_source, _seen,
                    base_cdef,
                )
                mutates_self = mutates_self or s2
                class_targets |= c2
                free |= f2
                break

    return mutates_self, frozenset(class_targets), frozenset(free)


def _decorator_free_var_mutations(
    decorator_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Module/free variables mutated by the wrapper a decorator returns
   . ``def logged(f): def wrap(*a): calls.append('x'); ...; return
    wrap`` — calling a ``@logged``-decorated function runs ``wrap``, which
    appends to the module list ``calls``. Collect the free vars each inner
    function mutates that are NOT local to the decorator (those are the
    closure case)."""
    scope = _factory_body_scope(decorator_def)
    out: set[str] = set()
    for node in ast.walk(decorator_def):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node is not decorator_def):
            out |= _free_vars_mutated_in_function(node) - scope
    return frozenset(out)


def _decorator_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The decorator NAMES applied to *func* (``@logged`` → ``logged``,
    ``@app.route(...)`` → ``route``-less, skipped). Bare-name and simple
    ``name(...)`` decorators are resolved by name."""
    names: list[str] = []
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            names.append(dec.func.id)
    return names


# ``obj <op>= x`` dispatches to the in-place operator dunder, falling back to the
# binary form when the in-place one is absent (``obj = obj.__add__(x)``). Maps the
# AST op node name to ``(inplace_dunder, fallback_dunder)``.
_AUGOP_DUNDERS: dict[str, tuple[str, str]] = {
    'Add': ('__iadd__', '__add__'),
    'Sub': ('__isub__', '__sub__'),
    'Mult': ('__imul__', '__mul__'),
    'Div': ('__itruediv__', '__truediv__'),
    'FloorDiv': ('__ifloordiv__', '__floordiv__'),
    'Mod': ('__imod__', '__mod__'),
    'Pow': ('__ipow__', '__pow__'),
    'MatMult': ('__imatmul__', '__matmul__'),
    'BitOr': ('__ior__', '__or__'),
    'BitAnd': ('__iand__', '__and__'),
    'BitXor': ('__ixor__', '__xor__'),
    'LShift': ('__ilshift__', '__lshift__'),
    'RShift': ('__irshift__', '__rshift__'),
}


def object_protocol_mutations(
    tree: ast.Module | None,
    resolve_class_source,
    instance_class,
    resolve_source,
    resolve_var_factory,
    decorated_class=None,
) -> ObjectProtocolResets:
    """Hidden mutations reached through the object protocol.

    Walks the cell's executable nodes and, for each object-protocol invocation —
    a ``with`` statement, a subscript op (``s[k]=v`` / ``del s[k]`` / ``v=s[k]``),
    a call (constructor / instance ``__call__`` / decorated function), or a
    method call — resolves the receiver's class (via *resolve_class_source* +
    *instance_class*), a ``@contextmanager`` generator or decorator (via
    *resolve_source*), or a reassignment-decorator factory (via
    *resolve_var_factory*), analyses the invoked method body, and returns the
    reset targets grouped by channel (see :class:`ObjectProtocolResets`).

    * *resolve_class_source* — ``class_name -> source`` (or ``None``).
    * *instance_class* — ``var -> class_name`` for a ``var = ClassName(...)``
      instance whose class is resolvable (or ``None``).
    * *resolve_source* — ``func_name -> source`` (or ``None``), for
      ``@contextmanager`` generators and decorator functions.
    * *resolve_var_factory* — ``var -> factory FunctionDef`` (or ``None``), for a
      reassignment decorator ``g = counting(g)``.
    * *decorated_class* — ``var -> class_name`` for a class-based decorator
      binding ``@Counter def task`` (or ``None``).
    """
    free_vars: set[str] = set()
    receivers: set[str] = set()
    class_defs: set[str] = set()
    if decorated_class is None:
        def decorated_class(_var):
            return None

    _class_cache: dict[str, ast.ClassDef | None] = {}

    def _classdef(name):
        if name not in _class_cache:
            _class_cache[name] = _resolve_class_def(name, resolve_class_source)
        return _class_cache[name]

    def _apply_method(cdef, class_name, method, recv_var, *, allow_self):
        si, class_targets, fv = _classify_method_mutations(
            method, class_name, cdef, resolve_class_source,
        )
        if fv:
            free_vars.update(fv)
        if class_targets:
            # The class-var reset target is the OWNING class (a base when the var
            # is inherited), which must re-run to recreate the class-level
            # container. When the var is inherited (owner != receiver's class),
            # also reset the receiver's class so its instances re-derive against
            # the fresh base — a subclass method mutating an inherited class var
            # via ``self`` needs both, since the reset cascade is one level deep
            #. For a non-inherited var the owner IS the receiver's class,
            # so this adds nothing.
            class_defs.update(class_targets)
            class_defs.add(class_name)
        if si and allow_self and recv_var is not None:
            receivers.add(recv_var)

    def _apply_ctor(cdef, class_name):
        """A construction ``X()`` — the fresh instance's self-init is discarded, so
        only class-var / free-var mutations in ``__init__`` (or a dataclass
        ``__post_init__``) persist."""
        for ctor in ('__init__', '__post_init__'):
            method = _class_method(cdef, ctor, resolve_class_source)
            if method is not None:
                _apply_method(cdef, class_name, method, None, allow_self=False)

    def _dispatch_dunder(recv_var, dunder):
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        method = _class_method(cdef, dunder, resolve_class_source)
        if method is not None:
            _apply_method(cdef, cls, method, recv_var, allow_self=True)

    def _dispatch_context(recv_var):
        """A context-managed instance ``recv_var`` (``with cm:`` or
        ``stack.enter_context(cm)``): analyse its ``__enter__`` / ``__exit__``."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        for dunder in ('__enter__', '__exit__'):
            method = _class_method(cdef, dunder, resolve_class_source)
            if method is not None:
                _apply_method(cdef, cls, method, recv_var, allow_self=True)

    def _return_class(fdef):
        """The ``(name, ClassDef)`` of a notebook class a factory function
        RETURNS (``def cm(): return Mgr()`` → ``Mgr``), or ``(None, None)``
       . Used for ``with cm() as x:`` where ``cm`` is a plain factory."""
        for sub in ast.walk(fdef):
            if (isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Name)):
                rcdef = _classdef(sub.value.func.id)
                if rcdef is not None:
                    return sub.value.func.id, rcdef
        return None, None

    def _dispatch_descriptor(cdef, attr, dunder):
        """A data descriptor's ``__set__`` / ``__get__`` receives ``self`` (the
        descriptor, a shared class attribute) and ``obj`` (the instance) — both
        parameters — so only its FREE-var side effects are attributed."""
        ddef = _descriptor_class(cdef, attr, resolve_class_source)
        if ddef is None:
            return
        method = _class_method(ddef, dunder, resolve_class_source)
        if method is not None:
            free_vars.update(_free_vars_mutated_in_function(method))

    def _dispatch_attr_set(recv_var, attr):
        """``recv.attr = v`` dispatching to a ``@property`` setter or a data
        descriptor's ``__set__``. A plain attribute assign resolves to
        neither and is a no-op."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        setter = _property_accessor(cdef, attr, 'setter', resolve_class_source)
        if setter is not None:
            _apply_method(cdef, cls, setter, recv_var, allow_self=True)
        else:
            _dispatch_descriptor(cdef, attr, '__set__')

    def _dispatch_attr_get(recv_var, attr):
        """``recv.attr`` (load) dispatching to a ``@property`` getter or a data
        descriptor's ``__get__`` with a side effect."""
        cls = instance_class(recv_var)
        cdef = _classdef(cls) if cls else None
        if cdef is None:
            return
        getter = _property_accessor(cdef, attr, 'getter', resolve_class_source)
        if getter is not None:
            _apply_method(cdef, cls, getter, recv_var, allow_self=True)
        else:
            _dispatch_descriptor(cdef, attr, '__get__')

    if tree is None:
        return ObjectProtocolResets(frozenset(), frozenset(), frozenset())

    nodes = list(_walk_executable(tree))

    for node in nodes:
        # --- with statements: __enter__ / __exit__ or a @contextmanager gen ----
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Name):
                    _dispatch_context(ctx.id)
                elif isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name):
                    nm = ctx.func.id
                    cdef = _classdef(nm)
                    if cdef is not None:
                        # ``with SomeCM():`` — anonymous instance, no receiver to
                        # reset; only class-var / free-var mutations persist.
                        for dunder in ('__enter__', '__exit__'):
                            method = _class_method(cdef, dunder, resolve_class_source)
                            if method is not None:
                                _apply_method(cdef, nm, method, None, allow_self=False)
                    else:
                        fdef = _resolve_function_def(nm, resolve_source)
                        if fdef is not None:
                            # A ``@contextmanager`` generator's free-var mutations,
                            free_vars.update(_free_vars_mutated_in_function(fdef))
                            # or a plain factory ``def cm(): return Mgr()`` — the
                            # returned instance's __enter__/__exit__ run anonymously
                            # (``with cm() as x:``), so only class-var / free-var
                            # mutations persist.
                            ret_name, ret_cdef = _return_class(fdef)
                            if ret_cdef is not None:
                                for dunder in ('__enter__', '__exit__'):
                                    method = _class_method(ret_cdef, dunder, resolve_class_source)
                                    if method is not None:
                                        _apply_method(ret_cdef, ret_name, method, None, allow_self=False)
        # --- subscript operations dispatching to custom dunders ----------------
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for leaf in _iter_store_targets(tgt):
                    if isinstance(leaf, ast.Subscript) and isinstance(leaf.value, ast.Name):
                        _dispatch_dunder(leaf.value.id, '__setitem__')
                    elif isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name):
                        _dispatch_attr_set(leaf.value.id, leaf.attr)
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                    _dispatch_dunder(tgt.value.id, '__delitem__')
        # --- ``obj <op>= x`` dispatching to an in-place operator dunder ---------
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            cls = instance_class(node.target.id)
            cdef = _classdef(cls) if cls else None
            dunders = _AUGOP_DUNDERS.get(type(node.op).__name__)
            if cdef is not None and dunders is not None:
                inplace_dunder, fallback_dunder = dunders
                method = _class_method(cdef, inplace_dunder, resolve_class_source)
                if method is not None:
                    # ``__iadd__`` mutates the receiver in place (returns self).
                    _apply_method(cdef, cls, method, node.target.id, allow_self=True)
                else:
                    # No in-place form: ``obj += x`` REASSIGNS obj to a fresh
                    # ``obj.__add__(x)`` (idempotent), so only its free-var /
                    # class-var side effects persist.
                    method = _class_method(cdef, fallback_dunder, resolve_class_source)
                    if method is not None:
                        _apply_method(cdef, cls, method, node.target.id, allow_self=False)
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
              and isinstance(node.value, ast.Name)):
            _dispatch_dunder(node.value.id, '__getitem__')
        # --- ``recv.attr`` load dispatching to a property getter / __get__ -----
        elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
              and isinstance(node.value, ast.Name)):
            _dispatch_attr_get(node.value.id, node.attr)
        # --- calls: constructor / instance __call__ / decorated fn / method ----
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                nm = func.id
                # ``next(it)`` advances the iterator via ``It.__next__``.
                if nm == 'next' and node.args and isinstance(node.args[0], ast.Name):
                    _dispatch_dunder(node.args[0].id, '__next__')
                    continue
                cdef = _classdef(nm)
                if cdef is not None:
                    _apply_ctor(cdef, nm)
                    continue
                # A class-based decorator ``@Counter def task`` — ``task`` is a
                # Counter INSTANCE holding the wrapped function, so it is a
                # stateful callable: calling it runs ``Counter.__call__`` (which
                # mutates ``self.n``). Re-run its producer (the decorated def) to
                # reset — the receiver's content is unhashable (holds a function),
                # so route a self-mutation to the class-def channel.
                deco_cls = decorated_class(nm)
                dcdef = _classdef(deco_cls) if deco_cls else None
                if dcdef is not None:
                    call_m = _class_method(dcdef, '__call__', resolve_class_source)
                    if call_m is not None:
                        si, ct, fv = _classify_method_mutations(
                            call_m, deco_cls, dcdef, resolve_class_source,
                        )
                        free_vars.update(fv)
                        class_defs.update(ct)
                        if si:
                            class_defs.add(nm)
                    continue
                # An instance called via __call__ (``a('z')``).
                cls = instance_class(nm)
                icdef = _classdef(cls) if cls else None
                if icdef is not None:
                    call_m = _class_method(icdef, '__call__', resolve_class_source)
                    if call_m is not None:
                        _apply_method(icdef, cls, call_m, nm, allow_self=True)
                # A decorated function whose wrapper mutates a free var, or a
                # reassignment decorator ``g = counting(g)``.
                fdef = _resolve_function_def(nm, resolve_source)
                if fdef is not None:
                    for dname in _decorator_names(fdef):
                        ddef = _resolve_function_def(dname, resolve_source)
                        if ddef is not None:
                            free_vars.update(_decorator_free_var_mutations(ddef))
                factory = resolve_var_factory(nm) if resolve_var_factory else None
                if factory is not None:
                    free_vars.update(_decorator_free_var_mutations(factory))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                recv = func.value.id
                method_name = func.attr
                # ``stack.enter_context(cm)`` runs ``cm.__enter__`` / ``__exit__``
                # regardless of what ``stack`` is (an ExitStack), so dispatch to
                # the ARGUMENT's context manager.
                if method_name == 'enter_context' and node.args and isinstance(node.args[0], ast.Name):
                    _dispatch_context(node.args[0].id)
                    continue
                own_class = _classdef(recv)
                if own_class is not None:
                    # A class-level call ``Registry.record()`` — no instance.
                    method = _class_method(own_class, method_name, resolve_class_source)
                    if method is not None:
                        _apply_method(own_class, recv, method, None, allow_self=False)
                else:
                    cls = instance_class(recv)
                    cdef = _classdef(cls) if cls else None
                    if cdef is not None:
                        method = _class_method(cdef, method_name, resolve_class_source)
                        if method is not None:
                            _apply_method(cdef, cls, method, recv, allow_self=True)

    # --- top-level ``class Sub(Base):`` triggering a base __init_subclass__ -----
    # A subclass def runs the nearest base's ``__init_subclass__(cls, ...)`` hook
    # during CLASS CREATION (before any node in this cell's body executes), so it
    # is invisible to the executable-node walk above. The hook receives ``cls``
    # (the fresh subclass, discarded on re-derivation) — analyse its body exactly
    # like a constructor (``allow_self=False``) so only its class-var / free-var
    # mutations persist. A class-var accumulator (``Base.registry``) routes to the
    # class-def channel (already under the cross-cell guard); a module/free
    # var (``registry.append``) is collected SEPARATELY so the caller can apply the
    # same guard — the simulator cannot see this hidden cross-cell mutation.
    # Only ``__init_subclass__`` is handled here (metaclass hooks / __set_name__
    # are separate follow-ups). ClassDefs are deferred scopes so the walk above
    # never yields them — scan ``tree.body`` directly.
    init_subclass_free: set[str] = set()
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.bases):
            continue
        for base_name in _class_bases(node):
            base_cdef = _classdef(base_name)
            if base_cdef is None:
                continue
            hook = _class_method(base_cdef, '__init_subclass__', resolve_class_source)
            if hook is not None:
                _, class_targets, fv = _classify_method_mutations(
                    hook, base_name, base_cdef, resolve_class_source,
                )
                class_defs.update(class_targets)
                init_subclass_free.update(fv)
                break

    return ObjectProtocolResets(
        frozenset(free_vars), frozenset(receivers), frozenset(class_defs),
        frozenset(init_subclass_free),
    )


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


# ---------------------------------------------------------------------------
# Consumption detection (consumable / producer-re-execution engine)
# ---------------------------------------------------------------------------

# Builtins that merely *inspect* a name without advancing it. Everything else
# that receives the name as an argument is assumed to consume it — see
# ``consumed_input_names``.
_NON_CONSUMING_FUNCS = frozenset({
    'type', 'id', 'repr', 'isinstance', 'issubclass', 'hasattr', 'getattr',
    'setattr', 'delattr', 'callable', 'hash', 'dir', 'vars', 'print',
    'len', 'format',
})

# Methods that report on a consumable without drawing from it. Being wrong here
# means a genuine consumption goes undetected (the producer is not re-run and
# the stale-value bug survives), so the set stays small and unambiguous.
_NON_CONSUMING_METHODS = frozenset({
    # queue.Queue / SimpleQueue introspection
    'qsize', 'empty', 'full', 'task_done', 'join',
    # file-handle introspection (``seek``/``tell`` do not read bytes; ``seek``
    # in particular REWINDS, but the divergence probe compares positions and a
    # rewound handle legitimately reads from the new offset)
    'tell', 'fileno', 'seekable', 'readable', 'writable', 'flush', 'isatty',
})


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
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id in _NON_CONSUMING_FUNCS
                and name_node is not parent.func):
            return False
        if isinstance(parent, ast.Attribute):
            grand = parents.get(parent)
            # ``q.qsize()`` / ``fh.tell()`` — receiver of a reporting method.
            if (isinstance(grand, ast.Call) and grand.func is parent
                    and parent.attr in _NON_CONSUMING_METHODS):
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
   . Deferred scopes (def / class) are not descended into. Only a bare
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
        elif (isinstance(target, (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)):
            # NESTED 1:1 literal unpack -- ``(p, (q,)) = (x, (y,))``. Nesting
            # changes the shape of the unpack, not the aliasing: every leaf still
            # shares its partner's object. Binding only the outer level left
            # ``q`` unmapped, so a mutation through it (``q.append(9)``) was not
            # attributed to ``y`` and an idempotent re-run appended twice.
            for elt_target, elt_value in zip(target.elts, value.elts):
                _bind(elt_target, elt_value)

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
    object with, so the holder resets on an isolated re-run. Names that
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


def bare_alias_targets(tree: ast.Module | None) -> frozenset[str]:
    """Names bound by a top-level statement that does NOTHING but pointer-copy a
    bare ``Name`` — ``b = a``, ``b = c = a``, ``b, c = a, d``.

    Such a statement must never be cached. Two independent reasons, either alone
    sufficient:

    * **Correctness.** ``b = a`` binds *the same object* to a second name; Python
      guarantees ``b is a``. A cache hit rebinds ``b`` to a DESERIALISED COPY, so
      the two names silently stop being the same object and a later mutation
      through ``a`` (``a.fit(..)``, ``a.append(..)``) is invisible through ``b``.
      The window opens on the SECOND warm re-run — the first re-run still
      re-executes — which is why a one-repetition test reports it as working.
    * **Cost.** A pointer copy is free. Caching it serialises and deserialises a
      whole object to avoid a nanosecond of work, so refusing is also a strictly
      cheaper default. There is no configuration in which caching this shape wins.

    Deliberately NARROW — only bindings that are provably pure pointer copies:

    * every target of a ``Name``-valued assign must itself be a plain ``Name``, so
      ``b, c = a`` (an unpack that INDEXES ``a``, binding ``a[0]``/``a[1]``, not
      ``a``) is excluded;
    * the tuple form is restricted to a 1:1 literal unpack of bare ``Name``\\ s, so
      ``b, c = a, f()`` (``f()`` is real work worth caching) is excluded;
    * anything computed from a name — ``b = a.attr``, ``b = a[0]``, ``b = f(a)``,
      ``b = a.copy()`` — is excluded. Those CAN alias a live mutable object too,
      but they can equally be expensive, so the cost half of the argument does not
      transfer and they keep their cache.

    Self-binds (``x = x``) are skipped, matching :func:`_cell_alias_map`.

    The statement still executes and still participates in lineage: the caller
    only refuses to store/restore the value (``capture_and_track_variables`` runs
    unconditionally), so downstream cache keys and the upstream simulation are
    unaffected.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        # ``b = a`` / ``b = c = a``: each target names the very same object.
        if isinstance(value, ast.Name):
            if all(isinstance(t, ast.Name) for t in node.targets):
                out.update(
                    t.id for t in node.targets
                    if isinstance(t, ast.Name) and t.id != value.id
                )
            continue
        # ``b, c = a, d``: the RHS tuple is built and unpacked element-wise, so
        # every binding is its own pointer copy. Requires equal arity and bare
        # ``Name``\\ s on both sides (no ``*rest``, no computed element).
        if isinstance(value, (ast.Tuple, ast.List)) and len(node.targets) == 1:
            aliases = _literal_unpack_aliases(node.targets[0], value)
            if aliases:
                out.update(aliases)
    return frozenset(out)


def _literal_unpack_aliases(target: ast.expr, value: ast.expr) -> set[str] | None:
    """Names pointer-copied by a 1:1 literal unpack, recursing through nesting.

    ``b, c = a, d`` binds every name to the very same object the matching RHS
    name holds — and so does ``(p, (q,)) = (x, (y,))``. Nesting changes the shape
    of the unpack, not the fact that each leaf is a pointer copy.

    Handling only the FLAT form left the nested one looking like ordinary work,
    so it kept its cache. A later mutation through the nested alias
    (``q.append(9)``) was then recorded against a name cash did not know aliased
    ``y``, and an idempotent re-run appended a second time — ``[3, 4, 9, 9]``.

    Returns ``None`` when the shape is not a pure 1:1 literal unpack, which keeps
    the all-or-nothing rule this generalises: one ``*rest`` or computed element
    anywhere opts the whole statement out, exactly as before, so a genuinely
    expensive element (``b, c = a, f()``) keeps its cache rather than being
    refused because a sibling happened to be an alias.
    """
    if isinstance(target, ast.Name) and isinstance(value, ast.Name):
        return {target.id} if target.id != value.id else set()
    if (isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)):
        found: set[str] = set()
        for elt_target, elt_value in zip(target.elts, value.elts):
            nested = _literal_unpack_aliases(elt_target, elt_value)
            if nested is None:
                return None
            found |= nested
        return found
    return None


def _is_free_reference_expr(node: ast.expr) -> bool:
    """True for an expression that only DEREFERENCES existing state.

    These are the shapes that share BOTH halves of the argument, which is
    what makes them safe to refuse:

    * they alias — the result can be a live sub-object of a tracked variable, so
      restoring a deserialised copy silently breaks the identity Python
      guarantees; and
    * they are FREE to re-run — an attribute lookup or a constant-key subscript
      is a pointer dereference, so refusing to cache is also strictly cheaper.

    ``bare_alias_targets``' docstring lumps these in with ``b = f(a)`` and
    excludes them all on the grounds that they "can equally be expensive". That
    is true of a CALL, but not of a deref: whatever built ``obj.inner`` is cached
    at its own statement; re-reading the attribute costs nothing.

    Accepted, rooted in a bare ``Name`` so the base is a live tracked variable:

    * ``a.attr``, ``a.b.c``        — attribute chains
    * ``a[0]``, ``a['k']``         — subscript with a LITERAL key
    * ``a if cond else b``         — ternary whose branches are themselves free

    Deliberately NOT accepted:

    * any ``Call`` (``f(a)``, ``list(a)``, ``a.copy()``) — may do real work, so
      the cost half does not transfer. ``b = list(a)`` additionally aliases only
      one level down (``b[0] is a[0]`` while ``b is not a``), which refusing to
      cache the binding would not fix anyway.
    * a subscript with a NON-literal key (``a[i]``, ``a[mask]``, ``a[1:]``) —
      ``df[mask]`` is a filter that does real work and must keep its cache.
    """
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_free_reference_expr(node.value)
    if isinstance(node, ast.Subscript):
        # Literal key only: a computed key can be an expensive filter.
        key = node.slice
        if not isinstance(key, ast.Constant):
            return False
        return _is_free_reference_expr(node.value)
    if isinstance(node, ast.IfExp):
        return (_is_free_reference_expr(node.body)
                and _is_free_reference_expr(node.orelse))
    if isinstance(node, ast.Constant):
        return True   # the ``else None`` arm of a ternary
    return False


def _root_name(node: ast.expr) -> str | None:
    """The base variable a dereference chain is rooted in, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def reference_alias_targets(
    tree: ast.Module | None,
    user_ns: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Names bound by a pure DEREFERENCE of live state — the dereference half.

    ``b = obj.inner`` / ``b = holder['k']`` / ``b = lst[0]`` / ``b = obj if c
    else None`` each bind the *same object* that is already reachable through a
    tracked variable. A cache hit rebinds the target to a deserialised copy, so
    ``b is obj.inner`` stops holding and a later mutation through ``obj`` is
    invisible through ``b`` — measured divergence from a ``%cash_off`` kernel on
    the FIRST warm re-run, surviving a kernel restart.

    Same enforcement as :func:`bare_alias_targets`: the statement still executes
    and still participates in lineage; only store/restore is refused.
    """
    if tree is None:
        return frozenset()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        # The RHS must itself be a DEREFERENCE. Gating on the top-level node type
        # matters: ``_is_free_reference_expr`` accepts a bare ``Name`` (a valid
        # base) and a ``Constant`` (a valid ternary arm), but neither can alias
        # as a whole RHS -- ``b = a`` is bare_alias_targets' job, and ``x = 21``
        # binds a fresh immutable. Accepting Constant here refused caching for
        # every plain literal assignment in the suite.
        if not isinstance(node.value, (ast.Attribute, ast.Subscript, ast.IfExp)):
            continue
        if not _is_free_reference_expr(node.value):
            continue
        # A MODULE attribute (``v = mod.VERSION``) is not an alias hazard worth
        # refusing: module-level names are overwhelmingly immutable constants and
        # functions, so there is no live object whose identity a restore could
        # break -- and granular module-dependency invalidation relies on these
        # bindings being cached. Only skip when we can SEE it is a module; with
        # no namespace we keep the conservative refusal.
        if user_ns is not None:
            root = _root_name(node.value)
            if root is not None and isinstance(user_ns.get(root), types.ModuleType):
                continue
        if all(isinstance(t, ast.Name) for t in node.targets):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(out)


def alias_mutation_sources(tree: ast.Module | None) -> frozenset[str]:
    """Upstream variables whose object is mutated in place through an alias.

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
        base = _extract_receiver_base_name(call.func.value)
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
        base = _extract_receiver_base_name(call.func.value)
        if base:
            calls.add((base, call.func.attr))
    return frozenset(calls)


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
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        target_name = node.targets[0].id
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Name)
                and sub.id == target_name
                and isinstance(sub.ctx, ast.Load)
            ):
                return frozenset({target_name})
    return frozenset()


# ---------------------------------------------------------------------------
# Accumulator-loop shape detection
# ---------------------------------------------------------------------------
#
# CAS-259 history: this used to be consulted directly by a dispatcher in
# ``control_structures/processor.py`` that routed a matching loop through the
# statement cache as one unit BEFORE the cost-based
# ``_should_execute_loop_as_single_unit`` check ever ran -- so every
# accumulator loop, however cheap, skipped per-iteration decomposition and
# interception. CAS-259 deleted that dispatch. That was too broad a deletion:
# a CAS-259 follow-up review (measured on a 150-iteration, 4.6s-body loop)
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
    'append': frozenset({'list'}),
    'extend': frozenset({'list'}),
    'add': frozenset({'set'}),
    'update': frozenset({'dict', 'set'}),
}


def _empty_seed_kind(node: ast.expr) -> str | None:
    """Container kind a FRESH-EMPTY seed expression *node* produces, or ``None``.

    Recognises only the empty seed forms: ``[]`` / ``list()`` -> ``'list'``,
    ``{}`` / ``dict()`` -> ``'dict'``, ``set()`` -> ``'set'``. A non-empty
    literal (``[0]``, ``{1}``, ``{'k': 1}``) or any computed expression returns
    ``None`` so a pre-seeded accumulator is never matched.
    """
    if isinstance(node, ast.List) and not node.elts:
        return 'list'
    if isinstance(node, ast.Dict) and not node.keys:
        return 'dict'
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and not node.args and not node.keywords):
        return {'list': 'list', 'dict': 'dict', 'set': 'set'}.get(node.func.id)
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
    if not (isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)):
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
    for_node: ast.For, prev_node: ast.stmt | None,
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
    if not (isinstance(prev_node, ast.Assign) and len(prev_node.targets) == 1
            and isinstance(prev_node.targets[0], ast.Name)):
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
    if not (isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == acc):
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



#: ``source text -> globals that function's body mutates in place``.
#:
#: The verdict is purely syntactic -- "is this name a parameter, a plain local,
#: or free?" is read off the function's own AST with no reference to any
#: namespace -- so keying on the source text alone is sound, and it is what
#: keeps this affordable on the per-statement hot path.
_CALLEE_GLOBALS_BY_SOURCE: dict[str, frozenset[str]] = {}


def _globals_mutated_by_callees(names, resolve_source) -> frozenset[str]:
    """Union of the globals each named callee mutates in place."""
    out: set[str] = set()
    for name in names:
        try:
            source = resolve_source(name)
        except Exception:  # noqa: BLE001 - a resolver must never break analysis
            continue
        if not source:
            continue
        cached = _CALLEE_GLOBALS_BY_SOURCE.get(source)
        if cached is None:
            cached = callee_source_global_mutations(source)
            _CALLEE_GLOBALS_BY_SOURCE[source] = cached
        out |= cached
    return frozenset(out)


def _called_name_scopes(tree) -> tuple[frozenset[str], frozenset[str]]:
    """``(all_called, top_level_called)`` bare-``Name`` callees.

    Two walks, mirroring the two mutation visitors in
    :func:`analyze_statement` exactly, so a mutation PROPAGATED from a callee
    lands in the same two sets an inline one would: the full walk feeds
    ``all_mutated_vars``, and the walk that skips nested function/class bodies
    feeds ``top_level_mutated_vars``. A loop body is top level by this rule --
    which is correct, and is why the inline spelling of the same mutation is
    already surfaced there.
    """
    all_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            all_names.add(node.func.id)
    top_names: set[str] = set()
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(child):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                top_names.add(node.func.id)
    return frozenset(all_names), frozenset(top_names)



def callee_mutated_globals_for_tree(tree, resolve_source, user_ns=None) -> frozenset[str]:
    """Globals mutated in place by any function called anywhere in *tree*.

    The input/output half of CAS-265's propagation, and deliberately the SAME
    per-callee analysis :func:`analyze_statement` uses for the mutation half --
    one verdict feeding both, so a name can never be declared a mutation
    without also being declared a read and a write.
    """
    if tree is None:
        return frozenset()
    names = _globals_mutated_by_callees(_called_name_scopes(tree)[0], resolve_source)
    if user_ns is None:
        return names
    # Filter to names that are REAL notebook variables, mirroring
    # ``StatementProcessor._callee_mutated_globals``. Without this the analysis
    # invents variables: ``_free_vars_mutated_in_function`` reports every free
    # name a callee writes, including CLOSURE cells, which live in a cell object
    # and never appear in the namespace at all::
    #
    #     def make():
    #         total = 0
    #         def add(v):
    #             nonlocal total
    #             total += v
    #         return add
    #
    #     final = add(15)   -> declared inputs/outputs ['add','history','total']
    #
    # Declaring `total` an output makes reconstruction try to PRODUCE it, which
    # re-runs the statement -- measured, ``add(15)`` executed twice on a FIRST
    # run and `final` came out 55 where 40 is correct. A module is excluded for
    # the same reason it is everywhere else: never a value to serialise.
    import types as _types
    return frozenset(
        n for n in names
        if n in user_ns and not isinstance(user_ns[n], _types.ModuleType)
    )


def analyze_statement(
    code: str,
    tree: ast.Module | None,
    user_ns: Mapping[str, Any] | None = None,
    resolve_source=None,
) -> StatementAnalysis:
    """Return a :class:`StatementAnalysis` for *code* using pure-AST analysis.

    Analysis is pure AST with one exception: *user_ns*, when supplied, is read
    ONLY to tell a module apart from an ordinary object, so
    :func:`reference_alias_targets` can exempt ``v = mod.CONST`` from the alias
    refusal. Omitting it keeps the conservative refusal.

    Args:
        code: Python source code of the statement.
        tree: Optional pre-parsed AST.  When ``None`` the code is parsed
              here; a :class:`SyntaxError` produces an empty analysis
              rather than raising.
        user_ns: Optional live namespace, used only for the module check above.
        resolve_source: Optional ``name -> source`` for called functions. When
            supplied, globals a CALLEE mutates in place are propagated into
            the mutation sets as though the mutation had been written inline
            at the call site (CAS-265).

            This is the single seam that makes a callee's write visible to
            every consumer at once -- the checker's idempotent-rerun reset
            (``current_cell_mutated``), the loop handler's mutated-variable set
            (and therefore the loop's outputs, which is what records a producer
            for cross-cell reconstruction), and the cacheability decision's
            ``pure_mutations``. Wiring those separately was tried and each one
            alone makes the tree worse than leaving the write dropped.

            Omitting it (``None``, the default) keeps pure-AST behaviour
            exactly as before, so a caller with no way to resolve callee source
            is unchanged rather than silently degraded.
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

    # CAS-265: a global mutated INSIDE a called function is invisible to the
    # visitors above -- the mutation is not in this statement's source. Fold it
    # in here, at the one place every consumer already reads, so the write is
    # treated exactly as the inline spelling of it would be.
    if resolve_source is not None:
        all_called, top_called = _called_name_scopes(tree)
        all_mutated = all_mutated | _globals_mutated_by_callees(all_called, resolve_source)
        top_level_mutated = top_level_mutated | _globals_mutated_by_callees(
            top_called, resolve_source)

    # Top-level vars grown by an accumulator method (append/extend/add/update) —
    # the only mutations that earn the comprehension guidance hint (b).
    accumulator_mutated = frozenset(
        m.variable
        for m in top_level_visitor.mutations
        if m.kind == 'method_call' and m.method in ACCUMULATOR_METHODS
    )

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
        accumulator_mutated_vars=accumulator_mutated,
        alias_targets=bare_alias_targets(tree) | reference_alias_targets(tree, user_ns),
    )
