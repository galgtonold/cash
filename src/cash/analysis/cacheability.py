"""The analysis of one notebook statement: :func:`analyze_statement`.

Assembles a :class:`StatementAnalysis` from the concern modules --
:mod:`~cash.analysis.mutations`, :mod:`~cash.analysis.file_effects`,
:mod:`~cash.analysis.aliases`, :mod:`~cash.analysis.callee_effects` -- which are
pure AST. What needs the live namespace is in
:mod:`cash.analysis.namespace_effects`; the merge with annotations and runtime
state into a verdict is :mod:`cash.analysis.cacheability_decision`.
"""

from __future__ import annotations

import ast
import re
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._memo import STATEMENTS, LruMemo
from ..effects import EffectKind
from .aliases import aliased_sources, bare_alias_targets, cell_alias_map, reference_alias_targets
from .ast_util import called_names
from .callee_effects import callee_global_mutations
from .file_effects import WRITE_TEXT_MARKERS, SideEffectInfo, SideEffectVisitor
from .mutations import ACCUMULATOR_METHODS, MutationVisitor

__all__ = ["statement_writes_files", "StatementAnalysis", "alias_mutation_sources", "analyze_statement"]


def statement_writes_files(code: str, tree: "ast.Module | None" = None) -> bool:
    """True when *code* contains a file-WRITE side effect.

    Used by the upstream simulation to give file-writing statements a trace
    entry and by the re-execution planner to schedule stale writers — file
    writes have no variable edge, so lineage alone never re-runs them.
    Cheap: a textual marker pre-filter runs before the AST analysis.
    """
    if not any(m in code for m in WRITE_TEXT_MARKERS):
        return False
    try:
        analysis = analyze_statement(code, tree)
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(e.effect_kind is EffectKind.FILE_WRITE for e in analysis.side_effects)


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

    def skip_reasons(self, outputs: set[str], *, side_effects: bool = True) -> list[str]:
        """Render structured findings as human-readable skip reasons.

        Used to populate ``metrics['uncacheable_reasons']``.

        Args:
            outputs: Variable names that are *outputs* of this statement.
                     Mutations on outputs are expected and do not block
                     caching (the output itself gets a fresh lineage).
            side_effects: False leaves the side effects out -- the statement
                     carries ``# @cash:assume-safe``.
        """
        reasons: list[str] = []
        # An alias bind (``b = a``) is free to execute and MUST NOT be restored:
        # a hit rebinds the target to a deserialised copy, silently breaking the
        # ``b is a`` identity Python guarantees. Reported first — it is
        # a property of the statement's shape, not of its effects.
        if self.alias_targets:
            names = ", ".join(sorted(self.alias_targets))
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
                    "tip: assign the result to cache it — e.g. `out = [f(e) for e in it]` instead of a for-append loop"
                )
        if side_effects:
            reasons.extend(f"Side effect: {e.description} ({e.kind})" for e in self.side_effects)
        return reasons


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
    alias_map = cell_alias_map(tree)
    if not alias_map:
        return frozenset()
    try:
        mutated = set(analyze_statement(ast.unparse(tree), None).all_mutated_vars)
    except (SyntaxError, ValueError, TypeError):
        return frozenset()
    return aliased_sources(tree, mutated)


#: ``(code, the identifiers in it that name a module) -> StatementAnalysis``.
#: See ``analyze_statement``.
_ANALYSIS_MEMO: LruMemo[tuple[str, frozenset], StatementAnalysis] = LruMemo(STATEMENTS)


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def analyze_statement(
    code: str,
    tree: ast.Module | None,
    user_ns: Mapping[str, Any] | None = None,
    resolve_source=None,
) -> StatementAnalysis:
    """Memoised front of :func:`_analyze_statement` -- see there.

    A loop body's statements are analysed on every iteration, and the analysis
    is pure AST apart from telling a module apart from an ordinary object. A
    631-iteration loop spent 15% of its cash overhead re-walking the same four
    statements. So the result is keyed on the code and on
    which of its identifiers are bound to modules right now -- everything it
    reads from *user_ns*. Only without *resolve_source*, whose answers about
    callee source can change under it. The result is a frozen dataclass of
    immutable fields, so sharing it is safe.
    """
    modules: frozenset[str] = frozenset()
    if user_ns is not None:
        modules = frozenset(n for n in set(_IDENTIFIER.findall(code)) if isinstance(user_ns.get(n), types.ModuleType))
    if resolve_source is not None:
        return _analyze_statement(code, tree, modules, resolve_source)
    key = (code, modules)
    found = _ANALYSIS_MEMO.get(key)
    if found is None:
        found = _analyze_statement(code, tree, modules, resolve_source)
        _ANALYSIS_MEMO[key] = found
    return found


def _analyze_statement(
    code: str,
    tree: ast.Module | None,
    module_names: frozenset[str] = frozenset(),
    resolve_source=None,
) -> StatementAnalysis:
    """Return a :class:`StatementAnalysis` for *code* using pure-AST analysis.

    *module_names* are the identifiers in *code* bound to modules, which
    :func:`reference_alias_targets` exempts from the alias refusal
    (``v = mod.CONST``); :func:`analyze_statement` reads them from the
    namespace.

    Args:
        code: Python source code of the statement.
        tree: Optional pre-parsed AST.  When ``None`` the code is parsed
              here; a :class:`SyntaxError` produces an empty analysis
              rather than raising.
        module_names: Names bound to modules, for the check above.
        resolve_source: Optional ``name -> source`` for called functions. When
            supplied, globals a CALLEE mutates in place are propagated into
            the mutation sets as though the mutation had been written inline
            at the call site.

            This is the single seam that makes a callee's write visible to
            every consumer at once -- the checker's idempotent-rerun reset
            (``current_cell_mutated``), the loop handler's mutated-variable set
            (and therefore the loop's outputs, which is what records a producer
            for cross-cell reconstruction), and the cacheability decision's
            ``pure_mutations``. Wiring those separately was tried and each one
            alone makes the tree worse than leaving the write dropped.

            Omitted (``None``), only the statement's own text is analysed.
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
    full_visitor = MutationVisitor()
    full_visitor.visit(tree)
    all_mutated = frozenset(m.variable for m in full_visitor.mutations)

    # --- Top-level mutations (skip function/class bodies) ---
    top_level_visitor = MutationVisitor()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        top_level_visitor.visit(node)
    top_level_mutated = frozenset(m.variable for m in top_level_visitor.mutations)

    # A global mutated INSIDE a called function is invisible to the
    # visitors above -- the mutation is not in this statement's source. Fold it
    # in here, at the one place every consumer already reads, so the write is
    # treated exactly as the inline spelling of it would be.
    if resolve_source is not None:
        all_mutated = all_mutated | callee_global_mutations(tree, resolve_source)
        top_level_mutated = top_level_mutated | callee_global_mutations(tree, resolve_source, scope="top_level")

    # Top-level vars grown by an accumulator method (append/extend/add/update) —
    # the only mutations that earn the comprehension guidance hint (b).
    accumulator_mutated = frozenset(
        m.variable for m in top_level_visitor.mutations if m.kind == "method_call" and m.method in ACCUMULATOR_METHODS
    )

    # --- Side effects ---
    se_visitor = SideEffectVisitor()
    se_visitor.visit(tree)

    return StatementAnalysis(
        top_level_mutated_vars=top_level_mutated,
        all_mutated_vars=all_mutated,
        side_effects=tuple(se_visitor.effects),
        called_names=called_names(tree),
        accumulator_mutated_vars=accumulator_mutated,
        alias_targets=bare_alias_targets(tree) | reference_alias_targets(tree, module_names),
    )
