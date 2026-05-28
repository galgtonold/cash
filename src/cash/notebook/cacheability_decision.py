"""The runtime merge layer for the cacheability decision.

``cacheability.py`` owns the pure-AST half (``StatementAnalysis``).  This
module owns the **merge**: it combines that AST analysis with the user
annotation, the ``@stateful`` registry, the forbidden-function scan, and
the variable-lineage state into one verdict.

The merge has five reason-sources.  The first that triggers wins; later
sources are not consulted.  This matches the historical behavior of
``StatementProcessor._check_skip_conditions`` and keeps the per-statement
hot path short.

Reason-source order (deterministic):

1. ``@cash:no-cache`` annotation
2. Forbidden function calls (e.g. ``input``)
3. ``@stateful`` function calls
4. In-place mutations + side effects (from ``StatementAnalysis``)
5. Input variable missing lineage

The function takes the runtime hooks (purity lookup, forbidden scan) as
callables so this module does not have to import ``purity`` or
``analysis``.  That keeps the dependency picture in this file honest:
every input the decision reads is in the signature.

The "should this input be skipped for lineage-check purposes?" predicate
is inlined as ``_is_lineage_exempt`` — it's purely a property of the
value (module type, private callable, etc.) and has no production
override, so no hook indirection is justified.

See ``CONTEXT.md`` entry: *Cacheability decision*.
"""
from __future__ import annotations

import ast
import builtins
import logging
import types
from collections.abc import Callable, Mapping
from typing import Any

from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cacheability import StatementAnalysis

logger = logging.getLogger(__name__)

__all__ = ["decide_cacheability"]

# Builtin-ish names that never need lineage tracking.  Kept in module scope
# so the per-input loop does not rebuild ``set(dir(builtins))`` on every call.
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))
_SKIP_INPUT_NAMES: frozenset[str] = frozenset(
    {'get_ipython', '__builtins__', 'print', '__name__', '__doc__'}
)


def _is_lineage_exempt(var_name: str, val: Any) -> bool:
    """Return True if *val* is a kind of input that never needs lineage tracking.

    Modules, ``get_ipython``, and bound / private callables (whose source
    is captured by other means) are exempt.  Used by ``_has_missing_lineage``
    to decide whether absence of a lineage entry is a cacheability blocker.
    """
    if isinstance(val, types.ModuleType) or var_name == 'get_ipython':
        return True
    return bool(callable(val) and (var_name.startswith('_') or hasattr(val, '__self__')))


def decide_cacheability(
    *,
    code: str,
    tree: ast.Module | None,
    inputs: set[str],
    outputs: set[str],
    annotation: CacheAnnotation | None,
    analysis: StatementAnalysis,
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
    is_stateful_call: Callable[[str], bool],
    scan_forbidden: Callable[[str, Mapping[str, Any], ast.Module | None], list[str]],
) -> tuple[bool, list[str]]:
    """Return ``(cacheable, reasons)`` for a statement.

    ``cacheable`` is ``True`` only when *all* reason-sources are silent.
    ``reasons`` is the list of human-readable strings that populate
    ``metrics['uncacheable_reasons']``.  An empty list means "cacheable."
    """
    if annotation is not None and annotation.no_cache:
        return False, ['@cash:no-cache annotation']

    try:
        forbidden = scan_forbidden(code, user_ns, tree)
        if forbidden:
            return False, list(forbidden)
    except (TypeError, AttributeError, SyntaxError) as exc:
        logger.debug("Error scanning for forbidden functions: %s", exc)

    try:
        for name in analysis.called_names:
            if is_stateful_call(name):
                return False, ["Calls @stateful function"]
    except (TypeError, AttributeError) as exc:
        logger.debug("Error checking function purity: %s", exc)

    ast_reasons = analysis.skip_reasons(outputs)
    if ast_reasons:
        return False, ast_reasons

    if _has_missing_lineage(inputs, user_ns, variable_lineage):
        return False, ['Input variable missing lineage']

    return True, []


def _has_missing_lineage(
    inputs: set[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
) -> bool:
    """Return True if any input variable lacks tracked lineage."""
    for var_name in inputs:
        if var_name in _SKIP_INPUT_NAMES or var_name in _BUILTIN_NAMES:
            continue
        if var_name not in user_ns:
            return True
        if var_name not in variable_lineage:
            val = user_ns[var_name]
            if not _is_lineage_exempt(var_name, val):
                return True
    return False
