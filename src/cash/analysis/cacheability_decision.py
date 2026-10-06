"""The runtime merge layer for the cacheability decision.

``cacheability.py`` owns the pure-AST half (``StatementAnalysis``).  This
module owns the **merge**: it combines that AST analysis with the user
annotation, the ``@stateful`` registry, the forbidden-function scan, and
the variable-lineage state into one verdict.

The merge has five reason-sources.  The first that triggers wins; later
sources are not consulted, which keeps the per-statement hot path short.

Reason-source order (deterministic):

1. ``@cash:no-cache`` annotation
2. Calls the forbidden-function scan refuses: the clock, a fresh id, ``input``
3. ``@stateful`` function calls, and calls into user code that writes files
4. In-place mutations + side effects (from ``StatementAnalysis``)
5. Input variable missing lineage

``# @cash:assume-safe`` on the statement waives the side effects -- a call
into user code that writes files (3) and the side effects of (4) -- and
nothing else: a hit skips them, which is what the user has said is fine (a
POST that only runs a query, say). What the clock or ``input`` return (2) is
not a side effect but a value, and a hit would replay the first one.

The function takes the runtime hooks (purity lookup, forbidden scan) as
callables so this module does not have to import ``purity`` or
``analysis``.  That keeps the dependency picture in this file honest:
every input the decision reads is in the signature.

The "should this input be skipped for lineage-check purposes?" predicate
is inlined as ``is_lineage_exempt`` — it's purely a property of the
value (module type, private callable, etc.) and has no production
override, so no hook indirection is justified.
"""

from __future__ import annotations

import ast
import collections
import logging
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any

from cash import _plain_data
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.ast_util import called_dotted_names, parse_cached
from cash.analysis.cacheability import StatementAnalysis
from cash.analysis.namespace_effects import statement_user_writer_call
from cash.diagnostics import warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning
from cash.value_types import BUILTIN_NAMES, mro_kind

logger = logging.getLogger(__name__)

__all__ = [
    "analysis_failed",
    "decide_cacheability",
    "identity_coupled_reason",
    "receiver_is_identity_coupled",
]

# --- Identity-coupled library objects ---------------------------
#
# Some objects are only *correct* while they ARE the object a library global
# points at.  pyplot keeps a process-wide registry of the "current figure"
# (``matplotlib._pylab_helpers.Gcf``); ``plt.savefig()`` / ``plt.title()`` /
# ``plt.show()`` act on whatever that registry says is current — NOT on the
# user's variable.
#
# Caching such an object is actively harmful.  The RAM tier deep-copies every
# value it stores (``InMemoryBackend._safe_deep_copy``).  ``Figure.__getstate__``
# records that the figure was registered with pyplot, so ``Figure.__setstate__``
# on the COPY calls ``Gcf._set_new_active_manager`` and makes *the cache's
# private snapshot* the current figure.  From then on the user draws on their
# own figure while ``plt.savefig()`` writes the snapshot — a blank image, on the
# FIRST run, with no error.  Deep-copying a bare Axes does the same thing: it
# drags its ``.figure`` along.
#
# The trade is entirely one-sided — a Figure costs ~0.04s to build — so there is
# no scenario in which caching one pays for the risk.  Refuse outright.
#
# DETECTION: matplotlib is an OPTIONAL dependency, so this path must never
# import it (shipped an unimportable package exactly that way). We walk
# the MRO and compare ``module.qualname`` STRINGS, which imports nothing.  We
# match the BASE classes, not the leaf: ``Axes3D`` lives in ``mpl_toolkits`` but
# inherits ``_AxesBase``, and a user subclass leafs in ``__main__``.  Matching
# the bases also keeps this precise — ``Line2D`` is an ``Artist`` but is not
# identity-coupled, so it stays cacheable.
_IDENTITY_COUPLED_BASES: Mapping[str, str] = {
    "matplotlib.figure.FigureBase": "matplotlib Figure",  # Figure, SubFigure
    "matplotlib.axes._base._AxesBase": "matplotlib Axes",  # Axes + every projection
}

# ``fig, axes = plt.subplots(2, 2)`` binds ``axes`` to a numpy object-array of
# Axes, ``plt.subplot_mosaic(...)`` returns ``dict[str, Axes]``, and a function
# may return ``{"mean": ..., "fig": fig}``: a top-level type check alone would
# cache the Figure inside. So the check walks every item of every builtin
# container (and of object-dtype numpy arrays) at every depth. A skipped item
# would be a cached Figure, so there is no count or depth limit; the walk is
# iterative with an ``id()`` seen set, which makes it cycle-safe. Numeric numpy
# arrays and plain scalars are skipped by type, since they cannot hold an Axes.
_SCAN_CONTAINERS: tuple[type, ...] = (list, tuple, set, frozenset, collections.deque)
_SCALAR_TYPES: frozenset[type] = frozenset({int, float, complex, bool, str, bytes, type(None)})

_SKIP_INPUT_NAMES: frozenset[str] = frozenset({"get_ipython", "__builtins__", "print", "__name__", "__doc__"})


def is_lineage_exempt(var_name: str, val: Any) -> bool:
    """Return True if *val* is a kind of input that never needs lineage tracking.

    Modules, ``get_ipython``, and bound / private callables (whose source
    is captured by other means) are exempt.  Used by ``_has_missing_lineage``
    to decide whether absence of a lineage entry is a cacheability blocker.
    """
    if isinstance(val, types.ModuleType) or var_name == "get_ipython":
        return True
    return bool(callable(val) and (var_name.startswith("_") or hasattr(val, "__self__")))


#: The modules that define the classes above: a value cannot be an instance of
#: one before its module is imported.
_COUPLED_MODULES: tuple[str, ...] = tuple(base.rpartition(".")[0] for base in _IDENTITY_COUPLED_BASES)


def _coupled_classes_loaded() -> bool:
    """Whether any module defining an identity-coupled class is imported."""
    return any(name in sys.modules for name in _COUPLED_MODULES)


def _coupled_kind(value: Any) -> str | None:
    """Return the friendly name if *value* is itself identity-coupled, else None."""
    return mro_kind(value, _IDENTITY_COUPLED_BASES, ("matplotlib",))


_EXHAUSTED = object()


def _container_items(value: Any) -> Any:
    """The items a container can hold a coupled object in, or None for a leaf."""
    if isinstance(value, dict):
        return (*value.keys(), *value.values())
    if isinstance(value, _SCAN_CONTAINERS):
        return value
    if type(value).__module__ == "numpy" and getattr(getattr(value, "dtype", None), "kind", "") == "O":
        return value.flat
    return None


def _coupled_kind_in_container(value: Any) -> str | None:
    """Return the friendly name if a container holds a coupled object at any depth.

    Walks list/tuple/set/frozenset/deque, ``dict`` keys and values, and
    object-dtype numpy arrays, item by item and level by level, so
    ``rows = list(axes)`` (a list of ndarrays of Axes) and a figure as the
    ninth entry of a result dict are both found. Each container is visited
    once (an ``id()`` seen set), which keeps a self-referential container
    from looping.
    """
    root_items = _container_items(value)
    if root_items is None:
        return None
    # Nothing to find when no module that defines a coupled class is loaded:
    # an instance needs its class. And plain data holds no object at all,
    # which a level-at-a-time look at the types answers without visiting each
    # item in Python: a parsed log of two million pairs took 19 s to walk, once
    # for every statement that named it.
    if not _coupled_classes_loaded() or _plain_data.is_tree(value):
        return None
    seen = {id(value)}
    stack = [iter(root_items)]
    while stack:
        item = next(stack[-1], _EXHAUSTED)
        if item is _EXHAUSTED:
            stack.pop()
            continue
        if type(item) in _SCALAR_TYPES:
            continue
        kind = _coupled_kind(item)
        if kind is not None:
            return kind
        if id(item) in seen:
            continue
        items = _container_items(item)
        if items is not None:
            seen.add(id(item))
            stack.append(iter(items))
    return None


def identity_coupled_reason(var_name: str, value: Any) -> str | None:
    """Return an ``uncacheable_reasons`` string if *value* must never be cached.

    ``None`` means "no objection".  See ``_IDENTITY_COUPLED_BASES`` for why
    these objects are refused.  This is a *value* check, so it runs
    after execution — the object does not exist yet when
    :func:`decide_cacheability` runs on a first run.
    """
    kind = _coupled_kind(value) or _coupled_kind_in_container(value)
    if kind is None:
        return None
    return (
        f"Identity-coupled object '{var_name}' ({kind}); caching it would "
        "detach pyplot's current figure from yours and make plt.savefig() "
        "write a blank image."
    )


def receiver_is_identity_coupled(value: Any) -> bool:
    """True when *value* is a live matplotlib Figure/Axes (or a container of them).

    A method call on such a receiver DRAWS on the figure — it adds artists or
    sets Axes state — regardless of what it *returns*.  ``ax.hist(...)`` returns a
    ``(counts, bins, BarContainer)`` data tuple yet mutates the Axes exactly like
    ``ax.plot(...)`` returns a ``Line2D``; keying the mutation decision on the
    RECEIVER (this predicate) rather than the return type is what tells
    ``ax.hist()`` (Axes -> in-place draw) apart from ``df.hist()`` (DataFrame ->
    genuinely receiver-pure, must not bump ``df``).

    Reuses the identity-coupled scan (direct value + full container
    walk for the ``fig, axes = plt.subplots(2, 2)`` object-array spelling), so it
    imports no matplotlib and covers subclasses/projections.
    """
    return bool(_coupled_kind(value) or _coupled_kind_in_container(value))


#: Failures already reported, so a statement that runs every time does not
#: repeat the same warning.
_REPORTED_FAILURES: set[tuple[str, str]] = set()


def analysis_failed(check: str, exc: BaseException) -> str:
    """The uncacheable reason for a safety *check* that raised, warned once.

    A check whose job is to stop caching cannot answer "nothing found" when it
    crashed: the statement it could not judge runs uncached instead.
    """
    reason = f"cash could not {check} ({type(exc).__name__}: {exc}), so the statement runs uncached"
    logger.debug("analysis failed: %s", reason, exc_info=exc)
    key = (check, type(exc).__name__)
    if key not in _REPORTED_FAILURES:
        _REPORTED_FAILURES.add(key)
        try:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "NOTEBOOK-ANALYSIS-FAILED",
                f"{reason}.",
                "nothing to change in your code; please report the error so the check can handle it.",
            )
        except Exception:
            logger.debug("could not warn about a failed analysis", exc_info=True)
    return reason


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
        return False, ["@cash:no-cache annotation"]
    waived = bool(getattr(annotation, "assume_safe", False))

    try:
        forbidden = scan_forbidden(code, user_ns, tree)
    except Exception as exc:  # noqa: BLE001 - an unjudged statement is not a pure one
        return False, [analysis_failed("scan the statement for calls it must not cache", exc)]
    if forbidden:
        return False, list(forbidden)

    try:
        # Bare names first, then ``helpers.announce(...)`` -- how a @stateful
        # function is called once it moves into a module; *is_stateful_call*
        # resolves a dotted spelling through modules only.
        for name in (
            *analysis.called_names,
            *sorted(called_dotted_names(tree if tree is not None else parse_cached(code))),
        ):
            if is_stateful_call(name):
                return False, ["Calls @stateful function"]
        # Every spelling of the call: ``save(...)``, and ``helpers.save(...)``
        # through a project module -- how a function reaches the notebook
        # after it moves into a module. Offered only bare names, the module
        # spelling was stored and a later run restored it without the write.
        found = None if waived else statement_user_writer_call(code, user_ns, tree)
        if found:
            name, writer = found
            return False, [f"Calls {name}(), which writes files ({writer}): a cache hit would skip the write"]
    except Exception as exc:  # noqa: BLE001 - an unjudged statement is not a pure one
        return False, [analysis_failed("check the functions the statement calls", exc)]

    ast_reasons = analysis.skip_reasons(outputs, side_effects=not waived)
    if ast_reasons:
        return False, ast_reasons

    if _has_missing_lineage(inputs, user_ns, variable_lineage):
        return False, ["Input variable missing lineage"]

    return True, []


def _has_missing_lineage(
    inputs: set[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
) -> bool:
    """Return True if any input variable lacks tracked lineage."""
    for var_name in inputs:
        if var_name in _SKIP_INPUT_NAMES or var_name in BUILTIN_NAMES:
            continue
        if var_name not in user_ns:
            return True
        if var_name not in variable_lineage:
            val = user_ns[var_name]
            if not is_lineage_exempt(var_name, val):
                return True
    return False
