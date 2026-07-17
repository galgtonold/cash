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

__all__ = [
    "decide_cacheability",
    "identity_coupled_reason",
    "receiver_is_identity_coupled",
]

# --- Identity-coupled library objects (CAS-144) ---------------------------
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
# import it (CAS-129 shipped an unimportable package exactly that way).  We walk
# the MRO and compare ``module.qualname`` STRINGS, which imports nothing.  We
# match the BASE classes, not the leaf: ``Axes3D`` lives in ``mpl_toolkits`` but
# inherits ``_AxesBase``, and a user subclass leafs in ``__main__``.  Matching
# the bases also keeps this precise — ``Line2D`` is an ``Artist`` but is not
# identity-coupled, so it stays cacheable.
_IDENTITY_COUPLED_BASES: Mapping[str, str] = {
    'matplotlib.figure.FigureBase': 'matplotlib Figure',       # Figure, SubFigure
    'matplotlib.axes._base._AxesBase': 'matplotlib Axes',      # Axes + every projection
}

# ``fig, axes = plt.subplots(2, 2)`` binds ``axes`` to a numpy object-array of
# Axes, and ``axs = fig.subplots(2, 2)`` never binds the Figure at all — so a
# top-level type check alone would miss it and silently cache the Axes (which
# drags the Figure).  A *bounded* scan catches the common spellings without
# walking a million-element list on the hot path: these containers are
# homogeneous, so the first few elements settle it.  It also recurses a few
# levels and covers ``dict`` — ``rows = list(axes)`` nests a list of ndarrays,
# and ``plt.subplot_mosaic(...)`` returns ``dict[str, Axes]`` (and typically
# binds ONLY that dict, so nothing bare-Figure/Axes co-occurs to trip the check
# for the statement) — CAS-155.  The depth cap also makes the plain recursion
# cycle-safe (unlike ``deepcopy`` it has no memo).
_CONTAINER_SCAN_LIMIT = 8
_CONTAINER_SCAN_MAX_DEPTH = 4  # dict-of-list-of-Axes is 2 deep; leave headroom.

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


def _coupled_kind(value: Any) -> str | None:
    """Return the friendly name if *value* is itself identity-coupled, else None.

    Imports nothing: the match is on ``module.qualname`` strings taken from the
    MRO.  The ``startswith`` gate keeps the common case (int, str, DataFrame)
    to a couple of cheap string checks.
    """
    try:
        mro = type(value).__mro__
    except AttributeError:  # pragma: no cover - exotic metaclass
        return None
    for base in mro:
        module = getattr(base, '__module__', '') or ''
        if not module.startswith('matplotlib'):
            continue
        kind = _IDENTITY_COUPLED_BASES.get(f"{module}.{getattr(base, '__qualname__', '')}")
        if kind is not None:
            return kind
    return None


def _coupled_kind_in_container(value: Any, _depth: int = 0) -> str | None:
    """Return the friendly name if a container holds a coupled object.

    Bounded by ``_CONTAINER_SCAN_LIMIT`` per level and ``_CONTAINER_SCAN_MAX_DEPTH``
    levels deep.  Handles list/tuple/set/frozenset, ``dict`` (scanning values —
    ``subplot_mosaic`` returns ``dict[str, Axes]``), and object-dtype numpy
    arrays.  Only object-dtype arrays are scanned — a numeric array cannot hold
    an Axes, and checking ``dtype`` first keeps big numeric arrays off this path
    entirely.  Recurses so ``rows = list(axes)`` (a list of ndarrays of Axes)
    and other nestings are caught; the depth cap also keeps a pathological
    self-referential container from looping (CAS-155).
    """
    if _depth >= _CONTAINER_SCAN_MAX_DEPTH:
        return None
    if isinstance(value, dict):
        items: Any = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    elif type(value).__module__ == 'numpy' and getattr(getattr(value, 'dtype', None), 'kind', '') == 'O':
        items = value.flat
    else:
        return None

    for index, item in enumerate(items):
        if index >= _CONTAINER_SCAN_LIMIT:
            break
        kind = _coupled_kind(item) or _coupled_kind_in_container(item, _depth + 1)
        if kind is not None:
            return kind
    return None


def identity_coupled_reason(var_name: str, value: Any) -> str | None:
    """Return an ``uncacheable_reasons`` string if *value* must never be cached.

    ``None`` means "no objection".  See ``_IDENTITY_COUPLED_BASES`` for why
    these objects are refused (CAS-144).  This is a *value* check, so it runs
    after execution — the object does not exist yet when
    :func:`decide_cacheability` runs on a first run.
    """
    kind = _coupled_kind(value) or _coupled_kind_in_container(value)
    if kind is None:
        return None
    return (
        f"Identity-coupled object '{var_name}' ({kind}); caching it would "
        "detach pyplot's current figure from yours and make plt.savefig() "
        "write a blank image (CAS-144)."
    )


def receiver_is_identity_coupled(value: Any) -> bool:
    """True when *value* is a live matplotlib Figure/Axes (or a container of them).

    A method call on such a receiver DRAWS on the figure — it adds artists or
    sets Axes state — regardless of what it *returns*.  ``ax.hist(...)`` returns a
    ``(counts, bins, BarContainer)`` data tuple yet mutates the Axes exactly like
    ``ax.plot(...)`` returns a ``Line2D``; keying the mutation decision on the
    RECEIVER (this predicate) rather than the return type is what tells
    ``ax.hist()`` (Axes -> in-place draw) apart from ``df.hist()`` (DataFrame ->
    genuinely receiver-pure, must not bump ``df``) — CAS-194.

    Reuses the CAS-144 identity-coupled scan (direct value + bounded container
    walk for the ``fig, axes = plt.subplots(2, 2)`` object-array spelling), so it
    imports no matplotlib and covers subclasses/projections.
    """
    return bool(_coupled_kind(value) or _coupled_kind_in_container(value))


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
