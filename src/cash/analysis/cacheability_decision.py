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
from typing import Any, NamedTuple

from cash import _plain_data
from cash._user_attrs import instance_of
from cash.analysis.analysis_budget import ANALYSIS_BUDGET_REASON, AnalysisBudgetExceeded
from cash.analysis.annotations import CacheAnnotation
from cash.analysis.ast_util import called_dotted_names, parse_cached
from cash.analysis.cacheability import StatementAnalysis
from cash.analysis.namespace_effects import statement_user_pyplot_call, statement_user_writer_call
from cash.diagnostics import warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning, CashImpurityWarning
from cash.value_types import BUILTIN_NAMES, mro_kind

logger = logging.getLogger(__name__)

__all__ = [
    "CacheabilityVerdict",
    "analysis_failed",
    "cacheability_verdict",
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
    if instance_of(value, dict):
        return (*value.keys(), *value.values())
    if instance_of(value, _SCAN_CONTAINERS):
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
    if not _coupled_classes_loaded():
        return None
    from cash.notebook.shared_objects import object_free_leaves

    if _plain_data.is_tree(value, object_free_leaves()):
        return None
    seen = {id(value)}
    # Types already found to be neither coupled nor a container: whether an
    # item is either is a question of its type, but for an array, whose dtype
    # decides. A list of 1.7 million numpy timedeltas asked it of each item:
    # 3.6 s for a ``plt.hist(deltas)`` plain Python ran in 4.5 s.
    leaves = set(_SCALAR_TYPES)
    stack = [iter(root_items)]
    while stack:
        item = next(stack[-1], _EXHAUSTED)
        if item is _EXHAUSTED:
            stack.pop()
            continue
        item_type = type(item)
        if item_type in leaves:
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
        elif item_type.__name__ != "ndarray":
            leaves.add(item_type)
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
    crashed: the statement it could not judge runs uncached instead. One
    stopped by the statement's analysis budget (`analysis_budget`) did not
    crash: it is the budget's reason, with no warning.
    """
    if isinstance(exc, AnalysisBudgetExceeded):
        return ANALYSIS_BUDGET_REASON
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


class CacheabilityVerdict(NamedTuple):
    """:func:`cacheability_verdict`'s answer."""

    cacheable: bool
    reasons: list[str]
    #: Refused because what the statement binds can differ between two runs
    #: with the same key: it reads the clock or makes a fresh id, calls a
    #: ``@stateful`` function, or reads pyplot's current figure through a
    #: helper. It runs every time, and its outputs carry their value in their
    #: lineage, as a ``# @cash:no-cache`` statement's do, so what is cached
    #: below is keyed on the value this run gave.
    varies: bool = False
    #: Cacheable only if it does not touch pyplot when it runs: it calls a
    #: helper of the user's that can (``plot_tour`` behind ``run(...,
    #: plot=False)``), and the caller asked for that to be seen as it runs
    #: (*pyplot_seen_as_it_runs*). The reason to give when it does.
    unless_it_draws: str | None = None


def decide_cacheability(**kwargs: Any) -> tuple[bool, list[str]]:
    """Return ``(cacheable, reasons)`` for a statement: see
    :func:`cacheability_verdict`, which takes the same arguments."""
    verdict = cacheability_verdict(**kwargs)
    return verdict.cacheable, verdict.reasons


def cacheability_verdict(
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
    pyplot_seen_as_it_runs: bool = False,
) -> CacheabilityVerdict:
    """Return the verdict ``(cacheable, reasons, varies)`` for a statement.

    ``cacheable`` is ``True`` only when *all* reason-sources are silent.
    ``reasons`` is the list of human-readable strings that populate
    ``metrics['uncacheable_reasons']``.  An empty list means "cacheable."

    With *pyplot_seen_as_it_runs* (the caller records which code runs), a
    helper that can reach pyplot does not refuse the statement on its own:
    the verdict is cacheable with ``unless_it_draws`` set, and the caller
    refuses it after the run when pyplot's code ran. Refused for another
    reason, it is refused as before, for the helper.
    """
    if annotation is not None and annotation.no_cache:
        return CacheabilityVerdict(False, ["@cash:no-cache annotation"])
    waived = bool(getattr(annotation, "assume_safe", False))

    try:
        forbidden = scan_forbidden(code, user_ns, tree)
    except Exception as exc:  # noqa: BLE001 - an unjudged statement is not a pure one
        return CacheabilityVerdict(False, [analysis_failed("scan the statement for calls it must not cache", exc)])
    if forbidden:
        return CacheabilityVerdict(False, list(forbidden), varies=True)
    if not waived:
        try:
            ambient = _helper_ambient_reasons(code, user_ns)
        except Exception as exc:  # noqa: BLE001 - an unjudged statement is not a pure one
            return CacheabilityVerdict(False, [analysis_failed("check the functions the statement calls", exc)])
        if ambient:
            return CacheabilityVerdict(False, ambient, varies=True)

    try:
        # Bare names first, then ``helpers.announce(...)`` -- how a @stateful
        # function is called once it moves into a module; *is_stateful_call*
        # resolves a dotted spelling through modules only.
        for name in (
            *analysis.called_names,
            *sorted(called_dotted_names(tree if tree is not None else parse_cached(code))),
        ):
            if is_stateful_call(name):
                return CacheabilityVerdict(False, ["Calls @stateful function"], varies=True)
        # Every spelling of the call: ``save(...)``, and ``helpers.save(...)``
        # through a project module -- how a function reaches the notebook
        # after it moves into a module. Offered only bare names, the module
        # spelling was stored and a later run restored it without the write.
        found = None if waived else statement_user_writer_call(code, user_ns, tree)
        if found:
            name, writer = found
            return CacheabilityVerdict(
                False,
                [
                    f"Calls {name}(), which writes files ({writer}): a cache hit would skip the write "
                    "(# @cash:assume-safe caches it anyway)"
                ],
            )
        # A helper reading pyplot's current figure (``len(plt.gca().lines)``)
        # is display, as ``plt.gca()`` in the statement is: no key holds the
        # figure it reads.
        drawn = None if waived else statement_user_pyplot_call(code, user_ns, tree)
        drawn_reason = None
        if drawn:
            name, user = drawn
            drawn_reason = (
                f"Calls {name}(), which reads or changes pyplot's current figure through {user}: it runs every time"
            )
            if not pyplot_seen_as_it_runs:
                return CacheabilityVerdict(False, [drawn_reason], varies=True)
    except Exception as exc:  # noqa: BLE001 - an unjudged statement is not a pure one
        return CacheabilityVerdict(False, [analysis_failed("check the functions the statement calls", exc)])

    ast_reasons = analysis.skip_reasons(outputs, side_effects=not waived)
    if ast_reasons or _has_missing_lineage(inputs, user_ns, variable_lineage):
        if drawn_reason is not None:
            return CacheabilityVerdict(False, [drawn_reason], varies=True)
        if ast_reasons:
            return CacheabilityVerdict(False, ast_reasons)
        return CacheabilityVerdict(False, ["Input variable missing lineage"])

    return CacheabilityVerdict(True, [], unless_it_draws=drawn_reason)


#: (function's code, read) pairs already warned about this session.
_AMBIENT_WARNED: set[tuple[int, str]] = set()


def _helper_ambient_reasons(code: str, user_ns: Mapping[str, Any]) -> list[str]:
    """Why *code* runs every time: a function of the user's it calls reads
    the clock, a fresh UUID or random bytes (`helper_ambient_reads`). Cached,
    the first value came back on every run without a word; KEY-AMBIENT-READ
    says so once per function and read."""
    from cash.notebook.lineage_formula import helper_ambient_reads

    reasons = []
    for fn, issue in helper_ambient_reads(code, user_ns):
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", "a function"))
        read = issue.description.partition(" - ")[0]
        via = f"{name}()"
        reasons.append(f"Runs {via}, which reads {read}: a cached result would freeze it, so it runs every time")
        mark = (id(getattr(fn, "__code__", fn)), f"{issue.where}:{read}")
        if mark in _AMBIENT_WARNED:
            continue
        _AMBIENT_WARNED.add(mark)
        warn_diagnostic(
            CashImpurityWarning,
            code="KEY-AMBIENT-READ",
            what=(
                f"this statement runs {via}, which reads {read}. That value is not in "
                f"the cache key, so a cached result would return the first one on every run: "
                f"the statement runs every time instead."
            ),
            fix=(
                "pass the value in (stamp(x, now=time.time())) so the call can be cached, "
                "or mark the line `# @cash:assume-safe` to cache the first value on purpose."
            ),
        )
    return reasons


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
