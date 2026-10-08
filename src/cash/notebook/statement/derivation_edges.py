"""Derivation-edge detection and lineage-bump replay.

Some objects hold a *live* reference to another object that lineage tracking
never models:

* a numpy **view** (``v = a[100:200]``) whose ``v.base is a`` — mutating ``v``
  in place mutates ``a``;
* a pandas **ref-holder** (``g = df.groupby('k')``, ``r = df.rolling(3)``, ...)
  whose ``g.obj is df`` — mutating ``df`` in place changes what ``g`` aggregates.

Lineage freezes each variable's hash at *creation*, so a later in-place
mutation of one side never bumps the other, and a downstream consumer serves a
stale cached result.

This module keeps a *derivation edge store* on :class:`TrackingState`
(``derivation_edges``): ``bump_source_var -> {vars_to_bump_when_source_bumps}``.

Two responsibilities live here:

* **Detection** (runtime only — it can observe live ``.base`` / ``.obj``
  identity): :func:`detect_derivation_edges`.
* **Replay** (runtime *and* the upstream simulator, which never executes user
  code and only reads the recorded edges): :func:`bump_derived_lineages`.

The replay derives the bumped hash *deterministically from existing lineage
strings only* — it never recomputes ``source_hash`` — so the runtime and the
simulator stay byte-identical (honours the unified-cache-key rule; lineage
hashes are a separate artifact already computed outside ``compute_cache_key``).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Iterable
from typing import Any, Callable

from ..shared_objects import (
    _EXACT_VALUE_TYPES,
    VALUE_TYPES,
    _count_held,
    _excess,
    children_of,
    is_value,
    library_value_types,
)

logger = logging.getLogger(__name__)

__all__ = [
    "detect_derivation_edges",
    "bump_derived_lineages",
    "clear_edges_for",
    "is_uncacheable_alias",
]


def is_uncacheable_alias(value: Any, user_ns: dict, cash_held: Iterable[Any] = ()) -> bool:
    """True if *value* is, or holds, a live-alias of a ``user_ns`` object that
    must NOT be cache-restored.

    A numpy **view** and a pandas **ref-holder** (groupby/rolling/...) each hold
    a live reference to another in-memory object. Pickling and restoring them
    breaks that reference identity (the restored object aliases a stale *copy*),
    so a downstream mutation of the base is lost. These statements must always
    re-derive from the live base instead of restoring from cache.

    *value* is walked as the shared-object check walks an output (the builtin
    containers and the attributes of the notebook's own objects), so views
    inside a list (``parts = np.split(a, 2)``) count too.

    The alias is only *uncacheable* when its base is live: bound to a NAMED
    variable, or held by anything besides the views *value* holds -- an array
    inside a dict (``window(data['x'], 2)``), an attribute, another view, or
    *value* itself (``(buf, buf[:3])``). The containers in *cash_held* are
    cash's own (the call cache holding ``load()``'s result in
    ``x = load()[::2]``), so their references do not make a base live. Many fresh arrays (``np.linspace``,
    and in some builds ``np.arange``/ufunc results) carry a non-None ``.base``
    pointing at an anonymous internal buffer that only the view holds;
    restoring an independent copy of those is correct, so they stay cacheable.
    A ``.copy()`` (numpy ``base is None`` / independent pandas frame) is likewise
    not an alias — the over-invalidation guard.
    """
    np = _imported("numpy")
    refholder_types = _pandas_refholder_types()
    if np is None and not refholder_types:
        return False
    views, refholders = _aliases_in(value, np, refholder_types)
    for holder in refholders:
        src = getattr(holder, "obj", None)
        if src is not None and _find_name_by_identity(user_ns, src) is not None:
            return True
    del refholders
    return bool(views) and _bases_are_live(views, user_ns, list(cash_held))


def _aliases_in(value: Any, np: Any, refholder_types: tuple[type, ...]) -> tuple[list[Any], list[Any]]:
    """``(views, refholders)``: the numpy arrays with a ``.base`` and the
    pandas ref-holders that *value* is or holds."""
    value_types = VALUE_TYPES + library_value_types()
    views: list[Any] = []
    refholders: list[Any] = []
    seen: set[int] = set()
    stack = [value]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or is_value(obj, value_types):
            continue
        seen.add(id(obj))
        if np is not None and isinstance(obj, np.ndarray):
            if obj.base is not None:
                views.append(obj)
            continue
        if refholder_types and isinstance(obj, refholder_types):
            refholders.append(obj)
            continue
        children = children_of(obj)
        if children and not _EXACT_VALUE_TYPES.issuperset(map(type, children)):
            stack.extend(children)
    return views, refholders


def _bases_are_live(views: list[Any], user_ns: dict, cash_held: list[Any]) -> bool:
    """Whether an object on the ``.base`` chain of one of *views* is bound to
    a name of *user_ns*, or held by more than the *views* and the chain
    itself and the containers in *cash_held* (by reference count, `_excess`)."""
    nodes: dict[int, Any] = {}
    inbound: dict[int, int] = {}
    for view in views:
        base = view.base
        while base is not None:
            key = id(base)
            inbound[key] = inbound.get(key, 0) + 1
            if key in nodes:
                break
            nodes[key] = base
            if _find_name_by_identity(user_ns, base) is not None:
                return True
            base = getattr(base, "base", None)
    # No local reference to a base may be left while the counts are read.
    view = base = None
    _count_held(cash_held, nodes, inbound, VALUE_TYPES + library_value_types())
    return bool(_excess(nodes, inbound, list(nodes)))


def _find_name_by_identity(user_ns: dict, obj: Any) -> str | None:
    """Return the first user_ns name bound to *exactly* ``obj`` (identity)."""
    for name, val in user_ns.items():
        if name.startswith("_"):
            continue
        if val is obj:
            return name
    return None


def clear_edges_for(derivation_edges: dict[str, set[str]], var: str) -> None:
    """Drop *var*'s stale incoming and outgoing derivation edges.

    Called when *var* is freshly (re)assigned — otherwise ``g = other`` would
    keep a dead ``df -> g`` edge pointing at an object that is no longer a live
    alias.
    """
    derivation_edges.pop(var, None)
    for src in list(derivation_edges):
        targets = derivation_edges[src]
        if var in targets:
            targets.discard(var)
            if not targets:
                del derivation_edges[src]


def detect_derivation_edges(
    derivation_edges: dict[str, set[str]],
    out: str,
    value: Any,
    user_ns: dict,
) -> None:
    """Record derivation edges for output *out* with live *value*.

    IDENTITY checks only; numpy/pandas are lazy-imported so this stays a soft
    dependency. A ``.copy()`` gives numpy ``base is None`` and pandas an
    independent frame, so no edge is recorded — that is the over-invalidation
    guard and must not be weakened.
    """
    _detect_numpy_view_edge(derivation_edges, out, value, user_ns)
    _detect_pandas_refholder_edge(derivation_edges, out, value, user_ns)
    _detect_matplotlib_figure_edge(derivation_edges, out, value, user_ns)


def _detect_matplotlib_figure_edge(
    derivation_edges: dict[str, set[str]],
    out: str,
    value: Any,
    user_ns: dict,
) -> None:
    """``out`` is part of a named Figure (an Axes, or an array of them):
    drawing on it draws on the figure, so a mutation of ``out`` bumps it.

    ``ax.bar(names, totals)`` changes what ``fig.savefig`` writes, with no
    value edge from ``totals`` to ``fig``. At first a chart re-drew after
    an upstream edit only because ``fig`` ALSO drifted for no reason (it had
    recorded matplotlib's fonts, and its own PNG, as file dependencies); with
    that gone, this edge is the dependency.
    """
    artist_mod = _imported("matplotlib.artist")
    if artist_mod is None:
        return
    probe = value
    np = _imported("numpy")
    if np is not None and isinstance(value, np.ndarray) and value.dtype == object and value.size:
        probe = value.flat[0]  # plt.subplots(1, 2) -> array of Axes
    if not isinstance(probe, artist_mod.Artist):
        return
    fig = getattr(probe, "figure", None)
    if fig is None:
        return
    nm = _find_name_by_identity(user_ns, fig)
    if nm is not None and nm != out:
        derivation_edges.setdefault(out, set()).add(nm)


def _imported(name: str):
    """Return module *name* only if it is ALREADY imported, else ``None``.

    These detectors ask type questions -- "is this an ndarray?", "is this a
    groupby?" -- and importing the library to ask costs far more than the
    answer is worth. An instance of one of those types cannot exist unless
    its library is imported, so absence from ``sys.modules`` IS the answer.

    Measured: the FIRST cell executed under ``%cash_on`` cost 707ms, of which
    ~535ms was `_detect_pandas_refholder_edge` importing pandas for a cell
    that only summed integers.

    Deliberately not cached: a later cell may ``import pandas``, and from then
    on these detectors must start seeing its types.
    """
    return sys.modules.get(name)


def _detect_numpy_view_edge(
    derivation_edges: dict[str, set[str]],
    out: str,
    value: Any,
    user_ns: dict,
) -> None:
    """``out`` is a numpy view; a future mutation of ``out`` must bump
    the root base (and any intermediate NAMED view along the ``.base`` chain)."""
    np = _imported("numpy")
    if np is None:
        return
    if not isinstance(value, np.ndarray) or value.base is None:
        return
    # Walk the .base chain to the root; collect any intermediate array object
    # that is itself a named view (view-of-view).
    targets: set[str] = set()
    base = value.base
    seen: set[int] = set()
    root = None
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        root = base
        nm = _find_name_by_identity(user_ns, base)
        if nm is not None and nm != out:
            targets.add(nm)
        base = getattr(base, "base", None)
    # ``root`` is the ultimate base; ensure it is captured even if intermediate
    # links were unnamed.
    if root is not None:
        nm = _find_name_by_identity(user_ns, root)
        if nm is not None and nm != out:
            targets.add(nm)
    if targets:
        derivation_edges.setdefault(out, set()).update(targets)


_PANDAS_REFHOLDER_TYPES: tuple[type, ...] | None = None


def _pandas_refholder_types() -> tuple[type, ...]:
    """Lazily resolve the pandas ref-holder classes; cache the tuple."""
    global _PANDAS_REFHOLDER_TYPES
    if _PANDAS_REFHOLDER_TYPES is not None:
        return _PANDAS_REFHOLDER_TYPES
    if _imported("pandas") is None:
        # No pandas, so no groupby/rolling object can exist. Return WITHOUT
        # caching: a later cell may import pandas, and this must then resolve.
        return ()
    types_list: list[type] = []
    try:
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

        types_list += [DataFrameGroupBy, SeriesGroupBy]
    except ImportError:
        pass
    try:
        from pandas.core.window.rolling import Rolling

        types_list.append(Rolling)
    except ImportError:
        pass
    try:
        from pandas.core.window.expanding import Expanding

        types_list.append(Expanding)
    except ImportError:
        pass
    try:
        from pandas.core.window.ewm import ExponentialMovingWindow

        types_list.append(ExponentialMovingWindow)
    except ImportError:
        pass
    _PANDAS_REFHOLDER_TYPES = tuple(types_list)
    return _PANDAS_REFHOLDER_TYPES


def _detect_pandas_refholder_edge(
    derivation_edges: dict[str, set[str]],
    out: str,
    value: Any,
    user_ns: dict,
) -> None:
    """``out`` is a groupby/rolling/... that holds a live reference to
    a source frame; a future mutation of that FRAME must bump ``out``."""
    refholder_types = _pandas_refholder_types()
    if not refholder_types or not isinstance(value, refholder_types):
        return
    src = getattr(value, "obj", None)
    if src is None:
        return
    nm = _find_name_by_identity(user_ns, src)
    if nm is not None and nm != out:
        # Edge points FROM the frame name TO the derived object: mutating the
        # frame bumps the derived object.
        derivation_edges.setdefault(nm, set()).add(out)


def bump_derived_lineages(
    derivation_edges: dict[str, set[str]],
    lineage_map: dict[str, str],
    outputs: set[str],
    inputs: set[str],
    *,
    record: Callable[[str, str], None],
    present: Callable[[str], bool],
) -> set[str]:
    """Replay derivation bumps after outputs' lineages were written.

    For each OUTPUT ``out`` that has outgoing edges, bump each target ``t`` —
    but ONLY if ``t`` is not an input of the current statement. At creation
    (``v = a[...]``) the base ``a`` IS an input, so a plain view creation does
    not invalidate the base; at mutation (``v[:] = 9``) the base is NOT an
    input, so it is bumped. Transitive with a visited set (view-of-view,
    groupby-of-...). Targets no longer present are pruned lazily.

    Returns the set of vars whose lineage was bumped. The simulator unions this
    into the statement's ``outputs`` so the reexecution planner records the
    mutation statement as a producer of the aliased base and reschedules it on
    an isolated re-run (otherwise the base restores to its stale pre-mutation
    cache while the alias-mutation statement is orphaned).

    Parameters
    ----------
    lineage_map:
        The lineage dict to read current hashes from *and* the one ``record``
        writes back into (``variable_lineage`` at runtime, ``virtual_lineage``
        in the simulator).
    record(var, hash):
        Persist ``var``'s new lineage hash (runtime attaches the value;
        the simulator just writes the dict).
    present(var):
        Whether ``var`` still exists (``in user_ns`` at runtime; always-True in
        the simulator, which has no live namespace).
    """
    bumped: set[str] = set()
    if not derivation_edges:
        return bumped

    visited: set[str] = set()
    # Seed the walk with the statement's outputs; a bump can cascade
    # (mutating ``a`` bumps view ``v``, which may itself be a base for ``w``).
    frontier: list[str] = [o for o in outputs if o in derivation_edges]
    while frontier:
        src = frontier.pop()
        if src in visited:
            continue
        visited.add(src)
        src_lineage = lineage_map.get(src)
        if src_lineage is None:
            continue
        for target in sorted(derivation_edges.get(src, ())):
            if target in inputs:
                # Creation edge (``v = a[...]``): do NOT bump the base.
                continue
            if not present(target):
                continue
            old = lineage_map.get(target, "")
            new_hash = hashlib.sha256(f"{old}:{src_lineage}".encode("utf-8")).hexdigest()
            if new_hash == old:
                continue
            record(target, new_hash)
            bumped.add(target)
            # Cascade: the freshly-bumped target may itself be a bump source.
            if target not in visited and target in derivation_edges:
                frontier.append(target)
    return bumped
