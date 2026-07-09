"""Derivation-edge detection and lineage-bump replay (CAS-115 / CAS-89).

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
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "detect_derivation_edges",
    "bump_derived_lineages",
    "clear_edges_for",
    "is_uncacheable_alias",
]


def is_uncacheable_alias(value: Any, user_ns: dict) -> bool:
    """True if *value* is a live-alias of a NAMED ``user_ns`` object that must
    NOT be cache-restored.

    A numpy **view** and a pandas **ref-holder** (groupby/rolling/...) each hold
    a live reference to another in-memory object. Pickling and restoring them
    breaks that reference identity (the restored object aliases a stale *copy*),
    so a downstream mutation of the base is lost. These statements must always
    re-derive from the live base instead of restoring from cache.

    The alias is only *uncacheable* when its base resolves to a NAMED live
    variable that could be mutated elsewhere. Many fresh arrays (``np.linspace``,
    and in some builds ``np.arange``/ufunc results) carry a non-None ``.base``
    pointing at an anonymous internal buffer that no user variable references;
    restoring an independent copy of those is correct, so they stay cacheable.
    A ``.copy()`` (numpy ``base is None`` / independent pandas frame) is likewise
    not an alias — the over-invalidation guard.
    """
    try:
        import numpy as np
        if isinstance(value, np.ndarray) and value.base is not None:
            # Only a genuine alias of a NAMED live variable is uncacheable.
            base = value.base
            seen: set[int] = set()
            while base is not None and id(base) not in seen:
                seen.add(id(base))
                if _find_name_by_identity(user_ns, base) is not None:
                    return True
                base = getattr(base, "base", None)
            return False
    except ImportError:
        pass
    refholder_types = _pandas_refholder_types()
    if refholder_types and isinstance(value, refholder_types):
        src = getattr(value, "obj", None)
        return src is not None and _find_name_by_identity(user_ns, src) is not None
    return False


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


def _detect_numpy_view_edge(
    derivation_edges: dict[str, set[str]], out: str, value: Any, user_ns: dict,
) -> None:
    """CAS-89: ``out`` is a numpy view; a future mutation of ``out`` must bump
    the root base (and any intermediate NAMED view along the ``.base`` chain)."""
    try:
        import numpy as np
    except ImportError:
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
    derivation_edges: dict[str, set[str]], out: str, value: Any, user_ns: dict,
) -> None:
    """CAS-115: ``out`` is a groupby/rolling/... that holds a live reference to
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
            new_hash = hashlib.sha256(
                f"{old}:{src_lineage}".encode("utf-8")
            ).hexdigest()
            if new_hash == old:
                continue
            record(target, new_hash)
            bumped.add(target)
            # Cascade: the freshly-bumped target may itself be a bump source.
            if target not in visited and target in derivation_edges:
                frontier.append(target)
    return bumped
