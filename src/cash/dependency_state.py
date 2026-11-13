"""The dependency-state hash: a deep seam over Cash's registries.

``DependencyStateHasher`` folds three sources of state into one digest
that becomes the ``state_hash`` segment of a decorator cache key
(``func:state:dynamic:args``):

1. The node's own source hash (functions) or change marker (data sources).
2. Each graph dependency's state hash, recursively, in sorted order.
3. Transitive *helper* source hashes captured by the purity analyzer —
   this is what makes editing a plain (non-decorated) helper invalidate
   the parent's cache key.

The fold is pure and deterministic. Its single impure input — re-reading
the *live* source of each captured helper, so an in-process redefinition
(notebook cell rerun, REPL) is noticed — is isolated behind the
``HelperResolver`` seam. ``SysModulesHelperResolver`` is the production
adapter; tests inject a fake.

The hasher borrows Cash's registries *by reference*: it reads the same
dict objects Cash mutates over its lifetime, so a function registered
after the hasher is constructed is still seen.
"""

from __future__ import annotations

import contextvars
import hashlib
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .code_swaps import note_report
from .data_source import state_token_of

if TYPE_CHECKING:
    from .analysis.purity_report import PurityReport
    from .data_source import DataSource
    from .graph import DependencyGraph

__all__ = [
    "DependencyStateHasher",
    "OUTSIDE_READS_FOLDED",
    "HelperResolver",
    "SysModulesHelperResolver",
    "STATE_LEDGER",
    "ledger_note",
]

#: What the state segment of the key being built is made of, by name: the
#: function's own source, each cached function and helper it calls, the
#: globals it reads, its captures and defaults. Filled while a decorator key
#: is built, so a "code or state changed" miss can say WHICH of them moved.
#: ``None`` whenever no
#: key is being built. Values are kept raw (the fold's own digests and part
#: lists); they are only formatted on a miss, never on the hit path.
STATE_LEDGER: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_cash_state_ledger", default=None)


def ledger_note(label: Any, value: Any) -> None:
    """Record one named part of the state being keyed, if a key is being built."""
    ledger = STATE_LEDGER.get()
    if ledger is not None:
        ledger[label] = value


#: Data source tokens and declared dependencies re-resolved by `compute`:
#: what no watcher sees, so a state segment that folded one is not
#: remembered (`cash.decorator.state_memo`).
OUTSIDE_READS_FOLDED = [0]


@runtime_checkable
class HelperResolver(Protocol):
    """Port: current source hashes for the helpers a report captured.

    Implementations re-read the *live* definition of each helper, so an
    in-process redefinition changes the returned hash. A helper that
    cannot be resolved is omitted; the caller falls back to the
    analysis-time snapshot recorded on the report.
    """

    def current_hashes(self, report: PurityReport) -> dict[str, str]: ...


class SysModulesHelperResolver:
    """Production ``HelperResolver``: re-resolve via ``sys.modules``.

    Walks each helper's recorded ``(module_name, attr_chain)`` path from
    ``sys.modules`` and hashes the resolved callable's source via the
    injected ``hash_callable``. Resolution or hashing failures (helper
    deleted, renamed, or a C-extension swap) drop that helper from the
    result, leaving the caller on the recorded snapshot.

    Costs ~5-30us per helper; called once per cached function per call.
    """

    def __init__(self, hash_callable: Callable[[Callable[..., Any]], str]):
        self._hash_callable = hash_callable

    def current_hashes(self, report: PurityReport) -> dict[str, str]:
        current: dict[str, str] = {}
        for qual, (mod_name, attr_chain) in report.helper_resolution_paths.items():
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            obj: Any = mod
            for attr in attr_chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None or not callable(obj):
                continue
            try:
                current[qual] = self._hash_callable(obj)
            except (OSError, TypeError, AttributeError):
                continue
        # Helpers with no path to re-resolve by (closures from a factory) are
        # hashed from the live object when it is still alive.
        for qual, ref in getattr(report, "helper_objects", {}).items():
            if qual in current:
                continue
            obj = ref()
            if obj is None or not callable(obj):
                continue
            try:
                current[qual] = self._hash_callable(obj)
            except (OSError, TypeError, AttributeError):
                continue
        return current


class DependencyStateHasher:
    """Folds Cash's dependency state into a single deterministic digest.

    Holds the registries by reference (Cash mutates them in place) and
    one ``HelperResolver`` for the live-helper re-resolution. ``compute``
    is the whole interface.
    """

    def __init__(
        self,
        *,
        functions: Mapping[str, Any],
        data_sources: Mapping[str, DataSource],
        source_hashes: Mapping[str, str],
        purity_reports: Mapping[str, PurityReport],
        graph: DependencyGraph,
        helper_resolver: HelperResolver,
        declared_dep_snapshots: Mapping[str, str] | None = None,
        declared_dep_resolver: Callable[[str], str | None] | None = None,
        reached_callee: Callable[[str, str], Any] | None = None,
    ):
        self._functions = functions
        self._data_sources = data_sources
        self._source_hashes = source_hashes
        self._purity_reports = purity_reports
        self._graph = graph
        self._helper_resolver = helper_resolver
        # Declared plain-callable deps: snapshot map + live resolver.
        # Keep the passed dict BY REFERENCE (it is empty at construction and
        # filled by later registrations) - ``or {}`` would swap in a fresh dict
        # because an empty dict is falsy, severing the shared reference.
        self._declared_dep_snapshots = declared_dep_snapshots if declared_dep_snapshots is not None else {}
        self._declared_dep_resolver = declared_dep_resolver
        # (node, dep) -> the cached wrapper *node* reaches under *dep* when
        # this registry's function of that name is not it (another instance,
        # or an object a reload replaced), else None.
        self._reached_callee = reached_callee

    def compute(
        self,
        node: str,
        visited: set[str] | None = None,
        *,
        own_source_override: str | None = None,
        own_report: PurityReport | None = None,
        note: bool = False,
        covered: set[tuple[str, str]] | None = None,
    ) -> str:
        """Return the dependency state hash for *node*.

        ``visited`` guards against dependency cycles; a node seen twice
        contributes the empty string. Callers pass nothing — the set is
        seeded internally and threaded through the recursion.

        ``own_source_override`` replaces the ROOT node's own-source
        component: a wrapper must key on the function object it
        actually executes, not on whatever the live registry currently
        holds under the shared ``module.qualname`` slot — otherwise a
        stale wrapper plants its results under a redefined function's
        identity (and two same-qualname lambdas collide outright).
        Recursive dependency calls never receive the override, so helper
        and dependency state stays live.

        ``own_report`` is the ROOT's purity report, for the same reason: two
        closures from one factory share a name but not their helpers
        (`FunctionRegistry.report_for`).

        ``note`` records the root's parts in `STATE_LEDGER`; only the
        decorator's key build asks, never a nested identity lookup.

        ``covered`` collects each ``(caller, callee)`` edge left out because
        the caller's helpers already hold the callee (`_walked_as_helper`),
        for the key's other folds over the graph to leave out as well.
        """
        if visited is None:
            visited = set()
        if node in visited:
            return ""
        visited.add(node)

        hashes: list[str] = []

        # 1. Node's own state.
        if node in self._functions:
            if own_source_override is not None:
                hashes.append(own_source_override)
            else:
                hashes.append(self._source_hashes.get(node, ""))
        elif node in self._data_sources:
            # The source's change token (mtime / version / digest); warns if a
            # source mistakenly returns a bool, which can't track changes.
            hashes.append(state_token_of(self._data_sources[node]))
            OUTSIDE_READS_FOLDED[0] += 1
        elif node in self._declared_dep_snapshots:
            # A declared plain-callable dep: re-resolve its live
            # source hash so a disk edit + reload is seen; fall back to the
            # registration-time snapshot when resolution fails.
            live = self._declared_dep_resolver(node) if self._declared_dep_resolver is not None else None
            OUTSIDE_READS_FOLDED[0] += 1
            hashes.append(live if live is not None else self._declared_dep_snapshots[node])

        if note and hashes:
            ledger_note("source", hashes[0])

        report = own_report if own_report is not None else self._purity_reports.get(node)
        helpers = self._helper_values(report)

        # 2. Dependencies' state, sorted for determinism.
        for dep in sorted(self._graph.get_dependencies(node)):
            reached = self._reached_callee(node, dep) if self._reached_callee is not None else None
            if reached is None and self._walked_as_helper(dep, report, helpers, set()):
                # Already in step 3, whole: a cached function this node calls
                # as the plain function (``cache(func)(x)`` wraps it inside
                # the body). Whether it was wrapped before this node was
                # first keyed depends on the order the program ran in and on
                # which earlier calls were hits, so the edge is no key part.
                if covered is not None:
                    covered.add((node, dep))
                continue
            # A cached function this registry does not describe keys by its
            # own state, built by the instance that owns it for the object
            # the caller reaches.
            state = (
                f"reached:{reached._cash_state()}"
                if reached is not None
                else self.compute(dep, visited, covered=covered)
            )
            hashes.append(state)
            if note:
                ledger_note(("calls", dep), state)

        # 3. Transitive helper source hashes, re-resolved live per call.
        #    The node's own qualname is skipped (its source is already in
        #    step 1); unresolved helpers fall back to the recorded snapshot.
        # The code each helper ran when walked, compared again after the
        # body (`cash.code_swaps`).
        note_report(report)
        for qual in sorted(helpers):
            if qual == node:
                continue
            hashes.append(f"helper:{qual}:{helpers[qual]}")
            if note:
                ledger_note(("helper", qual), helpers[qual])

        return hashlib.sha256(":".join(hashes).encode("utf-8")).hexdigest()

    def _helper_values(self, report: PurityReport | None) -> dict[str, str]:
        """``{qualname: digest}`` of *report*'s helpers, live where they resolve."""
        if report is None or not report.helper_source_hashes:
            return {}
        current = self._helper_resolver.current_hashes(report)
        return {qual: current.get(qual, digest) for qual, digest in report.helper_source_hashes.items()}

    def _walked_as_helper(self, dep: str, report: PurityReport | None, helpers: dict[str, str], seen: set[str]) -> bool:
        """Does *helpers* (the caller's step 3) already hold all that *dep*'s
        own state hash would fold?

        True only for a cached function of this registry that the caller's
        analysis walked as a plain helper of the same name, the very object
        registered, whose own helpers all appear in *helpers* with the same
        digests, and whose own dependencies are covered the same way. A data
        source, a declared dependency, a function another instance caches or
        a helper digest that differs keeps the edge, so nothing it folds
        leaves the key.
        """
        if dep in seen:
            return True
        seen.add(dep)
        func = self._functions.get(dep)
        if func is None or report is None or dep not in helpers:
            return False
        if _resolve_helper(report, dep) is not func:
            return False
        own = self._purity_reports.get(dep)
        if own is not None and not set(own.environment_reads or ()) <= set(report.environment_reads or ()):
            return False
        for qual, digest in self._helper_values(own).items():
            if qual != dep and helpers.get(qual) != digest:
                return False
        for inner in self._graph.get_dependencies(dep):
            if self._reached_callee is not None and self._reached_callee(dep, inner) is not None:
                return False
            if not self._walked_as_helper(inner, report, helpers, seen):
                return False
        return True


def _resolve_helper(report: PurityReport, qual: str) -> Any:
    """The object *report*'s helper *qual* is now, or None if it does not resolve."""
    path = report.helper_resolution_paths.get(qual)
    if path is not None:
        obj: Any = sys.modules.get(path[0])
        for attr in path[1]:
            if obj is None:
                return None
            obj = getattr(obj, attr, None)
        return obj
    ref = getattr(report, "helper_objects", {}).get(qual)
    return ref() if ref is not None else None
