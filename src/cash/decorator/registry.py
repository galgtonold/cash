"""The dependency graph of cached functions: declared dependencies, TTLs
inherited along it, and the one-time analysis of each function."""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..analysis.code_analyzer import CodeAnalyzer
from ..analysis.purity_analyzer import PurityReport, bindings_changed, get_analyzer, resolve_binding
from ..data_source import DataSource, state_token_of
from ..exceptions import CashCacheIneffectiveWarning
from ..graph import DependencyGraph
from ..source_norm import bytecode_identity, callable_identity, compiled_identity
from .cached_function import CachedFunction, PurityMode
from .call_state import KeyBuildFailed
from .code_identity import func_key, hash_callable_source

if TYPE_CHECKING:
    from .reporting import Notices

logger = logging.getLogger(__name__)


def resolve_dynamic_dependencies(
    func_name: str,
    dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
    args: tuple,
    kwargs: dict,
) -> str:
    """The digest of the DataSources *dynamic_depends_on* resolves to for
    this call, or ``""`` with none. Raises `KeyBuildFailed` when a resolver
    fails or returns something else: the call then has no key, never a key
    without the dependency."""
    if not dynamic_depends_on:
        return ""

    dynamic_state_parts = []
    resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
    fix = (
        "fix the resolver -- it is called with exactly the same arguments "
        "as the function -- so that it returns a DataSource, a list of "
        "them, or None for no dependency."
    )
    for resolver in resolvers:
        # Any failure makes the call unkeyable, never a key without the
        # dependency: that key would keep hitting after the data changed.
        try:
            # Resolver receives the same args as the function
            ds_result = resolver(*args, **kwargs)
            dss = ds_result if isinstance(ds_result, list) else [ds_result]
            for ds in dss:
                if ds is None:
                    continue
                if not isinstance(ds, DataSource):
                    raise KeyBuildFailed(
                        "KEY-DYNAMIC-DEP-FAILED",
                        f"@cash.cache on {func_name}: a dynamic_depends_on resolver "
                        f"returned a {type(ds).__name__}, which is not a DataSource, "
                        f"so cash cannot tell when it changes and the call ran uncached.",
                        fix,
                    )
                dynamic_state_parts.append(state_token_of(ds))
        except KeyBuildFailed:
            raise
        except Exception as e:  # noqa: BLE001 - any failure here is the resolver's
            raise KeyBuildFailed(
                "KEY-DYNAMIC-DEP-FAILED",
                f"@cash.cache on {func_name}: dynamic_depends_on resolver raised "
                f"{type(e).__name__} ({e}), so cash cannot tell whether that "
                f"dependency changed and the call ran uncached.",
                fix,
            ) from e

    if dynamic_state_parts:
        # Sort to ensure deterministic order if multiple sources
        return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode("utf-8")).hexdigest()
    return ""


def warn_inert_dependency(notices: Notices, func_name: str, dep: Callable[..., Any]) -> None:
    """KEY-DEPENDS-ON-OPAQUE: a declared ``depends_on`` callable cash cannot read."""
    notices.warn_once(
        CashCacheIneffectiveWarning,
        func_name,
        "",
        f"@cash.cache on {func_name}: depends_on={getattr(dep, '__qualname__', dep)!r} "
        f"is a callable whose source cannot be read (builtin / C-extension), "
        f"so the declaration is inert and changes to it will NOT "
        f"invalidate the cache.",
        code="KEY-DEPENDS-ON-OPAQUE",
        fix="depend on something cash can read: pass the extension's "
        "version in as an argument, or declare a DataSource whose "
        "token is that version or build id.",
    )


class FunctionRegistry:
    """The cached functions of one `Cash`, the dependencies between them, and
    their one-time analysis: what the state segment of every key is built from."""

    def __init__(self) -> None:
        #: func_name -> its decoration's options and per-process state.
        self.cached: dict[str, CachedFunction] = {}
        #: func_name -> the decorated function.
        self.functions: dict[str, Callable[..., Any]] = {}
        #: func_name -> its source identity at decoration.
        self.source_hashes: dict[str, str] = {}
        #: DataSource id -> the source, from ``depends_on``.
        self.data_sources: dict[str, DataSource] = {}
        #: Edges from a cached function to what it calls or declares.
        self.graph = DependencyGraph()
        #: func_name -> its purity report. Helper source hashes from it fold
        #: into the key's state hash, so cross-process helper edits invalidate.
        #: For a closure this is only the latest one analysed under the name;
        #: read a function's own through `report_for`.
        self.purity_reports: dict[str, PurityReport] = {}
        #: closure -> its own purity report. Two closures from one factory
        #: share a name but can call different helpers through what they
        #: capture (``make(ha)`` and ``make(hb)`` reading ``cap.helper``), and
        #: the analyzer reports each closure on its own; filed under the name
        #: alone, the first report keyed both.
        self._closure_reports: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        #: Functions whose purity findings have been *surfaced*, on their
        #: first call.
        self.analyzed: set[str] = set()
        #: Closures whose own findings have been surfaced (see `needs_surfacing`).
        self._surfaced_closures: weakref.WeakSet = weakref.WeakSet()
        #: Functions whose graph edges and purity report are populated
        #: (separate from `analyzed`: a dependency can be populated to
        #: complete a parent's state hash long before it is called directly
        #: and surfaced). Keeps the cache key stable from the first call.
        self.populated: set[str] = set()
        #: ONE lock for the one-time analysis, whatever function triggers it.
        #:
        #: The CACHE KEY depends on what the analysis populates (helper source
        #: hashes, graph edges), so a thread that built a key while another was
        #: still analysing would get a different key for the same call: an
        #: entry no later run looks up, and under `use_locking=True` a second
        #: execution, since each key is single-flighted on its own.
        #:
        #: RLock, not Lock: analysis walks the dependency graph and re-enters
        #: this same guard for the callees it populates on the way.
        #:
        #: One lock rather than one per function, deliberately. Analysis of f
        #: populates f's whole callee closure, so per-function locks could be
        #: taken in two orders by two threads and deadlock. It is a one-time,
        #: source-reading step measured in milliseconds; serialising unrelated
        #: first calls behind it costs nothing worth a lock-ordering rule.
        self.analysis_lock = threading.RLock()
        self._effective_ttl_cache: dict[str, int | None] = {}
        #: Declared plain-callable dependencies (``depends_on=[proxy_fn]``
        #: where proxy_fn is NOT a decorated cached function): a source-hash
        #: snapshot at registration, and a ``(module, attr_chain)`` path for
        #: live re-resolution, so editing the dep on disk and reloading
        #: invalidates the parent key.
        self.declared_dep_snapshots: dict[str, str] = {}
        self._declared_dep_paths: dict[str, tuple[str, tuple[str, ...]]] = {}

    def report_for(self, func: Callable[..., Any], func_name: str) -> PurityReport | None:
        """*func*'s own purity report: a closure's, else the one under *func_name*."""
        if getattr(func, "__closure__", None):
            try:
                report = self._closure_reports.get(func)
            except TypeError:
                report = None
            if report is not None:
                return report
        return self.purity_reports.get(func_name)

    def needs_population(self, func: Callable[..., Any], func_name: str) -> bool:
        """Has *func* no report yet: its name was never analysed, or it is a
        closure analysed only through a sibling from the same factory?"""
        if func_name not in self.populated:
            return True
        if not getattr(func, "__closure__", None):
            return False
        try:
            return func not in self._closure_reports
        except TypeError:
            return False

    def needs_surfacing(self, func: Callable[..., Any], func_name: str) -> bool:
        """Are *func*'s purity findings still to be shown, on this call?

        Once per name, and once per closure: a second closure from one
        factory reaches its own helpers (`report_for`), so what the first
        one's findings said does not cover it.
        """
        if func_name not in self.analyzed:
            return True
        if not getattr(func, "__closure__", None):
            return False
        try:
            return func not in self._surfaced_closures
        except TypeError:
            return False

    def mark_surfaced(self, func: Callable[..., Any], func_name: str) -> None:
        """Record that *func*'s findings were shown (`needs_surfacing`)."""
        self.analyzed.add(func_name)
        if getattr(func, "__closure__", None):
            try:
                self._surfaced_closures.add(func)
            except TypeError:
                pass  # not weak-referenceable: the name stands for it

    def purity_mode(self, func_name: str) -> PurityMode:
        cf = self.cached.get(func_name)
        return cf.purity if cf is not None else "warn"

    def is_frozen(self, func_name: str | None) -> bool:
        cf = self.cached.get(func_name) if func_name is not None else None
        return cf is not None and cf.frozen

    def effective_ttl(self, func_name: str, own_ttl: int | None) -> int | None:
        """The TTL actually used for *func_name*: the minimum of its own TTL and
        the TTLs of cached functions it (transitively) depends on.

        A function whose result derives from a TTL'd dependency must refresh at
        least as often as that dependency - otherwise, because ``depends_on``
        tracks source (not runtime freshness), the downstream keeps returning a
        stale value after the dependency's TTL refresh. Functions with no TTL'd
        dependency are unaffected (effective TTL == own TTL)."""
        cached = self._effective_ttl_cache.get(func_name)
        if cached is not None or func_name in self._effective_ttl_cache:
            return cached
        ttls = [t for t in self._collect_dep_ttls(func_name, set()) if t is not None]
        if own_ttl is not None:
            ttls.append(own_ttl)
        eff = min(ttls) if ttls else None
        self._effective_ttl_cache[func_name] = eff
        return eff

    def _collect_dep_ttls(self, func_name: str, visited: set[str]) -> list[int | None]:
        """TTLs of every cached function reachable from *func_name* via the
        dependency graph (cycle-guarded)."""
        if func_name in visited:
            return []
        visited.add(func_name)
        out: list[int | None] = []
        for dep in self.graph.get_dependencies(func_name):
            cf = self.cached.get(dep)
            if cf is not None:
                out.append(cf.ttl)
                out.extend(self._collect_dep_ttls(dep, visited))
        return out

    def register(
        self, cf: CachedFunction, depends_on: list[Callable[..., Any] | DataSource] | None
    ) -> list[Callable[..., Any]]:
        """Register *cf* and its ``depends_on``; return the declared callables
        that are inert, for the caller to warn about (`warn_inert_dependency`)."""
        func, func_name = cf.func, cf.name
        previous = self.cached.get(func_name)
        if previous is not None:
            cf.carry_over(previous)
        self.cached[func_name] = cf
        self.functions[func_name] = func
        new_hash = callable_identity(func)
        old_hash = self.source_hashes.get(func_name)
        if old_hash and old_hash != new_hash:
            self.analyzed.discard(func_name)
            self.populated.discard(func_name)
        self.source_hashes[func_name] = new_hash
        self.graph.add_node(func_name)
        inert = self._register_static_dependencies(func_name, depends_on)
        # A downstream that depends on this function inherits its TTL
        # (effective TTL = min over the dependency closure).
        self._effective_ttl_cache.clear()
        return inert

    def _register_static_dependencies(
        self, func_name: str, depends_on: list[Callable[..., Any] | DataSource] | None
    ) -> list[Callable[..., Any]]:
        """Record *func_name*'s ``depends_on``; return the declared callables
        that are inert (see `_register_declared_callable_dep`)."""
        inert: list[Callable[..., Any]] = []
        if not depends_on:
            return inert
        for dep in depends_on:
            if isinstance(dep, DataSource):
                dep_id = dep.get_id()
                self.data_sources[dep_id] = dep
                self.graph.add_dependency(func_name, dep_id)
            elif callable(dep):
                dep_key = func_key(dep)
                self.graph.add_dependency(func_name, dep_key)
                # A declared callable dep that is NOT a decorated cached function
                # would contribute nothing to the state hash (the hasher only
                # folds functions/data-sources), silently breaking the documented
                # ``depends_on`` promise. Snapshot its source + a live
                # resolution path so edits/reloads invalidate the parent key.
                if dep_key not in self.functions and not self._register_declared_callable_dep(dep, dep_key):
                    inert.append(dep)
        return inert

    def _register_declared_callable_dep(self, dep: Callable[..., Any], dep_key: str) -> bool:
        """Record a plain-callable ``depends_on`` dependency's source identity.

        Stores a source-hash snapshot and a ``(module, attr_chain)`` path for
        live re-resolution (so an on-disk edit + ``importlib.reload`` is seen).
        If the dep has neither source nor bytecode (builtin / C-extension), its
        identity is only its ``module.qualname``, which a rebuilt extension
        does not change: nothing is recorded, and False says the declared
        dependency is inert, for the caller to warn about.
        """
        snapshot = hash_callable_source(dep)
        if bytecode_identity(dep) is None and snapshot == compiled_identity(dep):
            return False
        self.declared_dep_snapshots[dep_key] = snapshot
        module = getattr(dep, "__module__", None)
        qualname = getattr(dep, "__qualname__", None) or getattr(dep, "__name__", None)
        if module and qualname and "<locals>" not in qualname:
            self._declared_dep_paths[dep_key] = (module, tuple(qualname.split(".")))
        return True

    def resolve_declared_dep_hash(self, dep_key: str) -> str | None:
        """Re-resolve a declared plain-callable dep's live source hash.

        Walks the stored ``(module, attr_chain)`` path via ``sys.modules`` and
        hashes the resolved callable's current source. Returns ``None`` on any
        resolution/hash failure so the hasher falls back to the snapshot.
        """
        path = self._declared_dep_paths.get(dep_key)
        if path is None:
            return None
        mod_name, attr_chain = path
        obj: Any = sys.modules.get(mod_name)
        if obj is None:
            return None
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        if not callable(obj):
            return None
        try:
            return hash_callable_source(obj)
        except (OSError, TypeError, ValueError):
            return None

    def code_functions(self, func: Callable, func_name: str) -> list[Any]:
        """The functions whose code a call of *func_name* runs, as far as cash
        follows it: its own, its helpers', and those of the cached functions it
        depends on, transitively."""
        found: list[Any] = []
        seen_names: set[str] = set()
        stack: list[tuple[str, Any]] = [(func_name, func)]
        while stack:
            name, fn = stack.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            if fn is not None:
                found.append(fn)
            report = self.report_for(fn, name) if fn is not None else self.purity_reports.get(name)
            if report is not None:
                for ref in report.helper_objects.values():
                    helper = ref()
                    if helper is not None:
                        found.append(helper)
                for module_name, chain in report.helper_resolution_paths.values():
                    target = resolve_binding(module_name, chain)
                    if callable(target):
                        found.append(target)
            for dep in self.graph.get_dependencies(name):
                if dep in self.functions and dep not in seen_names:
                    stack.append((dep, self.functions[dep]))
        return found

    def ensure_closure_analyzed(self, func: Callable[..., Any]) -> None:
        """Populate graph edges + purity reports for *func* and every cached
        function transitively reachable from it, WITHOUT surfacing warnings.

        Idempotent per source version (guarded by ``self.populated``). Always
        traverses the dependency edges - even when the root is already
        populated - so a dependency invalidated by a source edit gets
        re-populated. The local ``seen`` set bounds cyclic graphs.

        Under ``self.analysis_lock`` because the CACHE KEY is built from what
        this populates. Concurrent first calls otherwise resolved two different
        keys for one call -- the threads that arrived mid-population, and the
        one doing it -- which is what made ``use_locking=True`` look like it
        admitted exactly two threads at every thread count. Both paths into
        this must hold the lock: the wrapper's one-time analysis AND
        ``explain()``, which populates the closure directly and would otherwise
        race it back apart.
        """
        with self.analysis_lock:
            self._ensure_closure_analyzed_locked(func)

    def _ensure_closure_analyzed_locked(self, func: Callable[..., Any]) -> None:
        """The body of ``FunctionRegistry.ensure_closure_analyzed``, with the lock already held."""
        stack = [func]
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            fname = func_key(f)
            if fname in seen:
                continue
            seen.add(fname)
            if self.needs_population(f, fname):
                self.populate(f, fname)
            for dep in self.graph.get_dependencies(fname):
                dep_func = self.functions.get(dep)
                if dep_func is not None:
                    stack.append(dep_func)

    def refresh_helper_bindings(self, func: Callable[..., Any], func_name: str) -> str | None:
        """Re-analyse any report whose call-site bindings moved; name a mock.

        Walks *func* and its cached-dependency closure. A report is built once
        per function, from whatever its call sites' names held at that moment,
        so a helper patched before the first call stayed "the helper" after
        the real one came back, and the helpers below a patched one were never
        walked. When a binding no longer holds the object analysed, that
        function is analysed again from the current bindings (the analyzer's
        own cache applies the same check).

        Returns a description when the tree reaches a mock -- there is no code
        to key and its answer is whatever the test configured, so the caller
        runs this call uncached -- and None otherwise. Never raises: a failure
        here leaves the reports as they were.

        Per call: one ``sys.modules`` lookup, an attribute chain and an
        identity test per binding, no hashing.
        """
        try:
            reason: str | None = None
            stack: list[tuple[Callable[..., Any], str]] = [(func, func_name)]
            seen: set[str] = set()
            while stack:
                f, name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                report = self.report_for(f, name)
                if report is not None and report.helper_bindings and bindings_changed(report):
                    with self.analysis_lock:
                        self.populate(f, name)
                    report = self.report_for(f, name)
                if report is not None and report.unkeyable and reason is None:
                    reason = report.unkeyable[0]
                for dep in self.graph.get_dependencies(name):
                    dep_func = self.functions.get(dep)
                    if dep_func is not None:
                        stack.append((dep_func, dep))
            return reason
        except Exception:  # noqa: BLE001 - never break a call over this
            logger.debug("[CORE] binding refresh failed for %s", func_name, exc_info=True)
            return None

    def populate(self, func: Callable[..., Any], func_name: str) -> None:
        """Record *func*'s cached-call graph edges and purity report (no
        surfacing). The analyzer caches by source hash globally, so this is
        cheap on repeated registrations.
        """
        self.populated.add(func_name)
        called_names = CodeAnalyzer.find_called_functions(func, self.functions, include_references=True)
        for called in called_names:
            if called != func_name:
                self.graph.add_dependency(func_name, called)
        try:
            report = get_analyzer().analyze(func)
        except (OSError, TypeError, SyntaxError, RecursionError) as e:
            # Analyzer must never break caching. On error, treat as clean -
            # the user's compute still runs.
            logger.debug("Purity analyzer failed for %s: %s", func_name, e)
            report = PurityReport()
        self.purity_reports[func_name] = report
        if getattr(func, "__closure__", None):
            try:
                self._closure_reports[func] = report
            except TypeError:
                pass  # not weak-referenceable: the name's report stands for it
