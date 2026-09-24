"""The dependency graph of cached functions: declared dependencies, TTLs
inherited along it, and the one-time analysis of each function."""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Callable
from typing import Any

from ..analysis.code_analyzer import CodeAnalyzer
from ..data_source import DataSource, state_token_of
from ..exceptions import CashCacheIneffectiveWarning
from ..purity_analyzer import PurityReport, bindings_changed, get_analyzer, resolve_binding
from ..source_norm import bytecode_identity, compiled_identity
from .cached_function import PurityMode
from .call_state import KeyBuildFailed
from .code_identity import func_key, hash_callable_source

logger = logging.getLogger(__name__)


class RegistryMixin:
    """Dependencies between cached functions and their one-time analysis."""

    def _purity_mode(self, func_name: str) -> PurityMode:
        cf = self._cached.get(func_name)
        return cf.purity if cf is not None else "warn"

    def _is_frozen(self, func_name: str | None) -> bool:
        cf = self._cached.get(func_name) if func_name is not None else None
        return cf is not None and cf.frozen

    def _effective_ttl(self, func_name: str, own_ttl: int | None) -> int | None:
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
            cf = self._cached.get(dep)
            if cf is not None:
                out.append(cf.ttl)
                out.extend(self._collect_dep_ttls(dep, visited))
        return out

    def _register_static_dependencies(
        self, func_name: str, depends_on: list[Callable[..., Any] | DataSource] | None
    ) -> None:
        if not depends_on:
            return
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
                if dep_key not in self.functions:
                    self._register_declared_callable_dep(dep, dep_key, func_name)

    def _register_declared_callable_dep(self, dep: Callable[..., Any], dep_key: str, func_name: str) -> None:
        """Record a plain-callable ``depends_on`` dependency's source identity.

        Stores a source-hash snapshot and a ``(module, attr_chain)`` path for
        live re-resolution (so an on-disk edit + ``importlib.reload`` is seen).
        If the dep has neither source nor bytecode (builtin / C-extension), its
        identity is only its ``module.qualname``, which a rebuilt extension
        does not change: warn once that the declared dependency is inert
        rather than silently ignore it.
        """
        snapshot = hash_callable_source(dep)
        if bytecode_identity(dep) is None and snapshot == compiled_identity(dep):
            self._warn_once(
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
            return
        self._declared_dep_snapshots[dep_key] = snapshot
        module = getattr(dep, "__module__", None)
        qualname = getattr(dep, "__qualname__", None) or getattr(dep, "__name__", None)
        if module and qualname and "<locals>" not in qualname:
            self._declared_dep_paths[dep_key] = (module, tuple(qualname.split(".")))

    def _resolve_declared_dep_hash(self, dep_key: str) -> str | None:
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

    def _resolve_dynamic_dependencies(
        self,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
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

    def _code_functions(self, func: Callable, func_name: str) -> list[Any]:
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
            report = self._purity_reports.get(name)
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

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Populate analysis for *func* + its transitive cached-dependency
        closure, then surface *func*'s own purity issues.

        Populating the WHOLE closure (not just *func*) before the first cache
        key is computed is what keeps the key stable from the very first call.
        The state hash folds in each dependency's purity-report
        ``helper_source_hashes``; filled lazily on each dependency's own first
        call, the key would deepen only after the chain warmed, and a fresh
        process would miss the first call to every cached function even with a
        valid entry on disk.

        Surfacing stays per-function: each dependency warns/raises on its OWN
        first direct call, not here, so eager population doesn't change which
        warnings fire or when.
        """
        self._ensure_closure_analyzed(func)
        func_name = func_key(func)
        report = self._purity_reports.get(func_name) or PurityReport()
        mode = self._purity_mode(func_name)
        self._surface_purity(func_name, report, mode)

    def _ensure_closure_analyzed(self, func: Callable[..., Any]) -> None:
        """Populate graph edges + purity reports for *func* and every cached
        function transitively reachable from it, WITHOUT surfacing warnings.

        Idempotent per source version (guarded by ``self._populated``). Always
        traverses the dependency edges - even when the root is already
        populated - so a dependency invalidated by a source edit gets
        re-populated. The local ``seen`` set bounds cyclic graphs.

        Under ``self._analysis_lock`` because the CACHE KEY is built from what
        this populates. Concurrent first calls otherwise resolved two different
        keys for one call -- the threads that arrived mid-population, and the
        one doing it -- which is what made ``use_locking=True`` look like it
        admitted exactly two threads at every thread count. Both paths into
        this must hold the lock: the wrapper's one-time analysis AND
        ``explain()``, which populates the closure directly and would otherwise
        race it back apart.
        """
        with self._analysis_lock:
            self._ensure_closure_analyzed_locked(func)

    def _ensure_closure_analyzed_locked(self, func: Callable[..., Any]) -> None:
        """The body of ``_ensure_closure_analyzed``, with the lock already held."""
        stack = [func]
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            fname = func_key(f)
            if fname in seen:
                continue
            seen.add(fname)
            if fname not in self._populated:
                self._populate_analysis(f, fname)
            for dep in self.graph.get_dependencies(fname):
                dep_func = self.functions.get(dep)
                if dep_func is not None:
                    stack.append(dep_func)

    def _refresh_helper_bindings(self, func: Callable[..., Any], func_name: str) -> str | None:
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
                report = self._purity_reports.get(name)
                if report is not None and report.helper_bindings and bindings_changed(report):
                    with self._analysis_lock:
                        self._populate_analysis(f, name)
                    report = self._purity_reports.get(name)
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

    def _populate_analysis(self, func: Callable[..., Any], func_name: str) -> None:
        """Record *func*'s cached-call graph edges and purity report (no
        surfacing). The analyzer caches by source hash globally, so this is
        cheap on repeated registrations.
        """
        self._populated.add(func_name)
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
        self._purity_reports[func_name] = report
