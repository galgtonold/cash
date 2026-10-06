"""The function ``@cash.cache`` hands back: the caching wrapper, the stats
layer around it, and the API it carries (``cache_info``, ``cache_clear``,
``explain``)."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._active import ACTIVE_CONFIG
from .cached_function import new_stats
from .call_state import CACHE_MISS, CALL_ENTRY, enter_cached_call, exit_cached_call
from .script_pickling import expose_script_function, refuse_pickling_by_value

if TYPE_CHECKING:
    from ..backends import CacheBackend
    from ..config.schema import CashConfig
    from .backend_slot import BackendSlot
    from .cached_function import CachedFunction
    from .explain import CacheExplanation, Explainer
    from .maintenance import Maintenance
    from .registry import FunctionRegistry
    from .reporting import Notices
    from .rng import RngWatch
    from .runtime import CallRunner, KeyBuilder


def _backend_cache_dir(backend: CacheBackend | None) -> str | None:
    """The directory *backend* keeps entries in -- its disk tier's, if tiered."""
    directory = backend.local_dir if backend is not None else None
    return os.path.abspath(directory) if directory else None


def _count(stats: dict[str, Any], slot: list) -> None:
    """Add the entry THIS call logged to *stats*, if it logged one: a call
    that raised before its lookup, or went through uncounted, leaves it
    empty."""
    call = slot[0]
    if call is None:
        return
    if call["cache_hit"]:
        stats["hits"] += 1
        stats["total_time_saved"] += call.get("time_saved", 0.0)
        stats["lookup_seconds"] += call.get("execution_time", 0.0)
    else:
        stats["misses"] += 1
        stats["miss_overhead_seconds"] += call.get("cash_seconds") or 0.0
        missed = call["miss_reason"]
        stats["miss_reasons"][missed.kind] += 1
        if missed.changed:
            stats["changed"][missed.changed] += 1
        if call.get("not_stored"):
            stats["not_stored"][call["not_stored"]] += 1
        elif call.get("not_persisted"):
            stats["not_persisted"][call["not_persisted"]] += 1


class Wrappers:
    """Builds the wrapped function for one `Cash` instance.

    *config* and *locking* are read on every call, not once: a setting
    changed after decoration (``disable``, ``use_locking``) applies to the
    next call.

    What a wrapper closes over reaches the instance's collaborators through
    this object, so pickling a wrapper by value meets `__reduce__` first and
    says what to do instead of reporting a lock.
    """

    def __init__(
        self,
        config: Callable[[], CashConfig],
        locking: Callable[[], bool],
        registry: FunctionRegistry,
        backend_slot: BackendSlot,
        keys: KeyBuilder,
        runner: CallRunner,
        rng: RngWatch,
        explainer: Explainer,
        maintenance: Maintenance,
        notices: Notices,
    ) -> None:
        self._config = config
        self._locking = locking
        self._registry = registry
        self._backend_slot = backend_slot
        self._keys = keys
        self._runner = runner
        self._rng = rng
        self._explainer = explainer
        self._maintenance = maintenance
        self._notices = notices

    def __reduce__(self):
        """Refused like `Cash`'s own (`refuse_pickling_by_value`)."""
        refuse_pickling_by_value()

    def wrap(self, cf: CachedFunction) -> Callable:
        """The function ``@cash.cache`` returns for *cf*."""
        stats_wrapper = self._with_stats(cf, self.caching_wrapper(cf))
        self._attach_api(cf, stats_wrapper)
        return stats_wrapper

    def caching_wrapper(self, spec: CachedFunction) -> Callable:
        """Build and return the caching wrapper for *spec*, sync or async.

        One wrapper for both: everything before and after the body is the
        same sync code (`CallRunner.lookup`, `CallRunner.body_scope`,
        `CallRunner.finish_miss`), and the two variants differ only in whether
        they await the body, so they cannot drift apart.
        """
        if inspect.iscoroutinefunction(spec.func):
            return self._async_caching_wrapper(spec)
        return self._sync_caching_wrapper(spec)

    def _async_caching_wrapper(self, spec: CachedFunction) -> Callable:
        """The caching wrapper of a coroutine function: the body is awaited."""
        func = spec.func
        runner = self._runner
        locking = self._locking

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = runner.lookup(spec, args, kwargs, async_body=True)
            if call.outcome is not CACHE_MISS:
                # A result the key path produced by calling `func` itself
                # (no key) is a coroutine here: await it before handing back.
                if inspect.iscoroutine(call.outcome):
                    return await call.outcome
                return call.outcome

            async def compute() -> Any:
                with runner.body_scope(spec, call) as run:
                    run.res = await func(*args, **kwargs)
                return runner.finish_miss(spec, call, run)

            if not locking():
                return await compute()
            return await runner.single_flight(spec, call, compute)

        return async_wrapper

    def _sync_caching_wrapper(self, spec: CachedFunction) -> Callable:
        """The caching wrapper of a plain function."""
        func = spec.func
        runner = self._runner
        locking = self._locking

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = runner.lookup(spec, args, kwargs, async_body=False)
            if call.outcome is not CACHE_MISS:
                return call.outcome

            def compute() -> Any:
                with runner.body_scope(spec, call) as run:
                    run.res = func(*args, **kwargs)
                return runner.finish_miss(spec, call, run)

            if locking():
                return runner.compute_with_lock(spec, call, compute)
            return compute()

        return wrapper

    def _with_stats(self, cf: CachedFunction, wrapper: Callable) -> Callable:
        """Wrap *wrapper* with hit/miss stat tracking.

        Dispatches on whether *func* is a coroutine function so the stats
        update (from the entry `CallLog.log` left in this call's
        `CALL_ENTRY` slot) happens AFTER the await for async, and
        synchronously otherwise.
        """
        func, func_name, allow_random = cf.func, cf.name, cf.allow_random
        # Read through `cf` by the end-of-run summary too.
        stats = cf.stats
        warn_unseeded = self._rng.warn_unseeded_estimator_result

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = self._config()
                if settings.disable:
                    stats["bypassed"] += 1
                    return await func(*args, **kwargs)
                token = ACTIVE_CONFIG.set(settings)
                slot: list = [None]
                slot_token = CALL_ENTRY.set(slot)
                enter_cached_call()
                try:
                    result = await wrapper(*args, **kwargs)
                finally:
                    exit_cached_call()
                    CALL_ENTRY.reset(slot_token)
                    ACTIVE_CONFIG.reset(token)
                    _count(stats, slot)
                warn_unseeded(func_name, result, allow_random)
                return result

            return stats_wrapper

        @functools.wraps(func)
        def sync_stats_wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = self._config()
            # Before anything else: disabled means the function, and
            # nothing of cash's -- no key, no analysis, no lookup, no store.
            if settings.disable:
                stats["bypassed"] += 1
                return func(*args, **kwargs)
            # This instance's settings for the checks the call makes; see
            # ACTIVE_CONFIG.
            token = ACTIVE_CONFIG.set(settings)
            slot: list = [None]
            slot_token = CALL_ENTRY.set(slot)
            enter_cached_call()
            try:
                result = wrapper(*args, **kwargs)
            finally:
                exit_cached_call()
                CALL_ENTRY.reset(slot_token)
                ACTIVE_CONFIG.reset(token)
                _count(stats, slot)
            warn_unseeded(func_name, result, allow_random)
            return result

        return sync_stats_wrapper

    def _attach_api(self, cf: CachedFunction, stats_wrapper: Callable) -> None:
        """Give *stats_wrapper* its introspection API and the markers the
        rest of cash reads off a cached function.

        * ``cache_info()`` - hit/miss stats plus a rolling list of recent
          warnings emitted for this function.
        * ``cache_clear()`` - drop backend entries, reset stats, drop the
          warning log + dedup marks so re-warnings can fire.
        * ``explain(*args, **kwargs)`` - return a `CacheExplanation`
          for that specific call (sync, even on async wrappers).
        """
        func, func_name = cf.func, cf.name
        stats_wrapper.cache_info = self._cache_info(cf)
        stats_wrapper.cache_clear = self._cache_clear(cf)
        stats_wrapper.explain = self._explain(cf)
        stats_wrapper.__wrapped__ = func
        # Marker so the purity analyzer treats a call to this wrapper as a
        # dependency-graph edge rather than recursing into cash's own wrapper
        # machinery. functools.wraps copies __module__, which would
        # otherwise make the wrapper look like same-package user code.
        stats_wrapper._cash_cached = True
        # Declared TTL, exposed so the notebook statement cache can see it. A
        # ``ttl=0`` function must recompute every call; without this the
        # statement ``x = f()`` gets cached with no TTL under %cash_on and
        # freezes the value the decorator promised to refresh.
        stats_wrapper._cash_declared_ttl = cf.ttl
        # What a call depends on besides its arguments, built by THIS instance
        # for THIS function object, and the TTL it refreshes at: how a cached
        # function reached without an edge in the caller's own registry (on
        # another instance, passed in, held in a table, or still held after a
        # reload) reaches the caller's key.
        stats_wrapper._cash_state = lambda: self._keys.callee_state(cf)
        stats_wrapper._cash_effective_ttl = lambda: self._registry.effective_ttl(func_name, cf.ttl)
        expose_script_function(func, stats_wrapper)
        cf.wrapper = stats_wrapper

    def _cache_info(self, cf: CachedFunction) -> Callable[[], dict[str, Any]]:
        """*cf*'s ``cache_info``."""
        stats = cf.stats

        def cache_info() -> dict[str, Any]:
            """Return this function's hit and miss counts and recent warnings.

            Returns:
                A dict with ``hits``, ``misses``, ``hit_rate``,
                ``total_time_saved`` (seconds), ``miss_reasons`` (count per
                reason) and ``warnings`` (the last 20, each with
                ``category``, ``code``, ``message`` and ``timestamp``).
            """
            total = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total if total > 0 else 0.0
            warnings_log = self._notices.log_of(cf)
            return {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": hit_rate,
                "total_time_saved": stats["total_time_saved"],
                "miss_reasons": {str(kind): n for kind, n in stats["miss_reasons"].items()},
                "warnings": warnings_log,
            }

        return cache_info

    def _cache_clear(self, cf: CachedFunction) -> Callable[[], None]:
        """*cf*'s ``cache_clear``."""
        stats = cf.stats

        def cache_clear() -> None:
            """Delete this function's cache entries and reset its statistics
            and warning log, so its warnings are shown again."""
            stats.update(new_stats())
            self._maintenance.delete_function_entries(cf.name)
            self._notices.forget(cf)

        return cache_clear

    def _explain(self, cf: CachedFunction) -> Callable[..., CacheExplanation]:
        """*cf*'s ``explain``."""

        def explain(*args: Any, **kwargs: Any) -> CacheExplanation:
            """Return a `CacheExplanation` of whether a call with these
            arguments would hit, and why. Runs nothing and changes nothing."""
            token = ACTIVE_CONFIG.set(self._config())
            try:
                explanation = self._explainer.explain(cf, args, kwargs)
            finally:
                ACTIVE_CONFIG.reset(token)
            return dataclasses.replace(explanation, cache_dir=_backend_cache_dir(self._backend_slot.backend))

        return explain
