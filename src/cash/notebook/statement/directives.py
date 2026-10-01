"""What a statement's ``# @cash:`` directives and its callees say about how
it is cached: its TTL, whether it is forced to disk, skipped, allowed to
draw unseeded, or opted in to caching an estimator fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...analysis.annotations import CacheAnnotation

__all__ = ["statement_directives", "ttl_floor_from_called_functions"]


def statement_directives(
    annotation: CacheAnnotation | None,
    ttl: int | None,
    persist_all: bool,
) -> tuple[int | None, bool, bool, bool, bool]:
    """Return ``(effective_ttl, force_persist, skip_cache, allow_random, cache_fit)``.

    *persist_all* (config / ``%cash_persist`` magic) forces persistence
    for every statement, as if each carried ``# @cash:persist``.

    ``allow_random`` (``# @cash:allow-random``) is *advisory only* — it
    suppresses the unseeded-randomness warning and nothing else.  It must
    never reach the cacheability decision: an unseeded random statement is
    cacheable by design, with or without the directive.

    ``cache_fit`` (``# @cash:cache-fit``) opts a bare ``estimator.fit(X, y)``
    statement IN to the estimator-fit caching path.  It is off by
    default: without it a bare fit is skip-cached and simply re-executes,
    which is net-neutral.  It does NOT keep aliases correct:
    ``backup = model`` is an ordinary assignment whose own restore rebinds a
    pre-fit copy, independently of the fit.
    """
    effective_ttl = ttl
    force_persist = persist_all
    skip_cache = False
    allow_random = False
    cache_fit = False
    if annotation:
        if annotation.ttl is not None:
            effective_ttl = annotation.ttl
        force_persist = force_persist or annotation.persist
        skip_cache = annotation.no_cache
        allow_random = annotation.allow_random
        cache_fit = annotation.cache_fit
    return effective_ttl, force_persist, skip_cache, allow_random, cache_fit


def ttl_floor_from_called_functions(inputs: set[str], effective_ttl: int | None, user_ns: dict[str, Any]) -> int | None:
    """Lower *effective_ttl* to the TTL of any ``@cash.cache`` function called here.

    A statement ``x = f()`` where ``f`` is decorated ``@cash.cache(ttl=0)`` was
    cached with no TTL under %cash_on, so the statement restore froze ``x`` at
    the first result — silently overriding the freshness the decorator
    promised. The call target appears in ``inputs`` (the analyzer
    lists ``f`` for ``x = f()``); if it is a cash wrapper with a smaller
    declared TTL, the statement must expire at least as often. ``ttl=0`` then
    rides the existing immediate-expiry path, so every run is a miss
    and the decorated body runs every time, as ``ttl=0`` asks.

    Only LOWERS the TTL and only for a wrapper carrying an explicit TTL, so a
    plain ``@cash.cache`` (ttl=None) call is completely unaffected — the
    statement caches exactly as before.
    """
    floor = effective_ttl
    for name in inputs:
        fn = user_ns.get(name)
        if fn is None or not getattr(fn, "_cash_cached", False):
            continue
        declared = getattr(fn, "_cash_declared_ttl", None)
        if declared is None:
            continue
        floor = declared if floor is None else min(floor, declared)
    return floor
