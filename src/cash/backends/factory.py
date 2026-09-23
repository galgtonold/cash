"""Build a concrete :class:`CacheBackend` from a :class:`CashConfig`.

This is the only place that translates declarative config into live
backend instances. Cash's constructor delegates here; the runtime
``cash.configure()`` mutation API delegates here on backend-affecting
field changes.

Three input shapes:

1. ``config.tiers`` is non-empty
   → ``TieredBackend([_build_tier(t) for t in config.tiers])``
   The simple-mode ``config.backend`` field is ignored.

2. ``config.tiers`` empty AND ``config.backend == "tiered"`` (default)
   → ``TieredBackend([InMemoryBackend, FileBackend])`` constructed from
   the top-level fields (cache_dir, compress, max_cache_size, ...).

3. ``config.tiers`` empty AND ``config.backend`` is some other type
   → a single backend of that type, built from the per-backend
   simple-mode connection fields (``redis_host`` / ``s3_bucket`` / ...).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ..exceptions import DependencyNotFoundError
from ._base import CacheBackend
from .adaptive_caps import resolve_disk_cap, resolve_ram_cap
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .persistence_policy import PersistencePolicy
from .sqlite_backend import SQLiteBackend
from .tiered_backend import TieredBackend

if TYPE_CHECKING:
    from cash.config import CashConfig, TierConfig

logger = logging.getLogger(__name__)

__all__ = ["apply_persistence_settings", "build_backend_from_config"]


def build_backend_from_config(config: "CashConfig") -> CacheBackend:
    """Construct a backend stack from *config*. See module docstring."""
    if config.tiers:
        return _build_tiered_from_tier_list(config)
    if config.backend == "tiered":
        return _build_default_tiered(config)
    return _build_single_backend(config.backend, config)


# ---------------------------------------------------------------------------
# Cap resolution — a single ``max_cache_size`` no longer caps every
# tier at a flat 1 GiB. The disk tier scales to free disk, the RAM tier to
# system memory; an explicit ``max_cache_size`` pins the DISK tier only, and
# the RAM tier keeps its own modest auto cap either way.
# ---------------------------------------------------------------------------


def _sqlite_db_path(cache_dir: str) -> str:
    """The database file for a cache directory: ``<cache_dir>/cache.db``.

    A file inside the directory, so the CLI, which looks for entries inside a
    directory, sees it. The directory is created here because SQLite will not
    make it.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        logger.debug("[SQLITE] could not create %s", cache_dir)
    return os.path.join(cache_dir, "cache.db")


def _resolve_disk_cap(config: "CashConfig") -> int | None:
    """Byte cap for a disk tier's LRU.

    ``max_cache_size`` explicit → that value (backward compatible: it has
    always capped the disk tier). ``None`` (auto) → a generous fraction of
    free space on the cache volume, so the disk tier can actually retain
    what the user persists instead of thrashing under a flat 1 GiB.
    """
    explicit = config.max_cache_size
    if explicit is not None:
        return explicit

    return resolve_disk_cap(config.cache_dir)


def _disk_cap_is_adaptive(config: "CashConfig") -> bool:
    """Did the disk cap come from the policy rather than from the user?

    Only an adaptive cap may be re-derived once the backend knows its own
    footprint (see ``FileBackend._ensure_size_scanned``). An explicit
    ``max_cache_size`` is the user's number and must stay exactly where they
    put it.
    """
    return config.max_cache_size is None


def _resolve_ram_cap() -> int:
    """Byte cap for the RAM tier — always its own modest, machine-scaled cap.

    Independent of ``max_cache_size`` (which caps disk): a user pinning the
    disk cap should not accidentally make the RAM tier unbounded, and the RAM
    tier should stay a small fraction of system memory regardless.
    """

    return resolve_ram_cap()


# ---------------------------------------------------------------------------
# Single-backend construction (simple mode, non-default backend)
# ---------------------------------------------------------------------------


def _build_single_backend(backend_type: str, config: "CashConfig") -> CacheBackend:
    """Build one bare backend instance from the simple-mode top-level fields."""
    if backend_type == "memory":
        return InMemoryBackend(max_entries=config.max_memory_entries)
    if backend_type == "file":
        return FileBackend(
            cache_dir=config.cache_dir,
            compress=config.compress,
            max_size_bytes=_resolve_disk_cap(config),
            flush_interval=config.flush_interval,
            adaptive_cap=_disk_cap_is_adaptive(config),
        )
    if backend_type == "sqlite":
        return SQLiteBackend(
            db_path=_sqlite_db_path(config.cache_dir),
            max_size_bytes=_resolve_disk_cap(config),
        )
    if backend_type == "redis":
        return _build_redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            prefix=config.redis_prefix,
        )
    if backend_type == "s3":
        return _build_s3(
            bucket=config.s3_bucket,
            region=config.s3_region,
            prefix=config.s3_prefix,
        )
    raise ValueError(
        f"Unknown backend type {backend_type!r}. Set config.backend to one of: tiered, memory, file, sqlite, redis, s3."
    )


# ---------------------------------------------------------------------------
# Default tiered stack (RAM + file) — built from the top-level fields
# ---------------------------------------------------------------------------


def _build_default_tiered(config: "CashConfig") -> TieredBackend:
    ram = InMemoryBackend(
        max_entries=config.max_memory_entries,
        max_size_bytes=_resolve_ram_cap(),
    )
    disk = FileBackend(
        cache_dir=config.cache_dir,
        compress=config.compress,
        max_size_bytes=_resolve_disk_cap(config),
        flush_interval=config.flush_interval,
        adaptive_cap=_disk_cap_is_adaptive(config),
    )

    return _build_tiered([ram, disk], config)


def _build_tiered(backends: list[CacheBackend], config: "CashConfig") -> TieredBackend:
    return TieredBackend(backends, policy=PersistencePolicy.from_config(config))


def apply_persistence_settings(backend: CacheBackend, config: "CashConfig") -> None:
    """Give a running `TieredBackend` the persistence policy *config* asks for.

    For ``cash.configure``: changing the policy must not rebuild the stack,
    which would drop the RAM tier. Any other backend has no policy.
    """
    if isinstance(backend, TieredBackend):
        backend.policy = PersistencePolicy.from_config(config)


# ---------------------------------------------------------------------------
# Advanced-mode: build from explicit tier list
# ---------------------------------------------------------------------------


def _build_tiered_from_tier_list(config: "CashConfig") -> TieredBackend:
    backends: list[CacheBackend] = []
    for tier in config.tiers:
        backends.append(_build_tier(tier, config))
    return _build_tiered(backends, config)


def _build_tier(tier: "TierConfig", config: "CashConfig") -> CacheBackend:
    """Build one tier from a TierConfig spec.

    Falls back to the top-level CashConfig fields when the tier didn't
    specify its own — so a minimal ``[[tool.cash.tiers]]\ntype = "redis"``
    can still pull host/port from the simple-mode top-level fields.
    """
    t = tier.type
    if t == "memory":
        return InMemoryBackend(
            max_entries=tier.max_entries if tier.max_entries is not None else config.max_memory_entries,
            max_size_bytes=tier.max_size_bytes if tier.max_size_bytes is not None else _resolve_ram_cap(),
        )
    if t == "file":
        return FileBackend(
            cache_dir=tier.cache_dir or config.cache_dir,
            compress=tier.compress if tier.compress is not None else config.compress,
            max_size_bytes=tier.max_size_bytes if tier.max_size_bytes is not None else _resolve_disk_cap(config),
            flush_interval=tier.flush_interval if tier.flush_interval is not None else config.flush_interval,
            # Dropped until round 18: a `default_ttl` on a file tier was
            # accepted, shown by `cash info`, and never applied.
            default_ttl=tier.default_ttl,
        )
    if t == "sqlite":
        return SQLiteBackend(
            db_path=tier.db_path or _sqlite_db_path(tier.cache_dir or config.cache_dir),
            max_size_bytes=tier.max_size_bytes if tier.max_size_bytes is not None else _resolve_disk_cap(config),
            default_ttl=tier.default_ttl,
        )
    if t == "redis":
        return _build_redis(
            host=tier.host or config.redis_host,
            port=tier.port if tier.port is not None else config.redis_port,
            db=tier.db if tier.db is not None else config.redis_db,
            password=tier.password if tier.password is not None else config.redis_password,
            prefix=tier.prefix or config.redis_prefix,
        )
    if t == "s3":
        return _build_s3(
            bucket=tier.bucket or config.s3_bucket,
            region=tier.region or config.s3_region,
            prefix=tier.prefix or config.s3_prefix,
        )
    if t == "tiered":
        # Nested tiered is unusual but legal; we don't recurse into config
        # so a tier of type=tiered with no further config is just a default
        # tiered stack.
        return _build_default_tiered(config)
    raise ValueError(f"Unknown tier type: {t!r}")


# ---------------------------------------------------------------------------
# Optional backend imports — kept lazy so missing extras don't break
# importing this module.
# ---------------------------------------------------------------------------


def _build_redis(**kwargs: Any) -> CacheBackend:
    try:
        from .redis_backend import RedisBackend
    except ImportError as exc:
        raise DependencyNotFoundError(
            "Redis backend requires `pip install cash-lib[redis]` (the `redis` package)."
        ) from exc
    # Drop None values so RedisBackend's own defaults apply.
    return RedisBackend(**{k: v for k, v in kwargs.items() if v is not None})


def _build_s3(*, bucket: str, region: str, prefix: str) -> CacheBackend:
    try:
        from .s3_backend import S3Backend
    except ImportError as exc:
        raise DependencyNotFoundError("S3 backend requires `pip install cash-lib[s3]` (the `boto3` package).") from exc
    if not bucket:
        raise ValueError("S3 backend requires a non-empty bucket name (set s3_bucket)")
    kwargs: dict[str, Any] = {"bucket": bucket, "prefix": prefix}
    if region:
        kwargs["region_name"] = region
    return S3Backend(**kwargs)
