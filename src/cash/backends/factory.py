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
from typing import TYPE_CHECKING, Any

from ._base import CacheBackend
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .sqlite_backend import SQLiteBackend
from .tiered_backend import TieredBackend

if TYPE_CHECKING:
    from cash.config import CashConfig, TierConfig

logger = logging.getLogger(__name__)

__all__ = ["build_backend_from_config"]


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
    from .adaptive_caps import resolve_disk_cap
    return resolve_disk_cap(config.cache_dir)


def _disk_cap_is_adaptive(config: "CashConfig") -> bool:
    """Did the disk cap come from the policy rather than from the user?

    Only an adaptive cap may be re-derived once the backend knows its own
    footprint (see ``FileBackend._ensure_size_scanned``). An explicit
    ``max_cache_size`` is the user's number and must stay exactly where they
    put it.
    """
    return config.max_cache_size is None


def _resolve_ram_cap(config: "CashConfig") -> int:
    """Byte cap for the RAM tier — always its own modest, machine-scaled cap.

    Independent of ``max_cache_size`` (which caps disk): a user pinning the
    disk cap should not accidentally make the RAM tier unbounded, and the RAM
    tier should stay a small fraction of system memory regardless.
    """
    from .adaptive_caps import resolve_ram_cap
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
            db_path=config.cache_dir,  # reuse cache_dir as path for simple mode
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
        f"Unknown backend type {backend_type!r}. "
        "Set config.backend to one of: tiered, memory, file, sqlite, redis, s3."
    )


# ---------------------------------------------------------------------------
# Default tiered stack (RAM + file) — built from the top-level fields
# ---------------------------------------------------------------------------

def _build_default_tiered(config: "CashConfig") -> TieredBackend:
    ram = InMemoryBackend(
        max_entries=config.max_memory_entries,
        max_size_bytes=_resolve_ram_cap(config),
    )
    disk = FileBackend(
        cache_dir=config.cache_dir,
        compress=config.compress,
        max_size_bytes=_resolve_disk_cap(config),
        flush_interval=config.flush_interval,
        adaptive_cap=_disk_cap_is_adaptive(config),
    )

    if not config.smart_persistence:
        return TieredBackend([ram, disk])
    return _build_smart_tiered([ram, disk], config)


# Compute floor for the smart-persistence stack: nothing under this many
# seconds is promoted past RAM.
#
# NOT because "disk I/O costs more than rerunning" -- that was the original
# rationale here and it is false. Measured on an NVMe volume, a small entry
# round-trips in 0.61 ms from a cold FileBackend and 0.14 ms warm, so I/O is
# ~3% of rerunning a 20 ms statement, not more than it.
#
# The floor earns its place for a different reason: statements under it are
# cheap *by definition*, so persisting them buys almost no time and costs a
# file each. Measured on 01_nyc_taxi_analysis, dropping this to 10 ms:
#
#     floor=100ms   restored  9/140   still computed 3.19 s   cache   0.2 MiB
#     floor= 10ms   restored 15/140   still computed 3.33 s   cache  70.9 MiB
#
# Six more statements come back, not one second is saved, and the cache grows
# 355x. The cost is worse than linear too: FileBackend._ensure_initialized
# unpickles every .meta file at kernel start, so entry count is paid on every
# start forever.
#
# So: the number is right, the old reason for it was not. Anyone tempted to
# lower it on the strength of the I/O argument should re-read the table above
# first -- and if they lower it anyway, measure the cache size and the
# kernel-start scan, not just the restore count.
_SMART_PERSIST_COMPUTE_FLOOR_S = 0.1


def _config_number(config: "CashConfig", attr: str, default: float) -> float:
    """Read a numeric config attribute, defaulting on anything non-numeric.

    Guards against test doubles (``MagicMock`` auto-attributes coerce to
    ``1.0`` under ``float()``, so an ``isinstance`` check is required rather
    than a ``try: float(...)``)."""
    value = getattr(config, attr, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _build_smart_tiered(backends: list[CacheBackend], config: "CashConfig") -> TieredBackend:
    """Wrap *backends* in a serialization-aware smart-persistence policy."""
    min_savings = _config_number(config, "min_cache_savings_pct", 0.20)
    return TieredBackend(
        backends,
        promotion_policy=_build_smart_persistence_policy(config),
        min_persist_compute_s=_SMART_PERSIST_COMPUTE_FLOOR_S,
        min_persist_savings_pct=min_savings,
    )


def _build_smart_persistence_policy(config: "CashConfig"):
    """The 2-arg fallback policy for entries with no cost-model family.

    Serialization-aware, using the fitted ``cost_model`` (same rule the
    statement processor's Gate A applies) instead of the old raw-bandwidth
    arithmetic that made bigger objects *less* likely to persist. Since the
    2-arg signature carries no type, it assumes the slowest (``_GENERIC``)
    family — a conservative floor. The set path recomputes with the real type
    whenever ``metadata['cost_model_family']`` is present.
    """
    min_persist_compute_s = _SMART_PERSIST_COMPUTE_FLOOR_S
    min_savings = _config_number(config, "min_cache_savings_pct", 0.20)

    def policy(execution_time: float, size_bytes: int) -> bool:
        if execution_time < min_persist_compute_s:
            return False
        from cash.notebook import cost_model

        est_restore = cost_model.estimated_restore_time("", size_bytes, "disk")
        return execution_time - est_restore > min_savings * execution_time

    return policy


# ---------------------------------------------------------------------------
# Advanced-mode: build from explicit tier list
# ---------------------------------------------------------------------------

def _build_tiered_from_tier_list(config: "CashConfig") -> TieredBackend:
    backends: list[CacheBackend] = []
    for tier in config.tiers:
        backends.append(_build_tier(tier, config))
    if not config.smart_persistence:
        return TieredBackend(backends)
    return _build_smart_tiered(backends, config)


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
            max_size_bytes=tier.max_size_bytes if tier.max_size_bytes is not None else _resolve_ram_cap(config),
        )
    if t == "file":
        return FileBackend(
            cache_dir=tier.cache_dir or config.cache_dir,
            compress=tier.compress if tier.compress is not None else config.compress,
            max_size_bytes=tier.max_size_bytes if tier.max_size_bytes is not None else _resolve_disk_cap(config),
            flush_interval=tier.flush_interval if tier.flush_interval is not None else config.flush_interval,
        )
    if t == "sqlite":
        return SQLiteBackend(
            db_path=tier.db_path or tier.cache_dir or config.cache_dir,
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
        from cash.exceptions import DependencyNotFoundError
        raise DependencyNotFoundError(
            "Redis backend requires `pip install cash-lib[redis]` (the `redis` package)."
        ) from exc
    # Drop None values so RedisBackend's own defaults apply.
    return RedisBackend(**{k: v for k, v in kwargs.items() if v is not None})


def _build_s3(*, bucket: str, region: str, prefix: str) -> CacheBackend:
    try:
        from .s3_backend import S3Backend
    except ImportError as exc:
        from cash.exceptions import DependencyNotFoundError
        raise DependencyNotFoundError(
            "S3 backend requires `pip install cash-lib[s3]` (the `boto3` package)."
        ) from exc
    if not bucket:
        raise ValueError("S3 backend requires a non-empty bucket name (set s3_bucket)")
    kwargs: dict[str, Any] = {"bucket": bucket, "prefix": prefix}
    if region:
        kwargs["region_name"] = region
    return S3Backend(**kwargs)
