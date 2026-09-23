"""Build a :class:`CacheBackend` from a :class:`CashConfig`.

The only place that turns config into backends, by one path: `tier_specs`
lists the tiers the config describes -- ``config.tiers`` when set, otherwise
the ones ``config.backend`` names (``"tiered"``: RAM then disk) -- each with
the top-level fields filled in where the tier leaves them unset. A tier list
is built as a `TieredBackend` over its tiers, even a list of one; a single
``backend`` type, as that backend alone.

Comparing two configs' `tier_specs` is how ``cash.configure`` tells whether a
change needs the backend rebuilt.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ..exceptions import DependencyNotFoundError
from ._base import CacheBackend
from .adaptive_caps import resolve_disk_cap, resolve_ram_cap
from .cache_dir import DB_FILENAME
from .file_backend import FileBackend
from .memory_backend import InMemoryBackend
from .persistence_policy import PersistencePolicy
from .sqlite_backend import SQLiteBackend
from .tiered_backend import TieredBackend

if TYPE_CHECKING:
    from cash.config import CashConfig, TierConfig

logger = logging.getLogger(__name__)

__all__ = ["apply_persistence_settings", "build_backend_from_config", "build_tiered", "tier_specs"]

#: The tiers ``backend = "tiered"`` stands for.
DEFAULT_STACK = ("memory", "file")

#: A tier as `tier_specs` resolves it: its type, and the settings it is built
#: from, with the top-level fallbacks applied.
TierSpec = tuple[str, tuple[tuple[str, Any], ...]]


def build_backend_from_config(config: CashConfig) -> CacheBackend:
    """The backend *config* describes. See the module docstring."""
    tiers = [_build(kind, dict(settings)) for kind, settings in tier_specs(config)]
    return build_tiered(tiers, config) if config.tiers or len(tiers) > 1 else tiers[0]


def build_tiered(backends: list[CacheBackend], config: CashConfig) -> TieredBackend:
    """A `TieredBackend` over *backends*, with the persistence policy *config* sets."""
    return TieredBackend(backends, policy=PersistencePolicy.from_config(config))


def apply_persistence_settings(backend: CacheBackend, config: CashConfig) -> None:
    """Give a running `TieredBackend` the persistence policy *config* asks for.

    For ``cash.configure``: changing the policy must not rebuild the stack,
    which would drop the RAM tier. Any other backend has no policy.
    """
    if isinstance(backend, TieredBackend):
        backend.policy = PersistencePolicy.from_config(config)


def tier_specs(config: CashConfig) -> list[TierSpec]:
    """Each tier *config* describes, fully resolved against the top-level fields.

    Pure: nothing is measured or created, so two configs can be compared.
    """
    from cash.config import TierConfig

    tiers = list(config.tiers) or [TierConfig(type=t) for t in _stack_of(config.backend)]
    return [(t.type, tuple(sorted(_settings(t, config).items()))) for t in tiers]


def _stack_of(backend: str) -> tuple[str, ...]:
    return DEFAULT_STACK if backend == "tiered" else (backend,)


def _pick(own: Any, fallback: Any) -> Any:
    return own if own is not None else fallback


def _settings(tier: TierConfig, config: CashConfig) -> dict[str, Any]:
    """What a tier of this type is built from: its own setting, else the top-level one."""
    t = tier.type
    if t == "memory":
        return {
            "max_entries": _pick(tier.max_entries, config.max_memory_entries),
            "max_size_bytes": tier.max_size_bytes,
        }
    if t in ("file", "sqlite"):
        out = {
            "cache_dir": tier.cache_dir or config.cache_dir,
            "max_size_bytes": _pick(tier.max_size_bytes, config.max_cache_size),
            "default_ttl": tier.default_ttl,
        }
        if t == "file":
            out["compress"] = _pick(tier.compress, config.compress)
            out["flush_interval"] = _pick(tier.flush_interval, config.flush_interval)
        else:
            out["db_path"] = tier.db_path
        return out
    if t == "redis":
        return {
            "host": tier.host or config.redis_host,
            "port": _pick(tier.port, config.redis_port),
            "db": _pick(tier.db, config.redis_db),
            "password": _pick(tier.password, config.redis_password),
            "prefix": tier.prefix or config.redis_prefix,
        }
    if t == "s3":
        return {
            "bucket": tier.bucket or config.s3_bucket,
            "region": tier.region or config.s3_region,
            "prefix": tier.prefix or config.s3_prefix,
        }
    raise ValueError(f"Unknown tier type {t!r}: one of memory, file, sqlite, redis, s3.")


def _build(kind: str, s: dict[str, Any]) -> CacheBackend:
    """One backend from its resolved settings.

    A size left unset is sized to the machine: the RAM tier to system memory,
    a disk tier to the free space on its volume, re-derived as the cache grows
    (``adaptive_cap``). A size that was set is kept exactly.
    """
    if kind == "memory":
        cap = s["max_size_bytes"]
        return InMemoryBackend(max_entries=s["max_entries"], max_size_bytes=resolve_ram_cap() if cap is None else cap)
    cap = s.get("max_size_bytes")
    if kind in ("file", "sqlite") and cap is None:
        cap = resolve_disk_cap(s["cache_dir"])
    if kind == "file":
        return FileBackend(
            cache_dir=s["cache_dir"],
            compress=s["compress"],
            max_size_bytes=cap,
            flush_interval=s["flush_interval"],
            adaptive_cap=s["max_size_bytes"] is None,
            default_ttl=s["default_ttl"],
        )
    if kind == "sqlite":
        return SQLiteBackend(
            db_path=s["db_path"] or _sqlite_db_path(s["cache_dir"]),
            max_size_bytes=cap,
            default_ttl=s["default_ttl"],
        )
    if kind == "redis":
        return _build_redis(**s)
    return _build_s3(**s)


def _sqlite_db_path(cache_dir: str) -> str:
    """The database inside *cache_dir*, where the CLI looks for it.
    The directory is created here because SQLite will not make it."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        logger.debug("[SQLITE] could not create %s", cache_dir)
    return os.path.join(cache_dir, DB_FILENAME)


# The remote backends are imported lazily, so a missing extra does not break
# importing this module.


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
