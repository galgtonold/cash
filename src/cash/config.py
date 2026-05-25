"""Cash configuration system.

Resolution precedence (highest priority wins):

    1. Explicit constructor kwargs       (Cash(redis_host="..."))
    2. Environment variables             (CASH_* and CASH_TIER_<N>_*)
    3. Project config                    (./pyproject.toml [tool.cash])
    4. User config                       (~/.config/cash/config.toml or
                                          %APPDATA%/cash/config.toml on Windows)
    5. CashConfig dataclass defaults

Every field on ``CashConfig`` is settable through every layer. The
``tiers`` list is settable as a whole from TOML and field-by-field from
env vars (``CASH_TIER_0_TYPE=redis``, ``CASH_TIER_0_HOST=...``).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CashConfig",
    "TierConfig",
    "get_config",
    "create_default_config",
]


# ---------------------------------------------------------------------------
# Supported tier types — must align with cash.backends.* classes.
# ---------------------------------------------------------------------------

_SUPPORTED_TIER_TYPES = frozenset({"memory", "file", "sqlite", "redis", "s3", "tiered"})


@dataclass
class TierConfig:
    """One backend tier inside a tiered stack.

    Only the fields relevant to the chosen ``type`` are read by the
    backend factory; the rest are left at their defaults so the same
    ``TierConfig`` shape can carry every backend type without
    forcing the user to wrap each one in a sum type.

    Used inside ``CashConfig.tiers`` to describe a multi-backend
    pipeline (e.g. ``[memory, redis, s3]``).
    """

    type: str
    """Backend type. One of ``"memory"``, ``"file"``, ``"sqlite"``,
    ``"redis"``, ``"s3"``, ``"tiered"``. Raises ``ValueError`` on
    unknown values."""

    # memory / file / sqlite shared:
    max_size_bytes: int | None = None
    """Per-tier size cap in bytes. Also serves as the
    *promotion hint* — values larger than this skip this tier when
    used inside a tiered stack."""

    default_ttl: int | None = None
    """Default TTL in seconds for entries in this tier. Overridden
    per-call by ``@cash.cache(ttl=...)``."""

    # memory:
    max_entries: int | None = None
    """memory tier only — LRU entry cap."""

    # file:
    cache_dir: str | None = None
    """file tier only — directory for the cache files. Defaults to
    ``CashConfig.cache_dir`` when omitted."""

    compress: bool | None = None
    """file tier only — gzip compression on/off. Defaults to
    ``CashConfig.compress`` when omitted."""

    flush_interval: int | None = None
    """file tier only — metadata flush interval (seconds)."""

    # sqlite:
    db_path: str | None = None
    """sqlite tier only — path to the .db file."""

    wal_mode: bool | None = None
    """sqlite tier only — enable WAL journal mode for better
    concurrency. Default True."""

    # redis:
    host: str | None = None
    """redis tier only — server hostname."""

    port: int | None = None
    """redis tier only — server port."""

    db: int | None = None
    """redis tier only — logical database number."""

    password: str | None = None
    """redis tier only — AUTH password (if enabled)."""

    prefix: str | None = None
    """redis tier only — key prefix."""

    # s3:
    bucket: str | None = None
    """s3 tier only — bucket name."""

    region: str | None = None
    """s3 tier only — AWS region (e.g. ``"us-east-1"``)."""

    def __post_init__(self) -> None:
        if self.type not in _SUPPORTED_TIER_TYPES:
            raise ValueError(
                f"Unknown tier type: {self.type!r}. "
                f"Supported: {sorted(_SUPPORTED_TIER_TYPES)}"
            )


@dataclass
class CashConfig:
    """Cash configuration — single source of truth for every tunable.

    Every field below is also exposed as a `CASH_<UPPERCASE>` env var
    (e.g. `CASH_CACHE_DIR`, `CASH_DEBUG=1`) and as a TOML key under
    `[tool.cash]` (project) / `[cash]` (XDG user config). Resolution
    precedence: kwargs > env vars > project TOML > user TOML >
    defaults (see the module docstring above).

    Fields are grouped below by purpose:

    * **Cache location & file backend** — where to store, compress
      vs raw, size cap, flush cadence.
    * **Cost-aware policy** — the smart-persistence rules that
      decide which results are expensive enough to write past RAM.
    * **Observability** — debug toggles.
    * **Backend selection (simple mode)** — pick one backend with
      connection details inline.
    * **Advanced — tier stack** — define an explicit `[memory →
      redis → s3]` (or any) pipeline.
    """

    # --- Cache location & file backend defaults ---
    cache_dir: str = ".cash"
    """Directory for the on-disk cache. Resolved to an absolute
    path at startup so ``os.chdir()`` won't break later writes."""

    compress: bool = False
    """When True, the file backend gzip-compresses each entry.
    Trades CPU for disk space; usually only worth it for plain-text
    blobs (CSV, JSON). Already-compressed formats (Parquet, joblib)
    don't shrink much."""

    max_cache_size: int = 1024 ** 3
    """Maximum total cache size in bytes (default 1 GiB). When the
    file backend exceeds this, it evicts least-recently-accessed
    entries until it fits."""

    max_memory_entries: int | None = None
    """LRU entry cap for the in-memory tier. ``None`` (default)
    means unlimited — eviction is driven by ``psutil`` memory
    pressure instead. Set an integer to force a hard count limit."""

    flush_interval: int = 5
    """Seconds between metadata flushes for the file backend. Lower
    values reduce data loss on crash but increase disk I/O. Set to
    0 to flush after every write (slowest, safest)."""

    # --- Cost-aware caching policy ---
    smart_persistence: bool = True
    """When True (default), the tiered backend decides per-entry
    whether to persist past RAM based on compute time vs storage
    cost. Set False to persist everything (legacy behavior — wastes
    disk for cheap-to-compute values)."""

    smart_persistence_threshold: float = 1.0
    """Compute-time threshold (seconds) above which results
    *always* persist past RAM, regardless of size. The default of
    1 s means anything that took > 1 s to compute is worth saving."""

    min_execution_time_to_cache_seconds: float = 0.01
    """Hard floor (seconds). Compute under this duration is never
    promoted past RAM — disk I/O alone would cost more than
    rerunning. Default 10 ms."""

    min_cache_savings_pct: float = 0.20
    """Required time-savings fraction (0.0 – 1.0) for a tier to be
    considered worthwhile. If a cache hit only saves 20% of the
    compute cost, the entry isn't promoted. Default 0.20."""

    min_cache_fixed_budget_seconds: float = 0.05
    """Per-call I/O budget (seconds). If the predicted serialise +
    write cost exceeds this fraction of the compute saved, skip
    the write. Pairs with ``min_cache_savings_pct``."""

    # --- Observability ---
    debug: bool = False
    """When True, every cache decision logs at INFO level with the
    reason ('HIT', 'MISS — file changed', 'SKIPPED — too cheap').
    Useful for diagnosing 'why didn't this hit?' mysteries. Hot
    field — flippable at runtime via ``cash.configure(debug=True)``."""

    # --- Backend selection (simple mode) ---
    backend: str = "tiered"
    """Backend selector. ``"tiered"`` (default) builds a RAM + disk
    stack from ``cache_dir`` and ``compress``. ``"memory"`` /
    ``"file"`` / ``"sqlite"`` / ``"redis"`` / ``"s3"`` builds a
    single backend of that type from the connection fields below.
    Ignored when ``tiers`` is non-empty."""

    # --- Redis connection details (simple mode) ---
    redis_host: str = "localhost"
    """Redis server hostname. Used when ``backend="redis"`` or a
    tier entry has ``type="redis"``."""

    redis_port: int = 6379
    """Redis server port."""

    redis_db: int = 0
    """Redis logical database number (0 – 15 in default config)."""

    redis_password: str | None = None
    """Redis password if AUTH is enabled. Prefer the env var
    ``CASH_REDIS_PASSWORD`` over committing this to TOML."""

    redis_prefix: str = "cash:"
    """Key prefix for every entry written to Redis. Lets multiple
    apps share one Redis without collisions. Strip with
    ``redis_prefix=""`` if you really want a flat namespace."""

    # --- S3 connection details (simple mode) ---
    s3_bucket: str = ""
    """S3 bucket name. Required when ``backend="s3"``."""

    s3_region: str = ""
    """S3 region (e.g. ``"us-east-1"``). Optional if the bucket's
    region is discoverable from the AWS config / IMDS."""

    s3_prefix: str = "cash/"
    """Object key prefix for every entry written to S3. Same role
    as ``redis_prefix``."""

    # --- Advanced: explicit tier list ---
    tiers: list[TierConfig] = field(default_factory=list)
    """Explicit tier stack — a list of ``TierConfig`` entries that
    build a custom backend pipeline (e.g. ``[memory, redis, s3]``).
    When non-empty, takes precedence over ``backend`` + the
    simple-mode connection fields. Use this for multi-region or
    multi-team setups where one backend isn't enough."""

    # --- Internal: source tracking for `cash --info` ---
    _source: str = "defaults"
    """Internal — records which config layers contributed (e.g.
    ``"kwargs+env+project"``). Surfaced by ``python -m cash info``."""

    def to_dict(self) -> dict[str, Any]:
        """Public field-value mapping (skips internal ``_source``)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if f.name == "tiers":
                out[f.name] = [t.__dict__ for t in val]
            else:
                out[f.name] = val
        return out


# ---------------------------------------------------------------------------
# Internal: type coercion helpers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _coerce(field_type: Any, raw: str) -> Any:
    """Convert a string from env vars/TOML into the field's declared type.

    Raises ``ValueError`` on failure so the caller can skip the value
    and log a warning rather than poison the whole config load.
    """
    # Handle Optional[X] — unwrap to X
    origin = getattr(field_type, "__origin__", None)
    if origin is not None:
        # Optional[X] / Union[X, None]
        args = [a for a in getattr(field_type, "__args__", ()) if a is not type(None)]
        if len(args) == 1:
            field_type = args[0]
    if field_type is bool:
        v = raw.lower()
        if v in _TRUTHY:
            return True
        if v in _FALSY:
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw  # str fallthrough


@functools.lru_cache(maxsize=None)
def _resolved_hints(dataclass_type: type) -> dict[str, Any]:
    """Resolve forward-reference type annotations (PEP 563)."""
    return typing.get_type_hints(dataclass_type)


def _field_type(name: str, dataclass_type: type = CashConfig) -> Any:
    hints = _resolved_hints(dataclass_type)
    return hints.get(name, str)


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

def _load_toml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Recognised sections (in order):
      - ``[tool.cash]`` (pyproject.toml convention)
      - ``[cash]`` (standalone config file)
      - flat top-level (last-resort fallback)

    Returns the merged dict (empty if file missing or unparseable).
    """
    if not path.exists():
        return {}
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.debug("Neither tomllib nor tomli available, skipping TOML config")
            return {}

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:  # noqa: BLE001 — malformed TOML, log and skip
        logger.warning("Error loading config from %s: %s", path, e)
        return {}

    # Try [tool.cash] first (pyproject convention), then [cash], then flat.
    if isinstance(data.get("tool"), dict) and isinstance(data["tool"].get("cash"), dict):
        return dict(data["tool"]["cash"])
    if isinstance(data.get("cash"), dict):
        return dict(data["cash"])
    return data


# ---------------------------------------------------------------------------
# Env var loader
# ---------------------------------------------------------------------------

_TIER_ENV_RE = re.compile(r"^CASH_TIER_(\d+)_(.+)$")


def _load_env_config() -> dict[str, Any]:
    """Read CASH_* env vars into a dict matching CashConfig field names.

    Two flavours:
      - ``CASH_<FIELD>`` → top-level field on CashConfig
      - ``CASH_TIER_<N>_<FIELD>`` → tier list entry at index N

    Tier env vars are collected into a ``tiers`` key as a list of partial
    dicts, indexed by N. The caller merges these into whatever the TOML
    layer produced.
    """
    out: dict[str, Any] = {}
    field_names = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    tier_field_names = {f.name for f in fields(TierConfig)}

    tier_overrides: dict[int, dict[str, Any]] = {}

    for env_key, raw in os.environ.items():
        if not env_key.startswith("CASH_"):
            continue

        # Tier override?
        m = _TIER_ENV_RE.match(env_key)
        if m:
            idx = int(m.group(1))
            field_name = m.group(2).lower()
            if field_name not in tier_field_names:
                logger.warning(
                    "Unknown tier field in env var %s=%r — skipping",
                    env_key, raw,
                )
                continue
            try:
                value = _coerce(_field_type(field_name, TierConfig), raw)
            except ValueError as e:
                logger.warning("Invalid value for %s=%r: %s", env_key, raw, e)
                continue
            tier_overrides.setdefault(idx, {})[field_name] = value
            continue

        # Top-level field
        key = env_key[len("CASH_"):].lower()
        if key not in field_names:
            # Silently ignore unknown CASH_* vars — they may be from
            # other tools that namespace with CASH_ too.
            continue
        try:
            value = _coerce(_field_type(key), raw)
        except ValueError as e:
            logger.warning("Invalid value for %s=%r: %s", env_key, raw, e)
            continue
        out[key] = value

    if tier_overrides:
        # Stable order by index (fill gaps with empty dicts so positional
        # alignment is preserved when later merged with TOML tiers).
        max_idx = max(tier_overrides) + 1
        out["tiers"] = [tier_overrides.get(i, {}) for i in range(max_idx)]

    return out


# ---------------------------------------------------------------------------
# Path resolution for the two TOML sources
# ---------------------------------------------------------------------------

def _default_project_config_path() -> Path | None:
    """Walk upward from cwd to find a ``pyproject.toml`` and return its path.

    The first parent containing one wins. None if we never find one (a
    standalone script with no project structure).
    """
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        candidate = d / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _default_user_config_path() -> Path:
    """The XDG-spec user config location."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "cash" / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "cash" / "config.toml"
    return Path.home() / ".config" / "cash" / "config.toml"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Sentinel to distinguish "use default path" from "explicitly None".
_USE_DEFAULT_PATH: Any = object()


def get_config(
    config_path: str | Path | None = None,
    *,
    user_config_path: Any = _USE_DEFAULT_PATH,
    project_config_path: Any = _USE_DEFAULT_PATH,
    overrides: dict[str, Any] | None = None,
) -> CashConfig:
    """Resolve the merged Cash configuration.

    Args:
        config_path: Legacy parameter — if given, treated as a user-level
            TOML file (overrides ``user_config_path``).
        user_config_path: Path to the user-scoped config (the XDG
            location). Pass ``None`` to skip the user layer entirely;
            omit to use the default location.
        project_config_path: Path to the project-scoped config
            (``pyproject.toml`` with ``[tool.cash]``). Pass ``None`` to
            skip; omit to walk up from cwd.
        overrides: Highest-priority overrides (mirrors what
            ``Cash(**kwargs)`` does internally).

    Returns:
        The merged `CashConfig`.
    """
    sources: list[str] = []

    # Layer 1: defaults from CashConfig dataclass
    merged: dict[str, Any] = {
        f.name: getattr(CashConfig(), f.name)
        for f in fields(CashConfig)
        if not f.name.startswith("_")
    }

    # Layer 2: user TOML
    if user_config_path is _USE_DEFAULT_PATH:
        user_path = _default_user_config_path()
    else:
        user_path = user_config_path
    if user_path is not None:
        user_data = _load_toml_config(Path(user_path))
        if user_data:
            _merge(merged, user_data)
            sources.append(f"user:{user_path}")

    # Layer 2b: legacy config_path argument (treated as user override)
    if config_path is not None:
        legacy_data = _load_toml_config(Path(config_path))
        if legacy_data:
            _merge(merged, legacy_data)
            sources.append(f"file:{config_path}")

    # Layer 3: project TOML
    if project_config_path is _USE_DEFAULT_PATH:
        project_path = _default_project_config_path()
    else:
        project_path = project_config_path
    if project_path is not None:
        project_data = _load_toml_config(Path(project_path))
        if project_data:
            _merge(merged, project_data)
            sources.append(f"project:{project_path}")

    # Layer 4: env vars
    env_data = _load_env_config()
    if env_data:
        _merge(merged, env_data)
        sources.append("env")

    # Layer 5: explicit overrides (kwargs)
    if overrides:
        _merge(merged, overrides)
        sources.append("kwargs")

    # Materialise the dict into a CashConfig.
    return _build_config(merged, source=",".join(sources) if sources else "defaults")


def _merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Apply *update* on top of *base* in place.

    Special-case ``tiers``: list of partial dicts in *update* is merged
    element-wise on top of *base*'s tier list so env-var partial
    overrides combine with TOML-declared tier configs.
    """
    for k, v in update.items():
        if k == "tiers" and isinstance(v, list):
            base_tiers = base.get("tiers", []) or []
            base_tiers = [dict(t) if isinstance(t, dict) else t for t in base_tiers]
            # Extend to length of update if needed.
            while len(base_tiers) < len(v):
                base_tiers.append({})
            for i, partial in enumerate(v):
                if isinstance(partial, dict):
                    base_tiers[i] = {**base_tiers[i], **partial}
                elif isinstance(partial, TierConfig):
                    base_tiers[i] = partial.__dict__
                else:
                    base_tiers[i] = partial
            base["tiers"] = base_tiers
        else:
            base[k] = v


def _build_config(merged: dict[str, Any], source: str) -> CashConfig:
    """Materialise the merged dict into a typed CashConfig instance."""
    cfg = CashConfig()
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    for key, value in merged.items():
        if key == "tiers":
            tiers: list[TierConfig] = []
            for entry in value or []:
                if isinstance(entry, TierConfig):
                    tiers.append(entry)
                elif isinstance(entry, dict) and entry.get("type"):
                    # Filter to known TierConfig fields so unrelated keys
                    # in the TOML don't blow up __init__.
                    tier_field_names = {f.name for f in fields(TierConfig)}
                    clean = {k: v for k, v in entry.items() if k in tier_field_names}
                    tiers.append(TierConfig(**clean))
                # else: malformed entry — silently skip
            cfg.tiers = tiers
        elif key in valid:
            setattr(cfg, key, value)
    cfg._source = source
    return cfg


# ---------------------------------------------------------------------------
# Default config file template
# ---------------------------------------------------------------------------

def create_default_config(path: str | None = None) -> str:
    """Write a documented default config TOML to *path* (or to the user
    config location if *path* is None) and return the path written."""
    if path is None:
        path = str(_default_user_config_path())

    content = '''# Cash configuration — see https://github.com/your-repo/cash
#
# Resolution priority (highest wins):
#   1. Cash(**kwargs)
#   2. CASH_* environment variables
#   3. ./pyproject.toml [tool.cash]
#   4. This file
#   5. Built-in defaults

[cash]
# Where the disk cache lives. Add this dir to .gitignore.
cache_dir = ".cash"

# Verbose logging for cache decisions.
debug = false

# gzip data files on disk.
compress = false

# Max disk cache size (bytes). LRU eviction kicks in above this.
max_cache_size = 1073741824

# Smart persistence — only promote past RAM when the compute was slow
# enough to be worth disk/network I/O for.
smart_persistence = true
smart_persistence_threshold = 1.0

# Backend selection. One of: tiered (default, builds [memory, file]),
# memory, file, sqlite, redis, s3.
# backend = "tiered"

# Redis (used when backend = "redis" or a redis tier is declared).
# redis_host = "localhost"
# redis_port = 6379
# redis_prefix = "cash:"

# S3 (used when backend = "s3" or an s3 tier is declared).
# s3_bucket = ""
# s3_region = ""
# s3_prefix = "cash/"

# Advanced: declare an explicit tier stack instead of the simple
# `backend = "..."` form. Each tier is one [[cash.tiers]] table.
# [[cash.tiers]]
# type = "memory"
# max_entries = 10000
#
# [[cash.tiers]]
# type = "redis"
# host = "redis.internal"
# port = 6379
'''

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)
