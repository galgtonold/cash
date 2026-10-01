"""The settings themselves: `CashConfig`, `TierConfig`, and what a value may be.

Each field's type, default and documentation, the choices and ranges a value
must fall in, and the coercion of a string (from a file or the environment)
into the field's type. Where values come from is `cash.config.resolve`.
"""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass, field, fields
from typing import Any

from ..units import parse_size
from .notices import config_notice

__all__ = [
    "SIZE_FIELDS",
    "TIER_TYPES",
    "CashConfig",
    "TierConfig",
    "check_choice",
    "coerce",
    "config_provenance",
    "declared_type",
    "validate_value",
]


#: The backend types a tier can be (``cash.backends.factory``).
TIER_TYPES = frozenset({"memory", "file", "sqlite", "redis", "s3"})

#: Settings whose value must be one of a fixed set (see ``validate_value``).
#: ``backend = "tiered"`` is the RAM + disk stack; any other is one tier.
_NAMED_CHOICES = {"backend": TIER_TYPES | {"tiered"}, "type": TIER_TYPES}


#: The smallest value each numeric setting can take (tier keys included), and
#: the largest where there is one. A size or count of 0 or less would cap
#: everything out -- nothing reached disk, and the messages then said "up to
#: -1 B" -- and a negative interval or timeout means nothing. ``None`` stays
#: the way to say "no cap".
_RANGES: dict[str, tuple[float, float | None]] = {
    "max_cache_size": (1, None),
    "max_size_bytes": (1, None),
    "max_memory_entries": (1, None),
    "max_entries": (1, None),
    "flush_interval": (0, None),
    "shutdown_write_timeout": (0, None),
    "default_ttl": (0, None),
    "min_execution_time_to_cache_seconds": (0, None),
    "call_cost_floor_seconds": (0, None),
    "loop_split_max_iter_seconds": (0, None),
    "loop_split_min_remaining_seconds": (0, None),
    "min_cache_savings_pct": (0, 1),
    "min_cache_fixed_budget_seconds": (0, None),
    "remote_revalidate_max_age_seconds": (0, None),
    "redis_port": (0, 65535),
    "port": (0, 65535),
    "redis_db": (0, None),
    "db": (0, None),
}


#: Settings that name a path, where ``""`` would mean the current directory:
#: the cache landed among the user's source files.
_PATH_SETTINGS = frozenset({"cache_dir", "db_path"})


def check_choice(name: str, value: Any) -> None:
    """``ValueError`` when *name* must be one of a fixed set and *value* is
    not, is a number outside the range *name* allows (`_RANGES`), or is an
    empty path."""
    if name in _PATH_SETTINGS and isinstance(value, str) and not value.strip():
        raise ValueError(f"{name}={value!r}: an empty path (leave it unset for the default)")
    allowed = _NAMED_CHOICES.get(name)
    if allowed is not None and isinstance(value, str) and value not in allowed:
        raise ValueError(f"{name}={value!r}: not one of {', '.join(sorted(allowed))}")
    bounds = _RANGES.get(name)
    if bounds is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    low, high = bounds
    # `not (x >= low)` also refuses NaN.
    if not (value >= low) or (high is not None and value > high):
        span = f"between {low} and {high}" if high is not None else f"{low} or more"
        unset = " (leave it unset for no cap)" if low == 1 else ""
        raise ValueError(f"{name}={value!r}: must be {span}{unset}")


@dataclass
class TierConfig:
    """One entry of the ``tiers`` setting: a backend in a tier stack.

    Only the keys for the tier's ``type`` are read. A key left unset
    comes from the top-level setting of the same meaning (``cache_dir``,
    ``max_cache_size``, ``redis_host``, ...). Set tiers in a config file
    (``[[tool.cash.tiers]]``) or with ``CASH_TIER_<N>_<KEY>`` variables.

    Attributes:
        type: Backend type: ``"memory"``, ``"file"``, ``"sqlite"``,
            ``"redis"`` or ``"s3"``.
        max_size_bytes: memory, file and sqlite tiers: size cap in bytes
            (or a size such as ``"2GB"``). In a tiered stack, a value
            bigger than this skips the tier. Unset, a memory tier is sized
            to the machine and a file or sqlite tier uses
            ``max_cache_size``.
        default_ttl: file and sqlite tiers: TTL in seconds for entries
            stored without one. A ``ttl=`` on the decorator takes
            precedence.
        max_entries: memory tier: entry-count cap. Defaults to
            ``max_memory_entries``.
        cache_dir: file and sqlite tiers: the cache directory. Defaults to
            ``cache_dir``.
        compress: file tier: gzip each entry. Defaults to ``compress``.
        flush_interval: file tier: seconds between metadata flushes.
            Defaults to ``flush_interval``.
        db_path: sqlite tier: path to the database file. Defaults to a file
            inside ``cache_dir``.
        wal_mode: sqlite tier: accepted but not used; a tier built from
            configuration always uses WAL journal mode.
        host: redis tier: server hostname. Defaults to ``redis_host``.
        port: redis tier: server port. Defaults to ``redis_port``.
        db: redis tier: database number. Defaults to ``redis_db``.
        password: redis tier: password. Defaults to ``redis_password``.
        prefix: redis and s3 tiers: key prefix. Defaults to
            ``redis_prefix`` or ``s3_prefix``.
        bucket: s3 tier: bucket name. Defaults to ``s3_bucket``.
        region: s3 tier: AWS region, such as ``"us-east-1"``. Defaults to
            ``s3_region``.
    """

    # The Attributes section above is the one description of each key: the
    # API reference renders it, and the config template reads it
    # (_tier_key_docs).
    type: str

    # memory / file / sqlite shared:
    max_size_bytes: int | None = None
    default_ttl: int | None = None

    # memory:
    max_entries: int | None = None

    # file:
    cache_dir: str | None = None
    compress: bool | None = None
    flush_interval: int | None = None

    # sqlite:
    db_path: str | None = None
    wal_mode: bool | None = None

    # redis:
    host: str | None = None
    port: int | None = None
    db: int | None = None
    password: str | None = None
    prefix: str | None = None

    # s3:
    bucket: str | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        if self.type not in TIER_TYPES:
            raise ValueError(f"Unknown tier type: {self.type!r}. Supported: {sorted(TIER_TYPES)}")
        # The factory is what reads a tier's keys, so it says which ones a
        # type uses. Imported here: the factory imports this module.
        from ..backends.factory import TIER_FIELDS

        unused = sorted(
            f.name
            for f in fields(self)
            if f.name != "type" and getattr(self, f.name) is not None and f.name not in TIER_FIELDS[self.type]
        )
        if unused:
            config_notice(
                "CONFIG-INVALID",
                f"a {self.type} tier sets {', '.join(unused)}, which a {self.type} tier does not use, "
                f"so {'it does' if len(unused) == 1 else 'they do'} nothing.",
                f"remove {'it' if len(unused) == 1 else 'them'}; a {self.type} tier is built from "
                f"{', '.join(sorted(TIER_FIELDS[self.type]))}.",
            )


@dataclass
class CashConfig:
    """Every cash setting, as resolved from all configuration layers.

    Precedence, highest first: code (``Cash(...)``, ``cash.configure``),
    ``CASH_<FIELD>`` environment variables, a file named with
    ``Cash(config_path=...)``, ``[tool.cash]`` in the project's
    ``pyproject.toml``, the user config file, then these defaults.
    Every field, its variable and which path it affects are listed in
    the Configuration guide.
    """

    # --- Cache location & file backend defaults ---
    cache_dir: str = ".cash"
    """Directory for the disk cache. A relative path is resolved once, so a
    later ``os.chdir()`` does not move the cache."""

    compress: bool = False
    """gzip each entry on disk. Trades CPU for space; worth it mainly for
    text-like values."""

    max_cache_size: int | None = None
    """Disk cache cap in bytes, or a size such as ``"5GB"``. ``None`` sizes it
    to the machine: a quarter of the room on the cache volume (free space
    plus what the cache holds), between 8 GiB and 100 GiB. The RAM tier has
    its own cap either way. At the cap, the entries cheapest to recompute
    per byte, weighted by their hits, are evicted first (GDSF)."""

    max_memory_entries: int | None = None
    """Entry-count cap for the RAM tier, evicting the least recently used.
    ``None`` means no count limit; the RAM tier is still capped in bytes."""

    flush_interval: int = 5
    """Seconds between the disk tier's metadata flushes. ``0`` flushes after
    every write."""

    shutdown_write_timeout: float = 60.0
    """Seconds a finishing process waits for its background writes before
    exiting without them, warning ``CACHE-WRITE-ABANDONED``. Bounded so a
    write that cannot finish never keeps a finished process alive."""

    # --- Cost-aware caching policy ---
    persist_all: bool = False
    """Notebook only: cache every statement, skipping the cost floors, as if
    each had ``# @cash:persist``. ``%cash_persist on`` does the same for a
    session."""

    summary: bool = False
    """At exit, print a per-function hit/miss table to stderr, with why each
    function missed. ``CASH_SUMMARY=1 python script.py`` needs no code
    change."""

    check_cash_keys: bool = True
    """Check each ``__cash_key__`` against the data it stands for: the first
    time an object is keyed by it in a process, its content is read on a
    background thread and compared with what the same key held before.
    Different data behind one key warns ``KEY-STALE-CASH-KEY``."""

    disable: bool = False
    """Run every ``@cash.cache`` call uncached (no key, lookup or store), and
    make ``%cash_on`` decline. ``CASH_DISABLE=1 pytest`` checks that tests
    pass without the cache. Read on every call."""

    min_execution_time_to_cache_seconds: float = 0.01
    """Notebook only: a statement faster than this (seconds) is not cached
    at all. Whether a cached value is written to disk is decided by the
    cost model."""

    call_cost_floor_seconds: float = 0.003
    """Notebook only: a call inside a statement is cached only when it runs
    at least this long (seconds)."""

    loop_split_max_iter_seconds: float = 0.006
    """Notebook only: a loop whose iterations each take less than this
    (seconds) may be cached as one unit instead of per call."""

    loop_split_min_remaining_seconds: float = 0.1
    """Notebook only: a loop is cached as one unit only when at least this
    much work (seconds) remains in it."""

    min_cache_savings_pct: float = 0.20
    """Notebook only: fraction (0.0 to 1.0) of the compute time a predicted
    restore must save for a value to be written to disk."""

    min_cache_fixed_budget_seconds: float = 0.05
    """Notebook only: a predicted restore time below this (seconds) is always
    acceptable, however short the compute. Above it,
    ``min_cache_savings_pct`` decides."""

    remote_revalidate_max_age_seconds: float = 0.0
    """Seconds a remote file's state may be reused before the store is asked
    again. ``0`` checks on every hit. A higher value lets a change go unseen
    for that long."""

    # --- Observability ---
    debug: bool = False
    """Log every cache decision with its reason, including one line per
    decorated call. ``%cash_debug on`` sets it in a notebook."""

    verbose: bool = False
    """Log one line per decorated call (a hit, or a miss and why) to the
    ``cash.calls`` logger, without the other debug records. ``debug``
    implies it."""

    analytics: bool = True
    """Notebook only: record each statement's hit, miss and timing in
    ``analytics.db`` under the per-user cache root, for the
    ``cash.show_stats()`` dashboard. Off, no file is created."""

    # --- Backend selection ---
    backend: str = "tiered"
    """``"tiered"`` is a RAM tier in front of a disk tier. ``"memory"``,
    ``"file"``, ``"sqlite"``, ``"redis"`` or ``"s3"`` is that one backend.
    Ignored when ``tiers`` is set."""

    # --- Redis connection details (simple mode) ---
    redis_host: str = "localhost"
    """Redis server hostname. Used when ``backend="redis"`` or a
    tier entry has ``type="redis"``."""

    redis_port: int = 6379
    """Redis server port."""

    redis_db: int = 0
    """Redis database number."""

    redis_password: str | None = None
    """Redis password if AUTH is enabled. Prefer the env var
    ``CASH_REDIS_PASSWORD`` over committing this to TOML."""

    redis_prefix: str = "cash:"
    """Key prefix for every entry written to Redis, so several apps can
    share one server."""

    # --- S3 connection details (simple mode) ---
    s3_bucket: str = ""
    """S3 bucket name. Required when ``backend="s3"``."""

    s3_region: str = ""
    """S3 region, such as ``"us-east-1"``. Empty uses your AWS
    configuration."""

    s3_prefix: str = "cash/"
    """Object key prefix for every entry written to S3."""

    # --- Advanced: explicit tier list ---
    tiers: list[TierConfig] = field(default_factory=list)
    """Your own tier stack, fastest first, replacing the one ``backend``
    names; each entry is a `TierConfig`. Set it in a config file or with
    ``CASH_TIER_<N>_<KEY>`` variables; there is no ``CASH_TIERS``."""

    # --- Internal: source tracking for `cash --info` ---
    _source: str = "defaults"
    """Internal — records which config layers contributed (e.g.
    ``"kwargs+env+project"``). Surfaced by ``python -m cash info``."""

    _origins: dict[str, str] = field(default_factory=dict)
    """Internal — for each setting a layer set, the layer that won (a file
    path, ``CASH_<NAME>``, ``Cash(...)``). ``cash info`` lists them."""

    _files: list[tuple[str, str, str]] = field(default_factory=list)
    """Internal — every config file looked for: ``(layer, path, outcome)``."""

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


def config_provenance(cfg: CashConfig) -> tuple[str, dict[str, str], list[tuple[str, str, str]]]:
    """Where *cfg* came from, for ``cash info``: its ``_source``, ``_origins`` and ``_files``."""
    return cfg._source, cfg._origins, cfg._files


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


#: Fields that hold a number of BYTES, and so also accept ``"2GB"`` /
#: ``"512MiB"``. People write sizes that way (``CASH_MAX_CACHE_SIZE=500MB``),
#: and a bare integer of bytes is the one spelling nobody reads correctly at a
#: glance.
SIZE_FIELDS = frozenset({"max_cache_size", "max_size_bytes"})


def coerce(field_type: Any, raw: str, name: str | None = None) -> Any:
    """Convert a string from env vars/TOML into the field's declared type.

    Raises ``ValueError`` on failure so the caller can skip the value
    and log a warning rather than poison the whole config load.
    """
    if name in SIZE_FIELDS:
        return parse_size(raw)
    # Handle Optional[X] / X | None — unwrap to the single non-None member.
    # typing.get_args normalises both typing.Union and PEP 604 (``int | None``)
    # unions across Python versions. The previous ``__origin__`` check missed
    # PEP 604 unions on 3.10–3.13 (types.UnionType has no ``__origin__`` there),
    # so int/float/bool fields stayed uncoerced strings.
    union_args = [a for a in typing.get_args(field_type) if a is not type(None)]
    if len(union_args) == 1:
        field_type = union_args[0]
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


def declared_type(name: str, dataclass_type: type = CashConfig) -> Any:
    hints = _resolved_hints(dataclass_type)
    return hints.get(name, str)


def validate_value(name: str, value: Any, dataclass_type: type = CashConfig) -> Any:
    """*value* as field *name*'s declared type, or ``ValueError`` saying why.

    Strings are coerced the way environment variables are (``"true"``,
    ``"8"``, ``"2GB"`` for a size); anything else must already BE the type.
    Nothing checked this for a constructor argument or a TOML value, so
    ``Cash(max_cache_size="2GB")`` was stored as a string and every disk
    write then failed comparing it with an int -- the cache silently kept
    nothing on disk for the rest of the process.
    """
    checked = _typed_value(name, value, dataclass_type)
    check_choice(name, checked)
    return checked


def _typed_value(name: str, value: Any, dataclass_type: type) -> Any:
    """`validate_value` before the choice and range checks."""
    field_type = declared_type(name, dataclass_type)
    args = typing.get_args(field_type)
    optional = type(None) in args
    base = next((a for a in args if a is not type(None)), field_type) if args else field_type
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} cannot be None")
    if isinstance(value, str) and base is not str:
        try:
            return coerce(field_type, value, name)
        except ValueError as exc:
            raise ValueError(f"{name}={value!r}: {exc}") from None
    if base is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)  # `debug=1` is ordinary code
    elif base is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif base is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif base is str:
        if isinstance(value, str):
            # A named choice is checked by the caller, so a file or env layer
            # reports it (CONFIG-INVALID) and falls back, as every other bad
            # value does, instead of the backend factory raising out of
            # `import cash` and `cash info`.
            return value
    else:
        return value  # lists, nested configs: checked elsewhere
    expected = getattr(base, "__name__", str(base))
    hint = " (or a size string such as '2GB')" if name in SIZE_FIELDS else ""
    raise ValueError(f"{name}={value!r} is a {type(value).__name__}; expected {expected}{hint}")
