"""Cash configuration system.

Resolution precedence (highest priority wins):

    1. Explicit constructor kwargs       (Cash(redis_host="..."))
    2. Environment variables             (CASH_* and CASH_TIER_<N>_*)
    3. A file named in code              (Cash(config_path="..."))
    4. Project config                    (./pyproject.toml [tool.cash])
    5. User config                       (~/.config/cash/config.toml or
                                          %APPDATA%/cash/config.toml on Windows)
    6. CashConfig dataclass defaults

Every field on ``CashConfig`` is settable through every layer. The
``tiers`` list is settable as a whole from TOML and field-by-field from
env vars (``CASH_TIER_0_TYPE=redis``, ``CASH_TIER_0_HOST=...``).
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import functools
import inspect
import logging
import os
import re
import sys
import textwrap
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ._location import (
    default_project_config_path,
    default_user_config_path,
    installed_entry_point_cache_dir,
    project_anchor,
)
from .diagnostics import warn_diagnostic
from .exceptions import CashCacheIneffectiveWarning
from .tracking.tracker_context import untracked

logger = logging.getLogger(__name__)

__all__ = [
    "CashConfig",
    "TierConfig",
    "get_config",
    "create_default_config",
]


#: The backend types a tier can be (``cash.backends.factory``).
_SUPPORTED_TIER_TYPES = frozenset({"memory", "file", "sqlite", "redis", "s3"})

#: Settings whose value must be one of a fixed set (see ``validate_value``).
#: ``backend = "tiered"`` is the RAM + disk stack; any other is one tier.
_NAMED_CHOICES = {"backend": _SUPPORTED_TIER_TYPES | {"tiered"}, "type": _SUPPORTED_TIER_TYPES}


#: The `TierConfig` fields each tier type is built from
#: (``backends.factory._settings``). Any other field set on a tier does
#: nothing, and is reported (CONFIG-INVALID) rather than silently ignored.
_TIER_FIELDS: dict[str, frozenset[str]] = {
    "memory": frozenset({"max_size_bytes", "max_entries"}),
    "file": frozenset({"max_size_bytes", "default_ttl", "cache_dir", "compress", "flush_interval"}),
    "sqlite": frozenset({"max_size_bytes", "default_ttl", "cache_dir", "db_path", "wal_mode"}),
    "redis": frozenset({"host", "port", "db", "password", "prefix"}),
    "s3": frozenset({"bucket", "region", "prefix"}),
}


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
    "file_hash_full_max_bytes": (0, None),
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


def _check_choice(name: str, value: Any) -> None:
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
        if self.type not in _SUPPORTED_TIER_TYPES:
            raise ValueError(f"Unknown tier type: {self.type!r}. Supported: {sorted(_SUPPORTED_TIER_TYPES)}")
        unused = sorted(
            f.name
            for f in fields(self)
            if f.name != "type" and getattr(self, f.name) is not None and f.name not in _TIER_FIELDS[self.type]
        )
        if unused:
            _config_notice(
                "CONFIG-INVALID",
                f"a {self.type} tier sets {', '.join(unused)}, which a {self.type} tier does not use, "
                f"so {'it does' if len(unused) == 1 else 'they do'} nothing.",
                f"remove {'it' if len(unused) == 1 else 'them'}; a {self.type} tier is built from "
                f"{', '.join(sorted(_TIER_FIELDS[self.type]))}.",
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

    file_hash_full_max_bytes: int = 256 * 1024 * 1024
    """Largest tracked file hashed in full to check freshness.

    Larger files hash three sampled regions plus their size and timestamps,
    which misses a same-size edit outside those regions that keeps the
    modification time. Only the first check of a file in a process pays for
    the hash; later checks of an unchanged file cost a ``stat``."""

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


# ---------------------------------------------------------------------------
# Internal: type coercion helpers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


#: Fields that hold a number of BYTES, and so also accept ``"2GB"`` /
#: ``"512MiB"``. People write sizes that way (``CASH_MAX_CACHE_SIZE=500MB``),
#: and a bare integer of bytes is the one spelling nobody reads correctly at a
#: glance.
SIZE_FIELDS = frozenset({"max_cache_size", "file_hash_full_max_bytes", "max_size_bytes"})

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]i?b|b)?\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,  # SI
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,  # binary
}


def parse_size(raw: str) -> int:
    """``"2GB"`` -> 2_000_000_000, ``"512MiB"`` -> 536_870_912, ``"1024"`` -> 1024.

    KB/MB/GB/TB are powers of 1000 and KiB/MiB/GiB/TiB powers of 1024, as the
    units say. Case-insensitive. Raises ``ValueError`` on anything else.
    """
    m = _SIZE_RE.match(raw)
    if m is None:
        raise ValueError(f"not a size: {raw!r} (write bytes, or e.g. '2GB' / '512MiB')")
    number, unit = m.group(1), (m.group(2) or "b").lower()
    return int(float(number) * _SIZE_UNITS[unit])


def human_bytes(n: int | None) -> str:
    """Format a byte count with a sensible unit (e.g. ``7.8 KiB``, ``8.0 GiB``).

    Used in the cap warnings so a message never reads ``~0 MiB`` for a small
    cap or an unwieldy raw byte count for a large one.
    """
    size = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # unreachable; keeps type-checkers happy


def format_size(n: int) -> str:
    """*n* bytes the way `parse_size` reads it back: ``2000000000`` -> ``"2 GB"``.

    A size written ``"2GB"`` must not come back as ``1.9 GB``, which reads as
    mis-parsed: a whole number of a decimal or a binary unit is shown in that
    unit; anything else in binary units, labelled as such.
    """
    for name in ("TB", "TiB", "GB", "GiB", "MB", "MiB", "KB", "KiB"):
        unit = _SIZE_UNITS[name.lower()]
        if n >= unit and (n * 10) % unit == 0:
            return f"{n / unit:g} {name}"
    return human_bytes(n)


def _coerce(field_type: Any, raw: str, name: str | None = None) -> Any:
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


def _field_type(name: str, dataclass_type: type = CashConfig) -> Any:
    hints = _resolved_hints(dataclass_type)
    return hints.get(name, str)


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

#: ``(code, message)`` pairs already said in this process: the one dedup for
#: every config notice. A config is resolved more than once -- the module
#: default, then each ``Cash(...)`` -- and must not repeat its complaints.
_CONFIG_NOTICES: set[tuple[str, str]] = set()


def _config_notice(code: str, what: str, fix: str) -> None:
    """Warn, once per process, about a setting cash could not use."""
    if (code, what) in _CONFIG_NOTICES:
        return
    _CONFIG_NOTICES.add((code, what))
    try:
        warn_diagnostic(CashCacheIneffectiveWarning, code, what, fix)
    except Exception:  # a notice must never break a config load
        logger.debug("Could not emit %s", code, exc_info=True)


def _did_you_mean(key: str, valid: Any) -> str:
    match = difflib.get_close_matches(key, sorted(valid), n=1, cutoff=0.6)
    return f" Did you mean `{match[0]}`?" if match else ""


def _warn_toml_unreadable(path: Path) -> None:
    """Say that a config file was found and is being ignored."""
    _config_notice(
        "CONFIG-TOML-UNREADABLE",
        f"cash found {path} but cannot read it: this is Python "
        f"{sys.version_info.major}.{sys.version_info.minor}, whose standard "
        f"library has no TOML parser, and `tomli` is not installed. Every "
        f"setting in that file is being ignored, including cache_dir -- so "
        f"cash is running on defaults that the file was written to change.",
        "pip install tomli -- or cash-lib[toml], which `cash-lib[all]` "
        "includes (cash keeps no required dependencies, so a bare install "
        "cannot pull one in for you) -- or set the values through CASH_* "
        "environment variables instead, or run on Python 3.11+ where the "
        "parser is in the standard library.",
    )


_CASH_SECTION_RE = re.compile(r"^\s*(\[\s*(tool\s*\.\s*)?cash\s*[\].]|tool\s*\.\s*cash\s*\.)", re.MULTILINE)


def _may_hold_cash_settings(path: Path) -> bool:
    """Would a parser find cash settings in *path*? Answered without one.

    A ``[tool.cash]`` / ``[cash]`` table (or a ``tool.cash.`` dotted key)
    anywhere says yes. A ``pyproject.toml`` without one says no: it belongs to
    the project, not to cash. Any other file was named as a cash config file,
    so anything but comments counts. Unreadable says yes -- the notice errs
    toward being given.
    """
    try:
        # -sig: a byte-order mark would hide a `[tool.cash]` on line 1.
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return True
    if _CASH_SECTION_RE.search(text):
        return True
    if path.name == "pyproject.toml":
        return False
    return any(line.strip() and not line.lstrip().startswith("#") for line in text.splitlines())


#: What `_load_toml_layer` found. A ``[tool.cash]`` or ``[cash]`` table holds
#: cash's settings, so a key in it that is not one is a mistake worth naming.
TOML_SECTION = "section"
TOML_NOT_CASH = "no [tool.cash] or [cash] table"
TOML_MISSING = "not found"
TOML_UNREADABLE = "not read"


def _load_toml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Settings are read from ``[tool.cash]`` (the pyproject.toml convention)
    or ``[cash]``, nowhere else.

    Returns the merged dict (empty if file missing or unparseable).
    """
    return _load_toml_layer(path)[0]


def _load_toml_layer(path: Path) -> tuple[dict[str, Any], str]:
    """`_load_toml_config`, plus which of the ``TOML_*`` outcomes it was."""
    if not path.exists():
        return {}, TOML_MISSING
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # ``tomllib`` is 3.11+, and cash has no required dependencies by
            # design, so on 3.10 without ``tomli`` there is nothing that can
            # read this file, and every setting in it would be silently
            # ignored. Said only when the file may hold cash settings: nearly
            # every project has a pyproject.toml, and a notice about one with
            # no [tool.cash] would teach people to filter it out.
            if _may_hold_cash_settings(path):
                _warn_toml_unreadable(path)
            return {}, TOML_UNREADABLE

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:  # noqa: BLE001 — malformed TOML, say so and skip
        if _may_hold_cash_settings(path):
            _warn_toml_malformed(path, e)
        return {}, TOML_UNREADABLE

    if isinstance(data.get("tool"), dict) and isinstance(data["tool"].get("cash"), dict):
        return dict(data["tool"]["cash"]), TOML_SECTION
    if isinstance(data.get("cash"), dict):
        return dict(data["cash"]), TOML_SECTION
    if data and path.name != "pyproject.toml":
        # A file named as cash's config, with its keys where cash does not
        # read them: say so, rather than run on defaults without a word.
        _config_notice(
            "CONFIG-INVALID",
            f"{path} has no [cash] table, so cash reads none of its settings "
            f"({', '.join(sorted(data)[:5])}{', ...' if len(data) > 5 else ''}).",
            "put the settings under a [cash] table (or [tool.cash] in a pyproject.toml).",
        )
    return {}, TOML_NOT_CASH


def _warn_toml_malformed(path: Path, exc: Exception) -> None:
    """A config file that does not parse: say where, and name a BOM.

    A UTF-8 byte-order mark is invisible in every editor and is what Windows
    PowerShell 5.1 writes for ``-Encoding utf8``; TOML forbids it, and the
    parser's own complaint -- an invalid statement at line 1, column 1 --
    points at a character nobody can see.
    """
    try:
        bom = path.read_bytes()[:3] == b"\xef\xbb\xbf"
    except OSError:
        bom = False
    if bom:
        what = (
            f"cash cannot read {path}: the file starts with a UTF-8 byte-order "
            f"mark (BOM), which TOML does not allow ({exc}). Every setting in "
            f"it is being ignored."
        )
        fix = (
            'save it as UTF-8 without a BOM. In an editor that is "UTF-8" '
            'rather than "UTF-8 with BOM"; Windows PowerShell 5.1 writes a '
            "BOM for `-Encoding utf8`, and PowerShell 7 does not."
        )
    else:
        what = f"cash cannot read {path}: it is not valid TOML ({exc}). Every setting in it is being ignored."
        fix = "correct the file at the line and column named."
    _config_notice("CONFIG-INVALID", what, fix)


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
        if not raw.strip():
            # Set but empty -- `CASH_CACHE_DIR=${X:-}` in CI, `docker -e
            # CASH_CACHE_DIR=` -- is how a shell says "not set".
            continue

        # Tier override?
        m = _TIER_ENV_RE.match(env_key)
        if m:
            idx = int(m.group(1))
            field_name = m.group(2).lower()
            if field_name not in tier_field_names:
                _config_notice(
                    "CONFIG-UNKNOWN-KEY",
                    f"the environment sets {env_key}, and `{field_name}` is not a "
                    f"tier setting, so it does nothing."
                    f"{_did_you_mean(field_name, tier_field_names)}",
                    "rename or unset it; the tier settings are listed under Configuration > Tiers in the docs.",
                )
                continue
            try:
                value = _coerce(_field_type(field_name, TierConfig), raw, field_name)
                _check_choice(field_name, value)
            except ValueError as e:
                _config_notice(
                    "CONFIG-INVALID",
                    f"the environment sets {env_key}={raw!r}, which cash cannot use: {e}. It is being ignored.",
                    f"correct or unset {env_key}.",
                )
                continue
            tier_overrides.setdefault(idx, {})[field_name] = value
            continue

        # Top-level field
        key = env_key[len("CASH_") :].lower()
        if key not in field_names:
            # Silently ignore unknown CASH_* vars — they may be from
            # other tools that namespace with CASH_ too.
            continue
        try:
            value = _coerce(_field_type(key), raw, key)
            _check_choice(key, value)
        except ValueError as e:
            _config_notice(
                "CONFIG-INVALID",
                f"the environment sets {env_key}={raw!r}, which cash cannot use: {e}. It is being ignored.",
                f"correct or unset {env_key}.",
            )
            continue
        out[key] = value

    if tier_overrides:
        # Stable order by index (fill gaps with empty dicts so positional
        # alignment is preserved when later merged with TOML tiers).
        max_idx = max(tier_overrides) + 1
        out["tiers"] = [tier_overrides.get(i, {}) for i in range(max_idx)]

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Sentinel to distinguish "use default path" from "explicitly None".
_USE_DEFAULT_PATH: Any = object()

#: ``cache_dir`` origin meaning "leave a relative path alone -- it is relative to
#: the caller's cwd, like every other path they type".
_CALLER_RELATIVE: Any = object()


def _anchor_cache_dir(cache_dir: Any, origin: Path | object) -> Any:
    """Resolve a relative *cache_dir* against whatever set it.

    Three answers, by who wrote the value:

    * **the default** (nobody wrote it) -- relative to the project anchor, so
      ``.cash`` means "this project's cache" rather than "a cache wherever this
      job was launched from".
    * **a config file** -- relative to that file's directory, the ordinary rule
      for paths in config files, so ``[tool.cash] cache_dir`` does not move
      with the cwd.
    * **an env var or a kwarg** -- relative to the current directory, as every
      other command-line path is, and made absolute NOW. The user typed it in
      the shell or in the code that is running now. Left relative, it moved
      with every later ``os.chdir()``: the backend was built in one directory
      and worker processes (handed the absolute path) used another, and a
      ``configure()`` after a chdir rebuilt the cache somewhere new.

    An absolute path is returned untouched in all three cases.
    """
    if not isinstance(cache_dir, str) or not cache_dir:
        return cache_dir
    # Before anchoring: `~/crunch-cache` in a shipped config file must not
    # become a directory named `~` beside that file, inside site-packages.
    expanded = os.path.expanduser(cache_dir)
    if expanded != cache_dir:
        # Windows expands `~/b` to `C:\Users\me/b`; normalise it to one
        # separator style so it compares equal to the same path built by hand.
        cache_dir = os.path.normpath(expanded)
    if os.path.isabs(cache_dir):
        return cache_dir
    if origin is _CALLER_RELATIVE:
        return os.path.abspath(cache_dir)
    if not isinstance(origin, Path):
        return cache_dir
    return os.path.normpath(str(origin / cache_dir))


#: The tier keys that are paths, resolved the way ``cache_dir`` is.
_TIER_PATH_KEYS = ("cache_dir", "db_path")


def _anchor_tier_paths(tiers: Any, origin: Path | object) -> Any:
    """*tiers* with each tier's path keys resolved against *origin*, the
    layer that set them (`_anchor_cache_dir`). A tier path left relative was
    resolved against whatever the cwd was when the backend was built."""
    if not isinstance(tiers, list):
        return tiers
    out = []
    for tier in tiers:
        if isinstance(tier, dict):
            tier = {k: (_anchor_cache_dir(v, origin) if k in _TIER_PATH_KEYS else v) for k, v in tier.items()}
        elif isinstance(tier, TierConfig):
            tier = dataclasses.replace(
                tier, **{k: _anchor_cache_dir(getattr(tier, k), origin) for k in _TIER_PATH_KEYS}
            )
        out.append(tier)
    return out


def get_config(
    config_path: str | Path | None = None,
    *,
    user_config_path: Any = _USE_DEFAULT_PATH,
    project_config_path: Any = _USE_DEFAULT_PATH,
    overrides: dict[str, Any] | None = None,
    anchor: Path | None = None,
) -> CashConfig:
    """Resolve the configuration from every layer and return it.

    Args:
        config_path: A TOML file to read above the project and user files,
            as ``Cash(config_path=...)`` does.
        overrides: Settings that win over every layer, as keyword
            arguments to ``Cash(...)`` do.

    ``user_config_path`` and ``project_config_path`` replace the default
    file locations; they exist for tests.
    """

    with untracked():
        return _resolve_config(
            config_path,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
            overrides=overrides,
            anchor=anchor,
        )


def _resolve_config(
    config_path: str | Path | None = None,
    *,
    user_config_path: Any = _USE_DEFAULT_PATH,
    project_config_path: Any = _USE_DEFAULT_PATH,
    overrides: dict[str, Any] | None = None,
    anchor: Path | None = None,
) -> CashConfig:
    """Resolve the merged Cash configuration.

    Args:
        config_path: A config file named in code (the form documented as
            ``Cash(config_path=...)``). Merged on top of the user AND the
            project layer -- a file named explicitly outranks the one found
            by walking up -- and below environment variables and kwargs.
        user_config_path: Path to the user-scoped config (the XDG
            location). Pass ``None`` to skip the user layer entirely;
            omit to use the default location.
        project_config_path: Path to the project-scoped config
            (``pyproject.toml`` with ``[tool.cash]``). Pass ``None`` to
            skip; omit to walk up from cwd.
        overrides: Highest-priority overrides (mirrors what
            ``Cash(**kwargs)`` does internally).
        anchor: The project anchor to resolve as, in place of this
            process's (`project_anchor`): where the default ``cache_dir`` is
            and where the walk for ``pyproject.toml`` starts. The CLI passes a
            notebook's directory, the anchor of a kernel started for it.

    Returns:
        The merged `CashConfig`.
    """
    sources: list[str] = []
    #: setting -> the layer that set it last (the one that won).
    origins: dict[str, str] = {}
    #: every config file looked for, and what was in it.
    files: list[tuple[str, str, str]] = []

    def file_layer(layer: str, path: Any) -> dict[str, Any]:
        data, found = _load_toml_layer(Path(path))
        files.append((layer, str(path), found))
        return _validated_layer(data, str(path), strict=False, unknown_keys=found == TOML_SECTION)

    if config_path is not None and not Path(config_path).exists():
        # Named in code, so it was meant to exist: a tool that forgot to ship
        # its config file would otherwise run on defaults without a word.
        _config_notice(
            "CONFIG-FILE-MISSING",
            f"Cash(config_path=...) names {config_path}, a file that does not "
            f"exist, so none of its settings apply: cash is running on the "
            f"other layers and its defaults.",
            "check the path -- for a packaged tool, that the file is included "
            "in the package (package data) and located relative to the module "
            "(Path(__file__).parent / 'cash.toml'), not the working directory.",
        )

    user_path = default_user_config_path() if user_config_path is _USE_DEFAULT_PATH else user_config_path
    project_path = (
        default_project_config_path(anchor) if project_config_path is _USE_DEFAULT_PATH else project_config_path
    )
    env_data = _load_env_config()
    kwarg_data = _validated_layer(overrides, "Cash(...) arguments", strict=True) if overrides else {}

    # The layers, lowest priority first: (source label, settings, where each
    # setting came from, what a relative cache_dir in it is relative to). A
    # file named in code outranks the pyproject.toml found by walking up, so a
    # package can ship its own settings; the environment and Cash(...)
    # arguments outrank both, and their paths are relative to the cwd.
    layers: list[tuple[str, dict[str, Any], Any, Path | object]] = []
    for layer, path, label in (
        ("user", user_path, "user"),
        ("project", project_path, "project"),
        ("config_path", config_path, "file"),
    ):
        if path is not None:
            layers.append((f"{label}:{path}", file_layer(layer, path), str(path), Path(path).parent))
    # Relative to the cwd -- or, for a given anchor, to it: the directory a
    # process anchored there (a notebook's kernel) runs in.
    here: Path | object = _CALLER_RELATIVE if anchor is None else Path(anchor)
    layers.append(("env", env_data, None, here))
    layers.append(("kwargs", kwarg_data, "Cash(...)", here))

    merged: dict[str, Any] = {
        f.name: getattr(CashConfig(), f.name) for f in fields(CashConfig) if not f.name.startswith("_")
    }
    # Where a relative ``cache_dir`` is resolved from: the project anchor for
    # the default ``.cash``, else the layer that set it.
    cache_dir_origin: Path | object = project_anchor() if anchor is None else anchor
    #: Only when no layer set ``cache_dir`` may an installed console script
    #: be redirected to a per-user location.
    cache_dir_was_configured = False
    for source, data, origin, relative_to in layers:
        if not data:
            continue
        if "tiers" in data:
            data = {**data, "tiers": _anchor_tier_paths(data["tiers"], relative_to)}
        _merge(merged, data)
        sources.append(source)
        for key in data:
            if origin is not None:
                origins[key] = origin
            else:  # the environment names each variable
                origins[key] = "CASH_TIER_<N>_*" if key == "tiers" else f"CASH_{key.upper()}"
        if "cache_dir" in data:
            cache_dir_origin = relative_to
            cache_dir_was_configured = True

    # Not for a given anchor: that resolves as a process anchored there (a
    # notebook's kernel), not as the installed tool this one is.
    if not cache_dir_was_configured and anchor is None:
        installed = installed_entry_point_cache_dir()
        if installed is not None:
            merged["cache_dir"] = str(installed)
            cache_dir_origin = _CALLER_RELATIVE  # already absolute
            sources.append("entry-point")
            origins["cache_dir"] = "installed tool, run outside any project"
    merged["cache_dir"] = _anchor_cache_dir(merged.get("cache_dir"), cache_dir_origin)

    # Materialise the dict into a CashConfig.
    cfg = _build_config(merged, source=",".join(sources) if sources else "defaults")
    cfg._origins = origins
    cfg._files = files
    return cfg


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
    _check_choice(name, checked)
    return checked


def _typed_value(name: str, value: Any, dataclass_type: type) -> Any:
    """`validate_value` before the choice and range checks."""
    field_type = _field_type(name, dataclass_type)
    args = typing.get_args(field_type)
    optional = type(None) in args
    base = next((a for a in args if a is not type(None)), field_type) if args else field_type
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} cannot be None")
    if isinstance(value, str) and base is not str:
        try:
            return _coerce(field_type, value, name)
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


def _validated_layer(data: dict[str, Any], label: str, *, strict: bool, unknown_keys: bool = False) -> dict[str, Any]:
    """*data* with every known field checked by `validate_value`.

    A bad value RAISES when the caller's own code supplied it (*strict*) --
    that is a bug at the call site, and the place to say so -- and is reported
    (CONFIG-INVALID) and dropped when it came from a file or the environment,
    which must not stop a program from running.

    With *unknown_keys* -- a ``[tool.cash]`` table or a cash config file, where
    every key is meant to be cash's -- a key that is not a setting is reported
    (CONFIG-UNKNOWN-KEY, with the nearest real name). Under *strict* it
    raises, as a bad value does: ``Cash(ttl=60)`` or a misspelt
    ``Cash(max_cache_szie=...)`` would otherwise be dropped without a word.
    """
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    tier_valid = {f.name for f in fields(TierConfig)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        try:
            if key == "tiers" and strict and not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"tiers={value!r}: expected a list of tier tables, such as "
                    '[{"type": "memory"}, {"type": "file"}], or backend="memory" for a single tier'
                )
            if key == "tiers" and isinstance(value, (list, tuple)):
                tiers = []
                for i, t in enumerate(value):
                    if not isinstance(t, dict):
                        tiers.append(t)
                        continue
                    for k in t:
                        if k in tier_valid:
                            continue
                        if strict:
                            raise ValueError(_not_a_setting(f"tiers[{i}].{k}", k, tier_valid))
                        if unknown_keys:
                            _unknown_key(label, f"tiers[{i}].{k}", k, tier_valid)
                    tiers.append(
                        {k: (validate_value(k, v, TierConfig) if k in tier_valid else v) for k, v in t.items()}
                    )
                    if tiers[-1].get("type") not in _SUPPORTED_TIER_TYPES:
                        raise ValueError(
                            f"tiers[{i}].type={t.get('type')!r}: not one of {', '.join(sorted(_SUPPORTED_TIER_TYPES))}"
                        )
                out[key] = tiers
            elif key in valid:
                out[key] = validate_value(key, value)
            elif strict:
                raise ValueError(_not_a_setting(key, key, valid))
            elif unknown_keys:
                _unknown_key(label, key, key, valid)
            else:
                out[key] = value
        except ValueError as exc:
            if strict:
                raise ValueError(f"cash config ({label}): {exc}") from None
            _config_notice(
                "CONFIG-INVALID",
                f"{label} sets {exc}. That setting is being ignored, so its default applies.",
                "correct the value; `cash info` shows every setting in effect and where it came from.",
            )
    return out


#: Names people pass as settings that belong somewhere else.
_NOT_SETTINGS = {
    "ttl": " ttl is set per function, @cash.cache(ttl=...), or as default_ttl on a file tier.",
}


def _not_a_setting(shown: str, key: str, valid: Any) -> str:
    return f"`{shown}` is not a cash setting.{_NOT_SETTINGS.get(key) or _did_you_mean(key, valid)}"


def _unknown_key(label: str, shown: str, key: str, valid: Any) -> None:
    _config_notice(
        "CONFIG-UNKNOWN-KEY",
        f"{label} sets `{shown}`, which is not a cash setting, so it does nothing.{_did_you_mean(key, valid)}",
        "rename or remove it; `cash info` shows every setting in effect and where it came from.",
    )


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


def _build_tiers(entries: list[Any]) -> list[TierConfig]:
    """The usable tiers of *entries*, naming each one left out (CONFIG-INVALID).

    A tier with no ``type`` is what ``CASH_TIER_<N>_*`` variables leave when no
    file declares tier N; dropping it without a word changed the stack.
    """
    names = {f.name for f in fields(TierConfig)}
    tiers: list[TierConfig] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, TierConfig):
            tiers.append(entry)
            continue
        problem = None
        if not isinstance(entry, dict):
            problem = f"is {entry!r}, not a table of tier settings"
        elif not entry.get("type"):
            problem = f"has no type ({entry!r})"
        else:
            try:
                tiers.append(TierConfig(**{k: v for k, v in entry.items() if k in names}))
            except (ValueError, TypeError) as exc:
                problem = f"cannot be used ({exc})"
        if problem:
            _config_notice(
                "CONFIG-INVALID",
                f"tiers[{i}] {problem}, so it is left out of the tier stack.",
                "give every tier a type (memory, file, sqlite, redis or s3); a tier set "
                "only through CASH_TIER_<N>_* variables needs CASH_TIER_<N>_TYPE.",
            )
    return tiers


def validated_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """*overrides* as ``Cash(**overrides)`` would apply them, for ``cash.configure``.

    The constructor's path, so the two cannot disagree: every value checked
    (``ValueError`` on a bad one, as code gave it), ``tiers`` built into
    `TierConfig`s, and ``cache_dir`` with ``~`` expanded and otherwise
    relative to the cwd, like any path given in code.
    """
    checked = _validated_layer(overrides, "cash.configure(...)", strict=True)
    if "tiers" in checked:
        checked["tiers"] = _build_tiers(checked["tiers"] or [])
    if "cache_dir" in checked:
        checked["cache_dir"] = _anchor_cache_dir(checked["cache_dir"], _CALLER_RELATIVE)
    if "tiers" in checked:
        checked["tiers"] = _anchor_tier_paths(checked["tiers"], _CALLER_RELATIVE)
    return checked


def _build_config(merged: dict[str, Any], source: str) -> CashConfig:
    """Materialise the merged dict into a typed CashConfig instance."""
    cfg = CashConfig()
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    for key, value in merged.items():
        if key == "tiers":
            cfg.tiers = _build_tiers(value or [])
        elif key in valid:
            setattr(cfg, key, value)
    cfg._source = source
    return cfg


# ---------------------------------------------------------------------------
# Default config file template
# ---------------------------------------------------------------------------


_CONFIG_DOCS_URL = "https://cash-lib.readthedocs.io/en/latest/getting-started/configuration/"

#: The ``[[cash.tiers]]`` example in the template: a RAM tier in front of Redis.
_TIER_EXAMPLE = """\
[[cash.tiers]]
type = "memory"
max_entries = 10000

[[cash.tiers]]
type = "redis"
host = "redis.internal"
port = 6379"""


def _field_docs(cls: type) -> dict[str, str]:
    """The docstring under each field of dataclass *cls*, read from its source.

    Python drops attribute docstrings at compile time, so the only copy is
    the source. Empty when the source is not available (a frozen app).
    """

    try:
        body = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0].body
    except (OSError, TypeError):
        return {}
    docs = {}
    for node, nxt in zip(body, body[1:]):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[node.target.id] = inspect.cleandoc(nxt.value.value)
    return docs


def _tier_key_docs() -> dict[str, str]:
    """Each ``TierConfig`` key's entry in its docstring's Attributes section.

    Empty when docstrings are stripped (``python -OO``).
    """
    doc = inspect.cleandoc(TierConfig.__doc__ or "")
    section = doc.partition("\nAttributes:\n")[2]
    docs: dict[str, str] = {}
    name = None
    for line in section.splitlines():
        entry = re.match(r"    (\w+): (.*)", line)
        if entry:
            name = entry[1]
            docs[name] = entry[2]
        elif name and line.startswith("        "):
            docs[name] += " " + line.strip()
        else:
            break
    return docs


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


def _comment(text: str) -> list[str]:
    return [f"# {line}".rstrip() for line in text.splitlines()]


def _default_config_text() -> str:
    """The template `create_default_config` writes, built from `CashConfig`.

    Every setting is commented out at its default, under its field docstring,
    so the file changes nothing until a line is uncommented and a new default
    in cash still reaches a user who never touched that line.
    """
    # The layer list in this module's docstring, so the two cannot disagree.
    precedence = (__doc__ or "").partition("highest priority wins):")[2].strip("\n").split("\n\n")[0]
    lines = [
        f"# Cash configuration -- see {_CONFIG_DOCS_URL}",
        "#",
        "# Resolution order (highest wins):",
        *_comment(textwrap.dedent(precedence).strip("\n")),
        "#",
        "# Every setting below is commented out at its default. Uncomment a line",
        "# to change it. A setting marked (unset) has no default value.",
        "",
        "[cash]",
    ]
    docs = _field_docs(CashConfig)
    defaults = CashConfig()
    for f in fields(CashConfig):
        if f.name.startswith("_") or f.name == "tiers":
            continue
        value = getattr(defaults, f.name)
        lines += ["", *_comment(docs.get(f.name, ""))]
        lines.append(f"# {f.name} = " + ("(unset)" if value is None else _toml_value(value)))

    tier_docs = _tier_key_docs()
    lines += ["", *_comment(docs.get("tiers", "")), "#", "# Keys of a [[cash.tiers]] table:"]
    for f in fields(TierConfig):
        first = tier_docs.get(f.name, "")
        lines += _comment(textwrap.fill(f"{f.name}: {first}", 72, initial_indent="  ", subsequent_indent="      "))
    lines += ["#", *_comment(_TIER_EXAMPLE)]
    return "\n".join(lines) + "\n"


def create_default_config(path: str | None = None, *, force: bool = False) -> str:
    """Write a documented config template to *path* and return the path.

    *path* defaults to the user config file (``~/.config/cash/config.toml``,
    or ``%APPDATA%\\cash\\config.toml`` on Windows). Every setting is listed
    commented out at its default, with its documentation above it.

    Raises ``FileExistsError`` if the file exists, unless ``force=True``.
    """
    out_path = Path(path) if path is not None else default_user_config_path()
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass force=True to overwrite it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_default_config_text(), encoding="utf-8")
    return str(out_path)
