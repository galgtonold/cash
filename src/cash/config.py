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
from .tracking.file_tracker import untracked

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


def _check_choice(name: str, value: Any) -> None:
    """``ValueError`` when *name* must be one of a fixed set and *value* is not."""
    allowed = _NAMED_CHOICES.get(name)
    if allowed is not None and isinstance(value, str) and value not in allowed:
        raise ValueError(f"{name}={value!r}: not one of {', '.join(sorted(allowed))}")


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
    ``"redis"``, ``"s3"``. Raises ``ValueError`` on unknown values."""

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
            raise ValueError(f"Unknown tier type: {self.type!r}. Supported: {sorted(_SUPPORTED_TIER_TYPES)}")


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
    * **Cost-aware policy** — which results are expensive enough to
      cache, and to write past RAM.
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

    max_cache_size: int | None = None
    """Maximum total **disk** cache size in bytes. ``None`` (default)
    means **auto**: cash scales the cap to the machine — a generous
    fraction of the free space on the cache volume for the disk tier,
    and a modest fraction of system RAM for the memory tier (see
    ``cash.backends.adaptive_caps``). Set an integer to pin the disk
    cap explicitly; the memory tier keeps its own auto/modest cap.
    When the file backend exceeds the resolved cap it evicts the
    entries least worth keeping until it fits: those that are cheapest
    to recompute per byte, weighted by how often they are hit (GDSF)."""

    max_memory_entries: int | None = None
    """LRU entry cap for the in-memory tier. ``None`` (default)
    means unlimited — eviction is driven by ``psutil`` memory
    pressure instead. Set an integer to force a hard count limit."""

    flush_interval: int = 5
    """Seconds between metadata flushes for the file backend. Lower
    values reduce data loss on crash but increase disk I/O. Set to
    0 to flush after every write (slowest, safest)."""

    file_hash_full_max_bytes: int = 256 * 1024 * 1024
    """Largest tracked file hashed IN FULL when checking freshness.

    Above this, the content hash covers three deterministic head/middle/tail
    regions plus the size, and the file's timestamps are used as a backstop —
    which keeps a freshness check on a multi-GB parquet cheap, and leaves one
    hole: a same-size edit *outside* the sampled regions that leaves the mtime
    as it was is invisible. `cp -p`, `rsync -a` and `tar -x` restore the mtime;
    a write through ``np.memmap(mode="r+")`` on Windows never moves it at all.
    On Linux and macOS the inode change time closes that; on Windows nothing
    does.

    The default covers the ordinary CSV, parquet or ``.npy`` outright. A full
    hash costs about 0.72 ms per MiB — about 0.2 s at 256 MiB — but only the
    FIRST check of a file pays it: the digest is memoized per process, so
    later checks of an unchanged file cost a ``stat``. Lower it if your inputs
    are large, on a slow mount, and re-read by many short-lived processes."""

    shutdown_write_timeout: float = 60.0
    """Seconds a finishing process waits for its background cache
    writes before exiting without them.

    Generous, because dropping a large result still on its way to
    disk wastes real compute — but FINITE, because a cache write
    that cannot complete must never keep a finished process alive.
    An unwritable cache directory once left a 24-second job still
    running at 420 seconds, its answer already printed. Expiry is
    reported as ``CACHE-WRITE-ABANDONED``; raise this only when the
    storage really is that slow."""

    # --- Cost-aware caching policy ---
    persist_all: bool = False
    """When True, cache **every** notebook statement, bypassing the
    cost-aware floors (``min_execution_time_to_cache_seconds`` and the
    size-aware skip) - equivalent to putting ``# @cash:persist`` on every
    statement. Useful for deterministic benchmarks, reproducibility, and
    debugging cache behavior where you want every statement to leave a
    restorable entry. Wasteful for cheap-to-compute values in normal use.
    Hot field - flippable at runtime via ``cash.configure(persist_all=True)``
    or the ``%cash_persist on`` magic."""

    summary: bool = False
    """Print a per-function hit/miss summary when the process exits.

    Off by default -- a library that prints uninvited is a library people
    filter. On, it answers the question a script user cannot otherwise
    answer: *which parts of my run recomputed just now?* A notebook shows
    that per statement in the badge; a script showed nothing, and the user
    who needed it resorted to adding ``print`` calls to each branch.

    Reachable four ways from this one field, because the config layer maps
    every field to an env var and a TOML key: ``cash.configure(summary=True)``,
    ``Cash(summary=True)``, ``CASH_SUMMARY=1``, or ``summary = true`` in
    ``cash.toml``. The env var matters most -- it is the only one that needs
    no edit to the code you are already running."""

    disable: bool = False
    """Run every ``@cash.cache`` function uncached: no key, no lookup, no
    store, no analysis -- the call goes straight through, and ``%cash_on``
    declines to switch the notebook path on.

    What a test suite needs to prove the code rather than the cache: a test
    that calls a cached function twice and compares the results is comparing
    one result with itself, and passes even when the function ignores its
    seed. ``CASH_DISABLE=1 pytest`` is the run that catches that. Read per
    call, so ``cash.configure(disable=True)`` takes effect immediately."""

    min_execution_time_to_cache_seconds: float = 0.01
    """Floor (seconds) a notebook statement must take to be cached at
    all, even in RAM. Default 10 ms. Whether a cached value is also
    written to disk is decided separately, by the cost model, which
    persists nothing that took under 0.1 s."""

    call_cost_floor_seconds: float = 0.003
    """Floor (seconds) a single CALL's own execution must clear before its
    result is stored. Below it, the ~2.2 ms of interception overhead is most
    of what a hit would save, so caching cannot pay back. Default 3 ms."""

    loop_split_max_iter_seconds: float = 0.006
    """Ceiling (seconds) on measured per-iteration cost above which a loop is
    NOT split. At or above it a call clears ``call_cost_floor_seconds`` once
    decomposition overhead is subtracted, so per-call caching already covers
    the loop — and covers it better, being incremental where a split's tail is
    all-or-nothing. Default 6 ms."""

    loop_split_min_remaining_seconds: float = 0.1
    """Projected cost (seconds) of a loop's remaining iterations below which a
    split is not worth persisting a verdict for. Default 100 ms."""

    min_cache_savings_pct: float = 0.20
    """Fraction (0.0 – 1.0) of the compute time a restore must save for
    a value to be written past RAM: restoring has to beat recomputing
    by this much, as the cost model predicts. Default 0.20."""

    min_cache_fixed_budget_seconds: float = 0.05
    """Per-call I/O budget (seconds). If the predicted serialise +
    write cost exceeds this fraction of the compute saved, skip
    the write. Pairs with ``min_cache_savings_pct``."""

    remote_revalidate_max_age_seconds: float = 0.0
    """How long a remote object's state token may be reused before the
    store is asked again (seconds). ``0`` (the default) revalidates on
    every cache hit, which is the only setting that cannot serve stale
    data.

    Raising it trades correctness for latency, so raise it deliberately:
    for the window's duration, a changed object goes unnoticed. It exists
    for the case the per-source ``max_age`` can't reach — reads cash
    tracked *automatically*, where you never construct the source and so
    have nowhere to put ``immutable=True``. Strict in CI, relaxed in a
    long exploratory session over a slow link.

    A very large value is the effective "stop checking" switch; there is
    deliberately no boolean for it, because "never revalidate" is a
    window, not a different mode."""

    # --- Observability ---
    debug: bool = False
    """When True, every cache decision logs at INFO level with the
    reason ('HIT', 'MISS — file changed', 'SKIPPED — too cheap').
    Useful for diagnosing 'why didn't this hit?' mysteries. Hot
    field — flippable at runtime via ``cash.configure(debug=True)``."""

    verbose: bool = False
    """When True, log one line per decorated call -- a hit, or a miss and
    why -- to the ``cash.calls`` logger, without cash's other debug
    records. ``debug`` implies it. Settable wherever ``debug`` is: the
    constructor, ``cash.configure(verbose=True)``, ``CASH_VERBOSE=1``,
    ``verbose = true`` under ``[tool.cash]``."""

    analytics: bool = True
    """Record each notebook statement's hit, miss and timing in
    ``analytics.db`` under the per-user cache root (``~/.cache/cash`` on
    Linux), which the analytics dashboard (``cash.show_stats()``) reads.
    Telemetry only: turning it off changes no cached result, and no file is
    created. Read when a notebook session starts."""

    # --- Backend selection ---
    backend: str = "tiered"
    """Backend selector. ``"tiered"`` (default) is a RAM tier in front
    of a file tier. ``"memory"`` / ``"file"`` / ``"sqlite"`` /
    ``"redis"`` / ``"s3"`` is a single backend of that type. Either
    way the fields above and below configure it. Ignored when
    ``tiers`` is non-empty."""

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
    """Explicit tier stack, fastest first (e.g. ``[memory, redis,
    s3]``), replacing the one ``backend`` names. A setting a tier
    leaves unset comes from the top-level field of the same meaning
    (``cache_dir``, ``max_cache_size``, ``redis_host``, ...)."""

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
#: ``"512MiB"``. People write sizes that way -- a round-15 operator set
#: ``CASH_MAX_CACHE_SIZE=500MB`` -- and a bare integer of bytes is the one
#: spelling nobody reads correctly at a glance.
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


def format_size(n: int) -> str:
    """*n* bytes the way `parse_size` reads it back: ``2000000000`` -> ``"2 GB"``.

    A size written ``"2GB"`` was shown back as ``1.9 GB`` -- parsed in powers
    of 1000 and displayed in powers of 1024 under the decimal label, so the
    setting looked mis-read (round 18, three testers). A whole number of a
    decimal or a binary unit is shown in that unit; anything else in binary
    units, labelled as such.
    """
    for name in ("TB", "TiB", "GB", "GiB", "MB", "MiB", "KB", "KiB"):
        unit = _SIZE_UNITS[name.lower()]
        if n >= unit and (n * 10) % unit == 0:
            return f"{n / unit:g} {name}"
    # Local: import cycle config -> backends.adaptive_caps -> backends -> backends._base -> config.
    from .backends.adaptive_caps import human_bytes

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

#: One notice per process. A config file is read on every ``get_config()``.
_TOML_NOTICE_GIVEN = False

#: ``(code, message)`` pairs already said in this process. A config is
#: resolved more than once -- the module default, then each ``Cash(...)`` --
#: and each resolution used to repeat every complaint about it (round 18: the
#: same bad value, printed twice, as an uncoded log line).
_CONFIG_NOTICES: set[tuple[str, str]] = set()


def _config_notice(code: str, what: str, fix: str) -> None:
    """Warn, once per process, about a setting cash could not use."""
    if (code, what) in _CONFIG_NOTICES:
        return
    _CONFIG_NOTICES.add((code, what))
    try:
        warn_diagnostic(CashCacheIneffectiveWarning, code, what, fix)
    except Exception:  # noqa: BLE001 - a notice must never break a config load
        logger.debug("Could not emit %s", code, exc_info=True)


def _did_you_mean(key: str, valid: Any) -> str:
    match = difflib.get_close_matches(key, sorted(valid), n=1, cutoff=0.6)
    return f" Did you mean `{match[0]}`?" if match else ""


def _warn_toml_unreadable(path: Path) -> None:
    """Say that a config file was found and is being ignored."""
    global _TOML_NOTICE_GIVEN
    if _TOML_NOTICE_GIVEN:
        return
    _TOML_NOTICE_GIVEN = True
    try:
        warn_diagnostic(
            CashCacheIneffectiveWarning,
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
    except Exception:  # noqa: BLE001 - a notice must never break a config load
        logger.debug("Could not emit the unreadable-TOML notice", exc_info=True)


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
            # read this file. That used to be a debug line: the config existed,
            # was found, and was silently ignored -- every setting in it, on
            # the oldest Python cash supports. Reached only when a config file
            # with cash settings in it is actually there: nearly every project
            # has a pyproject.toml, and a notice about one with no [tool.cash]
            # would be wrong nearly every time -- the way to teach people to
            # filter it out before the one case it exists for.
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
    points at a character nobody can see (round 18).
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
      job was launched from". That difference is the whole of CAS-84.
    * **a config file** -- relative to that file's directory, the ordinary rule
      for paths in config files. Anchoring it to the cwd instead is what made
      ``[tool.cash] cache_dir`` useless for the case it was documented to fix.
    * **an env var or a kwarg** -- left exactly as written. The user typed it in
      the shell or in the code that is running now, so cwd-relative is what they
      meant, and it is what every other command-line path does.

    An absolute path is returned untouched in all three cases.
    """
    if not isinstance(cache_dir, str) or not cache_dir:
        return cache_dir
    # `~/crunch-cache` in a shipped config file became a directory literally
    # named `~` beside that file, inside site-packages (round 20).
    cache_dir = os.path.expanduser(cache_dir)
    if origin is _CALLER_RELATIVE or os.path.isabs(cache_dir):
        return cache_dir
    if not isinstance(origin, Path):
        return cache_dir
    return os.path.normpath(str(origin / cache_dir))


def get_config(
    config_path: str | Path | None = None,
    *,
    user_config_path: Any = _USE_DEFAULT_PATH,
    project_config_path: Any = _USE_DEFAULT_PATH,
    overrides: dict[str, Any] | None = None,
) -> CashConfig:
    """Resolve the merged Cash configuration -- see `_resolve_config`.

    Never recorded as a file dependency: this can run inside a cached call
    (a nested call's bookkeeping), and the files it reads are cash's, not the
    function's.
    """

    with untracked():
        return _resolve_config(
            config_path,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
            overrides=overrides,
        )


def _resolve_config(
    config_path: str | Path | None = None,
    *,
    user_config_path: Any = _USE_DEFAULT_PATH,
    project_config_path: Any = _USE_DEFAULT_PATH,
    overrides: dict[str, Any] | None = None,
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
        data = _validated_layer(data, str(path), strict=False, unknown_keys=found == TOML_SECTION)
        for key in data:
            origins[key] = str(path)
        return data

    # Where a relative ``cache_dir`` should be resolved FROM. Starts as the
    # project anchor (the default ``.cash`` belongs to the project, not to
    # wherever the job was launched); each layer that sets ``cache_dir``
    # replaces it with the directory that layer is written relative to.
    cache_dir_origin: Path | object = project_anchor()
    #: Did any layer below actually set ``cache_dir``? Only when none did is
    #: the value still the dataclass default, and only then may an installed
    #: console script be redirected to a per-user location.
    cache_dir_was_configured = False

    # Layer 1: defaults from CashConfig dataclass
    merged: dict[str, Any] = {
        f.name: getattr(CashConfig(), f.name) for f in fields(CashConfig) if not f.name.startswith("_")
    }

    # Layer 2: user TOML
    if user_config_path is _USE_DEFAULT_PATH:
        user_path = default_user_config_path()
    else:
        user_path = user_config_path
    if user_path is not None:
        user_data = file_layer("user", user_path)
        if user_data:
            _merge(merged, user_data)
            sources.append(f"user:{user_path}")
            if "cache_dir" in user_data:
                cache_dir_origin = Path(user_path).parent
                cache_dir_was_configured = True

    # Layer 3: project TOML
    if project_config_path is _USE_DEFAULT_PATH:
        project_path = default_project_config_path()
    else:
        project_path = project_config_path
    if project_path is not None:
        project_data = file_layer("project", project_path)
        if project_data:
            _merge(merged, project_data)
            sources.append(f"project:{project_path}")
            if "cache_dir" in project_data:
                cache_dir_origin = Path(project_path).parent
                cache_dir_was_configured = True

    # Layer 3b: explicit ``Cash(config_path=...)``, above the project file. A
    # file named in code outranks the one found by walking up from wherever
    # the process started: a package shipping its own cash settings had them
    # overridden by the pyproject.toml of whatever project launched it
    # (round 19). Environment variables and Cash(...) arguments still win.
    if config_path is not None:
        if not Path(config_path).exists():
            # Named in code, so it was meant to exist: a tool that forgot to
            # ship its config file ran on defaults -- its cache lifetime gone
            # -- and nothing said so (round 20).
            _config_notice(
                "CONFIG-FILE-MISSING",
                f"Cash(config_path=...) names {config_path}, a file that does not "
                f"exist, so none of its settings apply: cash is running on the "
                f"other layers and its defaults.",
                "check the path -- for a packaged tool, that the file is included "
                "in the package (package data) and located relative to the module "
                "(Path(__file__).parent / 'cash.toml'), not the working directory.",
            )
        override_data = file_layer("config_path", config_path)
        if override_data:
            _merge(merged, override_data)
            sources.append(f"file:{config_path}")
            if "cache_dir" in override_data:
                cache_dir_origin = Path(config_path).parent
                cache_dir_was_configured = True

    # Layer 4: env vars
    env_data = _load_env_config()
    if env_data:
        _merge(merged, env_data)
        sources.append("env")
        for key in env_data:
            origins[key] = "CASH_TIER_<N>_*" if key == "tiers" else f"CASH_{key.upper()}"
        if "cache_dir" in env_data:
            cache_dir_origin = _CALLER_RELATIVE
            cache_dir_was_configured = True

    # Layer 5: explicit overrides (kwargs)
    if overrides:
        overrides = _validated_layer(overrides, "Cash(...) arguments", strict=True)
        _merge(merged, overrides)
        sources.append("kwargs")
        for key in overrides:
            origins[key] = "Cash(...)"
        if "cache_dir" in overrides:
            cache_dir_origin = _CALLER_RELATIVE
            cache_dir_was_configured = True

    if not cache_dir_was_configured:
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
            if name in _NAMED_CHOICES and value not in _NAMED_CHOICES[name]:
                # Checked here so a file or env layer reports it
                # (CONFIG-INVALID) and falls back, as every other bad value
                # does. It used to reach the backend factory, which raised
                # ValueError out of `import cash` -- and out of `cash info`,
                # the command for finding out what is wrong.
                raise ValueError(f"{name}={value!r}: not one of {', '.join(sorted(_NAMED_CHOICES[name]))}")
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
    (CONFIG-UNKNOWN-KEY, with the nearest real name). It used to pass through
    to ``_build_config`` and be dropped there without a word, while
    ``cash.configure()`` raised on the same typo (round 18).
    """
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    tier_valid = {f.name for f in fields(TierConfig)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        try:
            if key == "tiers" and isinstance(value, list):
                tiers = []
                for i, t in enumerate(value):
                    if not isinstance(t, dict):
                        tiers.append(t)
                        continue
                    if unknown_keys:
                        for k in t:
                            if k not in tier_valid:
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
                    try:
                        tiers.append(TierConfig(**clean))
                    except (ValueError, TypeError):
                        logger.debug("dropping unusable tier %r", clean)
                # else: malformed entry — silently skip
            cfg.tiers = tiers
        elif key in valid:
            setattr(cfg, key, value)
    cfg._source = source
    return cfg


# ---------------------------------------------------------------------------
# Default config file template
# ---------------------------------------------------------------------------


_CONFIG_DOCS_URL = "https://cash-lib.readthedocs.io/en/stable/getting-started/configuration/"

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

    tier_docs = _field_docs(TierConfig)
    lines += ["", *_comment(docs.get("tiers", "")), "#", "# Keys of a [[cash.tiers]] table:"]
    for f in fields(TierConfig):
        first = " ".join(tier_docs.get(f.name, "").split("\n\n")[0].split())
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
