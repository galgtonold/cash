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
import site
import sys
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

    max_cache_size: int | None = None
    """Maximum total **disk** cache size in bytes. ``None`` (default)
    means **auto**: cash scales the cap to the machine — a generous
    fraction of the free space on the cache volume for the disk tier,
    and a modest fraction of system RAM for the memory tier (see
    ``cash.backends.adaptive_caps``). Set an integer to pin the disk
    cap explicitly; the memory tier keeps its own auto/modest cap.

    Historically this defaulted to a flat 1 GiB applied to *every*
    tier, which capped the disk tier at one medium DataFrame and put
    persist-heavy workloads into a write-and-evict treadmill.
    When the file backend exceeds the resolved cap it evicts
    least-recently-accessed entries until it fits."""

    max_memory_entries: int | None = None
    """LRU entry cap for the in-memory tier. ``None`` (default)
    means unlimited — eviction is driven by ``psutil`` memory
    pressure instead. Set an integer to force a hard count limit."""

    flush_interval: int = 5
    """Seconds between metadata flushes for the file backend. Lower
    values reduce data loss on crash but increase disk I/O. Set to
    0 to flush after every write (slowest, safest)."""

    file_hash_full_max_bytes: int = 64 * 1024 * 1024
    """Largest tracked file hashed IN FULL when checking freshness.

    Above this, the content hash covers three deterministic head/middle/tail
    regions plus the size, and the file's timestamps are used as a backstop —
    which keeps a freshness check on a multi-GB parquet cheap, and leaves one
    hole: a same-size edit *outside* the sampled regions whose mtime is then
    restored (`cp -p`, `rsync -a`, `tar -x`) is invisible. On Linux and macOS
    the inode change time closes that; on Windows it does not.

    The default covers the ordinary CSV or parquet outright. A full hash
    costs about 0.72 ms per MiB — 46 ms at 64 MiB — but only the FIRST
    check of a file pays it: the digest is memoized per process, so later
    checks of an unchanged file cost a ``stat``. Lower it if your inputs are
    large, on a slow mount, and re-read by many short-lived processes."""

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

    smart_persistence: bool = True
    """When True (default), the tiered backend decides per-entry
    whether to persist past RAM based on compute time vs storage
    cost, using the serialization-aware cost model with a 0.1 s
    compute floor.

    Setting it False does **not** persist everything: it drops to
    ``TieredBackend``'s own ``_default_promotion_policy``, which applies
    the same cost-model rule at the more conservative 1.0 s floor. So the
    practical effect is *less* persistence for mid-cost values, not more.
    Use ``persist_all=True`` (or ``%cash_persist on``) if you actually want
    everything written to disk."""

    min_execution_time_to_cache_seconds: float = 0.01
    """Hard floor (seconds). Compute under this duration is never
    promoted past RAM — disk I/O alone would cost more than
    rerunning. Default 10 ms."""

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
    """Required time-savings fraction (0.0 – 1.0) for a tier to be
    considered worthwhile. If a cache hit only saves 20% of the
    compute cost, the entry isn't promoted. Default 0.20."""

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


#: Fields that hold a number of BYTES, and so also accept ``"2GB"`` /
#: ``"512MiB"``. People write sizes that way -- a round-15 operator set
#: ``CASH_MAX_CACHE_SIZE=500MB`` -- and a bare integer of bytes is the one
#: spelling nobody reads correctly at a glance.
_SIZE_FIELDS = frozenset({"max_cache_size", "file_hash_full_max_bytes", "max_size_bytes"})

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]i?b|b)?\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "b": 1,
    "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,          # SI
    "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,       # binary
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


def _coerce(field_type: Any, raw: str, name: str | None = None) -> Any:
    """Convert a string from env vars/TOML into the field's declared type.

    Raises ``ValueError`` on failure so the caller can skip the value
    and log a warning rather than poison the whole config load.
    """
    if name in _SIZE_FIELDS:
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


def _warn_toml_unreadable(path: Path) -> None:
    """Say that a config file was found and is being ignored."""
    global _TOML_NOTICE_GIVEN
    if _TOML_NOTICE_GIVEN:
        return
    _TOML_NOTICE_GIVEN = True
    try:
        from .diagnostics import warn_diagnostic
        from .exceptions import CashCacheIneffectiveWarning
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CONFIG-TOML-UNREADABLE",
            f"cash found {path} but cannot read it: this is Python "
            f"{sys.version_info.major}.{sys.version_info.minor}, whose standard "
            f"library has no TOML parser, and `tomli` is not installed. Every "
            f"setting in that file is being ignored, including cache_dir -- so "
            f"cash is running on defaults that the file was written to change.",
            "pip install tomli (cash keeps no required dependencies, so it "
            "cannot install one for you), or set the values through CASH_* "
            "environment variables instead, or run on Python 3.11+ where the "
            "parser is in the standard library.",
        )
    except Exception:  # noqa: BLE001 - a notice must never break a config load
        logger.debug("Could not emit the unreadable-TOML notice", exc_info=True)


_CASH_SECTION_RE = re.compile(
    r"^\s*(\[\s*(tool\s*\.\s*)?cash\s*[\].]|tool\s*\.\s*cash\s*\.)", re.MULTILINE)


def _may_hold_cash_settings(path: Path) -> bool:
    """Would a parser find cash settings in *path*? Answered without one.

    A ``[tool.cash]`` / ``[cash]`` table (or a ``tool.cash.`` dotted key)
    anywhere says yes. A ``pyproject.toml`` without one says no: it belongs to
    the project, not to cash. Any other file is a cash config file, whose flat
    top-level keys are read too, so anything but comments counts. Unreadable
    says yes -- the notice errs toward being given.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    if _CASH_SECTION_RE.search(text):
        return True
    if path.name == "pyproject.toml":
        return False
    return any(line.strip() and not line.lstrip().startswith("#")
               for line in text.splitlines())


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
                value = _coerce(_field_type(field_name, TierConfig), raw, field_name)
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
            value = _coerce(_field_type(key), raw, key)
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

#: Files that mean "the project starts here".
_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


def _interactive_shell_is_running() -> bool:
    """True inside IPython, a Jupyter kernel, or anything else hosting one.

    ``__main__.__file__`` cannot be trusted there. IPython SETS it, temporarily,
    while it executes each of the profile's startup scripts -- so a kernel that
    imports cash from a startup file resolves an anchor inside
    ``~/.ipython/profile_default/startup`` and writes the session's cache there.
    Measured exactly that: a notebook whose entries went to the profile
    directory after a kernel restart, so nothing hit.

    There is no "running script" in an interactive session anyway. The cwd is
    the right answer, and it is the one a notebook has always had.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return get_ipython() is not None
    except Exception:  # noqa: BLE001 - a half-initialised IPython is not one
        return False


def _running_cash_cli() -> bool:
    """Is ``__main__`` cash's own command line (``python -m cash``)?

    It is a tool acting on the project you are standing in, wherever its
    source lives. From an editable checkout it looked like a local script,
    anchored to cash's own repository, and ``python -m cash clear --all`` run
    inside another project cleared the cash checkout's cache instead.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(spec, "name", None) == "cash.__main__"


def _running_script_dir() -> Path | None:
    """The directory of the script being run, or None if that is meaningless.

    None for an interactive interpreter, a notebook, ``python -c``, and for any
    ``__main__`` that lives inside the interpreter's own installation -- an
    installed console entry point, ``python -m pytest``, the Jupyter kernel
    launcher. Those all report a ``__file__`` somewhere under ``sys.prefix`` or
    site-packages, and anchoring a user's cache inside their virtualenv because
    they ran an installed tool would be a worse answer than the cwd.
    """
    if _interactive_shell_is_running() or _running_cash_cli():
        return None
    main = sys.modules.get("__main__")
    raw = getattr(main, "__file__", None)
    if not raw:
        # A spawned multiprocessing worker has no ``__main__.__file__`` and no
        # ``__spec__`` -- but it does inherit the parent's ``sys.argv[0]``.
        # Without this the parent anchored to its project and every pool worker
        # fell back to the cwd, so one run wrote into two cache directories and
        # neither side could see the other's entries. A round-16 tester
        # measured exactly that, 3/3, and it defeats the whole point of a
        # shared cache across a fan-out.
        #
        # Only a real file counts, which is what keeps the interpreter's own
        # invocations out: ``python -c`` leaves ``-c`` here, a REPL leaves the
        # empty string, and an installed console script is filtered below like
        # any other path inside the interpreter's installation.
        raw = sys.argv[0] if sys.argv else None
        if not raw or not str(raw).endswith(".py") or not os.path.isfile(raw):
            return None
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    if _is_installed_path(path):
        return None
    return path.parent


def _is_installed_path(path: Path) -> bool:
    """Does *path* live inside the interpreter's own installation?"""
    installed_roots = [Path(sys.prefix), Path(sys.base_prefix)]
    installed_roots += [Path(p) for p in site.getsitepackages()] if hasattr(site, "getsitepackages") else []
    user_site = getattr(site, "getusersitepackages", None)
    if user_site is not None:
        try:
            installed_roots.append(Path(user_site()))
        except Exception:  # noqa: BLE001 - a site module without a user site
            pass
    for root in installed_roots:
        try:
            if path.is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


def _running_installed_module() -> bool:
    """``python -m <module installed in site-packages>`` -- ``python -m pytest``."""
    main = sys.modules.get("__main__")
    if getattr(main, "__spec__", None) is None:
        return False
    if _running_cash_cli():
        return True
    raw = getattr(main, "__file__", None)
    if not raw:
        return False
    try:
        return _is_installed_path(Path(raw).resolve())
    except OSError:
        return False


def _running_installed_code() -> bool:
    """Is the program itself installed code -- a console script or ``-m`` module?

    Then the code being cached is the user's project code it runs, and the
    project the user is standing in is the best anchor there is. Not true of a
    notebook, a REPL or ``python -c``, which keep the cwd.
    """
    if _interactive_shell_is_running():
        return False
    return _running_console_script() is not None or _running_installed_module()


def _cwd_project_root() -> Path | None:
    """The first directory at or above the cwd holding a project marker."""
    try:
        here = Path.cwd()
    except OSError:
        return None
    for d in [here, *here.parents]:
        try:
            if any((d / marker).exists() for marker in _PROJECT_MARKERS):
                return d
        except OSError:
            continue
    return None


def project_anchor() -> Path:
    """The directory cash treats as "here" -- for the DEFAULT cache location
    and for finding ``pyproject.toml``.

    Both used to be resolved from ``os.getcwd()``, which made the cache a
    property of where you were standing rather than of what you were running.
    Run the same script from a different directory -- a cron job, a CI step, a
    colleague -- and the entire cache was silently discarded and a second one
    built: measured at ``6 of 6 restored`` dropping to ``0 of 6``, a fresh 232MB
    ``.cash``, no warning, indistinguishable from a cold run. Three separate
    round-15 projects hit it, one of them writing a ``.cash`` at the drive root
    because the cwd
    happened to be the drive root. The documented escape hatch --
    ``[tool.cash] cache_dir`` in ``pyproject.toml`` -- was found the same broken
    way, so it did not work in exactly the case that needed it.

    The anchor walks up from the RUNNING SCRIPT to its project root, so
    ``python /srv/etl/run.py`` uses the same cache from anywhere on the machine.
    Without a script (a notebook, a REPL) or without a project marker above it,
    the answer is the cwd or the script's own directory respectively -- both
    stable for the case they describe.

    When the program itself is INSTALLED code -- ``pytest``, ``cash``, a
    ``python -m`` module in site-packages -- there is no script of the user's to
    anchor to, but there is usually a project the user is standing in, and the
    code being cached is that project's. So it walks up from the cwd instead.
    That puts a test suite's cache beside its project whichever way ``pytest``
    was typed and from whichever subdirectory, where round 17 found one
    per-user cache shared by every project on the machine.
    """
    start = _running_script_dir()
    if start is None:
        if _running_installed_code():
            root = _cwd_project_root()
            if root is not None:
                return root
        return Path.cwd()
    for d in [start, *start.parents]:
        if any((d / marker).exists() for marker in _PROJECT_MARKERS):
            return d
    return start


def _running_console_script() -> str | None:
    """The name of the installed entry point being run, if that is what this is.

    A ``[project.scripts]`` console script lives in the interpreter's own
    ``bin`` / ``Scripts`` directory, so it has no project to anchor to and
    ``_running_script_dir`` correctly returns None for it -- leaving it on the
    cwd, which means a `pip install`ed tool drops a fresh ``.cash`` in every
    directory you happen to run it from, and never reuses one. A round-16
    tester reported that as blocking.

    Detected from ``sys.argv[0]`` rather than from the absence of an anchor,
    because that absence also covers a notebook, a REPL and ``python -c``,
    where the cwd is the right answer and always was.

    Deliberately NOT ``python -m tool``: its ``argv[0]`` is a module path
    inside site-packages, so it looks similar, but the invocation is a
    developer standing in a project far more often than it is an installed
    tool -- ``python -m pytest`` most of all. That shape keeps today's
    behaviour.

    Generic on purpose: it names ANY launcher in the script directory,
    ``pytest`` and ``cash`` included. Deciding what that means is
    ``_installed_entry_point_cache_dir``'s job.
    """
    argv0 = sys.argv[0] if sys.argv else None
    if not argv0:
        return None
    try:
        path = Path(argv0).resolve()
    except OSError:
        return None
    script_dirs = {Path(sys.prefix) / d for d in ("bin", "Scripts")}
    script_dirs |= {Path(sys.base_prefix) / d for d in ("bin", "Scripts")}
    if path.parent not in script_dirs:
        return None
    name = re.sub(r"[^A-Za-z0-9._-]", "-", path.stem).strip("-.")
    return name or None


def _per_user_cache_root() -> Path:
    """The platform's own place for caches, where a cache survives ``cd``."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")     # not APPDATA: caches do not roam
        if base:
            return Path(base) / "cash"
        return Path.home() / "AppData" / "Local" / "cash"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "cash"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "cash"
    return Path.home() / ".cache" / "cash"


def _installed_entry_point_cache_dir() -> Path | None:
    """Where an installed console script should cache, or None if not one.

    Per tool, under the platform cache root, so two installed tools do not
    share one directory and neither inherits the other's eviction pressure.

    Only reached when nothing else claimed ``cache_dir``: an explicit setting
    of any kind wins, and so does a project ``pyproject.toml`` found by walking
    up from the cwd -- which is how a tool run inside a project that declares
    ``[tool.cash] cache_dir`` still caches beside that project's code. The
    per-user location is the answer for "nothing here claims this run", not a
    blanket override.

    Two refinements from round 17, where every tester hit the first:

    * **Never for ``cash`` itself.** cash's own CLI is an installed console
      script too, so it resolved a per-user ``…/cash/cash`` that nothing writes
      to -- ``cash inspect`` found nothing and ``cash clear --all`` "succeeded"
      while the real cache kept serving. The CLI resolves like the context it
      is run in; ``--tool NAME`` reaches an installed tool's cache.
    * **Not inside a project.** ``pytest`` is a console script as well, and
      took every project's test suite into one shared per-user cache. Any
      launcher run inside a project anchors to that project instead (see
      ``project_anchor``); the per-user location is for a tool run from
      somewhere no project claims -- a home directory, a scratch directory, a
      drive root.
    """
    name = _running_console_script()
    if name is None or name.lower() == "cash":
        return None
    if _cwd_project_root() is not None:
        return None
    try:
        return _per_user_cache_root() / name
    except (OSError, RuntimeError):           # no home directory to speak of
        return None


def _default_project_config_path() -> Path | None:
    """Walk upward from the project anchor to find a ``pyproject.toml``.

    The first directory containing one wins. None if we never find one (a
    standalone script with no project structure).
    """
    anchor = project_anchor()
    for d in [anchor, *anchor.parents]:
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
    if origin is _CALLER_RELATIVE or os.path.isabs(cache_dir):
        return cache_dir
    if not isinstance(origin, Path):
        return cache_dir
    resolved = os.path.normpath(str(origin / cache_dir))
    _warn_if_cache_moved(resolved, cache_dir)
    return resolved


#: One notice per process, whatever builds a config how many times.
_MOVE_NOTICE_GIVEN = False


def _warn_if_cache_moved(resolved: str, relative: str) -> None:
    """Say when the anchored default leaves an existing cwd cache behind.

    Anchoring the default is the fix for a cache that silently split in two;
    relocating somebody's 500MB cache without a word would be the same class of
    surprise in the other direction. Fires only when there is really something
    to leave behind: the new location does not exist yet, the old one does, and
    it holds entries.
    """
    global _MOVE_NOTICE_GIVEN
    if _MOVE_NOTICE_GIVEN:
        return
    try:
        previous = Path.cwd() / relative
        if os.path.normcase(str(previous)) == os.path.normcase(resolved):
            return
        if os.path.exists(resolved) or not previous.is_dir():
            return
        if not any(previous.iterdir()):
            return
    except OSError:
        return
    _MOVE_NOTICE_GIVEN = True
    try:
        from .diagnostics import warn_diagnostic
        from .exceptions import CashCacheIneffectiveWarning
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-DIR-MOVED",
            f"cash keeps this project's cache at {resolved}, next to the code, "
            f"rather than at {previous} -- the directory this process happens to "
            f"be running in. The cache already at {previous} will not be used, "
            f"so this run is a cold one.",
            f"nothing to do if you did not know that cache was there. To keep "
            f"using it, set CASH_CACHE_DIR={previous} or move it to {resolved}; "
            f"to be rid of it, delete it once this run has repopulated the new "
            f"location.",
        )
    except Exception:  # noqa: BLE001 - a notice must never break a config load
        logger.debug("Could not emit the cache-relocation notice", exc_info=True)



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
    from .notebook.file_tracker import untracked
    with untracked():
        return _resolve_config(
            config_path, user_config_path=user_config_path,
            project_config_path=project_config_path, overrides=overrides,
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
        config_path: Convenience override for per-script use (the form
            documented as ``Cash(config_path=...)``). When given, the
            file is treated as a user-level TOML and merged on top of
            ``user_config_path``.
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
        user_data = _validated_layer(
            _load_toml_config(Path(user_path)), str(user_path), strict=False)
        if user_data:
            _merge(merged, user_data)
            sources.append(f"user:{user_path}")
            if "cache_dir" in user_data:
                cache_dir_origin = Path(user_path).parent
                cache_dir_was_configured = True

    # Layer 2b: explicit ``Cash(config_path=...)`` override (merged on
    # top of the user-scoped layer)
    if config_path is not None:
        override_data = _validated_layer(
            _load_toml_config(Path(config_path)), str(config_path), strict=False)
        if override_data:
            _merge(merged, override_data)
            sources.append(f"file:{config_path}")
            if "cache_dir" in override_data:
                cache_dir_origin = Path(config_path).parent
                cache_dir_was_configured = True

    # Layer 3: project TOML
    if project_config_path is _USE_DEFAULT_PATH:
        project_path = _default_project_config_path()
    else:
        project_path = project_config_path
    if project_path is not None:
        project_data = _validated_layer(
            _load_toml_config(Path(project_path)), str(project_path), strict=False)
        if project_data:
            _merge(merged, project_data)
            sources.append(f"project:{project_path}")
            if "cache_dir" in project_data:
                cache_dir_origin = Path(project_path).parent
                cache_dir_was_configured = True

    # Layer 4: env vars
    env_data = _load_env_config()
    if env_data:
        _merge(merged, env_data)
        sources.append("env")
        if "cache_dir" in env_data:
            cache_dir_origin = _CALLER_RELATIVE
            cache_dir_was_configured = True

    # Layer 5: explicit overrides (kwargs)
    if overrides:
        overrides = _validated_layer(overrides, "Cash(...) arguments", strict=True)
        _merge(merged, overrides)
        sources.append("kwargs")
        if "cache_dir" in overrides:
            cache_dir_origin = _CALLER_RELATIVE
            cache_dir_was_configured = True

    if not cache_dir_was_configured:
        installed = _installed_entry_point_cache_dir()
        if installed is not None:
            merged["cache_dir"] = str(installed)
            cache_dir_origin = _CALLER_RELATIVE      # already absolute
            sources.append("entry-point")
    merged["cache_dir"] = _anchor_cache_dir(merged.get("cache_dir"), cache_dir_origin)

    # Materialise the dict into a CashConfig.
    return _build_config(merged, source=",".join(sources) if sources else "defaults")


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
            return bool(value)          # `debug=1` is ordinary code
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
            return value
    else:
        return value                    # lists, nested configs: checked elsewhere
    expected = getattr(base, "__name__", str(base))
    hint = " (or a size string such as '2GB')" if name in _SIZE_FIELDS else ""
    raise ValueError(
        f"{name}={value!r} is a {type(value).__name__}; expected {expected}{hint}")


def _validated_layer(data: dict[str, Any], label: str, *, strict: bool) -> dict[str, Any]:
    """*data* with every known field checked by `validate_value`.

    Unknown keys pass through untouched (``_build_config`` ignores them; a
    ``pyproject.toml`` read flat holds plenty). A bad value RAISES when the
    caller's own code supplied it (*strict*) -- that is a bug at the call site,
    and the place to say so -- and is logged and dropped when it came from a
    file or the environment, which must not stop a program from running.
    """
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    tier_valid = {f.name for f in fields(TierConfig)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        try:
            if key == "tiers" and isinstance(value, list):
                out[key] = [
                    {k: (validate_value(k, v, TierConfig) if k in tier_valid else v)
                     for k, v in t.items()} if isinstance(t, dict) else t
                    for t in value
                ]
            elif key in valid:
                out[key] = validate_value(key, value)
            else:
                out[key] = value
        except ValueError as exc:
            if strict:
                raise ValueError(f"cash config ({label}): {exc}") from None
            logger.warning("Ignoring invalid cash setting in %s: %s", label, exc)
    return out


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
# Leave unset (the default) to auto-scale the cap to the machine — a
# fraction of free disk for the disk tier, a fraction of RAM for the
# memory tier. Uncomment to pin the disk cap explicitly, e.g. 5 GiB:
# max_cache_size = 5368709120

# Persist every notebook statement, bypassing the cost-aware floors
# (same as putting # @cash:persist on each statement). Off by default;
# also flippable at runtime via the %cash_persist magic.
persist_all = false

# Smart persistence — only promote past RAM when the compute was slow
# enough to be worth disk/network I/O for.
smart_persistence = true

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
