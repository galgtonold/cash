"""Reading settings from a config file and from ``CASH_*`` variables.

Each reader returns a plain dict of what its source sets, and reports what it
cannot read (CONFIG-INVALID, CONFIG-UNKNOWN-KEY). Merging the layers is
`cash.config.resolve`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from .notices import config_notice, did_you_mean
from .schema import CashConfig, TierConfig, check_choice, coerce, declared_type

__all__ = [
    "TOML_MISSING",
    "TOML_NOT_CASH",
    "TOML_SECTION",
    "TOML_UNREADABLE",
    "load_env_config",
    "load_toml_layer",
]


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


#: What `load_toml_layer` found. A ``[tool.cash]`` or ``[cash]`` table holds
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
    return load_toml_layer(path)[0]


def load_toml_layer(path: Path) -> tuple[dict[str, Any], str]:
    """`_load_toml_config`, plus which of the ``TOML_*`` outcomes it was."""
    if not path.exists():
        return {}, TOML_MISSING
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # cash depends on tomli there
        import tomli as tomllib

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
        config_notice(
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
    config_notice("CONFIG-INVALID", what, fix)


_TIER_ENV_RE = re.compile(r"^CASH_TIER_(\d+)_(.+)$")


def load_env_config() -> dict[str, Any]:
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
                config_notice(
                    "CONFIG-UNKNOWN-KEY",
                    f"the environment sets {env_key}, and `{field_name}` is not a "
                    f"tier setting, so it does nothing."
                    f"{did_you_mean(field_name, tier_field_names)}",
                    "rename or unset it; the tier settings are listed under Configuration > Tiers in the docs.",
                )
                continue
            try:
                value = coerce(declared_type(field_name, TierConfig), raw, field_name)
                check_choice(field_name, value)
            except ValueError as e:
                config_notice(
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
            value = coerce(declared_type(key), raw, key)
            check_choice(key, value)
        except ValueError as e:
            config_notice(
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
