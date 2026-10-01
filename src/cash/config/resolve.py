"""Merging every configuration layer into one `CashConfig` (`get_config`).

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

import dataclasses
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from .._location import (
    default_project_config_path,
    default_user_config_path,
    installed_entry_point_cache_dir,
    project_anchor,
)
from ..tracking.tracker_context import untracked
from .notices import config_notice, did_you_mean
from .schema import TIER_TYPES, CashConfig, TierConfig, validate_value
from .sources import TOML_SECTION, load_env_config, load_toml_layer

__all__ = ["get_config", "validated_overrides"]


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
        data, found = load_toml_layer(Path(path))
        files.append((layer, str(path), found))
        return _validated_layer(data, str(path), strict=False, unknown_keys=found == TOML_SECTION)

    if config_path is not None and not Path(config_path).exists():
        # Named in code, so it was meant to exist: a tool that forgot to ship
        # its config file would otherwise run on defaults without a word.
        config_notice(
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
    env_data = load_env_config()
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
                    if tiers[-1].get("type") not in TIER_TYPES:
                        raise ValueError(
                            f"tiers[{i}].type={t.get('type')!r}: not one of {', '.join(sorted(TIER_TYPES))}"
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
            config_notice(
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
    return f"`{shown}` is not a cash setting.{_NOT_SETTINGS.get(key) or did_you_mean(key, valid)}"


def _unknown_key(label: str, shown: str, key: str, valid: Any) -> None:
    config_notice(
        "CONFIG-UNKNOWN-KEY",
        f"{label} sets `{shown}`, which is not a cash setting, so it does nothing.{did_you_mean(key, valid)}",
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
            config_notice(
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
