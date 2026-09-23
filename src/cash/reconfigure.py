"""Changing a running `Cash` instance's settings (``cash.configure``).

Every setting is applied to the instance's config in place. The backend is
rebuilt only when the tiers the config describes changed (`tier_specs`), so a
setting no tier uses -- ``redis_host`` on a RAM + disk stack -- is stored for
later and rebuilds nothing, and the RAM tier survives every change that does
not concern it.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from .backends.factory import apply_persistence_settings, build_backend_from_config, tier_specs
from .config import CashConfig, validate_value

if TYPE_CHECKING:
    from .core import Cash

logger = logging.getLogger(__name__)

__all__ = ["apply_overrides"]


def apply_overrides(cash: Cash, overrides: dict[str, Any]) -> None:
    """Validate *overrides*, apply them to ``cash.config``, and rebuild the
    backend if the tiers it describes changed.

    Raises ``ValueError`` before changing anything if a key is not a setting
    or a value is not valid for it.
    """
    valid = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"{sorted(unknown)!r} is not a configurable field. Valid keys: {sorted(valid)!r}")
    checked = {key: (val if key == "tiers" else validate_value(key, val)) for key, val in overrides.items()}

    before = tier_specs(cash.config)
    for key, val in checked.items():
        setattr(cash.config, key, val)

    running = cash.backend_if_built
    if running is None:
        return  # built from the config on first use
    if tier_specs(cash.config) == before:
        if "min_cache_savings_pct" in checked:
            apply_persistence_settings(running, cash.config)
        return
    try:
        running.shutdown()  # drain its pending writes
    except Exception as e:  # noqa: BLE001 - the new backend must still be swapped in
        logger.warning("Old backend shutdown failed during configure(): %s", e)
    cash.backend = build_backend_from_config(cash.config)
