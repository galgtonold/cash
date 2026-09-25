"""Changing a running `Cash` instance's settings (``cash.configure``).

Every setting is checked the way ``Cash(**overrides)`` checks it
(`validated_overrides`) before any is applied, then applied to the instance's
config in place. The backend is rebuilt only when the tiers the config
describes changed (`tier_specs`), so a setting no tier uses -- ``redis_host``
on a RAM + disk stack -- is stored for later and rebuilds nothing, and the RAM
tier survives every change that does not concern it. A backend the caller
passed in as an object is never rebuilt.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from . import _active, _log
from .backends.factory import apply_persistence_settings, build_backend_from_config, built_from_config, tier_specs
from .config import validated_overrides

if TYPE_CHECKING:
    from .core import Cash

logger = logging.getLogger(__name__)

__all__ = ["apply_overrides"]


def apply_overrides(cash: Cash, overrides: dict[str, Any]) -> None:
    """Validate *overrides*, apply them to ``cash.config``, and rebuild the
    backend if the tiers it describes changed.

    Raises ``ValueError`` before changing anything if a key is not a setting,
    a value is not valid for it, or it would change the tiers of a backend
    the caller supplied. A running backend's replacement is built before
    anything changes too, so a configuration that cannot be built (an S3
    tier without a bucket, a Redis tier without the ``redis`` package) raises
    and leaves the old settings and the old backend working.
    """
    checked = validated_overrides(overrides)

    proposed = dataclasses.replace(cash.config, **checked)
    tiers_change = tier_specs(proposed) != tier_specs(cash.config)
    running = cash.backend_if_built
    if tiers_change and running is not None and not built_from_config(running):
        raise ValueError(
            f"cash.configure({', '.join(sorted(checked))}=...) changes the storage tiers, but this "
            f"Cash was given its backend ({type(running).__name__}) as an object, and cash does not "
            f"rebuild a backend it did not build. Configure that backend, or pass a new one."
        )
    # Built from a copy before the config is touched: if it raises, nothing
    # has changed and the running backend keeps serving. Building is cheap --
    # a backend creates its directory and threads on first use.
    replacement = build_backend_from_config(proposed) if tiers_change and running is not None else None

    for key, val in checked.items():
        setattr(cash.config, key, val)
    if cash is _active.default_cash():
        _active.publish_settings(cash.config, checked)
    if "debug" in checked or "verbose" in checked:
        debug, verbose = cash.config.debug, cash.config.verbose
        _log.follow(logging.DEBUG if debug else logging.INFO if verbose else None)

    if running is None:
        return  # built from the config on first use
    if replacement is None:
        if "min_cache_savings_pct" in checked:
            apply_persistence_settings(running, cash.config)
        return
    try:
        running.shutdown()  # drain its pending writes
    except Exception as e:  # noqa: BLE001 - the new backend must still be swapped in
        logger.warning("Old backend shutdown failed during configure(): %s", e)
    cash.backend = replacement
