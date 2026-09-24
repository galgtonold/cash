"""The default ``Cash`` instance, and the config and mode of the code
running now.

Modules below ``core`` need these answers (a file snapshot needs the size
above which it samples a file, a remote source its revalidation window, a
data source whether a warning may fire), and they ask here rather than
importing the ``cash`` package or ``core``, which import them.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

from cash.config import get_config

if TYPE_CHECKING:
    from cash.config import CashConfig
    from cash.core import Cash

__all__ = ["ACTIVE_CONFIG", "EXPLAINING", "active_config", "default_cash", "set_default_cash"]

#: The config of the `Cash` instance whose call is running, set by its
#: wrapper, so the settings of a `Cash(...)` of your own are the ones its
#: calls follow, not the default instance's.
ACTIVE_CONFIG: contextvars.ContextVar[CashConfig | None] = contextvars.ContextVar("cash_active_config", default=None)

#: Set while ``explain()`` builds a key: the same steps a real call takes, with
#: every warning they would give held back, because inspecting a call must not
#: warn. ``Notices.warn_once``, the carrier warnings and ``state_token_of`` check
#: it.
EXPLAINING: contextvars.ContextVar[bool] = contextvars.ContextVar("_cash_explaining", default=False)

_default: Cash | None = None


def default_cash() -> Cash | None:
    """The instance behind ``cash.cache`` and ``cash.configure``, or None
    while nothing has used it. Never creates one (``cash._get_global_cash``
    does)."""
    return _default


def set_default_cash(instance: Cash | None) -> None:
    """Make *instance* the default; None lets the next use build a fresh one."""
    global _default
    _default = instance


def active_config() -> CashConfig:
    """The settings that govern the code running now.

    The running call's instance first. Outside a call, the default
    instance's config, which ``cash.configure(...)`` updates in place. Before
    any default instance exists, a fresh merge of env and TOML
    (``get_config``), which reads the config files on each call.
    """
    config = ACTIVE_CONFIG.get()
    if config is not None:
        return config
    if _default is not None:
        return _default.config
    return get_config()
