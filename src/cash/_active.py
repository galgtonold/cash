"""The default ``Cash`` instance, and the config and mode of the code
running now.

Modules below ``core`` need these answers (a file snapshot needs the size
above which it samples a file, a remote source its revalidation window, a
data source whether a warning may fire), and they ask here rather than
importing the ``cash`` package or ``core``, which import them.

It also hands the default instance's runtime settings (``cash.configure``,
``cash.disabled``) to the worker processes started from here
(`publish_settings`).
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any

from cash.config import get_config

if TYPE_CHECKING:
    from cash.config import CashConfig
    from cash.core import Cash

__all__ = [
    "ACTIVE_CONFIG",
    "EXPLAINING",
    "active_config",
    "default_cash",
    "publish_settings",
    "set_default_cash",
]

logger = logging.getLogger(__name__)

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
    """Make *instance* the default; None lets the next use build a fresh one.

    The settings the replaced default was given at runtime stop reaching new
    worker processes. A worker's first default takes the settings its parent
    had changed (`publish_settings`).
    """
    global _default
    previous, _default = _default, instance
    if previous is instance:
        return
    if previous is not None:
        _forget_settings()
    elif instance is not None and _inherited:
        _apply_inherited(instance, _inherited)


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


# ---------------------------------------------------------------------------
# Runtime settings in worker processes
# ---------------------------------------------------------------------------
#
# A pool worker builds its own default instance from the environment and the
# config files, so what ``cash.configure(...)`` and ``cash.disabled()`` changed
# in the parent has to be carried to it. ``multiprocessing`` already carries
# per-process settings to every child it starts: each ``Process`` copies its
# creator's ``_config`` dict when it is created. A forked child has that copy
# in its memory, together with the parent's default instance. A spawned or
# forkserver child (and a joblib/loky worker) receives the process object
# pickled, ``_config`` included, and unpickling the settings stored there
# (`_WorkerSettings`) applies them to the child's default instance -- after
# the child imported the main script, before the pool's task loop starts.
#
# Environment variables would not do: a forkserver child is forked from the
# server, so it has the server's environment from when that started, not the
# parent's now -- a setting would miss the workers of an early forkserver and
# stick to every worker of one started inside a ``disabled()`` block.

#: The key in ``multiprocessing.current_process()._config``.
_SETTINGS_KEY = "cash_runtime_settings"

_settings_lock = threading.Lock()

#: The settings this process was started with by its parent, if any, for a
#: default instance built after they arrived.
_inherited: dict[str, Any] | None = None


class _WorkerSettings(dict):
    """The default instance's settings changed at runtime, ``{name: value}``,
    as a parent hands them to the processes it starts."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (_arrive, (dict(self),))


def _arrive(settings: dict[str, Any]) -> _WorkerSettings:
    """Unpickled in a child process: apply its parent's settings."""
    global _inherited
    _inherited = dict(settings)
    if _default is not None:
        _apply_inherited(_default, _inherited)
    return _WorkerSettings(settings)


def _apply_inherited(instance: Cash, settings: dict[str, Any]) -> None:
    from cash.reconfigure import apply_overrides  # reconfigure imports the backends, which import this module

    try:
        apply_overrides(instance, dict(settings))
    except Exception as exc:  # noqa: BLE001 - the worker still runs, on its own settings
        logger.warning("cash could not apply its parent process's settings %s in this worker: %s", settings, exc)


def _portable(name: str, value: Any) -> Any:
    """*value* as a child process started from here should apply it."""
    if name == "cache_dir" and isinstance(value, str) and value:
        return os.path.abspath(value)  # the child may start in another directory
    if isinstance(value, list):
        return list(value)
    return value


def publish_settings(config: CashConfig, names: Any) -> None:
    """Hand the default instance's *names* settings, as *config* now has them,
    to every worker process started from here on.

    Called whenever the default instance is reconfigured, so the values are
    the ones in force: the end of a ``disabled()`` block publishes the value it
    restores. A worker keeps the settings it started with.
    """
    import multiprocessing  # here, not at the top: `import cash` must not load it

    store = getattr(multiprocessing.current_process(), "_config", None)
    if not isinstance(store, dict):
        return
    with _settings_lock:
        settings = _WorkerSettings(store.get(_SETTINGS_KEY) or {})
        for name in names:
            settings[name] = _portable(name, getattr(config, name))
        # A new dict each time: a process created earlier keeps the copy it took.
        store[_SETTINGS_KEY] = settings


def _forget_settings() -> None:
    global _inherited
    _inherited = None
    mp = sys.modules.get("multiprocessing")
    store = getattr(mp.current_process(), "_config", None) if mp is not None else None
    if isinstance(store, dict):
        with _settings_lock:
            store.pop(_SETTINGS_KEY, None)
