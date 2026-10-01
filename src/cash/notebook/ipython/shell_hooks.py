"""How ``CashMagics`` hooks into an IPython shell, and unhooks a previous one.

The shell keeps what the last instance installed in ``shell._cash_hooks``
(:func:`stash_hooks`), so a re-instantiation in a running kernel
(``cash.reset_session()``, a second ``Cash()``, ``%load_ext`` after a reset)
can take it out again first (:func:`remove_previous_hooks`). Without that it
would capture an already-wrapped ``run_cell`` as its "original", nesting
wrappers on every reset, and stack duplicate event handlers.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

__all__ = ["register_event", "remove_previous_hooks", "signature_preserving_proxy", "stash_hooks"]

logger = logging.getLogger(__name__)


def remove_previous_hooks(shell: Any) -> None:
    """Undo what an earlier instance installed on *shell*, as it stashed it."""
    prior = getattr(shell, "_cash_hooks", None)
    if not isinstance(prior, dict):
        return
    try:
        shell.events.unregister("pre_run_cell", prior["capture_cell_id"])
    except (ValueError, KeyError, AttributeError, TypeError):
        pass
    try:
        shell.run_cell = prior["original_run_cell"]
    except (KeyError, AttributeError):
        pass
    # The async entry point too, so a re-patch captures the true-original
    # run_cell_async rather than nesting a wrapper on every reset.
    if "original_run_cell_async" in prior:
        try:
            shell.run_cell_async = prior["original_run_cell_async"]
        except (KeyError, AttributeError):
            pass
    if prior.get("flush_pending_writes") is not None:
        try:
            shell.events.unregister("post_run_cell", prior["flush_pending_writes"])
        except (ValueError, KeyError, AttributeError, TypeError):
            pass


def register_event(shell: Any, event: str, callback: Callable[..., Any], failure: str) -> None:
    """Register *callback* for *shell*'s *event*; log *failure* (a ``%s``
    format for the error) as a warning when the shell will not take it."""
    try:
        shell.events.register(event, callback)
    except (AttributeError, TypeError) as e:
        logger.warning(failure, e)


def signature_preserving_proxy(
    owner: Any,
    original: Any,
    handler_name: str,
    is_async: bool = False,
) -> Any:
    """Wrap *original* with a proxy that dispatches to ``owner.<handler_name>``.

    The proxy forwards ``*args, **kwargs`` verbatim but, thanks to
    ``functools.wraps``, presents *original*'s signature to
    ``inspect.signature`` (via ``__wrapped__``). That is load-bearing:
    ipykernel introspects ``run_cell`` to decide what to pass
    (``_accepts_parameters(run_cell, ["cell_id"])``) and treats a ``**kwargs``
    signature as "accepts every parameter". A bare ``(*args, **kwargs)``
    proxy would claim to accept ``cell_id`` even against an IPython too old
    to have it (<8.3), ipykernel would pass it, the forward would raise
    TypeError before ``execute_reply`` was sent, and the cell would hang at
    ``[*]``. With the original's signature, every verdict about the proxy is
    the verdict about the shell it replaced.

    The handler is resolved by name **at call time** rather than captured, so
    tests can swap ``owner._execute_cell`` out and still be routed through.
    """
    if is_async:

        @functools.wraps(original)
        async def proxy(*args: Any, **kwargs: Any) -> Any:
            return await getattr(owner, handler_name)(*args, **kwargs)
    else:

        @functools.wraps(original)
        def proxy(*args: Any, **kwargs: Any) -> Any:
            return getattr(owner, handler_name)(*args, **kwargs)

    return proxy


def stash_hooks(
    shell: Any,
    *,
    original_run_cell: Any,
    capture_cell_id: Callable[..., Any],
    flush_pending_writes: Callable[..., Any],
    original_run_cell_async: Any = None,
) -> None:
    """Keep on *shell* what :func:`remove_previous_hooks` needs to undo."""
    try:
        shell._cash_hooks = {
            "original_run_cell": original_run_cell,
            "capture_cell_id": capture_cell_id,
            "flush_pending_writes": flush_pending_writes,
        }
        if original_run_cell_async is not None:
            shell._cash_hooks["original_run_cell_async"] = original_run_cell_async
    except (AttributeError, TypeError):
        pass
