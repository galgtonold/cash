"""Observe side effects a cached function actually performs, on its first call.

Static analysis stops at library boundaries -- that is deliberate, since
folding every installed package into the walk would be both slow and useless.
The cost is that a side effect *inside* a library is reachable only by the
method's NAME (``session.post``, ``cur.execute``; see
``notebook.purity._WRITE_METHODS``), and a name cannot reach everything:
``session.get(...)`` collides with ``dict.get``, and an arbitrary vendor
function like ``client.emit_metric(...)`` has no effect-shaped name at all.

Measured, before this module existed: of 24 real side effects planted across a
decorator-cached function, 21 were flagged statically. Every miss was inside an
installed library, and the two that survived even a widened name list were a
network read through a client object and a vendor function whose return value
was used.

This closes that class from the other side. While the body of a *missing* call
runs, cash watches for effects it can see regardless of where the code lives:

* a file opened for writing (recorded by :class:`FileAccessTracker`, which
  already wraps ``open`` for read-tracking and now keeps the write side too)
* an outbound socket connection
* a subprocess being spawned

If any fire and the analyzer said nothing, cash warns once -- because the
second call will not do them. That is the whole hazard: the effect happened,
the value looks right, and nothing will tell you the effect stopped.

**Why this cannot replace the static pass.** It only sees what the first call
happens to do. A branch not taken performs no effect, so silence here is not
proof of purity -- it is one observation. The static analyzer reasons about
code that was never run, which is a different and complementary guarantee.

**What it deliberately does not do:**

* It does not block, refuse, or un-cache anything. By the time an effect is
  observed the function has already run and its result is already worth
  storing; refusing the entry would cost the user the compute and prevent
  nothing. Even under ``strict=True`` this warns rather than raising: raising
  after the effect has landed would discard a correct result to report
  something the raise could not have prevented.
* It does not watch other threads. Dispatch is via ``ContextVar``, so an
  effect started on a worker thread is not attributed to the caller -- both
  because attribution would be wrong and because a background thread of cash's
  own (the tiered backend's writer) must never be mistaken for the user's
  code. The static pass already flags the ordinary ``Thread(target=...)``
  shape.
"""
from __future__ import annotations

import contextvars
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: The observer whose block is currently executing, per thread and per
#: asyncio Task. Mirrors ``file_tracker._active_tracker`` on purpose: same
#: install-once-dispatch-dynamically shape, same isolation properties.
_active_observer: contextvars.ContextVar["EffectObserver | None"] = (
    contextvars.ContextVar("_cash_active_observer", default=None)
)

#: Patches are installed at most once per process and never removed. With no
#: active observer each wrapper is one ``ContextVar.get()`` and an ``is None``
#: test before delegating, which is why leaving them in place costs nothing.
_PATCHED = False


def _record(kind: str, detail: str) -> None:
    observer = _active_observer.get()
    if observer is not None:
        observer.record(kind, detail)


def _install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import socket
    import subprocess

    original_connect = socket.socket.connect

    # Both wrappers are signature-TRANSPARENT (`*a, **kw`) and forward
    # unchanged. Naming a parameter would break any caller that passes it by
    # keyword, and these sit on paths -- kernel launch, every outbound
    # connection -- where a signature mismatch is a hard failure a long way
    # from here. Recording is also wrapped: observing an effect must never be
    # able to break the call that performed it.
    def _tracked_connect(self, *a, **kw):
        try:
            _record("network", f"socket connect to {_describe_address(a[0] if a else None)}")
        except Exception:                                    # noqa: BLE001
            pass
        return original_connect(self, *a, **kw)

    original_popen_init = subprocess.Popen.__init__

    def _tracked_popen_init(self, *a, **kw):
        try:
            _record("subprocess", f"spawned {_describe_argv(a[0] if a else kw.get('args'))}")
        except Exception:                                    # noqa: BLE001
            pass
        return original_popen_init(self, *a, **kw)

    for owner, name, wrapper, original in (
        (socket.socket, "connect", _tracked_connect, original_connect),
        (subprocess.Popen, "__init__", _tracked_popen_init, original_popen_init),
    ):
        try:
            wrapper._cash_effect_patch = True          # type: ignore[attr-defined]
            wrapper._original_func = original          # type: ignore[attr-defined]
            setattr(owner, name, wrapper)
        except (AttributeError, TypeError) as exc:
            # A hardened runtime may refuse to patch a builtin type. Degrading
            # to "this effect class is unobserved" is correct: the static pass
            # still runs and nothing else changes.
            logger.debug("[EFFECTS] could not patch %s.%s: %s", owner, name, exc)


def _describe_address(address: Any) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)[:80]


def _describe_argv(args: Any) -> str:
    if isinstance(args, (list, tuple)) and args:
        return str(args[0])[:80]
    return str(args)[:80]


class EffectObserver:
    """Records side effects performed on this context while the block runs.

    Usage mirrors :class:`FileAccessTracker`::

        observer = EffectObserver(exclude_under=cache_dir)
        with observer:
            result = func(*args, **kwargs)
        observer.summary()   # None when nothing was observed
    """

    def __init__(self, exclude_under: str | None = None) -> None:
        self.effects: list[tuple[str, str]] = []
        # cash's own cache directory. A write in there is cash storing the
        # entry, not the user's function doing I/O, and reporting it would
        # make every cached function look impure.
        self._exclude = os.path.abspath(exclude_under) if exclude_under else None
        self._tokens: list[contextvars.Token] = []

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "EffectObserver":
        _install_patches()
        self._tokens.append(_active_observer.set(self))
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        if self._tokens:
            _active_observer.reset(self._tokens.pop())
        return False

    def suspend(self):
        """Stop observing until :meth:`resume`. Mirrors `FileAccessTracker`."""
        return _active_observer.set(None)

    def resume(self, token) -> None:
        _active_observer.reset(token)

    # -- recording ---------------------------------------------------------
    def record(self, kind: str, detail: str) -> None:
        if len(self.effects) >= 8:      # a summary, not a log
            return
        self.effects.append((kind, detail))

    def record_write(self, path: Any) -> None:
        """Record a file opened for writing, unless it is cash's own storage."""
        try:
            resolved = os.path.abspath(os.fspath(path))
        except (TypeError, ValueError):
            return
        if self._exclude and resolved.startswith(self._exclude):
            return
        self.record("file write", resolved)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str | None:
        """One line per distinct effect, or ``None`` when nothing was seen."""
        if not self.effects:
            return None
        seen: list[str] = []
        for kind, detail in self.effects:
            line = f"  {kind}: {detail}"
            if line not in seen:
                seen.append(line)
        return "\n".join(seen)
