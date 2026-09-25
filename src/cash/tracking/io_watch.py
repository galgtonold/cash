"""The one layer through which cash watches a process's I/O.

Three things want to see what running code does: the file tracker (which
files a cached call or a notebook statement reads), the effect observer
(whether a first call wrote a file, connected somewhere or started a process)
and the notebook's write observer (which files a statement produced). They
share this module instead of each installing their own interception.

**Audit events first.** ``sys.addaudithook`` reports every Python-level
``open`` (``builtins.open``, ``io.open``, ``pathlib``, ``os.open``, and C code
that opens through ``io``), directory listings (``os.listdir``,
``os.scandir``, ``glob.glob``), ``socket.connect`` and ``subprocess.Popen``,
whoever calls them and however they were imported. The hook is installed once
and cannot be removed, so it is gated: it dispatches only the events some
consumer currently needs, and one dictionary lookup decides that for every
other audited event in the process.

**Monkeypatches only where no event exists.** Readers that open files in
C/C++/Rust (pyarrow, polars, sqlite3, some pandas readers), existence probes
and ``Path.stat`` (``os.stat`` raises no event), and the executor ``submit``
methods that carry a tracker into worker threads and processes. Those are
installed when the first observation scope opens and the originals are put
back when the last one closes; ``%cash_on`` holds a scope for as long as it is
on, so a notebook does not reinstall them for every statement.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["Patches", "add_patcher", "hold", "holding", "release", "subscribe", "watch_outside_scopes"]

#: A consumer receives the event's argument tuple. The Python frame that made
#: the audited call is ``sys._getframe(CALLER_DEPTH)`` from inside it.
Consumer = Callable[[tuple], None]
CALLER_DEPTH = 2

_lock = threading.RLock()
#: event -> [(consumer, also outside scopes?)], in subscription order.
_subscribers: dict[str, list[tuple[Consumer, bool]]] = {}
#: (install, remove) pairs, run when the first scope opens / the last closes.
_patchers: list[tuple[Callable[[], None], Callable[[], None]]] = []
_holds = 0
_outside = False
_hook_installed = False
#: What the hook dispatches: one of the two tables below, rebuilt whenever
#: the subscriptions change. Rebinding a global is atomic, so the hook never
#: sees a half-built table.
_active: dict[str, tuple[Consumer, ...]] = {}
_held_table: dict[str, tuple[Consumer, ...]] = {}
_idle_table: dict[str, tuple[Consumer, ...]] = {}


def _hook(event: str, args: tuple) -> None:
    # Runs for every audited event in the process (every `id()`, every import,
    # every unpickled class): one dict lookup unless a consumer wants it.
    consumers = _active.get(event)
    if consumers is None:
        return
    for consumer in consumers:
        try:
            consumer(args)
        except Exception:  # noqa: BLE001 - an exception here would fail the user's own call
            pass


def _rebuild() -> None:
    global _held_table, _idle_table
    _held_table = {event: tuple(consumer for consumer, _ in subs) for event, subs in _subscribers.items() if subs}
    idle: dict[str, tuple[Consumer, ...]] = {}
    for event, subs in _subscribers.items():
        chosen = tuple(consumer for consumer, outside in subs if outside and _outside)
        if chosen:
            idle[event] = chosen
    _idle_table = idle
    _switch()


def _switch() -> None:
    global _active
    _active = _held_table if _holds else _idle_table


def _ensure_hook() -> None:
    global _hook_installed
    if not _hook_installed:
        sys.addaudithook(_hook)
        _hook_installed = True


def subscribe(event: str, consumer: Consumer, *, outside_scopes: bool = False) -> None:
    """Deliver *event* to *consumer* while a scope is open.

    With *outside_scopes*, also after :func:`watch_outside_scopes`, with no
    scope open: the file tracker notes reads made before any cached call.
    """
    with _lock:
        _subscribers.setdefault(event, []).append((consumer, outside_scopes))
        _rebuild()


def add_patcher(install: Callable[[], None], remove: Callable[[], None]) -> None:
    """Run *install* when observation starts and *remove* when it stops."""
    with _lock:
        _patchers.append((install, remove))
        if _holds:
            install()


def watch_outside_scopes() -> None:
    """Keep delivering the ``outside_scopes`` events with no scope open."""
    global _outside
    with _lock:
        _ensure_hook()
        if not _outside:
            _outside = True
            _rebuild()


def hold() -> None:
    """Open an observation scope: the first one installs everything."""
    global _holds
    with _lock:
        _ensure_hook()
        _holds += 1
        if _holds == 1:
            try:
                for install, _remove in _patchers:
                    install()
            finally:
                _switch()


def release() -> None:
    """Close a scope: the last one puts the patched originals back."""
    global _holds
    with _lock:
        if _holds == 0:
            return
        _holds -= 1
        if _holds == 0:
            _switch()
            if sys.is_finalizing():
                # A scope left open until exit -- a cached generator still
                # suspended mid-stream -- closes while the interpreter tears
                # its modules down (``sys.meta_path`` is already None). Nothing
                # runs afterwards that the patches could mislead, and putting
                # the originals back printed a traceback at exit.
                return
            for _install, remove in reversed(_patchers):
                remove()


def holding() -> bool:
    """Is any observation scope open?"""
    return _holds > 0


_ABSENT = object()


class Patches:
    """Attribute replacements that can all be undone.

    A replacement is put back only while it is still the one installed: if
    something wrapped it since (``mock.patch``, another tool), the wrapper
    stays and keeps dispatching to cash's, which does nothing with no scope
    open.
    """

    __slots__ = ("_installed", "_last", "version")

    def __init__(self) -> None:
        #: Moves whenever what is installed changes.
        self.version = 0
        self._installed: list[tuple[Any, str, Any, Any]] = []
        #: What the last `restore` put back, in install order (see `reinstall`).
        self._last: list[tuple[Any, str, Any, Any]] = []

    def replace(self, owner: Any, name: str, new: Any) -> bool:
        """Set ``owner.name`` to *new*, remembering what was there."""
        try:
            if isinstance(owner, type):
                # The class's own attribute, raw: a staticmethod stays one, and
                # an inherited one is deleted again rather than copied down.
                old = owner.__dict__.get(name, _ABSENT)
            else:
                old = getattr(owner, name)
            setattr(owner, name, new)
        except (AttributeError, TypeError):
            return False
        self._installed.append((owner, name, new, old))
        self.version += 1
        return True

    def installed(self) -> list[tuple[Any, str, Any, Any]]:
        """``(owner, name, replacement, original)`` for each replacement in place."""
        return list(self._installed)

    def restore(self) -> None:
        """Put back every original that is still covered by cash's replacement."""
        installed, self._installed = self._installed, []
        self.version += 1
        restored = []
        for entry in reversed(installed):
            owner, name, new, old = entry
            try:
                if _current(owner, name) is not new:
                    continue
                if old is _ABSENT:
                    delattr(owner, name)
                else:
                    setattr(owner, name, old)
            except (AttributeError, TypeError):
                continue
            restored.append(entry)
        restored.reverse()
        self._last = restored

    def reinstall(self) -> bool:
        """Set again what the last `restore` took down, if every original is
        still in place; False (and nothing done) otherwise.

        The cheap path for the next scope: building the same replacements
        again means looking every reader up anew.
        """
        last = self._last
        if not last or any(_current(owner, name) is not old for owner, name, _new, old in last):
            return False
        try:
            for owner, name, new, _old in last:
                setattr(owner, name, new)
        except (AttributeError, TypeError):
            return False
        self._installed = list(last)
        self.version += 1
        return True


def _current(owner: Any, name: str) -> Any:
    if isinstance(owner, type):
        return owner.__dict__.get(name, _ABSENT)
    return getattr(owner, name, _ABSENT)
