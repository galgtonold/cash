"""Observe side effects a cached function actually performs, on its first call.

Static analysis stops at library boundaries -- that is deliberate, since
folding every installed package into the walk would be both slow and useless.
The cost is that a side effect *inside* a library is reachable only by the
method's NAME (``session.post``, ``cur.execute``; see
``cash.effects.METHOD_VERBS``), and a name cannot reach everything:
``session.get(...)`` collides with ``dict.get``, and an arbitrary vendor
function like ``client.emit_metric(...)`` has no effect-shaped name at all.

Measured, before this module existed: of 24 real side effects planted across a
decorator-cached function, 21 were flagged statically. Every miss was inside an
installed library, and the two that survived even a widened name list were a
network read through a client object and a vendor function whose return value
was used.

This closes that class from the other side. While the body of a *missing* call
runs, cash watches for effects it can see regardless of where the code lives:

* a file opened for writing (reported by the file tracker's ``open`` audit
  consumer, which sees reads and writes alike)
* an outbound socket connection
* a subprocess being spawned

If any fire and the analyzer said nothing, cash warns once -- because the
second call will not do them. That is the whole hazard: the effect happened,
the value looks right, and nothing will tell you the effect stopped.

**Why this cannot replace the static pass.** It only sees what the first call
happens to do. A branch not taken performs no effect, so silence here is not
proof of purity -- it is one observation. The static analyzer reasons about
code that was never run, which is a different and complementary guarantee.

**What becomes of an observation.** A file write, a connection or a
subprocess is reported, never acted on: by the time it is seen the function
has run, and its result is correct and worth storing. Even under
``strict=True`` it warns rather than raising, since raising after the effect
landed would discard a correct result and prevent nothing. Two observations do
stop the result being stored (`ResultStore.refusal`), because storing it would
make a hit behave differently from the call: an argument the call changed in
place (`mutated_args`, which a hit would leave as it was), and a
``unittest.mock`` object called while the body ran (`mock_called`: the result
may be a test's fake). The observer only records; the decorator decides.

**What it deliberately does not do:**

* It does not block or interrupt anything while the body runs.
* It does not watch other threads. Dispatch is via ``ContextVar``, so an
  effect started on a worker thread is not attributed to the caller -- both
  because attribution would be wrong and because a background thread of cash's
  own (the tiered backend's writer) must never be mistaken for the user's
  code. The static pass already flags the ordinary ``Thread(target=...)``
  shape.
"""

from __future__ import annotations

import contextvars
import functools
import linecache
import logging
import os
import sys
from typing import Any

from .analysis.annotations import ASSUME_SAFE_RE
from .effects import EffectKind
from .install_paths import is_user_path
from .tracking import io_watch

logger = logging.getLogger(__name__)

#: The observer whose block is currently executing, per thread and per
#: asyncio Task. Mirrors ``tracker_context.active_tracker`` on purpose: same
#: dispatch-dynamically shape, same isolation properties.
active_observer: contextvars.ContextVar["EffectObserver | None"] = contextvars.ContextVar(
    "_cash_active_observer", default=None
)


def reset_in_any_context(var: contextvars.ContextVar, token: contextvars.Token) -> None:
    """``var.reset(token)``, also in a context other than the one that made *token*.

    A cached generator keeps its tracker and observer across ``yield``, and
    the caller may advance it from anywhere: another thread, or
    ``asyncio.to_thread``, which runs each ``next()`` in a fresh copy of the
    context. ``ContextVar.reset`` refuses a token from another context, and
    that ValueError surfaced in the caller's loop. There, set the value the
    token restores instead.
    """
    try:
        var.reset(token)
    except ValueError:
        old = token.old_value
        var.set(None if old is contextvars.Token.MISSING else old)


#: How each kind of effect the observer can see is named in its report. The
#: static findings are matched against these names, so an effect the static
#: warning already listed is not reported twice (see :func:`observed_label`).
_LABELS: dict[EffectKind, str] = {
    EffectKind.FILE_WRITE: "file write",
    EffectKind.NETWORK: "network",
    EffectKind.SUBPROCESS: "subprocess",
}


def observed_label(kind: EffectKind | None) -> str | None:
    """The name the observer reports an effect of *kind* under, or None when
    the observer cannot see that kind. A network or database read or write is
    seen as the connection it opens. (A database in a local file is seen as a
    file write, which this does not cover: it errs toward reporting.)"""
    if kind in (EffectKind.NETWORK_READ, EffectKind.NETWORK_WRITE, EffectKind.DB_READ, EffectKind.DB_WRITE):
        kind = EffectKind.NETWORK
    return _LABELS.get(kind)  # type: ignore[arg-type]


def _record(kind: str, detail: str) -> None:
    observer = active_observer.get()
    if observer is not None:
        observer.record_effect(kind, detail)


def _is_library_file(filename: str) -> bool:
    """Library or interpreter code, cash's own included -- never the user's line."""
    return not is_user_path(filename)


def line_waived(filename: str, lineno: int) -> bool:
    """Does ``# @cash:assume-safe`` cover *lineno* -- on it, or alone above it?"""
    if ASSUME_SAFE_RE.search(linecache.getline(filename, lineno)):
        return True
    above = linecache.getline(filename, lineno - 1)
    return above.lstrip().startswith("#") and bool(ASSUME_SAFE_RE.search(above))


def _on_connect(args: tuple) -> None:
    """The ``socket.connect`` audit event: ``(socket, address)``."""
    if active_observer.get() is not None:
        _record(_LABELS[EffectKind.NETWORK], f"socket connect to {_describe_address(args[1])}")


def _on_spawn(args: tuple) -> None:
    """The ``subprocess.Popen`` audit event: ``(executable, args, cwd, env)``."""
    if active_observer.get() is not None:
        _record(_LABELS[EffectKind.SUBPROCESS], f"spawned {_describe_argv(args[1])}")


# Audit events rather than wrappers on `socket.socket.connect` and
# `Popen.__init__`: they see every connection and spawn whatever reference the
# caller holds, and with no observer open nothing on those paths (kernel
# launch, every outbound connection) runs through cash's code at all.
io_watch.subscribe("socket.connect", _on_connect)
io_watch.subscribe("subprocess.Popen", _on_spawn)


def _describe_address(address: Any) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)[:80]


#: What ``shell=True`` puts in front of the command on POSIX.
_SHELLS = ("/bin/sh", "/system/bin/sh")


def _describe_argv(args: Any) -> str:
    if isinstance(args, (list, tuple)) and args:
        if len(args) >= 3 and args[0] in _SHELLS and args[1] == "-c":
            return str(args[2])[:80]  # `shell=True`: the command, not the shell
        return str(args[0])[:80]
    return str(args)[:80]


#: Calls made on any ``unittest.mock`` object since `_hook_mock_calls` ran.
#: Only its movement across a body is read, so a lost increment between two
#: threads cannot hide one.
_mock_calls = 0
_mock_patches = io_watch.Patches()


def _hook_mock_calls() -> None:
    """Count every call on a ``unittest.mock`` object, once the module exists.

    A mock stands in for the real thing wherever it was put: one level below
    what the body calls (``requests.Session.request``, ``HTTPAdapter.send``)
    or swapped into a global after the key's bindings were read. No binding
    the key reads can show those, but a mock that RAN cannot hide
    that it did: every call on one goes through
    ``CallableMixin._increment_mock_call``. Never imports ``unittest.mock``
    itself -- a program that has not imported it has no mocks. Installed
    while an observer is open, like the rest of cash's I/O watch.
    """
    module = sys.modules.get("unittest.mock")
    real = getattr(getattr(module, "CallableMixin", None), "_increment_mock_call", None)
    if real is None or getattr(real, "_cash_effect_patch", False):
        return

    @functools.wraps(real)
    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _mock_calls
        _mock_calls += 1
        return real(self, *args, **kwargs)

    counted._cash_effect_patch = True  # type: ignore[attr-defined]
    counted._original_func = real  # type: ignore[attr-defined]
    _mock_patches.replace(module.CallableMixin, "_increment_mock_call", counted)


io_watch.add_patcher(_hook_mock_calls, _mock_patches.restore)


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
        #: The frames that entered this observer. The user's lines that led to
        #: an effect are the frames above these, and no further: the caller of
        #: the cached function did not perform the effect.
        self._outer: list[Any] = []
        # cash's own cache directory. A write in there is cash storing the
        # entry, not the user's function doing I/O, and reporting it would
        # make every cached function look impure.
        self._exclude = os.path.abspath(exclude_under) if exclude_under else None
        self._tokens: list[contextvars.Token] = []
        #: A ``unittest.mock`` object was called while the block ran: the
        #: result may be a test's fake, and must not be stored as the answer.
        self.mock_called = False
        self._mock_calls_at: list[int] = []
        #: The call's arguments before the body ran, for naming the ones it
        #: changed in place: ``{parameter: content hash}`` of those that can
        #: change (`PurityChecks.argument_snapshot`), and ``{parameter: (value,
        #: identity snapshot)}`` for plain lists and dicts
        #: (`PurityChecks.argument_identities`). Set by the decorator; None when not
        #: taken.
        self.arg_snapshot: dict[str, str] | None = None
        self.arg_identities: dict[str, tuple[Any, list]] | None = None
        #: The parameters the call changed in place, once the decorator has
        #: compared (`PurityChecks.check_argument_mutation`); None when none did.
        self.mutated_args: list[str] | None = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "EffectObserver":
        io_watch.hold()
        _hook_mock_calls()  # `unittest.mock` may have been imported since the first hold
        self._mock_calls_at.append(_mock_calls)
        self._tokens.append(active_observer.set(self))
        self._outer.append(sys._getframe(1))
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        if self._tokens:
            reset_in_any_context(active_observer, self._tokens.pop())
            io_watch.release()
        if self._outer:
            self._outer.pop()  # a frame must not outlive its call
        if self._mock_calls_at and self._mock_calls_at.pop() != _mock_calls:
            self.mock_called = True
        return False

    def suspend(self):
        """Stop observing until :meth:`resume`. Mirrors `FileAccessTracker`."""
        return active_observer.set(None)

    def resume(self, token) -> None:
        reset_in_any_context(active_observer, token)

    # -- recording ---------------------------------------------------------
    def record(self, kind: str, detail: str) -> None:
        if len(self.effects) >= 8:  # a summary, not a log
            return
        self.effects.append((kind, detail))

    def record_effect(self, kind: str, detail: str) -> None:
        """Record an effect performed on the stack right now, naming the user's
        line that led to it -- unless ``# @cash:assume-safe`` waives any line
        on the way.

        The effect itself happens inside a library, where no waiver can be
        written; the user's code that called into it can carry one, the same
        statement-scoped waiver the static findings take. Before this the only
        way to quiet an observed effect was ``assume_safe=True``, which also
        silences every effect added to the function later.
        """
        sites = self._user_sites()
        if any(line_waived(filename, lineno) for filename, lineno in sites):
            return
        if sites:
            inner = f"{os.path.basename(sites[0][0])}:{sites[0][1]}"
            outer = f"{os.path.basename(sites[-1][0])}:{sites[-1][1]}"
            detail += f", at {inner}" + (f" (from {outer})" if outer != inner else "")
        self.record(kind, detail)

    def _user_sites(self) -> list[tuple[str, int]]:
        """``(file, line)`` of the user's frames inside the observed call,
        innermost first."""
        stop = self._outer[-1] if self._outer else None
        sites: list[tuple[str, int]] = []
        try:
            frame = sys._getframe(2)
        except ValueError:
            return sites
        while frame is not None and frame is not stop:
            filename = frame.f_code.co_filename
            if filename and not _is_library_file(filename):
                sites.append((filename, frame.f_lineno))
            frame = frame.f_back
        return sites

    def record_write(self, path: Any) -> None:
        """Record a file opened for writing, unless it is cash's own storage."""
        try:
            resolved = os.path.abspath(os.fspath(path))
        except (TypeError, ValueError):
            return
        if self._exclude and resolved.startswith(self._exclude):
            return
        self.record_effect(_LABELS[EffectKind.FILE_WRITE], resolved)

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
