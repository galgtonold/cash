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

import contextlib
import contextvars
import functools
import linecache
import logging
import os
import sys
from typing import Any

from .analysis.annotations import line_assumes_safe
from .effects import EffectKind
from .install_paths import is_user_code_file, is_user_path, norm_dir, normcase_path
from .loaded_modules import loaded_module
from .tracking import io_watch, process_registries
from .tracking.read_classification import is_runtime_cache_path

logger = logging.getLogger(__name__)

#: The observer whose block is currently executing, per thread and per
#: asyncio Task. Mirrors ``tracker_context.active_tracker`` on purpose: same
#: dispatch-dynamically shape, same isolation properties.
active_observer: contextvars.ContextVar["EffectObserver | None"] = contextvars.ContextVar(
    "_cash_active_observer", default=None
)


#: The observer whose effects a ``with cash.assume_safe():`` block open now
#: waives: the one active when the block was entered. Holding the observer,
#: not a flag, keeps the waiver to the call it was written in: a cached call
#: made inside the block opens its own observer, which this does not name.
waived_observer: contextvars.ContextVar["EffectObserver | None"] = contextvars.ContextVar(
    "_cash_waived_observer", default=None
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
    if line_assumes_safe(linecache.getline(filename, lineno)):
        return True
    above = linecache.getline(filename, lineno - 1)
    return above.lstrip().startswith("#") and line_assumes_safe(above)


#: ``code -> (its lines ``# @cash:assume-safe`` waives, whether its ``def``
#: line waives the whole body)`` for the user's functions; None when neither.
_CODE_WAIVERS: dict[Any, tuple[frozenset[int], bool] | None] = {}
#: Files of code that is not the user's: no waiver is read there.
_NOT_USER_FILES: set[str] = set()


def _def_line(filename: str, code: Any) -> int | None:
    """The line of *code*'s ``def``, past any decorators; None for code that
    is not a ``def`` (a lambda, a comprehension, a module)."""
    if code.co_name.startswith("<"):
        return None
    lineno = code.co_firstlineno
    for _ in range(64):
        text = linecache.getline(filename, lineno).lstrip()
        if not text:
            return None
        if text.startswith(("def ", "async def ")):
            return lineno
        lineno += 1
    return None


def code_waivers(code: Any) -> tuple[frozenset[int], bool] | None:
    """What ``# @cash:assume-safe`` waives in *code*, a function of the
    user's: the lines it is on (or alone above), and the whole body when it
    is on the ``def`` line. None when it waives nothing there."""
    try:
        return _CODE_WAIVERS[code]
    except KeyError:
        pass
    filename = code.co_filename
    if filename in _NOT_USER_FILES:
        return None
    if not is_user_code_file(filename) or _is_cash_file(filename):
        _NOT_USER_FILES.add(filename)
        return None
    lines = frozenset(line for _start, _end, line in code.co_lines() if line and line_waived(filename, line))
    first = _def_line(filename, code)
    whole = first is not None and line_waived(filename, first)
    found = (lines, whole) if lines or whole else None
    _CODE_WAIVERS[code] = found
    return found


def code_carries_waiver(code: Any) -> bool:
    """Does ``# @cash:assume-safe`` waive anything in *code*, or in a
    function, lambda or comprehension defined inside it?"""
    if code_waivers(code) is not None:
        return True
    return any(code_carries_waiver(const) for const in code.co_consts if type(const) is type(code))


def frame_waived(frame: Any) -> bool:
    """Is what *frame* does right now waived by ``# @cash:assume-safe``:
    on its current line, or on the ``def`` line of its function?"""
    found = code_waivers(frame.f_code)
    return found is not None and (found[1] or frame.f_lineno in found[0])


_CASH_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_cash_file(filename: str) -> bool:
    return os.path.abspath(filename).startswith(_CASH_DIR)


def _cash_frame(frame: Any) -> bool:
    """Does *frame* run cash's own code?"""
    name = frame.f_globals.get("__name__")
    return isinstance(name, str) and (name == "cash" or name.startswith("cash."))


class _AssumeSafe(contextlib.AbstractContextManager):
    """What `assume_safe` returns. One contextvar set and reset, and the
    process settings read where the block begins and ends
    (`EffectObserver.open_waiver`), only while a cached call is being
    observed; nothing at all otherwise."""

    def __init__(self) -> None:
        self._tokens: list[tuple[contextvars.Token, EffectObserver] | None] = []

    def __enter__(self) -> None:
        observer = active_observer.get()
        if observer is None:
            self._tokens.append(None)
            return
        self._tokens.append((waived_observer.set(observer), observer))
        observer.open_waiver()

    def __exit__(self, *exc_info: Any) -> None:
        entry = self._tokens.pop() if self._tokens else None
        if entry is not None:
            reset_in_any_context(waived_observer, entry[0])
            entry[1].close_waiver()


def assume_safe() -> contextlib.AbstractContextManager[None]:
    """Waive cash's purity findings for the lines inside the block.

    Inside a ``@cash.cache`` function, ``with cash.assume_safe():`` says the
    effects in the block are fine to skip on a cache hit, and that what the
    block reads may be frozen in the stored result. It waives what
    ``# @cash:assume-safe`` waives on one line, for every line in the block,
    and the effects the first call is seen to perform while the block runs,
    including those of helpers it calls. Findings the comment cannot waive
    stay.

    It is not part of the cache key: adding or removing the block, and the
    indent it brings, keeps the stored results. (Imported under another
    name, ``from cash import assume_safe as waive``, it still waives, but
    adding it recomputes once.) Outside a cached call it does nothing. Use
    the line comment for one line and ``@cash.cache(assume_safe=True)`` for
    the whole function.

    Returns:
        A context manager. Its ``as`` value is ``None``.

    Examples:
        ```python
        import cash

        @cash.cache
        def total(rows):
            result = sum(rows)
            with cash.assume_safe():
                with open("audit.log", "a") as fh:
                    fh.write(f"summed {len(rows)} rows\n")
            return result
        ```
    """
    return _AssumeSafe()


def _on_connect(args: tuple) -> None:
    """The ``socket.connect`` audit event: ``(socket, address)``."""
    if active_observer.get() is not None:
        _record(_LABELS[EffectKind.NETWORK], f"socket connect to {_describe_address(args[1])}")


def _on_spawn(args: tuple) -> None:
    """The ``subprocess.Popen`` audit event: ``(executable, args, cwd, env)``."""
    if active_observer.get() is not None:
        _record(_LABELS[EffectKind.SUBPROCESS], f"spawned {_describe_argv(args[1])}")


def _waiver_trace(frame: Any, event: str, arg: Any) -> Any:
    """The global trace function `WaiverTrace` installs: for the frames of
    the user's code that ``# @cash:assume-safe`` waives something in, a
    local one that tells the observed call where a waived stretch begins
    and ends (`EffectObserver.open_waiver`); None, and no line events, for
    every other frame."""
    found = code_waivers(frame.f_code)
    if found is None:
        return None
    observer = active_observer.get()
    if observer is None or _cash_frame(frame):
        return None
    lines, whole = found
    is_open = [whole]
    if whole:
        observer.open_waiver()

    def local(frame: Any, event: str, arg: Any) -> Any:
        if event == "line" and not whole:
            waived = frame.f_lineno in lines
            if waived is not is_open[0]:
                is_open[0] = waived
                if waived:
                    observer.open_waiver()
                else:
                    observer.close_waiver()
        elif event == "return" and is_open[0]:
            is_open[0] = False
            observer.close_waiver()
        return local

    return local


def _both(first: Any, second: Any) -> Any:
    """One local trace function that runs *first* and *second*."""
    if first is None:
        return second
    if second is None:
        return first
    pair = [first, second]

    def both(frame: Any, event: str, arg: Any) -> Any:
        for index, each in enumerate(pair):
            if each is not None:
                pair[index] = each(frame, event, arg)
        return both if pair[0] is not None or pair[1] is not None else None

    return both


class WaiverTrace:
    """While a missing call runs whose code carries ``# @cash:assume-safe``
    (`code_carries_waiver`), trace the frames of the functions that carry
    it on this thread, so the process settings a waived line, or a function
    waived from its ``def`` line, changes are left out
    (`process_registries.waive_since`) as its file writes are. A trace
    function already set (a debugger, coverage) keeps running alongside,
    and ``sys.gettrace()`` reads as that one meanwhile."""

    __slots__ = ("_installed", "_replaced")

    def __init__(self) -> None:
        self._installed: Any = None
        self._replaced: Any = None

    def begin(self) -> None:
        replaced = sys.gettrace()
        if replaced is None:
            installed: Any = _waiver_trace
        else:

            def installed(frame: Any, event: str, arg: Any) -> Any:
                return _both(replaced(frame, event, arg), _waiver_trace(frame, event, arg))

        process_registries.stand_in_trace(installed, replaced)
        self._installed, self._replaced = installed, replaced
        sys.settrace(installed)

    def end(self) -> None:
        installed = self._installed
        if installed is None:
            return
        self._installed = None
        # A trace function the body set since is the body's: left in place.
        if sys.gettrace() is installed:
            sys.settrace(self._replaced)
        process_registries.drop_trace_stand_in(installed)
        self._replaced = None


def _env_write_waived() -> bool | None:
    """Whether the observed call waived the environment write being
    audited (`EffectObserver.waives_here`); None outside an observed call.
    Asked first (`process_registries.register_env_waiver`): inside a cached
    call, a waiver around the call does not reach into it."""
    observer = active_observer.get()
    return None if observer is None else observer.waives_here()


process_registries.register_env_waiver(_env_write_waived, first=True)


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
    module = loaded_module("unittest.mock")
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
        self._effects: list[tuple[str, str]] = []
        #: Directories the block made (``mkdtemp``, ``os.mkdir``), and the
        #: writes into them, held back: a scratch directory the block removes
        #: before it returns takes those writes with it, and a hit skipping
        #: them changes nothing (`effects`). A file ``os.open`` wrote is held
        #: back on its own path the same way (`record_write`).
        self._created_dirs: list[str] = []
        self._scratch_writes: list[tuple[str, tuple[str, str]]] = []
        #: The frames that entered this observer. The user's lines that led to
        #: an effect are the frames above these, and no further: the caller of
        #: the cached function did not perform the effect.
        self._outer: list[Any] = []
        # cash's own cache directory. A write in there is cash storing the
        # entry, not the user's function doing I/O, and reporting it would
        # make every cached function look impure. A directory prefix
        # (`norm_dir`): ``.cash_exports`` beside ``.cash`` is the user's.
        self._exclude = norm_dir(exclude_under) if exclude_under else None
        self._tokens: list[contextvars.Token] = []
        #: The process settings as the call found them
        #: (`process_registries.snapshot`), set by the decorator: a waived
        #: stretch of the call is left out of them (`open_waiver`).
        self.registries: Any = None
        #: What the user's classes bound as the call began
        #: (`class_writes.ClassState`), set by the decorator: a waived
        #: stretch of the call is left out of it too.
        self.class_state: Any = None
        self._waiver_depth = 0
        self._waiver_start: list[Any] | None = None
        self._class_start: Any = None
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
        #: True when those identities, with the arguments that cannot change
        #: at all, cover every argument the check looks at: unchanged
        #: identities then prove nothing changed, with no hash after the body
        #: (`PurityChecks.argument_identities`).
        self.arg_identities_cover_all = False
        #: Random generators held by arguments a ``__cash_key__`` or a
        #: registered hasher keys, with their state before the body
        #: (`PurityChecks.held_generators`).
        self.held_generators: list | None = None
        #: ``[(name, value, state)]`` of what the cached callable binds into
        #: every call -- a bound method's ``self``, a partial's arguments --
        #: before the body (`PurityChecks.bound_state`).
        self.bound_before: list | None = None
        #: The parameters the call changed in place, once the decorator has
        #: compared (`PurityChecks.check_argument_mutation`); None when none did.
        self.mutated_args: list[str] | None = None
        #: Of `mutated_args`, those not stored with this call only because
        #: it was the first seen changing them: the next call takes their
        #: state before the body and is stored (`CallRunner.arguments_left`).
        self.stored_from_next: list[str] = []
        #: Whether the next such call may be stored where this one was not
        #: (`stored_from_next`, or a frame cash's own memo held): False after
        #: a first change no later call can write back either, so the
        #: refusal stands for every call (`CallRunner._settle_verdict`).
        self.next_may_store = False
        #: The parameters no hit writes a state back into, whatever the call
        #: did to them (`CallRunner._unwritable_args`): a finding that the
        #: body changes one stands, seen or not.
        self.unwritable_args: list[str] = []
        #: The arguments whose attributes the call bound or deleted, which
        #: its entry stores and a hit replays (`attribute_replay`): not a
        #: change a hit leaves undone, so nothing is said about them.
        self.replayed_args: list[str] = []
        #: Why a hit cannot replay the attributes the call bound on an
        #: argument, when that is what keeps it from being stored
        #: (`attribute_replay.AttrVerdict.why`); said in its miss reason.
        self.unreplayable: str | None = None

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

    def open_waiver(self) -> None:
        """A waived stretch of the call begins: a ``with cash.assume_safe():``
        block, a line ``# @cash:assume-safe`` is on, a function it is on the
        ``def`` line of (`WaiverTrace`). What the process settings change
        until it ends is left out (`process_registries.waive_since`). Nested
        stretches count as the outermost."""
        self._waiver_depth += 1
        if self._waiver_depth == 1 and self.registries is not None:
            self._waiver_start = process_registries.read_registries()
        if self._waiver_depth == 1 and self.class_state is not None:
            from .decorator import class_writes

            self._class_start = class_writes.read(self.class_state)

    def close_waiver(self) -> None:
        """The stretch `open_waiver` began ends."""
        if self._waiver_depth == 0:
            return
        self._waiver_depth -= 1
        if self._waiver_depth == 0 and self._waiver_start is not None:
            start, self._waiver_start = self._waiver_start, None
            process_registries.waive_since(self.registries, start)
        if self._waiver_depth == 0 and self._class_start is not None:
            from .decorator import class_writes

            class_start, self._class_start = self._class_start, None
            class_writes.waive_since(self.class_state, class_start)

    def drop_waivers(self) -> None:
        """Forget a stretch still open as the call ends: not waived."""
        self._waiver_depth = 0
        self._waiver_start = None
        self._class_start = None

    # -- recording ---------------------------------------------------------
    @property
    def effects(self) -> list[tuple[str, str]]:
        """What the block did, as ``(kind, detail)``: a write into a directory
        it made only while that directory is still there."""
        kept = [effect for held, effect in self._scratch_writes if os.path.lexists(held)]
        return (self._effects + kept)[:8]

    def record(self, kind: str, detail: str) -> None:
        if len(self._effects) >= 8:  # a summary, not a log
            return
        self._effects.append((kind, detail))

    def reword(self, kind: str, old: str, new: str) -> None:
        """Replace *old* by *new* in the effects of *kind* recorded so far."""
        self._effects[:] = [(k, d.replace(old, new) if k == kind else d) for k, d in self._effects]

    def forget(self, kind: str) -> None:
        """Take back the effects of *kind* recorded so far."""
        self._effects[:] = [effect for effect in self._effects if effect[0] != kind]

    def note_created_dir(self, path: Any) -> None:
        """A directory the block made (see `effects`)."""
        try:
            self._created_dirs.append(norm_dir(os.fspath(path)))
        except (TypeError, ValueError):
            pass

    def record_effect(self, kind: str, detail: str, scratch: str | None = None) -> None:
        """Record an effect performed on the stack right now, naming the user's
        line that led to it -- unless ``# @cash:assume-safe`` waives any line
        on the way, or a ``with cash.assume_safe():`` block entered during
        this observed call is open (`waived_observer`), whatever code the
        effect is in.

        The effect itself happens inside a library, where no waiver can be
        written; the user's code that called into it can carry one, the same
        statement-scoped waiver the static findings take. Before this the only
        way to quiet an observed effect was ``assume_safe=True``, which also
        silences every effect added to the function later.
        """
        if waived_observer.get() is self:
            return
        if self._frame_waived_on_way():
            return
        sites = self._user_sites()
        if any(line_waived(filename, lineno) for filename, lineno in sites):
            return
        if sites:
            inner = f"{os.path.basename(sites[0][0])}:{sites[0][1]}"
            outer = f"{os.path.basename(sites[-1][0])}:{sites[-1][1]}"
            detail += f", at {inner}" + (f" (from {outer})" if outer != inner else "")
        if scratch is not None:
            if len(self._scratch_writes) < 8:
                self._scratch_writes.append((scratch, (kind, detail)))
            return
        self.record(kind, detail)

    def waives_here(self) -> bool:
        """Is an effect performed on the stack right now waived, as
        `record_effect` waives it: a ``with cash.assume_safe():`` block
        entered during this observed call is open, or ``#
        @cash:assume-safe`` is on a line of the user's on the way."""
        if waived_observer.get() is self:
            return True
        # The user's frames from the write up to the cached function's own,
        # a cell's included (``is_user_code_file``: linecache holds its
        # lines), and no further: the first of cash's frames above them runs
        # the call, and a waiver in its caller is not the function's word.
        # (The frame that entered this observer is a suspended generator's,
        # not on the way.)
        frame = sys._getframe(1)
        seen_user = False
        while frame is not None:
            if _cash_frame(frame):
                if seen_user:
                    return False
            elif is_user_code_file(frame.f_code.co_filename):
                seen_user = True
                if line_waived(frame.f_code.co_filename, frame.f_lineno) or frame_waived(frame):
                    return True
            frame = frame.f_back
        return False

    def _frame_waived_on_way(self) -> bool:
        """Is a function of the user's inside the observed call, on the way
        to the effect being recorded, waived from its ``def`` line?"""
        stop = self._outer[-1] if self._outer else None
        try:
            frame = sys._getframe(2)
        except ValueError:
            return False
        while frame is not None and frame is not stop:
            found = code_waivers(frame.f_code)
            if found is not None and found[1]:
                return True
            frame = frame.f_back
        return False

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

    def record_write(self, path: Any, raw: bool = False) -> None:
        """Record a file opened for writing, unless it is cash's own storage
        or a runtime's own cache (`is_runtime_cache_path`): bytecode, numba's
        compiled index and data. A hit that skips such a write changes
        nothing but a later compile.

        A folder opened to write is an ``O_TMPFILE`` open
        (``tempfile.TemporaryFile``): a file with no name, which nothing
        after the call can find.

        *raw*: opened by ``os.open``, whose file is often a temporary one
        (``mkstemp``) the block removes again: such a write is kept only
        while the file is still there (`effects`), as the notebook keeps it
        (`write_observer.kept_writes`)."""
        try:
            resolved = os.path.abspath(os.fspath(path))
        except (TypeError, ValueError):
            return
        under = normcase_path(resolved) + "/"
        if self._exclude and under.startswith(self._exclude):
            return
        if is_runtime_cache_path(resolved):
            return
        if os.path.isdir(resolved):
            return
        scratch = next((d for d in self._created_dirs if under.startswith(d)), None)
        if raw and scratch is None:
            scratch = resolved
        self.record_effect(_LABELS[EffectKind.FILE_WRITE], resolved, scratch)

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
