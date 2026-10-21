"""Which files a block of user code wrote, observed as it ran.

A writer's provenance -- the files it produced, and their state then -- lets
the re-execution planner tell after a kernel restart that the writer's effect
is already on disk, instead of re-firing it and rebuilding everything it
needs. Reading the paths from the code covers ``df.to_csv(OUT / 'a.csv')``;
it does not cover ``save_chart(kind)``, whose ``fig.savefig(OUT /
f'{kind}.png')`` sits in a helper, nor a loop over kinds. Those were re-fired
after every restart, and a chart loop re-ran a 263 s sweep to redraw
charts nobody had asked for.

So watch the writes instead, through cash's one audit hook
(:mod:`cash.tracking.io_watch`), inert unless a collector is active in the
current context. Every file opened for writing and every rename/replace
destination (an atomic write) is recorded, whichever library did it -- pandas,
matplotlib and pathlib all open through ``io.open``. Writes a C extension
makes without Python's ``open`` are not seen; their writer keeps no provenance
and is re-fired as before.
"""

from __future__ import annotations

import contextvars
import inspect
import os
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cash.analysis.namespace_effects import code_waived_lines
from cash.install_paths import is_user_code_file
from cash.purity import is_pure
from cash.tracking import io_watch
from cash.tracking.read_classification import is_runtime_cache_path

__all__ = ["WrittenPaths", "children_left_running", "kept_writes", "observe_writes", "process_runs"]

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class WrittenPaths(set):
    """The absolute paths a block wrote; :attr:`appended` holds those it
    only ever opened to append to, :attr:`waived` those it only wrote from
    a line the user waived (``_write_waived``). :attr:`unseen` is the first
    thing it did that may write files no event reports (`_UNSEEN_WRITERS`),
    or None. :attr:`children_before` are the process's child processes as
    the block first started one, or None when it started none
    (`children_left_running`)."""

    def __init__(self) -> None:
        super().__init__()
        self.unseen: str | None = None
        self.children_before: frozenset[int] | None = None
        self.appended: set[str] = set()
        self._replaced: set[str] = set()
        self._unwaived: set[str] = set()

    @property
    def waived(self) -> frozenset[str]:
        return frozenset(self - self._unwaived)


#: The collectors of the blocks being observed, innermost last. A statement
#: inside a loop is recorded for both.
_active: contextvars.ContextVar[tuple[WrittenPaths, ...]] = contextvars.ContextVar("cash_write_observers", default=())


def _record(sinks: tuple[WrittenPaths, ...], path: object, appends: bool = False) -> None:
    if path is None or isinstance(path, int):
        return  # a descriptor names no file
    resolved = os.path.abspath(os.fsdecode(os.fspath(path)))
    if is_runtime_cache_path(resolved):
        # The ``.pyc`` an import writes the first time a module is loaded
        # (and numba's compiled index): the runtime's own cache, which a hit
        # that skips the write only leaves to be compiled again. Not the
        # statement's output, wherever ``PYTHONPYCACHEPREFIX`` puts it.
        return
    waived = _write_waived()
    for sink in sinks:
        sink.add(resolved)
        if not waived:
            sink._unwaived.add(resolved)
        if not appends:
            sink._replaced.add(resolved)
            sink.appended.discard(resolved)
        elif resolved not in sink._replaced:
            sink.appended.add(resolved)


#: Modules that open a file for the code that calls them (``Path.open``,
#: ``gzip.open``, ``codecs.open``, the ``open`` IPython puts in a kernel's
#: builtins): whose append it is, is their caller's. Cash's own frames too.
#: ``LOG.open('a')`` in a helper is the same append as ``open(LOG, 'a')``,
#: which the static check counts: plain Python writes the line on every run.
_PASS_THROUGH_OPENERS = frozenset(
    {"pathlib", "pathlib._local", "pathlib._abc", "io", "_pyio", "codecs", "gzip", "bz2", "lzma"}
    | {"IPython.core.interactiveshell"}
)


def _passes_through(frame) -> bool:
    name = frame.f_globals.get("__name__") or ""
    return name in _PASS_THROUGH_OPENERS or name == "cash" or name.startswith("cash.")


def _appended_by_user_code() -> bool:
    """Was the append being audited opened by the user's own code (a helper's
    ``open(path, mode)``), not by a library's (a log handler's)?

    Such an append is a write a hit would skip, like a replacing one, and the
    static check cannot see it when the mode is computed (``open(p, mode)``)."""
    try:
        frame = sys._getframe(io_watch.CALLER_DEPTH + 1)
    except ValueError:
        return False
    while frame is not None and _passes_through(frame):
        frame = frame.f_back
    return frame is not None and is_user_code_file(frame.f_code.co_filename)


#: ``id(code)`` -> ``(code, verdict)``: *verdict* is ``None`` when the code
#: object is not a function of the user's, else ``(waived lines, whole
#: body)`` (`code_waived_lines`). Asked for every frame of every write; the
#: code objects of a run are few. Keyed by identity, the code object kept
#: beside its verdict so its id is not reused: a code object's own hash is
#: computed afresh each time, over its constants and every code object
#: nested in them, ~100 us for a module's (E10's loop over 50,000 files
#: spent 12 s looking up the cell's and IPython's module frames).
_frame_verdicts: dict[int, tuple[types.CodeType, tuple[frozenset[int], bool] | None]] = {}


def _frame_verdict(code: types.CodeType) -> tuple[frozenset[int], bool] | None:
    seen = _frame_verdicts.get(id(code))
    if seen is not None and seen[0] is code:
        return seen[1]
    user = is_user_code_file(code.co_filename)
    if user and code.co_name == "<module>":
        verdict = _TOP  # the cell (or script) itself: nothing of the user's above
    else:
        verdict = code_waived_lines(code) if user else None
    if len(_frame_verdicts) > 4096:
        _frame_verdicts.clear()
    _frame_verdicts[id(code)] = (code, verdict)
    return verdict


_TOP: tuple[frozenset[int], bool] = (frozenset(), False)


def _write_waived() -> bool:
    """Did the user waive the write being audited? Any function of theirs on
    its way -- the one that opened the file, or one calling it from a line
    -- waives it with ``# @cash:assume-safe`` on that line (or its ``def``),
    a ``with cash.assume_safe():`` around it, or ``@cash.pure`` on itself.
    A waiver elsewhere (a print helper's line, a ``@pure`` function that
    only made an argument) is not the user's word on this write."""
    frame = sys._getframe(1)  # cash's own frames are no user code: walked past
    while frame is not None:
        code = frame.f_code
        verdict = _frame_verdict(code)
        if verdict is _TOP:
            return False
        if verdict is not None and not _passes_through(frame):
            lines, whole = verdict
            if whole or frame.f_lineno in lines:
                return True
            func = frame.f_globals.get(code.co_name)
            if func is not None and is_pure(func) and getattr(inspect.unwrap(func), "__code__", None) is code:
                return True
        frame = frame.f_back
    return False


def _names_a_folder(path: object) -> bool:
    """Is the *path* being opened to write a folder already? Then it is an
    ``O_TMPFILE`` open (``tempfile.TemporaryFile``): a file with no name,
    which nothing after the block can find. A folder that only appears later
    (moved into place) is kept."""
    try:
        return isinstance(path, (str, bytes, os.PathLike)) and os.path.isdir(path)
    except (OSError, ValueError, TypeError):
        return False


def _on_open(args: tuple) -> None:
    sinks = _active.get()
    if not sinks:
        return
    path, mode, flags = args
    if isinstance(mode, str):
        written = any(c in mode for c in "wax+")
        # An append the user's own ``open()`` made is a write a hit skips.
        appends = "a" in mode and not _appended_by_user_code()
    else:  # os.open
        written = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        # ``O_APPEND`` from the user's own code is an append like ``open(p, "a")``.
        appends = written and bool(flags & os.O_APPEND) and not flags & os.O_TRUNC and not _appended_by_user_code()
    if written and not _names_a_folder(path):
        _record(sinks, path, appends)


def _on_move(args: tuple) -> None:
    sinks = _active.get()
    if sinks:
        _record(sinks, args[1])  # the destination of an atomic write


#: Audit events of work whose writes no ``open`` event reports: another
#: process (a fork's child included), SQLite (its own file I/O in C). h5py and the like open through
#: the C library, which the native file hook sees where it runs. A call
#: through ctypes raises no event; the statement processor reads the
#: statement for one (``statement_calls_through_ctypes``).
_UNSEEN_WRITERS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.exec",
        "os.startfile",
        "_winapi.CreateProcess",
        "sqlite3.connect",
    }
)


#: The events of those that start another process.
_PROCESS_STARTS = _UNSEEN_WRITERS - {"sqlite3.connect"}


def _child_pids() -> frozenset[int]:
    try:
        import psutil

        return frozenset(child.pid for child in psutil.Process().children())
    except Exception:  # noqa: BLE001 - none known: every child is new
        return frozenset()


def _on_unseen_writer(event: str) -> io_watch.Consumer:
    starts = event in _PROCESS_STARTS

    def note(_args: tuple) -> None:
        sinks = _active.get()
        for sink in sinks:
            if sink.unseen is None:
                sink.unseen = event
        if starts and any(sink.children_before is None for sink in sinks):
            # Before the child exists: the event comes ahead of the start.
            children = _child_pids()
            for sink in sinks:
                if sink.children_before is None:
                    sink.children_before = children

    return note


def children_left_running(written: WrittenPaths) -> list[Any]:
    """The child processes *written*'s block started that still run, as
    ``psutil.Process`` objects: a ``subprocess.Popen`` it did not wait for."""
    before = written.children_before
    if before is None:
        return []
    try:
        import psutil

        return [child for child in psutil.Process().children() if child.pid not in before and process_runs(child)]
    except Exception:  # noqa: BLE001 - none cash can see
        return []


def process_runs(process: Any) -> bool:
    """Whether *process* still runs: not ended, nor ended and not yet reaped."""
    try:
        import psutil

        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001 - gone
        return False


io_watch.subscribe("open", _on_open)
for _event in _UNSEEN_WRITERS:
    io_watch.subscribe(_event, _on_unseen_writer(_event))
io_watch.subscribe("os.rename", _on_move)
io_watch.subscribe("os.replace", _on_move)


@contextmanager
def observe_writes() -> Iterator[WrittenPaths]:
    """Yield the set of absolute paths the block writes, filled as it runs."""
    written = WrittenPaths()
    token = _active.set(_active.get() + (written,))
    io_watch.hold()
    try:
        yield written
    finally:
        io_watch.release()
        _active.reset(token)


def kept_writes(
    written: "set[str] | frozenset[str]", storage_root: object = None, appended=frozenset(), waived=frozenset()
) -> list[str]:
    """The paths of *written* still there, outside cash's own storage
    (*storage_root*) and not in *appended*, sorted. A file written and
    removed again within the block (a temporary file) left nothing a cache
    hit would skip.

    A file only appended to by a library is left to the static writer check,
    which reads ``to_csv(mode='a')`` in the user's code: an append seen in a
    library's code is a log handler's, as often as not. One the user's own
    code opened (``open(p, mode)`` in a helper) is never in *appended*.
    Nor is a file the user said a hit may skip (*waived*, `WrittenPaths.waived`)."""
    if not written:
        return []
    root = None
    if isinstance(storage_root, (str, os.PathLike)):
        root = os.path.normcase(os.path.abspath(os.fspath(storage_root))) + os.sep
    return sorted(
        p
        for p in written
        if p not in appended
        and p not in waived
        and not (root and os.path.normcase(p).startswith(root))
        # A folder is kept: one moved into place (``tmp.rename(dest)``) is
        # what the block published.
        and os.path.lexists(p)
    )
