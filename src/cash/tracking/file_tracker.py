"""File-read tracking for data file dependencies.

Records which files each cached call or notebook statement reads. Python-level
opens and directory listings arrive as audit events through
:mod:`cash.tracking.io_watch`; readers that open files in C (pyarrow, polars,
sqlite3, some pandas readers), existence probes and ``Path.stat`` are wrapped
while a tracker is open. File hashes are incorporated into cache keys so that
changed data automatically invalidates dependent cached results.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread as cf_thread
import contextvars
import functools
import glob as glob_module
import importlib.abc
import importlib.util
import logging
import os
import pathlib
import stat
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from cash._clock import perf_counter as _perf_counter
from cash._paths import is_remote_url, normalize_path
from cash.effect_observer import active_observer as _active_effect_observer
from cash.install_paths import is_user_path
from cash.tracking import io_watch
from cash.tracking._memo import Memo
from cash.tracking.read_classification import (
    RUNTIME_CACHE_SEGMENT,
    RUNTIME_CACHE_SUFFIXES,
    SCRATCH_MEMMAP,
    incidental_read,
    is_cash_internal,
    is_pseudo_fs,
    regular_file_stat,
)
from cash.tracking.tracker_context import active_tracker

# A remote URL handed to a reader (``pd.read_parquet("s3://bucket/key")``)
# reaches us as the raw first argument. ``os.path.realpath`` would mangle it into
# a bogus local path, nothing would resolve, and ``snapshot_file_deps`` would
# drop it — leaving the entry with *no* dependency, hitting forever even after
# the object changed. URLs are routed to their own channel instead and tracked
# by the store's own validator (ETag / version id / generation). ``file://`` is
# excluded: it names a local path that can genuinely be stat'ed.

__all__ = ["FileDependencyRegistry", "PostImportHook", "FileAccessTracker", "FileDependencies", "file_registry"]

# Type alias for file dependency tracking: maps normalized file path -> mtime at read time
FileDependencies = dict[str, float]

logger = logging.getLogger(__name__)

# The wrappers installed while a tracker is open (see `install_patches`),
# and the lock that serialises installing and removing them with the post-
# import hook patching a module that was imported meanwhile.
_patches = io_watch.Patches()
_install_lock = threading.RLock()

#: Keyword names a reader may take its path under, first match wins. Covers
#: pandas (`filepath_or_buffer`, `path_or_buf`, `io` for read_excel, `path`),
#: numpy (`file`, `fname`), joblib (`filename`), pyarrow (`source`,
#: `input_file`) and polars (`source`).
_PATH_KWARGS = ("filepath_or_buffer", "path_or_buf", "source", "input_file", "path", "file", "fname", "filename", "io")


#: ``code -> {file: its stat when last read}`` for files read while a frame of
#: that code was on the stack, process-wide. A memo (``functools.lru_cache``, a
#: module dict) hands a later call the product of an earlier read, and the
#: later call reads nothing -- so its entry recorded no file and kept serving
#: after the file changed. What a helper read once is what
#: `credited_reads` answers when a call reaches it again, and the stat says
#: WHICH version it read: a memo filled before the file changed hands back the
#: old version's data. Reads outside any cached call count too
#: (`note_untracked_read`) -- `main()` logging its settings through the memo
#: before the first cached call is the ordinary way to fill one. A code past
#: `_READS_PER_CODE_MAX` files is marked ``None``: it reads per argument, and
#: every file it ever read is no one call's dependency.
_READS_PER_CODE_MAX = 16
_reads_by_code = Memo(4096)
_CASH_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _record_read(code: Any, abs_path: str, stat: Any) -> None:
    """Remember that *code* read *abs_path*, as it was (*stat*)."""
    memo = _reads_by_code
    reads = memo.get(code, ())
    if reads is None:
        return
    if code not in memo:
        reads = {}
        if not memo.put(code, reads):
            return
    if abs_path not in reads and len(reads) >= _READS_PER_CODE_MAX:
        memo.put(code, None)
    else:
        reads[abs_path] = stat  # the LATEST read: a memo refilled is current again


#: The decorator's code: `Cash` in core.py and the call steps in decorator/.
_DECORATOR_FILES = (
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "core.py")),
    os.path.normcase(os.path.join(_CASH_PACKAGE_DIR, "decorator")) + os.sep,
)


def _is_cash_wrapper(filename: str) -> bool:
    norm = os.path.normcase(filename)
    return norm == _DECORATOR_FILES[0] or norm.startswith(_DECORATOR_FILES[1])


def credit_read_to_stack(abs_path: str, tracker: "FileAccessTracker") -> None:
    """Credit a read to the user code on the stack, up to the cached call."""
    try:
        frame = sys._getframe(1)
    except ValueError:
        return
    stat = tracker.read_stats.get(abs_path)
    depth = 0
    while frame is not None and depth < 64:
        code = frame.f_code
        kind = _frame_kind(code.co_filename)
        if kind == "wrapper":
            break  # the cached call's own wrapper: the walk ends
        if kind == "user":
            tracker._note_reading_code(code)
            _record_read(code, abs_path, stat)
        frame, depth = frame.f_back, depth + 1


#: code filename -> ``wrapper``/``cash``/``user``/``other``; `_frame_kind`.
_frame_kinds = Memo(8192)


def _frame_kind(filename: str) -> str:
    """``wrapper`` (the cached call's own), ``cash``, ``user`` or ``other``,
    remembered per filename: every read walks the stack."""
    kind = _frame_kinds.get(filename)
    if kind is None:
        kind = (
            "wrapper"
            if _is_cash_wrapper(filename)
            else "cash"
            if filename and os.path.normcase(filename).startswith(_CASH_PACKAGE_DIR)
            else "user"
            if is_user_path(filename)
            else "other"
        )
        _frame_kinds.put(filename, kind)
    return kind


#: absolute path -> resolved path, for reads outside a tracker; `note_untracked_read`.
_untracked_realpaths = Memo(4096, reset=True)
#: resolved path -> (monotonic time, stat); `note_untracked_read`.
_untracked_stats = Memo(4096, reset=True)


def note_untracked_read(path: Any, frame: Any) -> None:
    """A read made outside every cached call, credited to the user code on the stack.

    *frame* is the frame that asked for the read. Only a read that user code
    started: a read by cash itself or by a library with no user frame above it
    (an import, a font cache) is nobody's input. Never raises -- it runs inside
    every ``open`` in the process.
    """
    try:
        codes = []
        depth = 0
        while frame is not None and depth < 64:
            kind = _frame_kind(frame.f_code.co_filename)
            if kind == "wrapper":
                break
            if kind == "cash" and not codes:
                return  # cash reading its own files
            if kind == "user":
                codes.append(frame.f_code)
            frame, depth = frame.f_back, depth + 1
        if not codes:
            return
        raw = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        if not isinstance(raw, str) or is_pseudo_fs(raw) or is_remote_url(raw):
            return
        # `realpath` is 60us on Windows, most of what this costs; resolved once
        # per absolute path (so a chdir still resolves anew).
        absolute = os.path.abspath(raw)
        abs_path = _untracked_realpaths.get(absolute)
        if abs_path is None:
            abs_path = normalize_path(os.path.realpath(absolute))
            _untracked_realpaths.put(absolute, abs_path)
        if is_pseudo_fs(abs_path) or is_cash_internal(abs_path):
            return
        # A stat is 15us, and a loop re-reading one file pays it every time.
        # Reusing one taken in the last second can only be too OLD, and an old
        # stat that differs from the file makes a store refused, never a stale
        # answer served (`Cash._credit_remembered_reads`).
        now = time.monotonic()
        seen = _untracked_stats.get(abs_path)
        if seen is not None and now - seen[0] < 1.0:
            stat = seen[1]
        else:
            stat = regular_file_stat(abs_path)
            _untracked_stats.put(abs_path, (now, stat))
        if stat is None:
            return
        for code in codes:
            _record_read(code, abs_path, stat)
    except Exception:  # noqa: BLE001 - attribution is an aid, never a failure
        logger.debug("[TRACKER] could not note an untracked read of %r", path, exc_info=True)


def credited_reads(code: Any) -> dict[str, Any] | None:
    """``{file: stat when read}`` for files read while *code* was on the stack;
    None when it reads per argument."""
    reads = _reads_by_code.get(code, ())
    return None if reads is None else dict(reads)


def install_read_watch() -> None:
    """Watch reads from now on, not only inside cached calls.

    A memo is usually filled before any cached call runs -- ``main()`` logging
    its settings -- and a read nobody was watching is a read no entry can
    depend on. Called when a function is decorated. It turns on the ``open``
    audit event outside tracker scopes and installs no patch: a read through
    a C-level reader outside every cached call is not seen.
    """
    io_watch.watch_outside_scopes()


#: Callers whose opens and listings are the interpreter's, not a read of data:
#: the import system (a module's source and bytecode, the ``sys.path``
#: directories it lists) and the source readers behind ``inspect.getsource``
#: and tracebacks, which the ``linecache`` wrappers below cover instead.
_NOT_A_READ = frozenset(
    {
        "_frozen_importlib",
        "_frozen_importlib_external",
        "importlib._bootstrap",
        "importlib._bootstrap_external",
        "zipimport",
        "tokenize",
        "linecache",
    }
)

#: Packages whose every open and listing is a tool's, never the notebook's.
#: coverage's tracer lists the directory of each source file it first meets
#: (on Windows, to learn the path's case), from inside the trace function, so
#: the statement running at that moment got the directory as an input. Which
#: directories that is depends on coverage's per-process cache: the same
#: import keyed differently in the next kernel, and the two lineage engines
#: disagreed on it.
_TOOL_PACKAGES = frozenset({"coverage"})


def _not_a_read(frame: Any) -> bool:
    """Whether the call *frame* made is the interpreter's or a tool's, not the user's."""
    name = frame.f_globals.get("__name__") or ""
    return name in _NOT_A_READ or name.partition(".")[0] in _TOOL_PACKAGES


def _audited_caller() -> Any:
    """The frame that made the audited call, seen from an audit consumer."""
    # +1: this helper's own frame.
    return sys._getframe(io_watch.CALLER_DEPTH + 1)


def _is_read_mode(mode: str) -> bool:
    # `w+` and `x+` start from an empty file, so nothing the code reads back
    # existed before it: a write, not an input. Pillow saves every image with
    # "w+b", and would have each `savefig` recorded as a dependency on its own
    # output.
    return "r" in mode or ("+" in mode and "w" not in mode and "x" not in mode)


def _on_open(args: tuple) -> None:
    """The ``open`` audit event: a read is a dependency, a write an effect.

    ``mode`` is the raw mode (``r``, ``w+``, ...) of any Python-level open;
    ``os.open`` reports None there and is not watched, and an integer "path"
    is a descriptor being wrapped, which names no file.
    """
    path, mode, _flags = args
    if not isinstance(mode, str) or not isinstance(path, (str, bytes, os.PathLike)):
        return
    caller = _audited_caller()
    if _not_a_read(caller):
        return
    if _is_read_mode(mode):
        tracker = active_tracker.get()
        if tracker is None:
            note_untracked_read(path, caller)
            return
        tracker._track_path(path)
        if "a" not in mode:
            # A file that was not there is an input too, and the docs say so --
            # but only the `os.path.exists` spelling recorded it.
            # `try: open(p) except FileNotFoundError:` kept serving its default
            # after the file appeared. The event comes before the open, so ask the disk.
            try:
                os.stat(path)
            except FileNotFoundError:
                tracker._track_absent(path)
            except (OSError, ValueError):
                pass
    elif any(ch in mode for ch in "wax"):
        # Not a dependency -- a WRITE is an effect, not an input, and folding
        # it into the key would invalidate a function on its own output. It is
        # recorded for the effect observer instead: a cache hit skips this
        # write, and if the analyzer said nothing (the write happened inside a
        # library it does not walk into) that is a silent behaviour change the
        # user should hear about. See cash.effect_observer.
        observer = _active_effect_observer.get()
        if observer is not None and not is_cash_internal(path):
            observer.record_write(path)


def _on_listing(args: tuple) -> None:
    """``os.listdir`` / ``os.scandir``: the listed directory is a dependency.

    Listing a directory is how a new file matching a pattern becomes visible:
    a cell that enumerates a directory and reads the matches gets file deps
    only for the files READ on the first run, so a NEW matching file would be
    invisible. Adding or removing an entry bumps the directory's own mtime on
    local filesystems, so the existing freshness check invalidates the
    reader. ``pathlib``'s ``glob`` / ``iterdir`` list through these too.
    """
    tracker = active_tracker.get()
    if tracker is None:
        return
    path = args[0]
    if path is None:
        path = "."
    elif not isinstance(path, (str, bytes, os.PathLike)):
        return  # a descriptor
    if _not_a_read(_audited_caller()):
        return
    tracker._track_path(path)


def _on_glob(args: tuple) -> None:
    """``glob.glob`` / ``glob.iglob``: track the directory the pattern enumerates."""
    tracker = active_tracker.get()
    if tracker is None:
        return
    base = _glob_base_dir(args[0])
    if base is not None:
        tracker._track_path(base)


def _glob_base_dir(pattern: Any) -> str | None:
    """Return the longest leading, magic-free directory of a glob *pattern*.

    ``gdir/*.num`` → ``gdir``; ``a/b*/c`` → ``a`` (deepest stable ancestor).
    The directory's mtime is what we track for membership changes.
    """

    try:
        parts = (os.fsdecode(pattern) if isinstance(pattern, bytes) else str(pattern)).replace("\\", "/").split("/")
    except (TypeError, ValueError):
        return None
    base: list[str] = []
    for p in parts[:-1]:  # exclude the filename component
        if glob_module.has_magic(p):
            break
        base.append(p)
    return "/".join(base) or "."


def subscribe_read_events() -> None:
    """Route the audit events of Python-level opens and listings to the tracker."""
    io_watch.subscribe("open", _on_open, outside_scopes=True)
    io_watch.subscribe("os.listdir", _on_listing)
    io_watch.subscribe("os.scandir", _on_listing)
    io_watch.subscribe("glob.glob", _on_glob)


def _dispatch_track(path: Any) -> None:
    """Module-level tracker-dispatching shim. Custom handler factories
    registered via :func:`cash.register_file_handler` receive this as
    their ``tracker_callback`` argument. The shim consults
    ``active_tracker`` at *call* time, so old-signature factories
    (whose wrappers do ``tracker_callback(path)``) transparently route
    to whichever tracker is active on the current asyncio task or
    thread — same isolation guarantees as the built-in handlers.
    """
    _tracker = active_tracker.get()
    if _tracker is not None:
        _tracker._track_path(path)


#: (module id, pattern) -> (namespace size, matched names); `_find_patch_targets`.
_patch_targets = Memo(1024)
#: (owner id, name, kind) -> (original, wrapper); `_install_wrapper`.
_wrappers = Memo(1024)
#: What the last full install was computed from; `install_patches`.
_installed_for: Any = None


def _find_patch_targets(func_pattern: str, module_obj: Any) -> list:
    """Return the list of attribute names to patch on *module_obj*.

    A wildcard's matches are remembered while the module's namespace keeps its
    size: `dir()` of pandas on every tracker that opens would be most of its
    cost.
    """
    if func_pattern.endswith("*"):
        namespace = getattr(module_obj, "__dict__", None)
        size = len(namespace) if isinstance(namespace, dict) else -1
        key = (id(module_obj), func_pattern)
        cached = _patch_targets.get(key)
        if cached is not None and size >= 0 and cached[0] == size:
            return cached[1]
        prefix = func_pattern[:-1]
        found = [name for name in dir(module_obj) if name.startswith(prefix)]
        _patch_targets.put(key, (size, found))
        return found
    if hasattr(module_obj, func_pattern):
        return [func_pattern]
    return []


def _mark_patch(wrapper: Any, original: Any) -> None:
    """Mark *wrapper* as cash's wrapper of *original*.

    The marker is how an install recognises its own wrapper (and skips it),
    and how call caching and cache keys leave cash's shims alone.
    """
    wrapper._is_file_tracker_patch = True
    wrapper._original_func = original


def _is_patch(obj: Any) -> bool:
    return bool(getattr(obj, "_is_file_tracker_patch", False))


def _install_wrapper(owner: Any, name: str, original: Any, kind: Any, make: Callable[[Any], Any]) -> None:
    """Set ``owner.name`` to cash's wrapper of *original*, made by *make*.

    Trackers open and close around every cached call, so a wrapper is built
    once per original and only set again after that.
    """
    key = (id(owner), name, kind)
    cached = _wrappers.get(key)
    if cached is not None and cached[0] is original:
        wrapper = cached[1]
    else:
        wrapper = make(original)
        _mark_patch(wrapper, original)
        _wrappers.put(key, (original, wrapper))
    if not _patches.replace(owner, name, wrapper):
        logger.debug("[FILE_TRACKER] Failed to patch %r.%s", owner, name)


def _install_module_patches(module_name: str, module_obj: Any) -> None:
    """Install dispatcher patches on a module while a tracker is open.
    Idempotent — skips any target whose current attribute is already a
    dispatcher wrapper.

    Called from `install_patches`, from :class:`_PatchingLoader.exec_module`
    (post-import) and when a handler is registered. The dispatcher wrappers
    route via ``active_tracker`` so they're tracker-agnostic — one install
    serves all trackers.
    """
    handlers = file_registry().get_handlers_for_module(module_name)

    with _install_lock:
        if not io_watch.holding():
            return  # an import finishing after the last tracker closed
        for func_pattern, factory in handlers:
            for name in _find_patch_targets(func_pattern, module_obj):
                original = getattr(module_obj, name, None)
                if original is None or not callable(original) or _is_patch(original):
                    continue
                _install_wrapper(module_obj, name, original, factory, lambda o, f=factory: f(o, _dispatch_track))


def _patch_attribute(owner: Any, name: str, make: Callable[[Any], Any]) -> None:
    """Wrap ``owner.name`` (defined on *owner* itself) with *make*, unless it is
    missing or still cash's wrapper -- still installed means something else
    wrapped it, so it was not put back."""
    original = owner.__dict__.get(name)
    if original is None or not callable(original) or _is_patch(original):
        return
    _install_wrapper(owner, name, original, name, make)


def _track_regular_file(path: Any) -> None:
    """Record *path* as read when a tracker is active and it is a regular file.

    For the metadata calls (``Path.stat``, ``os.path.getsize`` ...): what they
    report is the file's, so the file is a dependency -- a directory has no
    content to hash, and an absent path raised before this was reached.
    ``os.stat``, not ``os.path.isfile``: that one is patched to record a
    NEGATIVE answer as an absent dependency.
    """
    tracker = active_tracker.get()
    if tracker is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    try:
        if stat.S_ISREG(os.stat(path).st_mode):
            tracker._track_path(path)
    except (OSError, ValueError, TypeError):
        return


def _patch_pathlib_stat() -> None:
    """Track the file ``Path.stat()`` looks at.

    An export cell ending with
    ``print({p.name: p.stat().st_size for p in sorted(OUT.glob('*.csv'))})``:
    has the folder's listing as a dependency, and its names do not change, so
    after the exports above are rewritten the line would be served from the
    cache with the old sizes. Patched where ``stat`` is defined on ``Path``'s MRO,
    since pathlib has moved it between versions.
    """
    owner = next((k for k in pathlib.Path.__mro__ if "stat" in k.__dict__), None)
    if owner is None:
        return

    def make(original):
        @functools.wraps(original)
        def tracked_path_stat(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if active_tracker.get() is not None and stat.S_ISREG(result.st_mode):
                _track_regular_file(self)
            return result

        return tracked_path_stat

    _patch_attribute(owner, "stat", make)


def _patch_thread_pool_submit() -> None:
    """Run work submitted to a ``ThreadPoolExecutor`` under the submitter's context.

    The tracker is found through a ContextVar, and a pool's worker threads
    start with an empty context -- so ``ex.map(np.load, shards)`` inside a
    cached function read files no tracker saw, and editing a shard served the
    pre-edit result while the serial loop beside it invalidated.

    With a tracker active, ``submit`` (which ``Executor.map`` calls) wraps the
    call in ``copy_context().run``; with none, it is the original. A pool can
    opt out with ``_cash_internal = True``. Threads started directly with
    ``threading.Thread`` still begin empty -- documented, not patched.
    """

    def make(original):
        @functools.wraps(original)
        def submit(self, fn, /, *args, **kwargs):
            if active_tracker.get() is None or getattr(self, "_cash_internal", False):
                return original(self, fn, *args, **kwargs)
            return original(self, contextvars.copy_context().run, fn, *args, **kwargs)

        return submit

    _patch_attribute(cf_thread.ThreadPoolExecutor, "submit", make)


class _WorkerReads:
    """What a task run in a worker process returned, and the files it read."""

    __slots__ = ("value", "files", "absent")

    def __init__(self, value: Any, files: list[str], absent: list[str]) -> None:
        self.value, self.files, self.absent = value, files, absent

    def __reduce__(self):
        return (_WorkerReads, (self.value, self.files, self.absent))


class _ReadsInWorker:
    """A task sent to a worker process that brings back what it read.

    Module-level so it pickles by reference: the worker imports this module
    (0.14 s, once per worker) and runs the task under a tracker of its own,
    of the submitting tracker's class. A cached function the task calls there
    propagates its reads -- and on a hit, its recorded ones -- into that
    tracker like into any outer call's.
    """

    __slots__ = ("fn", "tracker_type")

    def __init__(self, fn: Callable[..., Any], tracker_type: type) -> None:
        self.fn = fn
        self.tracker_type = tracker_type

    def __reduce__(self):
        return (_ReadsInWorker, (self.fn, self.tracker_type))

    def __call__(self, *args: Any, **kwargs: Any) -> _WorkerReads:
        tracker = self.tracker_type()
        with tracker:
            value = self.fn(*args, **kwargs)
        return _WorkerReads(value, sorted(tracker.get_accessed_files()), sorted(tracker.get_absent_files()))


class _RelayFuture(concurrent.futures.Future):
    """The future a patched ``submit`` returns: the worker's plain result, and
    the cancellation reaching the future that actually runs."""

    def __init__(self, inner: concurrent.futures.Future) -> None:
        super().__init__()
        self._inner = inner

    def cancel(self) -> bool:
        return self._inner.cancel() and super().cancel()


def _patch_process_pool_submit() -> None:
    """Bring the files a ``ProcessPoolExecutor`` task read back to the submitter.

    A cached orchestrator that fans work out to a process pool read its data in
    the workers, where no tracker of the parent's can see: after a data fix in
    one input it served the pre-fix report, while the thread-pool version beside
    it invalidated. With a tracker active, ``submit`` (which
    ``Executor.map`` calls, chunked or not) sends a `_ReadsInWorker` instead of
    the bare function, and credits what it read to the submitting call when the
    result comes back -- before the caller can see the result, so before the
    call that waits on it is stored. ``multiprocessing.Pool`` and joblib are
    not wrapped: their reads stay unseen, and ``file_depends_on=`` names them.
    """
    # Local: this loads multiprocessing, which `import cash` must not pay for.
    import concurrent.futures.process as cf_process

    def make(original):
        @functools.wraps(original)
        def submit(self, fn, /, *args, **kwargs):
            tracker = active_tracker.get()
            if tracker is None or getattr(self, "_cash_internal", False):
                return original(self, fn, *args, **kwargs)
            inner = original(self, _ReadsInWorker(fn, type(tracker)), *args, **kwargs)
            outer = _RelayFuture(inner)

            def relay(done: concurrent.futures.Future) -> None:
                if done.cancelled():
                    outer.cancel()
                    return
                error = done.exception()
                if error is not None:
                    outer.set_exception(error)
                    return
                result = done.result()
                if isinstance(result, _WorkerReads):
                    for path in result.files:
                        tracker.add_tracked(path)
                    tracker.absent_files.update(result.absent)
                    result = result.value
                outer.set_result(result)

            inner.add_done_callback(relay)
            return outer

        return submit

    _patch_attribute(cf_process.ProcessPoolExecutor, "submit", make)


class FileDependencyRegistry:
    """
    Registry for file dependency handlers.
    Allows easy extension of file tracking to new libraries and functions.

    The process has one, :func:`file_registry`; ``Cash.register_file_handler``
    adds to it.
    """

    def __init__(self) -> None:
        self.handlers: dict[str, list[tuple[str, Callable[..., Any]]]] = {}  # module -> [(func_name, factory)]
        self._revision = 0
        self._ready = False
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default handlers for readers no audit event reports.

        ``open`` itself, and every reader that opens its file through it
        (``json``/``pickle``/``joblib``/``numpy`` loaders, ``Path.read_text``),
        arrive as the ``open`` audit event instead (see `_on_open`), as do
        ``os.listdir``, ``os.scandir`` and ``glob``. What is left here opens
        files in C, or answers from a cache or a stat that raises no event.
        """
        # Pandas: several readers go to C libraries (read_hdf, read_orc,
        # read_spss), and a remote URL is recorded here before any fetch.
        self.register("pandas", "read_*", self._create_path_arg_handler)

        # Polars
        self.register("polars", "read_csv", self._create_path_arg_handler)
        self.register("polars", "read_parquet", self._create_path_arg_handler)
        self.register("polars", "read_json", self._create_path_arg_handler)
        self.register("polars", "read_ndjson", self._create_path_arg_handler)
        self.register("polars", "read_ipc", self._create_path_arg_handler)
        self.register("polars", "read_avro", self._create_path_arg_handler)
        self.register("polars", "read_excel", self._create_path_arg_handler)
        self.register("polars", "scan_csv", self._create_path_arg_handler)
        self.register("polars", "scan_parquet", self._create_path_arg_handler)
        self.register("polars", "scan_ipc", self._create_path_arg_handler)
        self.register("polars", "scan_ndjson", self._create_path_arg_handler)

        # pyarrow reads in C++, so nothing passes through Python's open(): a
        # cached function that switched to pyarrow.csv for speed recorded no
        # file dependency at all, and a whole new export returned yesterday's
        # numbers. Path-taking readers only -- a class such as
        # ParquetFile is left alone, since replacing it with a function would
        # break isinstance checks.
        self.register("pyarrow.csv", "read_csv", self._create_path_arg_handler)
        self.register("pyarrow.csv", "open_csv", self._create_path_arg_handler)
        self.register("pyarrow.parquet", "read_table", self._create_path_arg_handler)
        self.register("pyarrow.parquet", "read_pandas", self._create_path_arg_handler)
        self.register("pyarrow.feather", "read_table", self._create_path_arg_handler)
        self.register("pyarrow.feather", "read_feather", self._create_path_arg_handler)
        self.register("pyarrow.json", "read_json", self._create_path_arg_handler)

        # pyarrow.dataset reads in C++ like the rest of pyarrow; `read_table`
        # was registered and `dataset()` was not, so one entry point of an
        # otherwise-covered library went stale.
        self.register("pyarrow.dataset", "dataset", self._create_path_arg_handler)

        # linecache answers from its own cache, so a file read once is not
        # opened again and no audit event reports the next lookup. Source files
        # are left out: linecache is what `inspect.getsource` (and every
        # traceback) reads with, and recording those made a module's own source
        # a data dependency of the functions in it. The opens it does make are
        # skipped by `_on_open` for the same reason.
        self.register("linecache", "getline", self._create_source_reader_handler)
        self.register("linecache", "getlines", self._create_source_reader_handler)

        # sqlite3 opens the database in C, so nothing reaches a patched
        # reader: a cached `select sum(x)` returned 1 where an uncached run
        # returned 101 after an INSERT, and `pd.read_sql_query` over the same
        # connection did too.
        # The connection's path is the dependency; a URI or ":memory:" has no
        # file behind it and `_track_path` drops what it cannot resolve.
        self.register("sqlite3", "connect", self._create_path_arg_handler)
        self.register("sqlite3.dbapi2", "connect", self._create_path_arg_handler)

        # Existence probes: "is there a config here?" The ABSENCE of a file is
        # an input -- it selects the defaults branch -- and it was the only
        # input cash could not see, because a file that is never opened
        # produces no read to track. An entry written by a run that found
        # nothing recorded no dependencies at all, so it looked valid
        # everywhere: directory B's answer came back in directory A,
        # silently. Only a NEGATIVE result is recorded; a probe that
        # says yes is followed by the read that tracks it properly. `os.stat`
        # raises no audit event.
        self.register("os.path", "exists", self._create_exists_handler)
        self.register("os.path", "isfile", self._create_exists_handler)
        self.register("genericpath", "exists", self._create_exists_handler)
        self.register("genericpath", "isfile", self._create_exists_handler)
        self._ready = True

    def register(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]):
        """
        Register a file tracking handler for a specific function.

        Args:
            module_name: Name of the module (e.g., 'pandas', 'builtins').
            func_name: Name of the function to track. Supports wildcards like 'read_*'.
            handler_factory: A function that takes (original_function, tracker_callback)
                             and returns a wrapper function.
        """
        if module_name not in self.handlers:
            self.handlers[module_name] = []
        self.handlers[module_name].append((func_name, handler_factory))
        self._revision += 1
        # Registered while a tracker is open: wrap the module now, as opening
        # the next tracker would, rather than miss the reads of this one.
        module = sys.modules.get(module_name)
        if module is not None and self._ready and io_watch.holding():
            _install_module_patches(module_name, module)

    def get_handlers_for_module(self, module_name: str) -> list[tuple[str, Callable[..., Any]]]:
        return self.handlers.get(module_name, [])

    # --- Standard Handler Factories ---

    @staticmethod
    def _create_path_arg_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Generic handler for functions where the first argument is a path.

        ``track_callback`` is part of the user-facing handler factory
        signature (see :meth:`FileDependencyRegistry.register`) so custom
        factories can record the access. The built-in handlers ignore the
        argument and consult ``active_tracker`` directly — that way one patch
        serves any number of concurrent trackers.
        """

        # Positional OR keyword. The wrapper used to demand the path as its
        # first positional parameter, so while it was installed
        # `pd.read_csv(filepath_or_buffer=p)`, `np.load(file=p)` or
        # `pq.read_table(source=p)` raised TypeError EVERYWHERE in the process,
        # inside cached code or not. Measured while adding the pyarrow readers.
        @functools.wraps(original_func)
        def tracked_func(*args, **kwargs):
            target = args[0] if args else next((kwargs[k] for k in _PATH_KWARGS if k in kwargs), None)
            if isinstance(target, (str, bytes, os.PathLike)):
                _tracker = active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(target)
                else:
                    note_untracked_read(target, sys._getframe(1))
            return original_func(*args, **kwargs)

        return tracked_func

    @staticmethod
    def _create_exists_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Record a path that was looked for and was NOT there.

        Only the negative case. A probe that finds the file is followed by the
        read that records it properly, and recording it here as well would add
        a second, weaker entry for the same path.

        This one wraps a genuinely hot function, so it does the cheapest thing
        that can work: call through first, and only consult the tracker when
        the answer was False.
        """

        @functools.wraps(original_func)
        def tracked_exists(path, *args, **kwargs):
            result = original_func(path, *args, **kwargs)
            if not result:
                _tracker = active_tracker.get()
                if _tracker is not None and isinstance(path, (str, bytes, os.PathLike)):
                    _tracker._track_absent(path)
            return result

        return tracked_exists

    #: Suffixes of files that hold code, not data (see `_create_source_reader_handler`).
    _SOURCE_SUFFIXES = (".py", ".pyc", ".pyw", ".pyi", ".pyx")

    @staticmethod
    def _create_source_reader_handler(original_func, track_callback):
        """Track a data file read through a SOURCE reader (``linecache``).

        Two things are skipped, both because this is the machinery
        `inspect.getsource` and every traceback read with:

        * Python files -- recording those makes a module's own source a data
          dependency of the functions defined in it.
        * Names that are not files at all. Compiled-from-memory code carries a
          pseudo-filename in angle brackets -- ``<string>``, ``<stdin>``,
          ``<ipython-input-3>``, and cash's own ``<cash-0beec9249e1e>`` for
          every notebook statement it executes. Recording those gave each
          statement a dependency on a file that cannot exist, which told the
          planner that a statement in an edited cell was still satisfied: the
          cell kept the previous run's live generator and printed an average
          over five values instead of two.

        So a name is tracked only if it is a real file on disk, which is the
        only thing linecache can usefully have read.
        """

        @functools.wraps(original_func)
        def tracked_source_reader(filename, *args, **kwargs):
            if isinstance(filename, (str, bytes, os.PathLike)):
                text = os.fsdecode(filename) if isinstance(filename, bytes) else str(filename)
                if not text.startswith("<") and not text.endswith(FileDependencyRegistry._SOURCE_SUFFIXES):
                    _tracker = active_tracker.get()
                    if _tracker is not None:
                        try:
                            real = os.path.isfile(text)
                        except (OSError, ValueError):
                            real = False
                        if real:
                            _tracker._track_path(filename)
            return original_func(filename, *args, **kwargs)

        return tracked_source_reader


_registry = FileDependencyRegistry()


def file_registry() -> FileDependencyRegistry:
    """The process's registry of reader handlers."""
    return _registry


class PostImportHook(importlib.abc.MetaPathFinder):
    """Intercepts imports of registered modules to patch them after loading.

    A single shared hook sits on ``sys.meta_path`` while a tracker is open
    (see `install_patches`). Module patching is tracker-agnostic —
    :func:`_install_module_patches` routes file reads via ``active_tracker``
    so the same patches serve every tracker.
    """

    def __init__(self) -> None:
        self._skip: set[str] = set()  # Avoid recursion

    def find_spec(self, fullname, path, target=None):
        if fullname in self._skip:
            return None

        # Only a module a handler is registered for: patching looks up
        # handlers by the exact module name, so wrapping the loader of any
        # other module -- every submodule of pandas, say -- patches nothing.
        if fullname not in file_registry().handlers:
            return None

        # It's a target. We need to let the real import happen, then patch.
        self._skip.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname, path)
        finally:
            self._skip.remove(fullname)

        if spec is None or spec.loader is None:
            return None

        # Wrap the loader
        spec.loader = _PatchingLoader(spec.loader, fullname)
        return spec


class _PatchingLoader:
    def __init__(self, original_loader, fullname):
        self.original_loader = original_loader
        self.fullname = fullname

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        # execute module
        self.original_loader.exec_module(module)

        # Now patch it via the module-level dispatcher installer —
        # tracker-agnostic, install-once-per-target, and only while a
        # tracker is still open.
        _install_module_patches(self.fullname, module)


# Shared meta_path hook, on ``sys.meta_path`` while a tracker is open.
_shared_import_hook = PostImportHook()


def install_patches() -> None:
    """Wrap the readers no audit event reports; run when the first tracker opens."""
    global _installed_for
    with _install_lock:
        registry = file_registry()
        # A cached call opens and closes a tracker every time it misses: while
        # the handlers and the imported modules are the ones the last install
        # saw, put the same wrappers back without looking anything up.
        basis = (id(registry), registry._revision, tuple(id(sys.modules.get(name)) for name in registry.handlers))
        if basis != _installed_for or not _patches.reinstall():
            for mod_name in registry.handlers:
                module = sys.modules.get(mod_name)
                if module is not None:
                    _install_module_patches(mod_name, module)
            _patch_pathlib_stat()
            # Work handed to a thread pool runs under the submitter's tracker, and
            # work handed to a process pool reports what it read back to it.
            _patch_thread_pool_submit()
            _patch_process_pool_submit()
            _installed_for = basis
        if _shared_import_hook not in sys.meta_path:
            sys.meta_path.insert(0, _shared_import_hook)


def remove_patches() -> None:
    """Put the originals back; run when the last tracker closes."""
    with _install_lock:
        try:
            sys.meta_path.remove(_shared_import_hook)
        except ValueError:
            pass
        _patches.restore()


class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **ContextVar dispatch**: Python-level opens and directory listings
    arrive as audit events (:mod:`cash.tracking.io_watch`); readers that open
    in C, existence probes and ``Path.stat`` get dispatcher wrappers, plus a
    meta-path import hook for libraries loaded later. Both consult a
    ``ContextVar`` (``active_tracker``) at *call* time to decide whether to
    record the access. ``__enter__`` sets that ContextVar to ``self`` and
    stores the token; ``__exit__`` ``reset()``s it.

    Because ``ContextVar`` values are isolated per ``asyncio.Task`` and
    per ``threading.Thread`` by default, concurrent trackers — e.g. two
    coroutines under ``asyncio.gather``, or two worker threads — each
    see only their own block's reads.

    **Installed while in use**: the first tracker to open installs the
    wrappers and the last one to close puts the originals back, so with no
    tracker open ``pd.read_csv`` is pandas' own function again.

    **Writes are not dependencies.** A file opened for writing is never
    recorded as an input: a statement's output is hashed directly, and keying
    on its own output file would invalidate it on every run. ``open()`` writes
    are reported to the active effect observer instead (see
    ``cash.effect_observer``).
    """

    def __init__(self, user_ns=None, propagate_to_parent: bool = False, hash_on_read: bool = False):
        self.accessed_files = set()
        # The top-level package of the code being cached, when *user_ns* is a
        # module's globals (the decorator): an installed tool's own files are
        # its data. A notebook's namespace is `__main__` -- no package.
        name = user_ns.get("__name__") if isinstance(user_ns, dict) else None
        self._own_package = name.split(".")[0] if isinstance(name, str) and name != "__main__" else None
        # The content hash of each regular file WHEN IT WAS FIRST READ, for a
        # caller that stores what the block read (the decorator). Taken at
        # store time instead, a file changed mid-call by a writer that moves no
        # timestamp -- an np.memmap write on Windows -- was fingerprinted as
        # the NEW file next to a result computed from the old one, and served
        # to every later process. Moving the hash here costs
        # nothing extra: the snapshot reuses it while the stat is unchanged.
        self._hash_on_read = hash_on_read
        self.read_digests: dict[str, str] = {}
        # When each of those digests was taken (``file_dep_is_fresh`` trusts an
        # unchanged file only if it had settled by then).
        self.read_hashed_at: dict[str, float] = {}
        #: Time spent hashing inside the block, which is cash's, not the body's.
        self.read_hash_seconds = 0.0
        # Remote URLs are kept in their own set, never in ``accessed_files``:
        # every consumer of that set stats/hashes its members, and a URL is not
        # a path. They are re-joined downstream as remote dependency entries.
        self.accessed_remote: set[str] = set()
        # Paths this block looked for and did not find. Recorded as their own
        # kind of dependency (``{'absent': True}``) so an entry computed
        # WITHOUT an optional file stops being valid once that file appears --
        # including when the same relative name resolves into a directory that
        # has one. See ``_track_absent``.
        self.absent_files: set[str] = set()
        # The stat of each regular file WHEN IT WAS FIRST READ. The entry's
        # fingerprint is taken when it is stored, after the body has finished,
        # so a file that changed in between was fingerprinted as if it were
        # what the body read -- and served, stale, forever after.
        # Comparing against this is what lets the store step refuse instead.
        self.read_stats: dict[str, tuple[int, int, int]] = {}
        self.user_ns = user_ns or {}
        # Stack of ContextVar tokens, one per active __enter__. Supports
        # re-entry of the same instance (an async function that reuses
        # a tracker across awaits, or a sync caller using `with` twice).
        self._token_stack: list[contextvars.Token] = []
        # When True, a file read in this block also registers with the
        # enclosing tracker(s) - so an OUTER cached function records the files
        # its (cold) inner cached calls read, otherwise the outer entry would
        # never invalidate when that file changes (file deps don't propagate
        # through depends_on). The decorator sets this; manual `with
        # FileAccessTracker()` nesting stays isolated by default.
        self._propagate_to_parent = propagate_to_parent
        self._parent_stack: list[Optional["FileAccessTracker"]] = []
        # The user code that read a file in THIS block (see
        # `credit_read_to_stack`): its recorded reads are live, not remembered.
        self.reading_codes: set[Any] = set()
        # Files a memo handed this block data from that was read from an
        # EARLIER version of the file (see `Cash._credit_remembered_reads`).
        self.stale_memo_reads: set[str] = set()

    def __enter__(self):
        # The first open tracker installs the wrappers (see `install_patches`).
        io_watch.hold()
        # Capture the enclosing tracker (if any) BEFORE we become active, so a
        # read inside this block also registers with the outer tracker(s).
        self._parent_stack.append(active_tracker.get())
        self._token_stack.append(active_tracker.set(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token_stack:
            active_tracker.reset(self._token_stack.pop())
            io_watch.release()
        if self._parent_stack:
            self._parent_stack.pop()

    def _propagation_parent(self) -> FileAccessTracker | None:
        """The enclosing tracker a record is passed up to, or None when this
        tracker is isolated (the default for manual nesting)."""
        if not self._propagate_to_parent or not self._parent_stack:
            return None
        parent = self._parent_stack[-1]
        return parent if parent is not self else None

    def suspend(self):
        """Stop tracking until :meth:`resume`, restoring the enclosing tracker.

        For a streaming cached generator: production is tracked, the caller's
        loop body is not. `__enter__` cannot be used per item -- it took 5.1us
        against 0.15us for the ContextVar swap alone, which on a 200k-item
        iterator is over a second of pure bookkeeping.
        """
        parent = self._parent_stack[-1] if self._parent_stack else None
        return active_tracker.set(parent)

    def resume(self, token) -> None:
        """Undo :meth:`suspend`."""
        active_tracker.reset(token)

    def get_accessed_files(self) -> set[str]:
        return self.accessed_files

    def inputs_changed_since_read(self) -> list[str]:
        """Regular files whose stat moved between their first read and now.

        Directories are left out on purpose: a function that lists a directory
        and writes its output into it moves the directory's mtime itself, and
        refusing to cache every such function would be a regression for no
        gain -- a new entry appearing mid-call is a smaller hole than a file
        rewritten under the reader.
        """
        moved = []
        for path, before in self.read_stats.items():
            if regular_file_stat(path) != before:
                moved.append(path)
        return moved

    def get_accessed_remote_urls(self) -> set[str]:
        """Remote URLs read in this block, tracked by store validator instead."""
        return self.accessed_remote

    def get_absent_files(self) -> set[str]:
        """Paths this block looked for and did not find."""
        return self.absent_files

    def _track_path(self, path):
        global _tracking_seconds
        started = _perf_counter()
        try:
            self._track_path_untimed(path)
        finally:
            _tracking_seconds += _perf_counter() - started

    def _track_path_untimed(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            # ``open(3)`` opens a file DESCRIPTOR: joblib and loky do, and
            # ``str(3)`` was recorded as a read of ``<cwd>/3`` -- a directory
            # every file under the cwd sits in, so every write there read as
            # an input.
            return
        raw_path = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if is_pseudo_fs(raw_path):
            # See `is_pseudo_fs`. Checked BEFORE realpath, which on
            # Windows rewrites /proc/... to C:/proc/... and would slip past.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", raw_path)
            return
        if is_remote_url(raw_path):
            # A remote URL is a real dependency, just not a stat-able one:
            # ``realpath`` would mangle it into a nonexistent local path and the
            # dependency would vanish. Record it on the remote channel, where it
            # is tracked by the store's own validator.
            self.add_tracked_remote(raw_path)
            return
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls. Resolved once per cell run
            # (``realpath_this_run``): a loop reads the same files again.
            # Local: import cycle tracking.file_tracker -> tracking.file_dep_snapshot -> ... -> tracking.file_tracker.
            from cash.tracking.file_dep_snapshot import realpath_of_read_this_run

            resolved, read_lstat = realpath_of_read_this_run(raw_path)
            abs_path = normalize_path(resolved)
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)
            return
        if is_pseudo_fs(abs_path):
            # See `is_pseudo_fs`: recording one of these makes the entry
            # permanently unfreshenable. Return before the relative-path arm
            # too — these paths are always absolute.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", abs_path)
            return
        if SCRATCH_MEMMAP in abs_path:
            # joblib's memmaps of a parallel call's arrays: deleted when the
            # call returns, so recorded, every entry that read them was stale
            # for ever -- a ``cross_val_predict(n_jobs=4)`` loop re-ran on
            # every run of the report cell.
            logger.debug("[TRACKER] Ignoring joblib scratch read %r", abs_path)
            return
        if RUNTIME_CACHE_SEGMENT in abs_path or abs_path.endswith(RUNTIME_CACHE_SUFFIXES):
            logger.debug("[TRACKER] Ignoring runtime-cache read %r", abs_path)
            return
        if is_cash_internal(abs_path):
            # See `is_cash_internal`. Checked after realpath so a relative
            # or symlinked cache path is caught too.
            logger.debug("[TRACKER] Ignoring cash-internal read %r", abs_path)
            return
        why = incidental_read(abs_path, self._own_package)
        if why is not None:
            logger.debug("[TRACKER] Ignoring %s read %r", why, abs_path)
            return
        self.add_tracked(abs_path, lstat=read_lstat)
        try:
            credit_read_to_stack(abs_path, self)
        except Exception:  # noqa: BLE001 - attribution is an aid; the read counts regardless
            logger.debug("[TRACKER] Could not credit %r to the stack", abs_path, exc_info=True)
        # a RELATIVE read path also records the UN-resolved relative
        # string as its own dependency. The realpath above is frozen to the cwd
        # at track time, so after an ``os.chdir`` edit it still looks fresh even
        # though a re-run would read a different file. The relative dep is
        # re-resolved against the CURRENT cwd at each freshness check
        # (``resolve_file_dep_path``), so a collision - a different file with the
        # same relative name in the new directory - is caught via its mtime/size.
        try:
            raw = str(path)
            if raw and not os.path.isabs(raw):
                rel = normalize_path(raw)
                if rel != abs_path:
                    self.add_tracked(rel)
            elif raw:
                # An ABSOLUTE path through a junction or symlink gets the same
                # treatment: its unresolved form is recorded too. The realpath
                # above resolved the link at WRITE time, so after the link is
                # re-pointed every check stats the old target -- which still
                # exists and has not changed; a rollback would return the
                # NEWER release's answer. Checked through the link as it
                # points NOW, the switch is seen; the realpath entry keeps
                # catching an edit to the target itself.
                link = normalize_path(os.path.abspath(raw))
                if os.path.normcase(link) != os.path.normcase(abs_path):
                    self.add_tracked(link)
        except (TypeError, ValueError, OSError):
            logger.debug("[TRACKER] Could not record unresolved path for %r", path)

    def add_tracked(self, abs_path: str, digest: str | None = None, lstat: Any = None) -> None:
        """Record *abs_path* on this tracker and, when propagation is enabled,
        on the enclosing tracker(s) too - so nested cached reads count as the
        outer cached function's deps. Manual tracker nesting stays isolated.

        *digest* is the content hash an inner tracker already took of the same
        read, handed up so the outer one does not hash the file again.

        *lstat* is the stat taken while resolving the path, of a regular file
        that is not a link, so it is the stat the file would have given."""
        self.accessed_files.add(abs_path)
        if abs_path not in self.read_stats and os.path.isabs(abs_path):
            # Absolute paths only: a relative twin is re-resolved against the
            # cwd at check time, and a chdir during the call would make its
            # stat look like a change that never happened.
            st = (
                regular_file_stat(abs_path)
                if lstat is None
                else (lstat.st_size, lstat.st_mtime_ns, getattr(lstat, "st_ctime_ns", 0))
            )
            if st is not None:
                self.read_stats[abs_path] = st
                if self._hash_on_read:
                    if digest is None:
                        self.read_hashed_at[abs_path] = time.time()
                        digest = self._digest_now(abs_path, st[0])
                    if digest is not None:
                        self.read_digests[abs_path] = digest
        elif digest is None:
            digest = self.read_digests.get(abs_path)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked(abs_path, digest)

    def _digest_now(self, abs_path: str, size: int) -> str | None:
        """The file's content hash as the body is about to read it."""
        # Local: import cycle tracking.file_tracker -> tracking.file_dep_snapshot -> ... -> tracking.file_tracker.
        from cash.tracking.file_dep_snapshot import file_content_hash

        t0 = _perf_counter()
        try:
            return file_content_hash(abs_path, size)
        finally:
            self.read_hash_seconds += _perf_counter() - t0

    def _note_reading_code(self, code: Any) -> None:
        self.reading_codes.add(code)
        parent = self._propagation_parent()
        if parent is not None:
            parent._note_reading_code(code)

    def _track_absent(self, path) -> None:
        """Record *path* as looked-for-and-missing.

        Kept as WRITTEN, not resolved: a relative probe is about "a file with
        this name, here", and that is exactly the thing that must be
        re-evaluated against the live cwd on the next run. Resolving it would
        freeze the directory the probe happened to run in -- which is the bug
        this exists to close, in mirror image.

        The same filters as ``_track_path``: kernel pseudo-filesystems and
        cash's own storage are not user dependencies.
        """
        try:
            raw = str(path)
        except (TypeError, ValueError):
            return
        if not raw or is_pseudo_fs(raw):
            return
        try:
            normalized = normalize_path(raw)
        except (TypeError, ValueError):
            return
        if is_cash_internal(normalized):
            return
        if os.path.isabs(raw):
            # An absolute probe is about one fixed file, so record it resolved
            # the way a read of it would be -- otherwise the two spellings of
            # the same path would not match.
            try:
                normalized = normalize_path(os.path.realpath(raw))
            except (TypeError, ValueError, OSError):
                pass
            if is_pseudo_fs(normalized) or is_cash_internal(normalized):
                return
        # A library probing for an optional file while it is imported
        # (matplotlib looks for a `matplotlibrc` in the working directory) or
        # inside its own package is not the user's question either.
        if incidental_read(os.path.abspath(raw), self._own_package) is not None:
            return
        self.add_tracked_absent(normalized)

    def add_tracked_absent(self, path: str) -> None:
        """Record an absent path here and, when propagating, on the parents."""
        self.absent_files.add(path)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked_absent(path)

    def add_tracked_remote(self, url: str) -> None:
        """Record a remote *url* read, propagating to the enclosing tracker."""
        self.accessed_remote.add(url)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked_remote(url)


#: Seconds spent recording reads; `tracking_seconds`.
_tracking_seconds = 0.0


def tracking_seconds() -> float:
    """Seconds cash has spent recording file reads in this process.

    Read before and after a statement, the difference is cash's own time inside
    it, which is not the statement's cost: a folder read recorded 16.5 s for a
    load that takes 1.8 s without cash, and a later hit credited all of it as
    saved.
    """
    return _tracking_seconds


# What feeds a tracker: the audit events Python raises for its own opens and
# listings, and wrappers on the readers that raise none, installed while one
# is open.
subscribe_read_events()
io_watch.add_patcher(install_patches, remove_patches)
