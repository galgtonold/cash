"""Wrappers on the readers that raise no audit event.

Readers that open files in C (pyarrow, polars, sqlite3, some pandas readers),
existence probes and ``Path.stat`` raise no audit event, so they are wrapped
while a tracker is open: `FileDependencyRegistry` says which functions and
how, and a meta-path hook wraps a registered module imported meanwhile. The
executor ``submit`` methods are wrapped too, to carry the tracker into worker
threads and bring back what worker processes read. Every wrapper looks the
tracker up in `active_tracker` at call time, so one install serves every
tracker.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread as cf_thread
import contextvars
import functools
import importlib.abc
import importlib.util
import logging
import os
import pathlib
import stat
import sys
import threading
from collections.abc import Callable
from typing import Any

from cash.tracking import io_watch
from cash.tracking._memo import Memo
from cash.tracking.read_credit import note_untracked_read
from cash.tracking.tracker_context import active_tracker

__all__ = ["FileDependencyRegistry", "PostImportHook", "file_registry", "install_patches", "remove_patches"]

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
        arrive as the ``open`` audit event instead (see `read_events`), as do
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
