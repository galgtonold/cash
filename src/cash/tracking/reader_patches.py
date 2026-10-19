"""Wrappers on the calls that raise no audit event.

These are wrapped while a tracker is open; this list is the one place that
names them:

- readers that open files in C: pandas, polars, pyarrow, soundfile and ``sqlite3``
  (`FileDependencyRegistry` says which functions and how, and a meta-path
  hook wraps a registered module imported meanwhile);
- existence probes: ``os.path.exists``, ``lexists``, ``isfile``, ``isdir``
  and ``os.access``;
- metadata calls: ``Path.stat``, ``os.stat``, ``os.lstat``,
  ``os.path.getsize``, ``getmtime``, ``getctime`` and the entries of
  ``os.scandir``;
- ``linecache.getline`` and ``getlines``, which read source files;
- the ``submit`` of thread and process pool executors and
  ``multiprocessing.Pool``, to carry the tracker into worker threads and
  bring back what worker processes read.

Every wrapper looks the tracker up in `active_tracker` at call time, so one
install serves every tracker.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread as cf_thread
import contextvars
import functools
import glob
import importlib.abc
import importlib.util
import logging
import os
import pathlib
import stat
import sys
import threading
import types
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from cash import _native
from cash._memo import PATCH_SITES, LruMemo
from cash._paths import CELL_NAME_PREFIX, is_pseudo_filename, is_remote_url
from cash.effect_observer import active_observer as _active_effect_observer
from cash.install_paths import is_user_code_file, is_user_path
from cash.remote_source import remember_read_options
from cash.tracking import io_watch
from cash.tracking.read_credit import frame_kind, note_untracked_read
from cash.tracking.tracker_context import active_tracker

__all__ = ["FileDependencyRegistry", "PostImportHook", "file_registry", "install_patches", "remove_patches"]

logger = logging.getLogger(__name__)

# The wrappers installed while a tracker is open (see `install_patches`),
# and the lock that serialises installing and removing them with the post-
# import hook patching a module that was imported meanwhile.
_cw = _native.cwatch
# Told each wrapper and its original: a swap between them is no change for a
# remembered hit (`cash.decorator.hit_memo`), as keys look through it.
_patches = io_watch.Patches(note=getattr(_cw, "world_swap", None))
_install_lock = threading.RLock()

#: Keyword names a reader may take its path under, first match wins. Covers
#: pandas (`filepath_or_buffer`, `path_or_buf`, `io` for read_excel, `path`),
#: numpy (`file`, `fname`), joblib (`filename`), pyarrow (`source`,
#: `input_file`) and polars (`source`).
_PATH_KWARGS = (
    "filepath_or_buffer",
    "path_or_buf",
    "source",
    "input_file",
    "path",
    "file",
    "fname",
    "filename",
    "io",
    "filename_or_obj",
    "fp",
)


def _dispatch_track(path: Any) -> None:
    """The ``tracker_callback`` every handler factory receives, including
    those registered with :meth:`Cash.register_file_handler`.

    It looks up ``active_tracker`` at *call* time, so a wrapper calling
    ``tracker_callback(path)`` records into whichever tracker is active on
    the current asyncio task or thread, as the built-in handlers do.
    """
    _tracker = active_tracker.get()
    if _tracker is not None:
        _tracker.track_path(path)


#: (module id, pattern) -> (namespace size, matched names); `_find_patch_targets`.
_patch_targets: LruMemo[tuple[int, str], tuple[int, list]] = LruMemo(PATCH_SITES)
#: (owner id, name, kind) -> (original, wrapper); `_install_wrapper`.
_wrappers: LruMemo[tuple[int, str, Any], tuple[Any, Any]] = LruMemo(PATCH_SITES)
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
        _patch_targets[key] = (size, found)
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
        _wrappers[key] = (original, wrapper)
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
        for class_name, make_init in _CLASS_READERS.get(module_name, ()):
            owner = getattr(module_obj, class_name, None)
            if isinstance(owner, type):
                _patch_attribute(owner, "__init__", make_init)


def _make_init_path_handler(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a reader class's ``__init__(self, source, ...)``."""
    path_handler = FileDependencyRegistry._create_path_arg_handler(lambda *a, **k: None, _dispatch_track)

    @functools.wraps(original)
    def tracked_init(self, *args, **kwargs):
        path_handler(*args, **kwargs)
        return original(self, *args, **kwargs)

    return tracked_init


def _make_soundfile_init_handler(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``soundfile.SoundFile.__init__(self, file, mode='r', ...)``.

    A file opened to read (``'r'``, ``'r+'``) is a dependency; one opened only
    to write (``'w'``, ``'x'``) is an output, not an input.
    """
    path_handler = FileDependencyRegistry._create_path_arg_handler(lambda *a, **k: None, _dispatch_track)

    @functools.wraps(original)
    def tracked_init(self, *args, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if not isinstance(mode, str) or "r" in mode:
            path_handler(*args, **kwargs)
        return original(self, *args, **kwargs)

    return tracked_init


def _opens_to_read(mode: Any) -> bool:
    """Does an ``open(path, mode)``-style *mode* read the file that is there?

    ``'r'``, ``'r+'`` and HDF5's ``'a'`` (read/write, created if missing) do;
    ``'w'``, ``'w-'`` and ``'x'`` only write, so the file is an output.
    """
    return mode is None or not isinstance(mode, str) or "r" in mode or "a" in mode


def _make_h5py_init_handler(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``h5py.File.__init__(self, name, mode='r', ...)``.

    libhdf5 opens the file in C, so no audit event reports it; on a system
    without the native file hook (macOS) this is the only record of the read.
    """
    path_handler = FileDependencyRegistry._create_path_arg_handler(lambda *a, **k: None, _dispatch_track)

    @functools.wraps(original)
    def tracked_init(self, *args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        if _opens_to_read(args[1] if len(args) > 1 else kwargs.get("mode", "r")):
            path_handler(name)
        return original(self, *args, **kwargs)

    return tracked_init


#: Reader CLASSES, by module, with the wrapper each one's ``__init__`` gets:
#: the class is left in place, so ``isinstance`` still works, and its
#: ``__init__`` records the path it is given.
_CLASS_READERS: dict[str, tuple[tuple[str, Callable[[Any], Any]], ...]] = {
    "pyarrow.parquet": (("ParquetFile", _make_init_path_handler),),
    "soundfile": (("SoundFile", _make_soundfile_init_handler),),
    "h5py": (("File", _make_h5py_init_handler),),
}


#: id(namespace) -> (its size when scanned, the names bound to a wrapped
#: reader); `_patch_aliases_in`.
_alias_scans: LruMemo[int, tuple[int, tuple[str, ...]]] = LruMemo(PATCH_SITES)
#: (`_patches` version, ``id(original) -> (original, wrapper)``); `_wrapped_originals`.
_originals_memo: list[Any] = [None, {}]


def _wrapped_originals() -> dict[int, tuple[Any, Any]]:
    """``id(original) -> (original, cash's wrapper)`` for every module-level
    reader wrapped now."""
    if _originals_memo[0] == _patches.version:
        return _originals_memo[1]
    found: dict[int, tuple[Any, Any]] = {}
    for owner, _name, new, old in _patches.installed():
        if not isinstance(owner, types.ModuleType) or not callable(old):
            continue
        if _is_patch(new) and getattr(new, "_original_func", None) is old:
            found[id(old)] = (old, new)
    _originals_memo[:] = [_patches.version, found]
    return found


def _is_user_module(module: Any) -> bool:
    """A module of the user's own code, loaded from a file: a script or an
    import. Not a notebook's namespace -- see `patch_reader_aliases`."""
    return isinstance(module, types.ModuleType) and is_user_path(getattr(module, "__file__", None))


def _patch_aliases_in(module: types.ModuleType) -> None:
    """Point the names in *module* that hold a wrapped reader at its wrapper.

    ``from pyarrow.parquet import read_table`` at the top of a module binds the
    ORIGINAL function, before any tracker opened, and wrapping the module
    attribute does nothing for that name: it must be rebound to be seen as
    ``pq.read_table(...)`` is. A plain alias (``reader = pl.read_csv``) is the same case. The wrapper
    stands for the library's function in a cache key, as the patched module
    attribute always has. Remembered per namespace while its size holds, so a
    module with no such name (most) costs one lookup.
    """
    ns = module.__dict__
    size = len(ns)
    known = _alias_scans.get(id(ns))
    originals = None
    if known is not None and known[0] == size:
        names: tuple[str, ...] = known[1]
    else:
        originals = _wrapped_originals()
        try:
            names = tuple(name for name, value in list(ns.items()) if id(value) in originals)
        except RuntimeError:  # changed size while scanned: another thread importing
            return
        _alias_scans[id(ns)] = (size, names)
    if not names:
        return
    if originals is None:
        originals = _wrapped_originals()
    for name in names:
        value = ns.get(name)
        hit = originals.get(id(value))
        if hit is not None and hit[0] is value:
            _patches.replace(module, name, hit[1])


def _modules_now(modules: Mapping[str, Any] | None = None) -> list[tuple[str, Any]]:
    """The items of *modules* (``sys.modules``), listed again when an import
    changed it while they were listed.

    Listing allocates a tuple per module, and an allocation can run the
    collector, whose finalizers run Python code and let another thread in:
    its import changed ``sys.modules`` mid-list, the tracker's install
    raised "dictionary changed size during iteration", and every read the
    worker tracked after that went unrecorded (macOS 3.10 CI)."""
    if modules is None:
        modules = sys.modules
    for _attempt in range(16):
        try:
            return list(modules.items())
        except RuntimeError:
            continue
    return list(dict(modules).items())


#: (number of modules when listed, the user's modules then); `_patch_user_module_aliases`.
_user_modules: list[Any] = [-1, ()]


def _patch_user_module_aliases() -> None:
    """`_patch_aliases_in` every module of the user's own code."""
    if not _wrapped_originals():
        return
    if _user_modules[0] != len(sys.modules):
        _user_modules[:] = [len(sys.modules), tuple(m for _name, m in _modules_now() if _is_user_module(m))]
    for module in _user_modules[1]:
        _patch_aliases_in(module)


#: (number of modules when listed, the library submodules then); `_patch_library_reexports`.
_library_modules: list[Any] = [-1, ()]
#: module name -> (its namespace size when scanned, the names bound to a
#: wrapped reader). Unbounded: one entry per imported library module, which
#: lives as long as the process.
_library_scans: dict[str, tuple[int, tuple[str, ...]]] = {}


def _library_tops() -> frozenset[str]:
    """The third-party packages a reader is registered for (polars, pandas, ...)."""
    tops = {name.split(".", 1)[0] for name in file_registry().handlers}
    return frozenset(t for t in tops if t not in sys.stdlib_module_names)


def _patch_library_reexports() -> None:
    """Wrap a reader where its library also exports it from a submodule.

    polars defines ``read_parquet`` in ``polars.io.parquet.functions`` and
    exports the same function from ``polars.io.parquet``, ``polars.io`` and
    ``polars``; cash registers ``polars.read_parquet`` only. ``from
    polars.io.parquet import read_parquet`` then bound the original, even under
    a tracker, and the read went unrecorded. Every submodule of a registered
    library that holds a wrapped original gets the wrapper too.
    """
    originals = _wrapped_originals()
    if not originals:
        return
    if _library_modules[0] != len(sys.modules):
        tops = _library_tops()
        found = []
        for name, module in _modules_now():
            top, dot, _ = name.partition(".")
            if dot and top in tops and isinstance(module, types.ModuleType):
                found.append((name, module))
        _library_modules[:] = [len(sys.modules), tuple(found)]
    for name, module in _library_modules[1]:
        ns = getattr(module, "__dict__", None)
        if not isinstance(ns, dict):
            continue
        size = len(ns)
        known = _library_scans.get(name)
        if known is not None and known[0] == size:
            names = known[1]
        else:
            try:
                names = tuple(n for n, v in list(ns.items()) if id(v) in originals)
            except RuntimeError:  # changed size while scanned: another thread importing
                continue
            _library_scans[name] = (size, names)
        for n in names:
            hit = originals.get(id(ns.get(n)))
            if hit is not None and hit[0] is ns.get(n):
                _patches.replace(module, n, hit[1])


def rebind_reader_names(ns: dict) -> None:
    """Point the names in a notebook's namespace that hold an original reader
    at cash's wrapper. Called before a cell runs, never inside a statement.

    A name bound before cash's first tracked statement (an import above
    ``%cash_on``, an IPython startup file) holds the original function, and
    a call through it recorded no file. The wrapper stands for the same
    library function in a key (it carries its module and name), and outside
    a tracker it only calls the original.
    """
    if not isinstance(ns, dict):
        return
    if not len(_wrappers):
        # No tracker has opened yet in this process: open and close one, so
        # the wrappers exist to point at.
        io_watch.hold()
        io_watch.release()
    wrapped = {}
    for key in _wrappers.keys():
        entry = _wrappers.peek(key)
        if entry is not None and isinstance(key[0], int) and callable(entry[0]):
            wrapped[id(entry[0])] = entry
    if not wrapped:
        return
    try:
        items = list(ns.items())
    except RuntimeError:
        return
    for name, value in items:
        hit = wrapped.get(id(value))
        if hit is not None and hit[0] is value and not name.startswith("_"):
            ns[name] = hit[1]


def patch_reader_aliases(ns: Any) -> None:
    """`_patch_aliases_in` the module whose globals a tracker was opened for
    (the decorator's cached function), as its tracker opens.

    Catches a name bound after the patches went in. A notebook's namespace is
    left alone: rebinding a name there in the middle of a statement would read
    as the statement assigning it, and change its lineage.
    """
    if not isinstance(ns, dict) or not io_watch.holding():
        return
    name = ns.get("__name__")
    module = sys.modules.get(name) if isinstance(name, str) else None
    if module is None or getattr(module, "__dict__", None) is not ns or not _is_user_module(module):
        return
    with _install_lock:
        _patch_aliases_in(module)


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

    What a metadata call reports is the file's, so the file is a dependency;
    a directory has no content to hash. ``os.stat``, not ``os.path.isfile``:
    that one is patched to record a NEGATIVE answer as an absent dependency.
    """
    tracker = active_tracker.get()
    if tracker is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    try:
        if stat.S_ISREG(os.stat(path).st_mode):
            tracker.track_path(path)
    except (OSError, ValueError, TypeError):
        return


def _patch_pathlib_stat() -> None:
    """Track the file ``Path.stat()`` looks at: a regular file by content, a
    directory by being there, and a path that is not there as absent -- which
    is what ``Path.exists()``, ``is_file()`` and ``is_dir()`` see through it.

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
            try:
                result = original(self, *args, **kwargs)
            except (FileNotFoundError, NotADirectoryError):
                # ``Path.exists()`` / ``is_file()`` answering False.
                _record_path_answer(self, None, sys._getframe(1))
                raise
            if stat.S_ISREG(result.st_mode):
                _record_path_answer(self, "content", sys._getframe(1))
            elif stat.S_ISDIR(result.st_mode):
                # ``Path("out").is_dir()``: a directory has no content to
                # depend on, but it being there is what the code asked.
                _record_path_answer(self, "dir", sys._getframe(1))
            return result

        return tracked_path_stat

    _patch_attribute(owner, "stat", make)


def _patch_thread_pool_submit() -> None:
    """Run work submitted to a ``ThreadPoolExecutor`` under the submitter's context.

    The tracker is found through a ContextVar, and a pool's worker threads
    start with an empty context, so without this ``ex.map(np.load, shards)``
    inside a cached function would read files no tracker sees.

    With a tracker active, ``submit`` (which ``Executor.map`` calls) wraps the
    call in ``copy_context().run``; with none, it is the original. A thread
    started directly with ``threading.Thread`` is `_patch_thread_start`'s.

    The tracker also counts the work until it is done (`work_submitted`):
    work still running when the block ends can read a file after the entry's
    fingerprints were taken, or change the result after it was stored.
    """

    def make(original):
        @functools.wraps(original)
        def submit(self, fn, /, *args, **kwargs):
            tracker = active_tracker.get()
            if tracker is None:
                return original(self, fn, *args, **kwargs)
            context = contextvars.copy_context()
            trackers = tracker.work_submitted(fn)

            def run(*args, **kwargs):
                try:
                    return context.run(fn, *args, **kwargs)
                finally:
                    tracker.work_finished(trackers)

            try:
                future = original(self, run, *args, **kwargs)
            except BaseException:
                tracker.work_finished(trackers)
                raise
            # Cancelled before it ran: `run` never will.
            future.add_done_callback(lambda done: done.cancelled() and tracker.work_finished(trackers))
            return future

        return submit

    _patch_attribute(cf_thread.ThreadPoolExecutor, "submit", make)


def _thread_function(thread: threading.Thread) -> Any:
    """What *thread* runs: its target, a ``Timer``'s function, or its class's
    own ``run``; None for a plain ``Thread`` with no target."""
    fn = getattr(thread, "_target", None)
    if fn is None and isinstance(thread, threading.Timer):
        fn = thread.function
    if fn is None and type(thread).run is not threading.Thread.run:
        fn = type(thread).run
    return fn


def _runs_user_code(thread: threading.Thread) -> bool:
    """Does *thread* run code of the user's: its target (a ``Timer``'s
    function), or its class's own ``run``? A library's thread (a pool's
    worker, a writer) runs its own loop, and any task handed to it later is
    the pool patches' to place."""
    fn = _thread_function(thread)
    for _ in range(4):  # a bound method's or a partial's function
        inner = getattr(fn, "__func__", None) or getattr(fn, "func", None)
        if inner is None:
            break
        fn = inner
    code = getattr(fn, "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


def _patch_thread_start() -> None:
    """Run a thread a tracked block starts, on code of the user's, under the
    block's context.

    A thread starts with an empty context, so the tracker the ContextVar
    holds is not there: ``Thread(target=lambda: open(p).read())`` in a cached
    function read a file no tracker saw. Only a module-level target was
    rescued, by the reads its code is remembered for (`credited_reads`); a
    nested function or a lambda was not, and the edited file's old result
    was served with no warning. The thread now runs in a copy of the
    starter's context, as ``ThreadPoolExecutor.submit`` gives each task.

    The tracker also counts the thread until its ``run`` ends, as it counts
    pool work (`work_submitted`): a thread the block does not join can read a
    file after the entry's fingerprints were taken.
    """

    def make(original):
        @functools.wraps(original)
        def start(self):
            tracker = active_tracker.get()
            if tracker is None or not _runs_user_code(self):
                return original(self)
            run, context = self.run, contextvars.copy_context()
            trackers = tracker.work_submitted(_thread_function(self))

            def counted_run():
                try:
                    context.run(run)
                finally:
                    tracker.work_finished(trackers)

            self.run = counted_run
            try:
                return original(self)
            except BaseException:
                self.run = run
                tracker.work_finished(trackers)
                raise

        return start

    _patch_attribute(threading.Thread, "start", make)


def _probe_handler(original_func: Callable[..., Any], kind: str) -> Callable[..., Any]:
    """Wrap an existence probe (``os.path.exists``, ``isfile``, ``isdir``,
    ``os.access``) to record its answer.

    False: the path is recorded as absent, so the entry is stale once it
    appears. True: as present (of *kind*), so the entry is stale once it is
    gone -- a flag file or an output folder that is checked but never read.
    A False from ``isfile``, ``isdir`` or ``access`` on a path that IS there
    (another kind, no permission) records what is there instead: that is
    what the answer rests on.

    This wraps a genuinely hot function, so it calls through first and only
    looks further while a tracker is open.
    """
    exact_negative = kind == "any" and getattr(original_func, "__name__", "") != "access"

    @functools.wraps(original_func)
    def tracked_probe(path, *args, **kwargs):
        result = original_func(path, *args, **kwargs)
        if active_tracker.get() is not None:
            if result:
                _record_path_answer(path, kind, sys._getframe(1))
            elif exact_negative:
                _record_path_answer(path, None, sys._getframe(1))
            else:
                _record_negative_probe(path, sys._getframe(1))
        return result

    return tracked_probe


#: Modules that only pass a path question on: the frame that matters is the
#: one that called them.
_PATH_MACHINERY = frozenset({"os", "posixpath", "ntpath", "genericpath", "pathlib", "pathlib._local", "pathlib._abc"})


#: The name prefix of a notebook statement cash compiled
#: (``cash.notebook.compiled_source.is_cash_filename``): code the user wrote in
#: a cell, though it has no file.
_NOTEBOOK_STATEMENT_PREFIX = CELL_NAME_PREFIX


def _who_asked(frame: Any) -> str:
    """`frame_kind` of the code that made the call *frame* made, looking
    through pathlib and ``os``, which only pass a path question on. A
    notebook statement is the user's code."""
    while frame is not None and frame.f_globals.get("__name__") in _PATH_MACHINERY:
        frame = frame.f_back
    if frame is None:
        return "other"
    filename = frame.f_code.co_filename
    if filename.startswith(_NOTEBOOK_STATEMENT_PREFIX):
        return "user"
    return frame_kind(filename)


def _asked_by_user_code(frame: Any) -> bool:
    """Did the user's own code make the call *frame* made?"""
    return _who_asked(frame) == "user"


def _record_path_answer(path: Any, answer: str | None, caller: Any) -> None:
    """Record what a metadata or existence call on *path* answered.

    This is the one rule every stat and probe wrapper applies. *caller* is
    the frame that made the call; *answer* is None when nothing was there,
    ``"content"`` when the call reported a file's size or times, and
    otherwise the kind it found there (``file``, ``dir``, ``any``) when it
    asked only whether something is there.

    Nothing there is recorded whoever asked, except cash itself: the path
    appearing changes the answer, and a path that never appears costs
    nothing. Something there is recorded only when the user's own code
    asked: libraries and cash stat and probe paths for themselves all the
    time -- ``inspect`` checks that a function's source file exists while
    cash keys a nested call -- and each would become a dependency of the
    cached call around it. "content" makes a regular file a dependency by
    its content, as a read does; a directory has none.
    """
    tracker = active_tracker.get()
    if tracker is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    who = _who_asked(caller)
    if answer is None:
        if who not in ("cash", "wrapper"):
            tracker.track_absent(path)
    elif who != "user":
        return
    elif answer == "content":
        _track_regular_file(path)
    else:
        tracker.track_present(path, answer)


def _record_negative_probe(path: Any, caller: Any) -> None:
    """A probe said no: absent if nothing is there, else present as what is."""
    try:
        st = os.stat(path)
    except (OSError, ValueError, TypeError):
        _record_path_answer(path, None, caller)
        return
    kind = "dir" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "any"
    _record_path_answer(path, kind, caller)


def _path_max() -> int:
    """The longest path the OS opens; longer strings are never file names."""
    try:
        return os.pathconf("/", "PC_PATH_MAX")
    except (AttributeError, OSError, ValueError):
        return 32767  # Windows' extended-length limit


_PATH_MAX = _path_max()


def _dataset_member(name: str) -> bool:
    """Is *name* part of a dataset directory's data? pyarrow's rule, which
    pandas and polars follow: names starting with ``.`` or ``_`` are
    bookkeeping (``_SUCCESS``, ``.crc`` files, ``_temporary/``)."""
    return not name.startswith((".", "_"))


def _is_remote_target(target: Any) -> bool:
    return isinstance(target, str) and is_remote_url(target)


def _storage_options(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """The fsspec options a reader was given: ``storage_options=`` (pandas,
    polars), or those of an fsspec filesystem passed as ``filesystem=``."""
    options = kwargs.get("storage_options")
    if isinstance(options, dict):
        return options
    options = getattr(kwargs.get("filesystem"), "storage_options", None)
    return options if isinstance(options, dict) else None


def track_dataset(tracker: Any, target: Any) -> None:
    """Record what a reader given *target* reads: a file, or every file of a
    directory or glob.

    A dataset reader (``pd.read_parquet("dd")``, ``pl.read_parquet("dd/*.parquet")``,
    ``ds.dataset("dd")``) reads every file under the directory or matching
    the pattern. Each file is a dependency, since rewriting one leaves the
    directory's mtime where it was, and each directory listed on the way is
    too, so a new file counts. A glob is never recorded as a path itself.
    """
    text = os.fsdecode(target) if isinstance(target, bytes) else os.fspath(target)
    if not isinstance(text, str) or is_remote_url(text):
        tracker.track_path(target)
        return
    if len(text) > _PATH_MAX or "\0" in text:
        # No file can have this name, so the reader takes the string as
        # content (``pd.read_html(page)``, ``pd.read_json(text)``) and reads
        # nothing from disk. The string itself is already in the key, as code
        # or as an input.
        return
    try:
        if os.path.isdir(text):
            tracker.track_path(target)
            # Listed, not untracked: each directory's listing is recorded as a
            # read (`read_events._on_listing`), which is what makes a new file
            # in any of them a change.
            from cash.tracking.file_dep_snapshot import walk_dataset_dir

            for root, _dirs, files in walk_dataset_dir(text, _dataset_member):
                for name in sorted(files):
                    if _dataset_member(name):
                        tracker.track_path(os.path.join(root, name))
            return
        if glob.has_magic(text) and not os.path.exists(text):
            # The pattern's directory is recorded by the ``glob.glob`` audit
            # event (`read_events._on_glob`), and each directory it lists.
            for match in sorted(glob.glob(text, recursive=True)):
                if os.path.isfile(match):
                    tracker.track_path(match)
            return
    except (OSError, ValueError, RecursionError):
        # RecursionError: glob recurses once per directory level, so a deep
        # pattern can exhaust the stack; the name is then recorded as is.
        pass
    tracker.track_path(target)


#: Database names SQLite opens no file for: in memory, and a private
#: temporary database.
_SQLITE_NO_FILE = (":memory:", b":memory:", "", b"")


def _sqlite_uri_path(uri: str) -> str | None:
    """The file a SQLite ``file:`` URI opens, or None for an in-memory one.

    SQLite reads a URI that does not start with ``file:`` as a plain file
    name, and one with an authority only when it is empty or ``localhost``.
    """
    if not uri.startswith("file:"):
        return uri
    import urllib.parse

    parts = urllib.parse.urlsplit(uri)
    if parts.netloc not in ("", "localhost"):
        return None
    if dict(urllib.parse.parse_qsl(parts.query)).get("mode") == "memory":
        return None
    path = urllib.parse.unquote(parts.path)
    if not path or path == ":memory:":
        return None
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]  # file:///C:/data/d.db
    return path


def _track_sqlite_wal(tracker: Any, database: Any) -> None:
    """Make ``<db>-wal`` part of a WAL-mode database's dependency.

    In WAL mode a commit goes to the ``-wal`` file and the main file stays as
    it was until a checkpoint, so the main file alone does not show a commit
    while any writer keeps its connection open. The header says
    the mode (bytes 18 and 19 are 2 for WAL); the ``-wal`` file is tracked by
    content when it is there and as absent when it is not. A rollback-journal
    database has no ``-wal``, and switching one to WAL rewrites its header.
    """
    try:
        path = os.fsdecode(database) if isinstance(database, bytes) else os.fspath(database)
        if not path or path == ":memory:":
            return
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            header = os.read(fd, 20)
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        return
    if len(header) < 20 or not header.startswith(b"SQLite format 3\x00") or header[18:20] != b"\x02\x02":
        return
    wal = path + "-wal"
    if os.path.lexists(wal):
        tracker.track_path(wal)
        # A reader running as root chowns the -wal it opens: its ctime moves
        # while its content does not.
        tracker.note_ctime_unreliable(wal)
    else:
        tracker.track_absent(wal)


class _WorkerReads:
    """What a task run in a worker process returned, the files it read, and
    the ``dynamic_depends_on=`` sources of the cached calls it made."""

    __slots__ = ("value", "files", "absent", "unresolved", "present", "remote", "dynamic")

    def __init__(
        self,
        value: Any,
        files: list[str],
        absent: list[str],
        unresolved: list[str],
        present: dict[str, str],
        remote: list[str],
        dynamic: tuple[list, dict, list[str]] = ([], {}, []),
    ) -> None:
        self.value, self.files, self.absent, self.unresolved = value, files, absent, unresolved
        self.present, self.remote = present, remote
        #: ``([(id, source, token)], resolutions, functions whose resolver
        #: failed)``, as the worker's tracker collected them.
        self.dynamic = dynamic

    def __reduce__(self):
        return (
            _WorkerReads,
            (self.value, self.files, self.absent, self.unresolved, self.present, self.remote, _portable(self.dynamic)),
        )

    def relay_to(self, tracker: Any) -> Any:
        """Credit what the worker read to *tracker*; the task's own value."""
        if tracker is not None:
            for path in self.files:
                tracker.add_tracked(path)
            for path in self.absent:
                tracker.add_tracked_absent(path)
            for path in self.unresolved:
                tracker.add_tracked_unresolved(path)
            for path, kind in self.present.items():
                tracker.add_tracked_present(path, kind)
            for url in self.remote:
                tracker.add_tracked_remote(url)
            sources, resolutions, failed = self.dynamic
            if hasattr(tracker, "add_dynamic_source"):
                for source_id, source, token in sources:
                    tracker.add_dynamic_source(source_id, source, token)
                for key, resolution in resolutions.items():
                    tracker.add_dynamic_resolution(key, resolution)
                for func_name in failed:
                    tracker.add_unresolved_dynamic(func_name)
        return self.value


def _dynamic_of(tracker: Any) -> tuple[list, dict, list[str]]:
    """What *tracker* collected of ``dynamic_depends_on=`` (`_WorkerReads.dynamic`)."""
    sources = [(key[0], source, token) for key, (source, token) in getattr(tracker, "dynamic_sources", {}).items()]
    return (
        sources,
        dict(getattr(tracker, "dynamic_resolutions", {})),
        sorted(getattr(tracker, "unresolved_dynamic", ())),
    )


def _portable(dynamic: tuple[list, dict, list[str]]) -> tuple[list, dict, list[str]]:
    """*dynamic* as it crosses back from a worker process: the resolver
    calls come back too, to be asked again by the caller's entry like its
    own (a resolver may hand out a new source object, which the old one
    never answers for); a source or resolver call that does not pickle
    becomes a dependency nothing can check, which keeps the caller from
    being stored, rather than breaking the pool's result."""
    import pickle

    sources, resolutions, failed = dynamic
    kept, failed = [], list(failed)
    portable: dict = {}
    for key, resolution in resolutions.items():
        try:
            pickle.dumps(resolution)
        except Exception:  # noqa: BLE001 - pickle raises whatever __reduce__ raises
            failed.append(str(key[0]) if isinstance(key, tuple) and key else "a dynamic_depends_on resolver")
            continue
        portable[key] = resolution
    for source_id, source, token in sources:
        try:
            pickle.dumps(source)
        except Exception:  # noqa: BLE001 - pickle raises whatever __reduce__ raises
            failed.append(source_id)
            continue
        kept.append((source_id, source, token))
    return kept, portable, failed


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
        _hold_for_worker_life()
        tracker = self.tracker_type()
        with tracker:
            value = self.fn(*args, **kwargs)
        return _WorkerReads(
            value,
            sorted(tracker.get_accessed_files()),
            sorted(tracker.get_absent_files()),
            sorted(tracker.get_unresolved_files()),
            dict(tracker.get_present_files()),
            sorted(tracker.get_accessed_remote_urls()),
            _dynamic_of(tracker),
        )


_held_for_worker = False


def _hold_for_worker_life() -> None:
    """In a worker process, keep cash's I/O watch installed between tasks.

    A tracker installs it on entry and removes it on exit, 20 us a task; a
    ``pool.imap`` of small items runs one task per item. A worker process
    exists to run tasks, so it is installed once and left. Not in the
    submitting process: a ``ThreadPool`` task runs there, and that process's
    reads outside a cached call must stay unwatched.
    """
    global _held_for_worker
    if _held_for_worker:
        return
    import multiprocessing

    if multiprocessing.parent_process() is not None:
        io_watch.hold()
        _held_for_worker = True


class _RelayFuture(concurrent.futures.Future):
    """The future a patched ``submit`` returns: the worker's plain result, and
    the cancellation reaching the future that actually runs."""

    def __init__(self, inner: concurrent.futures.Future) -> None:
        super().__init__()
        self._inner = inner

    def cancel(self) -> bool:
        inner = self._inner
        return inner is not None and inner.cancel() and super().cancel()


def _make_relaying_submit(original: Callable[..., Any]) -> Callable[..., Any]:
    """A process-pool ``submit`` that brings back what the task read.

    With a tracker active it sends a `_ReadsInWorker` instead of the bare
    function, and credits what it read to the submitting call when the result
    comes back -- before the caller can see the result, so before the call
    that waits on it is stored.
    """

    @functools.wraps(original)
    def submit(self, fn, /, *args, **kwargs):
        tracker = active_tracker.get()
        if tracker is None:
            return original(self, fn, *args, **kwargs)
        inner = original(self, _ReadsInWorker(fn, type(tracker)), *args, **kwargs)
        outer = _RelayFuture(inner)

        def relay(done: concurrent.futures.Future) -> None:
            # The inner future holds this callback, which holds *outer*: with
            # *outer* holding the inner one back, the three are a cycle that
            # keeps the task's value alive until the garbage collector runs,
            # and the notebook would count it as another holder of whatever
            # the caller keeps of it (`_unwrapping_set`). Done, the inner
            # future is not needed to cancel.
            outer._inner = None
            if done.cancelled():
                concurrent.futures.Future.cancel(outer)
                return
            error = done.exception()
            if error is not None:
                outer.set_exception(error)
                return
            result = done.result()
            if isinstance(result, _WorkerReads):
                result = result.relay_to(tracker)
            outer.set_result(result)

        inner.add_done_callback(relay)
        return outer

    return submit


def _patch_process_pool_submit() -> None:
    """Bring the files a ``ProcessPoolExecutor`` task read back to the submitter.

    A cached orchestrator that fans work out to a process pool reads its data
    in the workers, where no tracker of the parent's can see, so the workers
    report what they read and it is credited to the submitter. ``submit`` is what ``Executor.map`` calls, chunked or not.
    joblib's default backend (loky, behind ``Parallel(n_jobs=...)`` and every
    scikit-learn ``n_jobs=``) runs on an executor of the same shape and is
    wrapped the same way, as is ``multiprocessing.Pool``
    (`_patch_multiprocessing_pool`).

    Only once the pool's module is imported (`_POOL_MODULES`): one imported
    later, while a tracker is open, is patched as its import finishes, and
    one imported between trackers when the next opens. Importing it here
    loaded multiprocessing and subprocess, 9-12 ms of every process that
    caches a call, for a pool most of them never make.
    """
    cf_process = sys.modules.get("concurrent.futures.process")
    if cf_process is not None:
        _patch_attribute(cf_process.ProcessPoolExecutor, "submit", _make_relaying_submit)
    # joblib vendors loky; loky also ships on its own. Only once imported: a
    # program that never imported joblib has no loky workers to hear from.
    for name in ("joblib.externals.loky.process_executor", "loky.process_executor"):
        module = sys.modules.get(name)
        executor = getattr(module, "ProcessPoolExecutor", None)
        if executor is not None:
            _patch_attribute(executor, "submit", _make_relaying_submit)


def _patch_multiprocessing_pool() -> None:
    """Bring the files a ``multiprocessing.Pool`` task read back to the submitter.

    Every ``Pool`` method hands its work to one of two places:
    ``_guarded_task_generation`` (``map``, ``starmap``, ``imap`` and their
    async forms, once per chunk) and ``apply_async``. With a tracker active,
    both send a `_ReadsInWorker`, so a tracker runs once per chunk in the
    worker, not once per item. What comes back is unwrapped by the result
    object, which learns its tracker when it is made -- before its task is
    queued, so no result can arrive first -- and keeps it on itself, so a
    result that arrives after the cached call has returned is still
    unwrapped. ``ThreadPool`` is a ``Pool`` and is covered alike.
    """
    # `multiprocessing.Pool()` imports the module only when called -- inside
    # the cached call, after this ran: the import hook patches it then
    # (`_POOL_MODULES`).
    module = sys.modules.get("multiprocessing.pool")
    if module is None:
        return

    def wrap_task(original):
        @functools.wraps(original)
        def guarded_task_generation(self, result_job, func, iterable):
            tracker = active_tracker.get()
            if tracker is not None:
                func = _ReadsInWorker(func, type(tracker))
            return original(self, result_job, func, iterable)

        return guarded_task_generation

    def wrap_apply(original):
        @functools.wraps(original)
        def apply_async(self, func, *args, **kwargs):
            tracker = active_tracker.get()
            if tracker is not None:
                func = _ReadsInWorker(func, type(tracker))
            return original(self, func, *args, **kwargs)

        return apply_async

    def wrap_init(original):
        @functools.wraps(original)
        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            tracker = active_tracker.get()
            if tracker is not None:
                self._set = _unwrapping_set(self, tracker)

        return __init__

    _patch_attribute(module.Pool, "_guarded_task_generation", wrap_task)
    _patch_attribute(module.Pool, "apply_async", wrap_apply)
    _patch_attribute(module.ApplyResult, "__init__", wrap_init)
    _patch_attribute(module.IMapIterator, "__init__", wrap_init)


def _unwrapping_set(result: Any, tracker: Any) -> Callable[..., Any]:
    """A result object's ``_set(i, (success, value))`` that relays a worker's
    reads to *tracker* and passes the task's own value on.

    Kept on *result* itself, so it holds *result* only weakly: a closure over
    ``result._set`` (a method bound to *result*) would make the two a
    reference cycle, keeping the task's values alive until the garbage
    collector runs. The notebook tells an object nothing else holds by its
    reference count, so a model fitted through joblib's threading backend
    (scikit-learn's histogram binning) looked held elsewhere and was never
    cached.
    """
    original_set = type(result)._set
    owner = weakref.ref(result)

    def _set(i: Any, obj: Any) -> Any:
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], _WorkerReads):
            obj = (obj[0], obj[1].relay_to(tracker))
        return original_set(owner(), i, obj)

    return _set


class _TrackedDirEntry:
    """An ``os.DirEntry`` whose ``stat()`` records the regular file it
    describes. ``os.DirEntry`` cannot be subclassed or patched."""

    __slots__ = ("_entry",)

    def __init__(self, entry: os.DirEntry) -> None:
        self._entry = entry

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        result = self._entry.stat(follow_symlinks=follow_symlinks)
        tracker = active_tracker.get()
        if tracker is not None and stat.S_ISREG(result.st_mode):
            tracker.track_path(self._entry.path)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._entry, name)

    def __fspath__(self) -> Any:
        return self._entry.__fspath__()

    def __repr__(self) -> str:
        return repr(self._entry)


class _TrackedScandir:
    """The iterator ``os.scandir`` returns, yielding `_TrackedDirEntry`."""

    __slots__ = ("_it",)

    def __init__(self, it: Any) -> None:
        self._it = it

    def __iter__(self) -> _TrackedScandir:
        return self

    def __next__(self) -> _TrackedDirEntry:
        return _TrackedDirEntry(next(self._it))

    def __enter__(self) -> _TrackedScandir:
        self._it.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._it.__exit__(*exc)

    def close(self) -> None:
        self._it.close()


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
        # numbers. A reader class such as ParquetFile is not replaced -- a
        # function in its place would break isinstance checks -- its
        # ``__init__`` is wrapped instead (``_CLASS_READERS``).
        self.register("pyarrow.csv", "read_csv", self._create_path_arg_handler)
        self.register("pyarrow.csv", "open_csv", self._create_path_arg_handler)
        self.register("pyarrow.parquet", "read_table", self._create_path_arg_handler)
        self.register("pyarrow.parquet", "read_pandas", self._create_path_arg_handler)
        self.register("pyarrow.feather", "read_table", self._create_path_arg_handler)
        self.register("pyarrow.feather", "read_feather", self._create_path_arg_handler)
        self.register("pyarrow.json", "read_json", self._create_path_arg_handler)
        self.register("pyarrow.orc", "read_table", self._create_path_arg_handler)
        # Arrow IPC files are opened by path in C++ too: ``pa.memory_map`` is
        # the zero-copy way to read one, ``ipc.open_file`` the reader over it.
        self.register("pyarrow", "memory_map", self._create_path_arg_handler)
        self.register("pyarrow", "input_stream", self._create_path_arg_handler)
        self.register("pyarrow.ipc", "open_file", self._create_path_arg_handler)
        self.register("pyarrow.ipc", "open_stream", self._create_path_arg_handler)

        # pyarrow.dataset reads in C++ like the rest of pyarrow; `read_table`
        # was registered and `dataset()` was not, so one entry point of an
        # otherwise-covered library went stale.
        self.register("pyarrow.dataset", "dataset", self._create_path_arg_handler)

        # soundfile reads through libsndfile in C, so an edited .wav/.flac gave
        # a stale hit with no warning. ``read``, ``blocks`` and ``info`` take the
        # path first; a ``SoundFile`` opened to read is wrapped at ``__init__``
        # (``_CLASS_READERS``), which also covers the reads the functions make
        # through it.
        self.register("soundfile", "read", self._create_path_arg_handler)
        self.register("soundfile", "blocks", self._create_path_arg_handler)
        self.register("soundfile", "info", self._create_path_arg_handler)

        # Scientific formats read in C: libhdf5, libnetcdf, GDAL. On Linux and
        # Windows the native file hook sees those opens too; on macOS there is
        # no hook, and a changed .h5/.nc/.tif returned yesterday's answer with
        # no warning. ``h5py.File`` is a class (``_CLASS_READERS``); a file
        # opened only to write is an output, not an input.
        self.register("tables", "open_file", self._create_read_mode_path_handler)
        self.register("rasterio", "open", self._create_read_mode_path_handler)
        for name in ("open_dataset", "open_dataarray", "load_dataset", "load_dataarray", "open_mfdataset"):
            self.register("xarray", name, self._create_path_arg_handler)

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
        # The connection's path is the dependency, also when it is given as a
        # ``file:`` URI (``uri=True``); ":memory:" has no file behind it and
        # `track_path` drops what it cannot resolve.
        self.register("sqlite3", "connect", self._create_sqlite_connect_handler)
        self.register("sqlite3.dbapi2", "connect", self._create_sqlite_connect_handler)

        # Existence probes: "is there a config here?" The ABSENCE of a file is
        # an input -- it selects the defaults branch -- and a file that is
        # never opened produces no read to track. A probe that says YES is an
        # input too: a flag file or an output folder that is checked and never
        # read. `os.stat` raises no audit event.
        for module in ("os.path", "genericpath"):
            self.register(module, "exists", self._create_exists_handler)
            self.register(module, "lexists", self._create_exists_handler)
            self.register(module, "isfile", self._create_isfile_handler)
            self.register(module, "isdir", self._create_isdir_handler)
        self.register("os", "access", self._create_exists_handler)

        # What a file's metadata says is the file's: ``max(files,
        # key=os.path.getmtime)`` picks the newest export and
        # ``os.path.getsize(p)`` reports it, and an in-place rewrite moves
        # nothing else that is recorded.
        for module in ("os.path", "genericpath"):
            for name in ("getsize", "getmtime", "getctime"):
                self.register(module, name, self._create_metadata_handler)
        self.register("os", "stat", self._create_os_stat_handler)
        self.register("os", "lstat", self._create_os_stat_handler)
        # ``os.scandir`` raises its audit event (the listing), but an entry's
        # ``stat()`` is a C method on a type that cannot be patched.
        self.register("os", "scandir", self._create_scandir_handler)
        # A module with only a reader class still needs an entry, so it is
        # patched when it is imported (``_PatchingLoader``) and at each install.
        for module_name in _CLASS_READERS:
            self.handlers.setdefault(module_name, [])
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
    def _create_read_mode_path_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """``open(path, mode='r')``-style: the path is a dependency only when
        *mode* reads it (`_opens_to_read`)."""
        path_handler = FileDependencyRegistry._create_path_arg_handler(lambda *a, **k: None, track_callback)

        @functools.wraps(original_func)
        def tracked_func(*args, **kwargs):
            if _opens_to_read(args[1] if len(args) > 1 else kwargs.get("mode", "r")):
                path_handler(*args, **kwargs)
            return original_func(*args, **kwargs)

        return tracked_func

    @staticmethod
    def _create_path_arg_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Generic handler for functions where the first argument is a path.

        ``track_callback`` is part of the user-facing handler factory
        signature (see :meth:`FileDependencyRegistry.register`) so custom
        factories can record the access. The built-in handlers ignore the
        argument and consult ``active_tracker`` directly — that way one patch
        serves any number of concurrent trackers.
        """

        # Positional OR keyword: readers take the path either way
        # (`pd.read_csv(filepath_or_buffer=p)`, `np.load(file=p)`,
        # `pq.read_table(source=p)`), and the wrapper is installed process-wide,
        # so it must accept every call the original accepts.
        @functools.wraps(original_func)
        def tracked_func(*args, **kwargs):
            target = args[0] if args else next((kwargs[k] for k in _PATH_KWARGS if k in kwargs), None)
            remote = False
            if isinstance(target, (str, bytes, os.PathLike)):
                _tracker = active_tracker.get()
                if _tracker is not None:
                    track_dataset(_tracker, target)
                    remote = _is_remote_target(target)
                else:
                    note_untracked_read(target, sys._getframe(1))
            elif isinstance(target, (list, tuple)):
                # ``pl.read_parquet([a, b])``, ``ds.dataset([a, b])``.
                _tracker = active_tracker.get()
                for item in target:
                    if isinstance(item, (str, bytes, os.PathLike)):
                        if _tracker is not None:
                            track_dataset(_tracker, item)
                            remote = remote or _is_remote_target(item)
                        else:
                            note_untracked_read(item, sys._getframe(1))
            if not remote:
                return original_func(*args, **kwargs)
            options = _storage_options(kwargs)
            if options:
                # The freshness check must reach the store this read did.
                for item in target if isinstance(target, (list, tuple)) else (target,):
                    if _is_remote_target(item):
                        remember_read_options(item, options)
            # The fetch of a remote file recorded above is a tracked read --
            # its ETag is checked on every hit -- not a connection the key
            # cannot see, so the effect observer does not report it.
            token = _active_effect_observer.set(None)
            try:
                return original_func(*args, **kwargs)
            finally:
                _active_effect_observer.reset(token)

        return tracked_func

    @staticmethod
    def _create_sqlite_connect_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """``sqlite3.connect``: the database file, also when named by a URI.

        With ``uri=True`` the database is a ``file:`` URI --
        ``file:d.db?mode=ro`` for a read-only connection -- which as written
        names no file. The URI's path is the file; an in-memory database has
        none, and neither has ``":memory:"`` or ``""`` (a private temporary
        database) as a plain name. Read as file names they were dependencies
        on ``<cwd>/:memory:``, a missing file on POSIX and a name Windows
        cannot resolve: an unresolved dependency, so a statement calling a
        module that connects to ``":memory:"`` never hit after a restart.
        """
        path_handler = FileDependencyRegistry._create_path_arg_handler(original_func, track_callback)

        @functools.wraps(original_func)
        def tracked_connect(*args, **kwargs):
            uri = kwargs.get("uri", args[7] if len(args) > 7 else False)
            target = args[0] if args else kwargs.get("database")
            if not uri and target in _SQLITE_NO_FILE:
                return original_func(*args, **kwargs)
            if not uri or not isinstance(target, (str, bytes)):
                if isinstance(target, (str, bytes, os.PathLike)) and active_tracker.get() is not None:
                    _track_sqlite_wal(active_tracker.get(), target)
                return path_handler(*args, **kwargs)
            path = _sqlite_uri_path(os.fsdecode(target) if isinstance(target, bytes) else target)
            if path is not None:
                _tracker = active_tracker.get()
                if _tracker is not None:
                    _tracker.track_path(path)
                    _track_sqlite_wal(_tracker, path)
                else:
                    note_untracked_read(path, sys._getframe(1))
            return original_func(*args, **kwargs)

        return tracked_connect

    @staticmethod
    def _create_exists_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Record what an existence probe answered: a path that was not there
        (absent), or one that was (present, of any kind)."""
        return _probe_handler(original_func, "any")

    @staticmethod
    def _create_metadata_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """``os.path.getsize`` / ``getmtime`` / ``getctime``: the regular file
        they describe is a dependency, when the user's code asked."""

        @functools.wraps(original_func)
        def tracked_metadata(path, *args, **kwargs):
            try:
                result = original_func(path, *args, **kwargs)
            except (FileNotFoundError, NotADirectoryError):
                # `try: getsize(p) except OSError:` answers for a file that is
                # not there yet, so its appearing is a change.
                _record_path_answer(path, None, sys._getframe(1))
                raise
            _record_path_answer(path, "content", sys._getframe(1))
            return result

        return tracked_metadata

    @staticmethod
    def _create_os_stat_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """``os.stat(p).st_size`` in the user's own code: the regular file is a
        dependency, as through ``Path.stat``.

        A call the path machinery makes (``os.path.exists`` stats what it
        probes, ``getsize`` what it measures) is part of the question the
        user asked through it, which that call's own wrapper records.
        Otherwise `_record_path_answer` decides.
        """

        @functools.wraps(original_func)
        def tracked_os_stat(path, *args, **kwargs):
            try:
                result = original_func(path, *args, **kwargs)
            except (FileNotFoundError, NotADirectoryError):
                # `try: os.stat(p) except FileNotFoundError:` answers for a
                # file that is not there yet, so its appearing is a change.
                caller = sys._getframe(1)
                if kwargs.get("dir_fd") is None and caller.f_globals.get("__name__") not in _PATH_MACHINERY:
                    _record_path_answer(path, None, caller)
                raise
            caller = sys._getframe(1)
            if (
                stat.S_ISREG(result.st_mode)
                and kwargs.get("dir_fd") is None
                and caller.f_globals.get("__name__") not in _PATH_MACHINERY
            ):
                _record_path_answer(path, "content", caller)
            return result

        return tracked_os_stat

    @staticmethod
    def _create_scandir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """``os.scandir`` in the user's own code: an entry's ``stat()``
        reports the file's size and times, so the regular file it describes
        is a dependency, as through ``os.stat``. The listing itself is
        tracked by its audit event."""

        @functools.wraps(original_func)
        def tracked_scandir(*args, **kwargs):
            result = original_func(*args, **kwargs)
            caller = sys._getframe(1)
            if active_tracker.get() is None or caller.f_globals.get("__name__") in _PATH_MACHINERY:
                return result  # os.walk, pathlib: they never hand the user an entry's stat()
            if not _asked_by_user_code(caller):
                return result
            if args and isinstance(args[0], int):
                return result  # a descriptor: entries have no usable path
            return _TrackedScandir(result)

        return tracked_scandir

    @staticmethod
    def _create_isfile_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """`_create_exists_handler` for ``isfile``: present means a file."""
        return _probe_handler(original_func, "file")

    @staticmethod
    def _create_isdir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """`_create_exists_handler` for ``isdir``: present means a directory."""
        return _probe_handler(original_func, "dir")

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
          ``<ipython-input-3>``, and cash's own ``/<cash-0beec9249e1e>`` for
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
                if not is_pseudo_filename(text) and not text.endswith(FileDependencyRegistry._SOURCE_SUFFIXES):
                    _tracker = active_tracker.get()
                    if _tracker is not None:
                        try:
                            real = os.path.isfile(text)
                        except (OSError, ValueError):
                            real = False
                        if real:
                            _tracker.track_path(filename)
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
        if fullname not in file_registry().handlers and fullname not in _POOL_MODULES:
            return None

        # It's a target. We need to let the real import happen, then patch.
        self._skip.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._skip.remove(fullname)

        if spec is None or spec.loader is None:
            return None

        # Wrap the loader
        spec.loader = _PatchingLoader(spec.loader, fullname)
        return spec


class _PatchingLoader:
    """Runs the original loader, then patches the module.

    It stands in for the original loader during this one import only: the
    module's ``__loader__`` and ``__spec__.loader`` are set back to the
    original before it executes, so ``importlib.resources``, ``pkgutil``
    and a later reload see the real loader. Anything else asked of it
    goes to the original loader.
    """

    def __init__(self, original_loader, fullname):
        self.original_loader = original_loader
        self.fullname = fullname

    def __getattr__(self, name):
        return getattr(self.original_loader, name)

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.loader is self:
            spec.loader = self.original_loader
        if getattr(module, "__loader__", None) is self:
            module.__loader__ = self.original_loader
        self.original_loader.exec_module(module)

        # Now patch it via the module-level dispatcher installer —
        # tracker-agnostic, install-once-per-target, and only while a
        # tracker is still open.
        _install_module_patches(self.fullname, module)
        patch_pool = _POOL_MODULES.get(self.fullname)
        if patch_pool is not None:
            with _install_lock:
                if io_watch.holding():  # an import finishing after the last tracker closed
                    patch_pool()


# The pools whose tasks report their reads back, by the module that defines
# them: patched when the module is imported (`_patch_process_pool_submit`).
_POOL_MODULES: dict[str, Callable[[], None]] = {
    "concurrent.futures.process": _patch_process_pool_submit,
    "multiprocessing.pool": _patch_multiprocessing_pool,
}

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
        # The number of modules stands for the user's modules, whose names
        # bound to a reader are wrapped too (`_patch_user_module_aliases`).
        basis = (
            id(registry),
            registry._revision,
            tuple(id(sys.modules.get(name)) for name in registry.handlers),
            len(sys.modules),
        )
        if basis != _installed_for or not _patches.reinstall():
            for mod_name in registry.handlers:
                module = sys.modules.get(mod_name)
                if module is not None:
                    _install_module_patches(mod_name, module)
            _patch_pathlib_stat()
            # Work handed to a thread pool runs under the submitter's tracker, and
            # work handed to a process pool reports what it read back to it.
            _patch_thread_pool_submit()
            _patch_thread_start()
            _patch_process_pool_submit()
            _patch_multiprocessing_pool()
            _patch_user_module_aliases()
            _patch_library_reexports()
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
