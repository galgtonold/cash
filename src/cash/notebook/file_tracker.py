from __future__ import annotations

"""File-read interception for tracking data file dependencies.

Monkey-patches common I/O functions (``open``, ``pandas.read_csv``,
``numpy.load``, etc.) to record which files each statement reads.
File hashes are incorporated into cache keys so that changed data
automatically invalidates dependent cached results.
"""

import builtins
import concurrent.futures
import contextvars
import functools
import importlib
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

from cash.utils import is_remote_url, normalize_path

# A remote URL handed to a reader (``pd.read_parquet("s3://bucket/key")``)
# reaches us as the raw first argument. ``os.path.realpath`` would mangle it into
# a bogus local path, nothing would resolve, and ``snapshot_file_deps`` would
# drop it — leaving the entry with *no* dependency, hitting forever even after
# the object changed. URLs are routed to their own channel instead and tracked
# by the store's own validator (ETag / version id / generation). ``file://`` is
# excluded: it names a local path that can genuinely be stat'ed. See CAS-236.

__all__ =["FileDependencyRegistry", "PostImportHook", "FileAccessTracker", "FileDependencies"]

# Type alias for file dependency tracking: maps normalized file path -> mtime at read time
FileDependencies = dict[str, float]

logger = logging.getLogger(__name__)

# The effect observer for the current task/thread. Imported at module scope
# rather than inside the wrapper: `tracked_open` sits on the process-global
# `open`, so an import executed there would run during arbitrary user code --
# including at interpreter shutdown, while sys.modules is being torn down.
# `cash.effect_observer` imports nothing from cash, so this cannot cycle.
from cash.effect_observer import _active_observer as _active_effect_observer

# Active tracker for the current asyncio task / thread.
# Read by the patched I/O dispatchers to decide whether to record the
# access. Isolated per task/thread by contextvars semantics.
_active_tracker: contextvars.ContextVar[Optional["FileAccessTracker"]] = (
    contextvars.ContextVar("_active_tracker", default=None)
)


class untracked:
    """Run cash's OWN I/O without it becoming anyone's dependency.

    A nested cached call does its bookkeeping -- resolving configuration,
    reading ``pyproject.toml``, walking up for project markers -- while the
    OUTER call's tracker is live, so those reads were recorded as the outer
    entry's file dependencies: bump the project's version and every cached
    function that calls another one recomputed. The storage-path filters
    (``_is_cash_internal``) cannot help, because a config file is not storage.
    A class rather than ``contextlib.contextmanager`` so it costs one
    ContextVar swap, not a generator.
    """

    __slots__ = ("_token", "_observer_token")

    def __enter__(self) -> None:
        self._token = _active_tracker.set(None)
        # Nor anyone's observed side effect: cash writing its own bookkeeping
        # file inside a nested call is not the OUTER function writing a file.
        self._observer_token = _active_effect_observer.set(None)

    def __exit__(self, *exc: Any) -> None:
        _active_effect_observer.reset(self._observer_token)
        _active_tracker.reset(self._token)

# Per-target install lock: the dispatcher wrappers are installed once
# per (module/dict, attr-name) pair for the lifetime of the process.
# Subsequent ``__enter__`` calls skip already-installed targets (cheap
# attribute-sentinel check). New targets — i.e. modules imported after
# the first tracker entered, or new handlers registered via
# :func:`cash.register_file_handler` — are still picked up because
# ``_apply_patches`` runs on every ``__enter__`` and only the
# already-installed wrappers are no-op skipped.
_install_lock = threading.Lock()

# Kernel pseudo-filesystems are never data dependencies, and recording one is
# actively harmful: ``/proc/meminfo`` reports live memory and so changes on
# every read, which means an entry that captured it can NEVER be fresh again.
# The cache still stores, still finds the entry, and still throws it away —
# a silent, permanent cache defeat with no warning anywhere.
#
# cash reaches these paths through its OWN machinery, not the user's code:
# ``InMemoryBackend._check_and_evict`` calls ``psutil.virtual_memory()`` every
# ``check_interval`` (10) writes, and ``psutil`` reads ``/proc/meminfo`` on
# Linux. The tracker is active for the duration of the user's cached call, so
# that read is attributed to the user's result. A chunked iterator is the
# reliable trigger — it writes one entry per chunk plus a manifest, so a
# 10-chunk result crosses the eviction-check threshold inside a single call.
#
# Same shape as CAS-214 (cash's own ``open`` shim poisoning its cache key):
# the tracker cannot tell cash's internal reads from the user's, so paths that
# are definitionally not data get excluded here. Linux-only in effect; on
# Windows these prefixes never match.
_PSEUDO_FS_PREFIXES: tuple[str, ...] = ("/proc/", "/sys/", "/dev/")


#: Keyword names a reader may take its path under, first match wins. Covers
#: pandas (`filepath_or_buffer`, `path_or_buf`, `io` for read_excel, `path`),
#: numpy (`file`, `fname`), joblib (`filename`), pyarrow (`source`,
#: `input_file`) and polars (`source`).
_PATH_KWARGS = ("filepath_or_buffer", "path_or_buf", "source", "input_file",
                "path", "file", "fname", "filename", "io")


def _regular_file_stat(path: str) -> tuple[int, int, int] | None:
    """``(size, mtime_ns, ctime_ns)`` for a regular file, None otherwise.

    ``ctime_ns`` is the inode change time on POSIX, so an edit that restores
    the mtime still moves this tuple there; on Windows it is the creation
    time and adds nothing, which is the residual the NTFS change-time ticket
    (CAS-114) is about.
    """
    import stat as _stat
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    if not _stat.S_ISREG(st.st_mode):
        return None
    return (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", 0))


#: The folder joblib memory-maps a parallel call's large arguments into.
_SCRATCH_MEMMAP = "joblib_memmapping_folder_"

#: Caches a runtime keeps for ITSELF: the interpreter's bytecode and numba's
#: JIT index/data (``.nbi`` / ``.nbc``), which live in a ``__pycache__`` next to
#: the code or wherever ``NUMBA_CACHE_DIR`` points. Never the user's data, and
#: rewritten by any other process that runs the same function -- recorded,
#: scanpy's normalize made every step after it re-run after a restart (round
#: 28, r28s4, 3/3).
_RUNTIME_CACHE_SEGMENT = "/__pycache__/"
_RUNTIME_CACHE_SUFFIXES = (".nbi", ".nbc", ".pyc")


def _is_pseudo_fs(path: str) -> bool:
    """True for kernel pseudo-filesystem paths, which are machine state rather
    than data and must never become cache dependencies.

    Checked against the RAW path as well as the resolved one, because
    ``os.path.realpath`` is not identity-preserving here: on Windows it turns
    ``/proc/meminfo`` into ``C:/proc/meminfo``, which no ``/proc/`` prefix
    would match. Testing the raw string keeps the guard honest wherever the
    read comes from.
    """
    return str(path).replace("\\", "/").startswith(_PSEUDO_FS_PREFIXES)


# ---------------------------------------------------------------------------
# Reads made by the interpreter and by libraries for themselves
# ---------------------------------------------------------------------------
#
# The tracker sits on the process-wide ``open`` / ``os.listdir``, so it also
# saw what Python and libraries read for their OWN purposes while the user's
# statement ran: the import system listing every ``sys.path`` directory
# (including the notebook's own folder) and reading ~100 ``entry_points.txt``
# files, matplotlib loading its style sheets and font cache on import and its
# fonts on first draw, scikit-learn reading the template for an estimator's
# HTML display. Round 21 traced every "code or state changed" with no change
# to them: they happen only the FIRST time (the second run finds everything
# loaded), so the same statement got a different lineage on a re-run; a new
# file anywhere next to the notebook invalidated everything after an import;
# and a figure that looked changed had its ``savefig`` replayed without its
# plotting calls -- a blank chart. None of it is the user's data.
#
# What is NOT dropped: any read that could be the user's data. A library
# opening a file for the user (``PIL.Image.open(p)``, ``torch.load(p)``) reads
# a path outside that library, so it stays tracked; a user module reading its
# config at import stays tracked; and an installed tool reading its own data
# file stays tracked for that tool's own cached functions (``own_package``).

#: Modules that look up package METADATA or RESOURCES -- never user data.
_METADATA_MODULES: tuple[str, ...] = (
    "importlib.metadata", "importlib_metadata", "importlib.resources",
    "importlib_resources", "pkg_resources", "pkgutil",
)

#: Modules a read passes through between the code that asked for it and the OS.
_READ_PLUMBING: tuple[str, ...] = (
    "io", "_io", "codecs", "pathlib", "contextlib", "zipfile", "shutil",
    "tempfile", "os", "posixpath", "ntpath", "genericpath", "fnmatch", "glob",
    "cash",
)


def _in_modules(module: str, names: tuple[str, ...]) -> bool:
    """Is *module* one of *names* or inside one of them (``os`` but not ``osgeo``)?"""
    return any(module == n or module.startswith(n + ".") for n in names)


def _nc(path: str) -> str:
    """*path* case-folded where the OS is, with forward slashes.

    ``normcase`` on Windows turns ``/`` back into ``\\``, so it has to come
    first, or a directory prefix would never match a path under it.
    """
    return normalize_path(os.path.normcase(path))


def _norm_dir(path: str) -> str:
    return _nc(os.path.abspath(path)).rstrip("/") + "/"


@functools.lru_cache(maxsize=1)
def _interpreter_roots() -> tuple[str, ...]:
    """The standard library, its compiled extensions and zipped stdlib."""
    import sysconfig
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib"):
        if paths.get(key):
            roots.add(_norm_dir(paths[key]))
    for prefix in {sys.base_prefix, sys.base_exec_prefix}:
        roots.add(_norm_dir(os.path.join(prefix, "DLLs")))
    for entry in sys.path:
        if entry and entry.lower().endswith(".zip"):
            roots.add(_norm_dir(entry))
    return tuple(sorted(roots))


@functools.lru_cache(maxsize=1)
def _site_roots() -> tuple[str, ...]:
    """Where installed third-party packages live (site-packages).

    Often INSIDE the standard library directory (``Lib/site-packages`` on
    Windows, ``lib/python3.X/site-packages`` in conda), so the interpreter
    test has to exclude these, or it would swallow every installed package --
    and with it the own-package exemption.
    """
    import site
    import sysconfig
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            roots.add(_norm_dir(paths[key]))
    try:
        for entry in site.getsitepackages():
            roots.add(_norm_dir(entry))
    except AttributeError:          # virtualenv's old site.py
        pass
    try:
        roots.add(_norm_dir(site.getusersitepackages()))
    except (AttributeError, TypeError):
        pass
    # On Windows `getsitepackages()` also lists the installation prefix itself.
    # That is the standard library's parent -- or, for a venv created as the
    # project folder, the user's whole project -- never a package directory.
    prefixes = {_norm_dir(p) for p in (sys.prefix, sys.exec_prefix,
                                       sys.base_prefix, sys.base_exec_prefix)}
    return tuple(sorted(roots - prefixes))


@functools.lru_cache(maxsize=1)
def _installed_roots() -> tuple[str, ...]:
    """Where installed packages live: site-packages and the standard library."""
    return tuple(sorted(set(_interpreter_roots()) | set(_site_roots())))


def _under(path_nc: str, roots: tuple[str, ...]) -> bool:
    return any(path_nc.startswith(root) for root in roots)


@functools.lru_cache(maxsize=1)
def _tz_roots() -> tuple[str, ...]:
    """The system time zone database directories ``zoneinfo`` searches."""
    try:
        import zoneinfo
    except ImportError:
        return ()
    return tuple(sorted({_norm_dir(p) for p in zoneinfo.TZPATH if os.path.isdir(p)}))


def _installed_data_file(path_nc: str, own_package: str | None) -> bool:
    """Is *path_nc* a file of an installed package other than *own_package*,
    or of the system time zone database?

    Whoever reads it, it is library data, not the user's. Round 30 (r30s1):
    ``zoneinfo`` -- the standard library, so not "a library reading its own
    package" -- loaded ``tzdata/zoneinfo/UTC`` on the first load in a process
    and kept the zone for the rest of it. The load's lineage carried that file
    after a restart and not on a re-run in the same session, and everything
    below it missed once.
    """
    if _under(path_nc, _tz_roots()):
        return True
    for root in _site_roots():
        if path_nc.startswith(root):
            top = path_nc[len(root):].split("/", 1)[0]
            name = top.split(".", 1)[0].split("-", 1)[0]
            return "/" in path_nc[len(root):] and name != own_package
    return False


def _module_package_dir(module_name: str) -> str | None:
    """The directory of *module_name*'s top-level package, or None."""
    top = sys.modules.get(module_name.split(".")[0])
    if top is None:
        return None
    paths = getattr(top, "__path__", None)
    if paths:
        try:
            return _norm_dir(list(paths)[0])
        except (TypeError, IndexError):
            return None
    file = getattr(top, "__file__", None)
    return _norm_dir(os.path.dirname(file)) if file else None


#: module name -> (a metadata module?, read plumbing?, top-level name). A read
#: walks the whole stack, ~30 frames in a kernel, and a folder read does it for
#: every file: 5,030 reads re-classified the same modules (round 25, r25s4).
_MODULE_KIND: dict[str, tuple[bool, bool, str]] = {}


def incidental_read(path: str, own_package: str | None = None) -> str | None:
    """Why the read of *path* happening now is not the user's data, or None.

    Four cases, each measured in round 21: a file of the interpreter itself;
    a package metadata or resource lookup; a library reading files while it is
    being imported; and a library reading a file inside its own installed
    package directory. *own_package* is the top-level package of the code
    being cached -- its own files are its data, even when it is installed.
    """
    path_nc = _nc(path)
    if _under(path_nc, _interpreter_roots()) and not _under(path_nc, _site_roots()):
        return "interpreter"
    if _installed_data_file(path_nc, own_package):
        return "installed package data"
    installed = _installed_roots()
    frame = sys._getframe(1)
    reader_seen = False
    while frame is not None:
        module = frame.f_globals.get("__name__") or ""
        kind = _MODULE_KIND.get(module)
        if kind is None:
            kind = (_in_modules(module, _METADATA_MODULES), _in_modules(module, _READ_PLUMBING),
                    module.split(".")[0])
            if len(_MODULE_KIND) < 8192:
                _MODULE_KIND[module] = kind
        is_metadata, is_plumbing, top = kind
        if is_metadata:
            return "package metadata"
        code = frame.f_code
        # `__main__` is the user's notebook or script -- in a kernel its module
        # object is the ipykernel launcher in site-packages, which must not make
        # a notebook statement look like a library.
        if top not in (own_package, "cash", "__main__", ""):
            if code.co_name == "<module>":
                # Only a module the import system is executing RIGHT NOW: a
                # host that runs its main loop from module level (an execnet
                # worker, a launcher) must not turn every read into one.
                spec = frame.f_globals.get("__spec__")
                origin = getattr(spec, "origin", None)
                if (getattr(spec, "_initializing", False) and isinstance(origin, str)
                        and _under(_nc(origin), installed)):
                    return "library import"
            if not reader_seen and not is_plumbing:
                reader_seen = True
                package_dir = _module_package_dir(module) if module else None
                if (package_dir and _under(package_dir, installed)
                        and path_nc.startswith(package_dir)):
                    return "library resource"
        frame = frame.f_back
    return None


#: Path segments that belong to cash's OWN storage, never to the user's data.
#:
#: A cache HIT reads the entry's ``.data`` file to deserialise it, and that read
#: happens inside the enclosing statement's tracker window -- so without this
#: guard the statement acquires a dependency on a cash-internal file. The
#: consequence is not a stale value but an UNSTABLE LINEAGE: the dependency
#: exists on a run where the inner call hit and not on a run where it missed,
#: so the statement's output lineage differs between those runs. In a loop that
#: makes each iteration's key depend on whether the previous one was already
#: cached, and the loop converges one iteration per run -- measured 7, 6, 5, 4,
#: 3 real calls across five restarts of an 8-iteration loop, i.e. O(N) runs to
#: warm up.
#:
#: Same class as the ``/proc`` guard above: cash's own I/O must never become a
#: user-visible dependency. That one was found through a stale value, this one
#: through a cache that would not settle.
_CASH_INTERNAL_SEGMENTS: tuple[str, ...] = ("/.cash/", "/_global_cash/")

#: Cache directories that actually exist in this process, registered by the
#: backends that own them.
#:
#: The segment list above only recognises cash's storage when the directory is
#: NAMED ``.cash`` or ``_global_cash``. Any other ``cache_dir`` -- a path from
#: configuration, a temp directory in a test, ``~/caches/project-a`` -- went
#: unguarded, so cash's own entry files became dependencies of the user's
#: functions. That stayed invisible while entries were renamed into place,
#: since a renamed file is only ever observed at its final size; it surfaced
#: the moment a new entry was written in place, because ``O_CREAT`` makes the
#: file briefly observable at zero length and the next check reports "size
#: changed".
_CASH_CACHE_DIRS: set[str] = set()
_CASH_CACHE_DIRS_LOCK = threading.Lock()

#: What cash itself writes into a cache directory. Being in a registered
#: directory is NOT enough on its own: ``cache_dir`` can legitimately point at
#: a directory that also holds the user's data -- ``Cash(cache_dir=".")`` is
#: enough to do it -- and swallowing a real dependency is far worse than the
#: bug this guard exists to prevent. A missed dependency serves a stale value
#: silently; an extra one only costs a recompute.
_CASH_FILE_SUFFIXES: tuple[str, ...] = (
    ".entry",           # one file per entry (format v2)
    ".meta", ".data",   # the pair entries were stored as before v2
    ".part",            # a write still in flight
    ".db", ".db-wal", ".db-shm",   # SQLiteBackend
)
_CASH_FILE_NAMES: frozenset[str] = frozenset({
    "CACHE_VERSION",        # the on-disk format stamp
    "_loop_split.json",     # the notebook loop-split store
})


def register_cache_dir(path: str) -> None:
    """Declare *path* as cash's own storage, whatever it is called."""
    try:
        resolved = os.path.realpath(path)
    except OSError:
        resolved = os.path.abspath(path)
    with _CASH_CACHE_DIRS_LOCK:
        _CASH_CACHE_DIRS.add(resolved.replace("\\", "/").rstrip("/") + "/")


def _is_cash_storage_filename(name: str) -> bool:
    return name in _CASH_FILE_NAMES or name.endswith(_CASH_FILE_SUFFIXES)


def _is_cash_internal(path: str) -> bool:
    """True for a read or write of cash's own cache storage."""
    p = str(path).replace("\\", "/")
    if any(seg in p for seg in _CASH_INTERNAL_SEGMENTS):
        return True

    if not _is_cash_storage_filename(p.rsplit("/", 1)[-1]):
        return False
    with _CASH_CACHE_DIRS_LOCK:
        dirs = tuple(_CASH_CACHE_DIRS)
    if not dirs:
        return False
    # The recorded path may be relative while the registered one is absolute.
    absolute = p if os.path.isabs(p) else os.path.abspath(p).replace("\\", "/")
    return absolute.startswith(dirs)


#: ``code -> {file: its stat when last read}`` for files read while a frame of
#: that code was on the stack, process-wide. A memo (``functools.lru_cache``, a
#: module dict) hands a later call the product of an earlier read, and the
#: later call reads nothing -- so its entry recorded no file and kept serving
#: after the file changed (round 19). What a helper read once is what
#: `credited_reads` answers when a call reaches it again, and the stat says
#: WHICH version it read: a memo filled before the file changed hands back the
#: old version's data (round 20). Reads outside any cached call count too
#: (`_note_untracked_read`) -- `main()` logging its settings through the memo
#: before the first cached call is the ordinary way to fill one. A code past
#: `_READS_PER_CODE_MAX` files is marked ``None``: it reads per argument, and
#: every file it ever read is no one call's dependency.
_READS_BY_CODE: dict[Any, dict[str, Any] | None] = {}
_READS_BY_CODE_MAX = 4096
_READS_PER_CODE_MAX = 16
_CASH_PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIBRARY_ROOTS: tuple[str, ...] | None = None
_FILE_IS_USER: dict[str, bool] = {}


def _is_user_file(filename: str) -> bool:
    """Is *filename* code outside cash, the standard library and site-packages?"""
    verdict = _FILE_IS_USER.get(filename)
    if verdict is not None:
        return verdict
    global _LIBRARY_ROOTS
    if _LIBRARY_ROOTS is None:
        import sysconfig
        roots = {os.path.normcase(os.path.abspath(p)) for key, p in sysconfig.get_paths().items()
                 if key in ("stdlib", "platstdlib", "purelib", "platlib") and p}
        roots.add(_CASH_PACKAGE_DIR)
        _LIBRARY_ROOTS = tuple(sorted(roots))
    norm = os.path.normcase(os.path.abspath(filename)) if filename and not filename.startswith("<") else ""
    verdict = bool(norm) and not norm.startswith(_LIBRARY_ROOTS) and "site-packages" not in norm
    if len(_FILE_IS_USER) < 8192:
        _FILE_IS_USER[filename] = verdict
    return verdict


def _record_read(code: Any, abs_path: str, stat: Any) -> None:
    """Remember that *code* read *abs_path*, as it was (*stat*)."""
    reads = _READS_BY_CODE.get(code, ())
    if reads is None:
        return
    if code not in _READS_BY_CODE:
        if len(_READS_BY_CODE) >= _READS_BY_CODE_MAX:
            return
        reads = _READS_BY_CODE[code] = {}
    if abs_path not in reads and len(reads) >= _READS_PER_CODE_MAX:
        _READS_BY_CODE[code] = None
    else:
        reads[abs_path] = stat          # the LATEST read: a memo refilled is current again


def _is_cash_wrapper(filename: str) -> bool:
    norm = os.path.normcase(filename)
    return norm.endswith(os.path.join("cash", "core.py")) and norm.startswith(_CASH_PACKAGE_DIR)


def _credit_read_to_stack(abs_path: str, tracker: "FileAccessTracker") -> None:
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
            break                    # the cached call's own wrapper: the walk ends
        if kind == "user":
            tracker._note_reading_code(code)
            _record_read(code, abs_path, stat)
        frame, depth = frame.f_back, depth + 1


_UNTRACKED_REALPATH: dict[str, str] = {}
_UNTRACKED_STAT: dict[str, tuple[float, Any]] = {}
_FRAME_KIND: dict[str, str] = {}


def _frame_kind(filename: str) -> str:
    """``wrapper`` (the cached call's own), ``cash``, ``user`` or ``other``,
    remembered per filename: every read walks the stack."""
    kind = _FRAME_KIND.get(filename)
    if kind is None:
        kind = ("wrapper" if _is_cash_wrapper(filename)
                else "cash" if filename and os.path.normcase(filename).startswith(_CASH_PACKAGE_DIR)
                else "user" if _is_user_file(filename) else "other")
        if len(_FRAME_KIND) < 8192:
            _FRAME_KIND[filename] = kind
    return kind


#: Seconds spent recording reads, process-wide (see :func:`tracking_seconds`).
_TRACKING_SECONDS = [0.0]


def tracking_seconds() -> float:
    """Seconds cash has spent recording file reads in this process.

    Read before and after a statement, the difference is cash's own time inside
    it, which is not the statement's cost: a folder read recorded 16.5 s for a
    load that takes 1.8 s without cash, and a later hit credited all of it as
    saved (round 25, r25s4).
    """
    return _TRACKING_SECONDS[0]


def _note_untracked_read(path: Any) -> None:
    """A read made outside every cached call, credited to the user code on the stack.

    Only a read that user code started: a read by cash itself or by a library
    with no user frame above it (an import, a font cache) is nobody's input.
    Never raises -- it runs inside every ``open`` in the process.
    """
    try:
        frame = sys._getframe(2)
        codes = []
        depth = 0
        while frame is not None and depth < 64:
            kind = _frame_kind(frame.f_code.co_filename)
            if kind == "wrapper":
                break
            if kind == "cash" and not codes:
                return               # cash reading its own files
            if kind == "user":
                codes.append(frame.f_code)
            frame, depth = frame.f_back, depth + 1
        if not codes:
            return
        raw = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        if not isinstance(raw, str) or _is_pseudo_fs(raw) or is_remote_url(raw):
            return
        # `realpath` is 60us on Windows, most of what this costs; resolved once
        # per absolute path (so a chdir still resolves anew).
        absolute = os.path.abspath(raw)
        abs_path = _UNTRACKED_REALPATH.get(absolute)
        if abs_path is None:
            abs_path = normalize_path(os.path.realpath(absolute))
            if len(_UNTRACKED_REALPATH) >= 4096:
                _UNTRACKED_REALPATH.clear()
            _UNTRACKED_REALPATH[absolute] = abs_path
        if _is_pseudo_fs(abs_path) or _is_cash_internal(abs_path):
            return
        # A stat is 15us, and a loop re-reading one file pays it every time.
        # Reusing one taken in the last second can only be too OLD, and an old
        # stat that differs from the file makes a store refused, never a stale
        # answer served (`Cash._credit_remembered_reads`).
        now = time.monotonic()
        seen = _UNTRACKED_STAT.get(abs_path)
        if seen is not None and now - seen[0] < 1.0:
            stat = seen[1]
        else:
            stat = _regular_file_stat(abs_path)
            if len(_UNTRACKED_STAT) >= 4096:
                _UNTRACKED_STAT.clear()
            _UNTRACKED_STAT[abs_path] = (now, stat)
        if stat is None:
            return
        for code in codes:
            _record_read(code, abs_path, stat)
    except Exception:  # noqa: BLE001 - attribution is an aid, never a failure
        logger.debug("[TRACKER] could not note an untracked read of %r", path, exc_info=True)


def credited_reads(code: Any) -> dict[str, Any] | None:
    """``{file: stat when read}`` for files read while *code* was on the stack;
    None when it reads per argument."""
    reads = _READS_BY_CODE.get(code, ())
    return None if reads is None else dict(reads)


def install_read_watch() -> None:
    """Install the read patches now rather than at the first cached call.

    A memo is usually filled before any cached call runs -- ``main()`` logging
    its settings -- and a read the patches were not there to see is a read no
    entry can depend on. Called when a function is decorated.
    """
    with _install_lock:
        FileAccessTracker()._apply_patches()
    _ensure_import_hook_installed()


def _dispatch_track(path: Any) -> None:
    """Module-level tracker-dispatching shim. Custom handler factories
    registered via :func:`cash.register_file_handler` receive this as
    their ``tracker_callback`` argument. The shim consults
    ``_active_tracker`` at *call* time, so old-signature factories
    (whose wrappers do ``tracker_callback(path)``) transparently route
    to whichever tracker is active on the current asyncio task or
    thread — same isolation guarantees as the built-in handlers.
    """
    _tracker = _active_tracker.get()
    if _tracker is not None:
        _tracker._track_path(path)

def _find_patch_targets(func_pattern: str, module_obj: Any) -> list:
    """Return the list of attribute names to patch on *module_obj*."""
    if func_pattern.endswith('*'):
        prefix = func_pattern[:-1]
        return [name for name in dir(module_obj) if name.startswith(prefix)]
    if hasattr(module_obj, func_pattern):
        return [func_pattern]
    return []


def _install_module_patches(module_name: str, module_obj: Any) -> None:
    """Install dispatcher patches on a module. Idempotent — skips any
    target whose current attribute is already a dispatcher wrapper.

    Called from :meth:`FileAccessTracker._apply_patches` and from
    :class:`_PatchingLoader.exec_module` (post-import). The dispatcher
    wrappers route via ``_active_tracker`` so they're tracker-agnostic
    — one install serves all trackers.
    """
    registry = FileDependencyRegistry()
    handlers = registry.get_handlers_for_module(module_name)

    for func_pattern, factory in handlers:
        targets = _find_patch_targets(func_pattern, module_obj)
        for name in targets:
            if not hasattr(module_obj, name):
                continue
            original_func = getattr(module_obj, name)

            # Install-once skip.
            if getattr(original_func, '_is_file_tracker_patch', False):
                continue

            real_original = _unwrap_to_real(original_func)
            if not callable(real_original):
                continue

            wrapper = factory(real_original, _dispatch_track)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_original
            try:
                setattr(module_obj, name, wrapper)
            except (AttributeError, TypeError) as e:
                logger.debug("[FILE_TRACKER] Failed to patch %s.%s: %s", module_name, name, e)


def _patch_pathlib_accessor() -> None:
    """Route pathlib's own opener through the tracker on Python 3.10.

    On 3.11+ ``Path.open`` calls ``io.open`` directly, so the ``io.open`` patch
    covers pathlib. On 3.10 it goes through ``Path._accessor.open``, and
    ``_NormalAccessor.open = io.open`` captures the ORIGINAL at class-definition
    time — when ``pathlib`` is first imported, long before cash patches
    anything. Patching ``io.open`` therefore never reached pathlib there, and
    every pathlib read was invisible: a cell doing ``Path(p).read_text()``
    recorded no file dependency at all and was never invalidated when ``p``
    changed. Silent staleness, not a visible error.

    The accessor attribute holds a BUILTIN, which has no descriptor protocol,
    so pathlib calls it as ``acc.open(path, ...)``. A plain Python function
    installed in its place would bind as a method and swallow ``path`` as
    ``self`` — hence ``staticmethod``.
    """
    accessor = getattr(pathlib, '_NormalAccessor', None)
    if accessor is None:
        return  # 3.11+: patching io.open already covers pathlib

    original = getattr(accessor, 'open', None)
    if original is None or getattr(original, '_is_file_tracker_patch', False):
        return

    real_original = _unwrap_to_real(original)
    if not callable(real_original):
        return

    wrapper = FileDependencyRegistry()._create_open_handler(real_original, _dispatch_track)
    wrapper._is_file_tracker_patch = True
    wrapper._original_func = real_original
    try:
        accessor.open = staticmethod(wrapper)
    except (AttributeError, TypeError) as e:
        logger.debug("[FILE_TRACKER] Failed to patch pathlib accessor: %s", e)


def _patch_pathlib_listing() -> None:
    """Track the directories ``Path.glob`` / ``rglob`` / ``iterdir`` list.

    Listing a directory is how a new file matching a pattern becomes visible
    (see the ``os.scandir`` registration). pathlib mostly calls
    ``os.scandir`` at call time, which that patch covers -- but not on two
    versions: on 3.10 it lists through ``_NormalAccessor.scandir`` /
    ``.listdir``, and on 3.13 ``Path.glob`` goes through ``glob._Globber`` /
    ``glob._StringGlobber``, whose ``scandir`` holds the ORIGINAL
    ``os.scandir``, captured when the module was imported. A notebook reading
    ``sorted(Path("ledger").glob("*.csv"))`` recorded no directory there, and
    a new month's file was never seen: the concat was served from the cache
    without it (round 22, 3/3).
    """
    import glob as glob_module

    targets = [(getattr(pathlib, '_NormalAccessor', None), ('scandir', 'listdir'))]
    targets += [(getattr(glob_module, name, None), ('scandir',))
                for name in ('_Globber', '_StringGlobber')]
    factory = FileDependencyRegistry()._create_listdir_handler
    for owner, names in targets:
        if owner is None:
            continue
        for name in names:
            original = owner.__dict__.get(name)
            if isinstance(original, staticmethod):
                original = original.__func__
            if original is None or getattr(original, '_is_file_tracker_patch', False):
                continue
            real_original = _unwrap_to_real(original)
            if not callable(real_original):
                continue
            wrapper = factory(real_original, _dispatch_track)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_original
            try:
                setattr(owner, name, staticmethod(wrapper))
            except (AttributeError, TypeError) as e:
                logger.debug("[FILE_TRACKER] Failed to patch %s.%s: %s", owner.__name__, name, e)


def _track_regular_file(path: Any) -> None:
    """Record *path* as read when a tracker is active and it is a regular file.

    For the metadata calls (``Path.stat``, ``os.path.getsize`` ...): what they
    report is the file's, so the file is a dependency -- a directory has no
    content to hash, and an absent path raised before this was reached.
    ``os.stat``, not ``os.path.isfile``: that one is patched to record a
    NEGATIVE answer as an absent dependency.
    """
    tracker = _active_tracker.get()
    if tracker is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    try:
        if stat.S_ISREG(os.stat(path).st_mode):
            tracker._track_path(path)
    except (OSError, ValueError, TypeError):
        return


def _patch_pathlib_stat() -> None:
    """Track the file ``Path.stat()`` looks at.

    Round 24's r24s4 ended an export cell with
    ``print({p.name: p.stat().st_size for p in sorted(OUT.glob('*.csv'))})``:
    the folder's listing was a dependency and its names had not changed, so
    after the exports above were rewritten the line was served from the cache
    with the old sizes. Patched where ``stat`` is defined on ``Path``'s MRO,
    since pathlib has moved it between versions.
    """
    owner = next((k for k in pathlib.Path.__mro__ if 'stat' in k.__dict__), None)
    if owner is None:
        return
    original = owner.__dict__['stat']
    if getattr(original, '_is_file_tracker_patch', False) or not callable(original):
        return

    @functools.wraps(original)
    def tracked_path_stat(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _active_tracker.get() is not None and stat.S_ISREG(result.st_mode):
            _track_regular_file(self)
        return result

    tracked_path_stat._is_file_tracker_patch = True
    tracked_path_stat._original_func = original
    try:
        setattr(owner, 'stat', tracked_path_stat)
    except (AttributeError, TypeError) as e:
        logger.debug("[FILE_TRACKER] Failed to patch %s.stat: %s", owner.__name__, e)


def _patch_thread_pool_submit() -> None:
    """Run work submitted to a ``ThreadPoolExecutor`` under the submitter's context.

    The tracker is found through a ContextVar, and a pool's worker threads
    start with an empty context -- so ``ex.map(np.load, shards)`` inside a
    cached function read files no tracker saw, and editing a shard served the
    pre-edit result while the serial loop beside it invalidated (round 19).

    With a tracker active, ``submit`` (which ``Executor.map`` calls) wraps the
    call in ``copy_context().run``; with none, it is the original. A pool can
    opt out with ``_cash_internal = True``. Threads started directly with
    ``threading.Thread`` still begin empty -- documented, not patched.
    """
    import concurrent.futures.thread as cf_thread

    pool = cf_thread.ThreadPoolExecutor
    original = pool.__dict__.get("submit")
    if original is None or getattr(original, "_is_file_tracker_patch", False):
        return

    @functools.wraps(original)
    def submit(self, fn, /, *args, **kwargs):
        if _active_tracker.get() is None or getattr(self, "_cash_internal", False):
            return original(self, fn, *args, **kwargs)
        return original(self, contextvars.copy_context().run, fn, *args, **kwargs)

    submit._is_file_tracker_patch = True
    submit._original_func = original
    pool.submit = submit


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
    (0.14 s, once per worker) and runs the task under a tracker of its own.
    A cached function the task calls there propagates its reads -- and on a
    hit, its recorded ones -- into that tracker like into any outer call's.
    """
    __slots__ = ("fn",)

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def __reduce__(self):
        return (_ReadsInWorker, (self.fn,))

    def __call__(self, *args: Any, **kwargs: Any) -> _WorkerReads:
        tracker = FileAccessTracker()
        with tracker:
            value = self.fn(*args, **kwargs)
        return _WorkerReads(value, sorted(tracker.get_accessed_files()),
                            sorted(tracker.get_absent_files()))


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
    it invalidated (round 20). With a tracker active, ``submit`` (which
    ``Executor.map`` calls, chunked or not) sends a `_ReadsInWorker` instead of
    the bare function, and credits what it read to the submitting call when the
    result comes back -- before the caller can see the result, so before the
    call that waits on it is stored. ``multiprocessing.Pool`` and joblib are
    not wrapped: their reads stay unseen, and ``file_depends_on=`` names them.
    """
    import concurrent.futures.process as cf_process

    pool = cf_process.ProcessPoolExecutor
    original = pool.__dict__.get("submit")
    if original is None or getattr(original, "_is_file_tracker_patch", False):
        return

    @functools.wraps(original)
    def submit(self, fn, /, *args, **kwargs):
        tracker = _active_tracker.get()
        if tracker is None or getattr(self, "_cash_internal", False):
            return original(self, fn, *args, **kwargs)
        inner = original(self, _ReadsInWorker(fn), *args, **kwargs)
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
                    tracker._add_tracked(path)
                tracker.absent_files.update(result.absent)
                result = result.value
            outer.set_result(result)

        inner.add_done_callback(relay)
        return outer

    submit._is_file_tracker_patch = True
    submit._original_func = original
    pool.submit = submit


def _unwrap_to_real(func: Any) -> Any:
    """Walk a chain of FileAccessTracker wrappers down to the original
    callable. Returns ``func`` unchanged if it isn't a wrapper.

    Used by :func:`_install_module_patches` and
    :meth:`FileAccessTracker._patch_user_ns` so a fresh tracker can
    self-heal past wrappers left behind by a
    prior tracker that failed to unpatch (e.g. an exception during
    ``__exit__``, an orphaned tracker, etc.). Without this, the sentinel
    check ``_is_file_tracker_patch`` would cause the new tracker to skip
    the function entirely, leaving the leaked wrapper installed
    indefinitely and pinning a dead tracker instance in memory via its
    closure on ``_track_path``.
    """
    seen: set[int] = set()
    while getattr(func, '_is_file_tracker_patch', False):
        if id(func) in seen:  # broken/circular chain, bail
            break
        seen.add(id(func))
        next_func = getattr(func, '_original_func', None)
        if next_func is None or not callable(next_func):
            break
        func = next_func
    return func


class FileDependencyRegistry:
    """
    Registry for file dependency handlers.
    Allows easy extension of file tracking to new libraries and functions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.handlers = {} # Map module -> list of (func_name, handler_factory)
            cls._instance._initialize_defaults()
        return cls._instance

    def _initialize_defaults(self):
        """Initialize default handlers for common libraries."""
        # Builtins
        self.register('builtins', 'open', self._create_open_handler)

        # io (used by pathlib)
        self.register('io', 'open', self._create_open_handler)

        # Pandas
        self.register('pandas', 'read_*', self._create_path_arg_handler)

        # Polars
        self.register('polars', 'read_csv', self._create_path_arg_handler)
        self.register('polars', 'read_parquet', self._create_path_arg_handler)
        self.register('polars', 'read_json', self._create_path_arg_handler)
        self.register('polars', 'read_ndjson', self._create_path_arg_handler)
        self.register('polars', 'read_ipc', self._create_path_arg_handler)
        self.register('polars', 'read_avro', self._create_path_arg_handler)
        self.register('polars', 'read_excel', self._create_path_arg_handler)
        self.register('polars', 'scan_csv', self._create_path_arg_handler)
        self.register('polars', 'scan_parquet', self._create_path_arg_handler)
        self.register('polars', 'scan_ipc', self._create_path_arg_handler)
        self.register('polars', 'scan_ndjson', self._create_path_arg_handler)

        # Numpy
        # pyarrow reads in C++, so nothing passes through a patched open(): a
        # cached function that switched to pyarrow.csv for speed recorded no
        # file dependency at all, and a whole new export returned yesterday's
        # numbers (CAS-115). Path-taking readers only -- a class such as
        # ParquetFile is left alone, since replacing it with a function would
        # break isinstance checks.
        self.register('pyarrow.csv', 'read_csv', self._create_path_arg_handler)
        self.register('pyarrow.csv', 'open_csv', self._create_path_arg_handler)
        self.register('pyarrow.parquet', 'read_table', self._create_path_arg_handler)
        self.register('pyarrow.parquet', 'read_pandas', self._create_path_arg_handler)
        self.register('pyarrow.feather', 'read_table', self._create_path_arg_handler)
        self.register('pyarrow.feather', 'read_feather', self._create_path_arg_handler)
        self.register('pyarrow.json', 'read_json', self._create_path_arg_handler)

        self.register('numpy', 'load', self._create_path_arg_handler)
        self.register('numpy', 'loadtxt', self._create_path_arg_handler)
        self.register('numpy', 'genfromtxt', self._create_path_arg_handler)
        self.register('numpy', 'fromfile', self._create_path_arg_handler)

        # pyarrow.dataset reads in C++ like the rest of pyarrow; `read_table`
        # was registered and `dataset()` was not, so one entry point of an
        # otherwise-covered library went stale (found attacking the decorator
        # before round 26).
        self.register('pyarrow.dataset', 'dataset', self._create_path_arg_handler)

        # linecache reads through `tokenize._builtin_open`, a reference taken
        # at import time, so the patched `open` never sees it. Source files are
        # left out: linecache is what `inspect.getsource` (and every traceback)
        # reads with, and recording those made a module's own source a data
        # dependency of the functions in it.
        self.register('linecache', 'getline', self._create_source_reader_handler)
        self.register('linecache', 'getlines', self._create_source_reader_handler)

        # sqlite3 opens the database in C, so nothing reaches a patched
        # reader: a cached `select sum(x)` returned 1 where an uncached run
        # returned 101 after an INSERT, and `pd.read_sql_query` over the same
        # connection did too (found attacking the decorator before round 26).
        # The connection's path is the dependency; a URI or ":memory:" has no
        # file behind it and `_track_path` drops what it cannot resolve.
        self.register('sqlite3', 'connect', self._create_path_arg_handler)
        self.register('sqlite3.dbapi2', 'connect', self._create_path_arg_handler)

        # Joblib
        self.register('joblib', 'load', self._create_path_arg_handler)

        # Pickle
        self.register('pickle', 'load', self._create_path_arg_handler)

        # Json
        self.register('json', 'load', self._create_path_arg_handler)

        # Directory listing: a cell that enumerates a directory and
        # reads the matches gets file-deps only for the files READ on the first
        # run, so a NEW matching file is invisible. Track the enumerated
        # directory itself as a dependency - adding/removing an entry bumps the
        # directory's own mtime on local filesystems, so the existing mtime
        # freshness check invalidates the reader.
        self.register('glob', 'glob', self._create_glob_dir_handler)
        self.register('glob', 'iglob', self._create_glob_dir_handler)
        self.register('os', 'listdir', self._create_listdir_handler)
        self.register('os', 'scandir', self._create_listdir_handler)

        # Existence probes: "is there a config here?" The ABSENCE of a file is
        # an input -- it selects the defaults branch -- and it was the only
        # input cash could not see, because a file that is never opened
        # produces no read to track. An entry written by a run that found
        # nothing recorded no dependencies at all, so it looked valid
        # everywhere: a round-16 tester got directory B's answer in directory
        # A, silently, 4/4. Only a NEGATIVE result is recorded; a probe that
        # says yes is followed by the read that tracks it properly.
        self.register('os.path', 'exists', self._create_exists_handler)
        self.register('os.path', 'isfile', self._create_exists_handler)
        self.register('genericpath', 'exists', self._create_exists_handler)
        self.register('genericpath', 'isfile', self._create_exists_handler)

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

    def get_handlers_for_module(self, module_name: str) -> list[tuple[str, Callable[..., Any]]]:
        return self.handlers.get(module_name, [])

    # --- Standard Handler Factories ---

    @staticmethod
    def _create_open_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Handler for open()-like functions.

        ``track_callback`` is part of the user-facing handler factory
        signature (see :meth:`FileDependencyRegistry.register`) so
        custom factories can record the access. The built-in handlers
        ignore the argument and consult ``_active_tracker`` directly —
        that way one patch serves any number of concurrent trackers.
        """
        def tracked_open(file, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            # `w+` and `x+` start from an empty file, so nothing the code reads
            # back existed before it: a write, not an input. Pillow saves every
            # image with "w+b", and round 21 found each `savefig` recorded as a
            # dependency on its own output.
            if 'r' in mode or ('+' in mode and 'w' not in mode and 'x' not in mode):
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(file)
                    try:
                        return original_func(file, *args, **kwargs)
                    except FileNotFoundError:
                        # A file that was not there is an input too, and the
                        # docs say so -- but only the `os.path.exists` spelling
                        # recorded it. `try: open(p) except FileNotFoundError:`
                        # kept serving its default after the file appeared
                        # (found attacking the decorator before round 26).
                        if isinstance(file, (str, bytes, os.PathLike)):
                            _tracker._track_absent(file)
                        raise
                elif isinstance(file, (str, bytes, os.PathLike)):
                    _note_untracked_read(file)
            elif any(ch in mode for ch in ('w', 'a', 'x')):
                # Not a dependency -- a WRITE is an effect, not an input, and
                # folding it into the key would invalidate a function on its
                # own output. It is recorded for the effect observer instead:
                # a cache hit skips this write, and if the analyzer said
                # nothing (the write happened inside a library it does not walk
                # into) that is a silent behaviour change the user should hear
                # about. See cash.effect_observer.
                _observer = _active_effect_observer.get()
                if _observer is not None and not _is_cash_internal(file):
                    _observer.record_write(file)
            return original_func(file, *args, **kwargs)
        return tracked_open

    @staticmethod
    def _create_path_arg_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Generic handler for functions where the first argument is a path.

        ``track_callback`` is part of the user-facing factory signature
        — see :meth:`_create_open_handler`. The built-in wrapper
        consults ``_active_tracker`` directly.
        """
        # Positional OR keyword. The wrapper used to demand the path as its
        # first positional parameter, and these wrappers are installed once,
        # process-wide, on the first cached call -- so from then on
        # `pd.read_csv(filepath_or_buffer=p)`, `np.load(file=p)` or
        # `pq.read_table(source=p)` raised TypeError EVERYWHERE in the process,
        # inside cached code or not. Measured while adding the pyarrow readers.
        def tracked_func(*args, **kwargs):
            target = args[0] if args else next(
                (kwargs[k] for k in _PATH_KWARGS if k in kwargs), None)
            if isinstance(target, (str, bytes, os.PathLike)):
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(target)
                else:
                    _note_untracked_read(target)
            return original_func(*args, **kwargs)
        return tracked_func

    @staticmethod
    def _glob_base_dir(pattern: Any) -> str | None:
        """Return the longest leading, magic-free directory of a glob *pattern*.

        ``gdir/*.num`` → ``gdir``; ``a/b*/c`` → ``a`` (deepest stable ancestor).
        The directory's mtime is what we track for membership changes.
        """
        import glob as _glob
        try:
            parts = str(pattern).replace('\\', '/').split('/')
        except (TypeError, ValueError):
            return None
        base: list[str] = []
        for p in parts[:-1]:  # exclude the filename component
            if _glob.has_magic(p):
                break
            base.append(p)
        return '/'.join(base) or '.'

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
        def tracked_exists(path, *args, **kwargs):
            result = original_func(path, *args, **kwargs)
            if not result:
                _tracker = _active_tracker.get()
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
        def tracked_source_reader(filename, *args, **kwargs):
            if isinstance(filename, (str, bytes, os.PathLike)):
                text = os.fsdecode(filename) if isinstance(filename, bytes) else str(filename)
                if (not text.startswith("<")
                        and not text.endswith(FileDependencyRegistry._SOURCE_SUFFIXES)):
                    _tracker = _active_tracker.get()
                    if _tracker is not None:
                        try:
                            real = os.path.isfile(text)
                        except (OSError, ValueError):
                            real = False
                        if real:
                            _tracker._track_path(filename)
            return original_func(filename, *args, **kwargs)
        return tracked_source_reader

    @staticmethod
    def _create_glob_dir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Track the directory a ``glob`` pattern enumerates."""
        def tracked_glob(pathname, *args, **kwargs):
            _tracker = _active_tracker.get()
            if _tracker is not None:
                base = FileDependencyRegistry._glob_base_dir(pathname)
                if base is not None:
                    _tracker._track_path(base)
            return original_func(pathname, *args, **kwargs)
        return tracked_glob

    @staticmethod
    def _create_listdir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Track the directory passed to ``os.listdir`` / ``os.scandir``."""
        def tracked_listdir(path='.', *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(path)
            return original_func(path, *args, **kwargs)
        return tracked_listdir

class PostImportHook(importlib.abc.MetaPathFinder):
    """Intercepts imports of registered modules to patch them after loading.

    A single shared hook is installed once on ``sys.meta_path`` (see
    ``_shared_import_hook`` below). Module patching is tracker-agnostic
    — :func:`_install_module_patches` routes file reads via
    ``_active_tracker`` so the same patches serve every tracker.
    """
    def __init__(self) -> None:
        self._skip: set[str] = set()  # Avoid recursion

    def find_spec(self, fullname, path, target=None):
        if fullname in self._skip:
            return None

        # Only interest in registered modules
        # Note: We match top-level packages mainly.
        # e.g. 'pandas.io' -> we patch 'pandas' too?
        # The handlers are registered by module name.
        top_level = fullname.split('.')[0]

        targets = FileDependencyRegistry().handlers.keys()
        if fullname not in targets and top_level not in targets:
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
        # tracker-agnostic, install-once-per-target.
        _install_module_patches(self.fullname, module)


# Shared meta_path hook — installed at most once for the lifetime of
# the process by :meth:`FileAccessTracker.__enter__`. Tracker-agnostic.
_shared_import_hook: Optional[PostImportHook] = None
_shared_import_hook_lock = threading.Lock()


def _ensure_import_hook_installed() -> None:
    global _shared_import_hook
    with _shared_import_hook_lock:
        if _shared_import_hook is None:
            _shared_import_hook = PostImportHook()
            sys.meta_path.insert(0, _shared_import_hook)

class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **ContextVar dispatch (current design)**: On the first
    ``__enter__`` ever, install once-permanently dispatcher wrappers on
    ``builtins.open``, registered pandas/polars/numpy/joblib/pickle/json
    I/O functions, the user namespace ``open``, and a meta-path import
    hook for libraries loaded later. The wrappers consult a
    ``ContextVar`` (``_active_tracker``) at *call* time to decide
    whether to record the access. ``__enter__`` sets that ContextVar
    to ``self`` and stores the token; ``__exit__`` ``reset()``s it.

    Because ``ContextVar`` values are isolated per ``asyncio.Task`` and
    per ``threading.Thread`` by default, concurrent trackers — e.g. two
    coroutines under ``asyncio.gather``, or two worker threads — each
    see only their own block's reads.

    **No per-tracker unpatching**: the wrappers stay installed for the
    lifetime of the process. With no active tracker the dispatcher is a
    single ``ContextVar.get()`` + ``is None`` check before falling
    through to the original — sub-microsecond.

    **Limitation**: The heuristic does not track writes (e.g., ``to_csv``,
    ``np.save``).  Write-side dependencies are not needed for cache invalidation
    because the *output* of a statement is hashed directly, not the files it
    writes.
    """
    def __init__(self, user_ns=None, propagate_to_parent: bool = False,
                 hash_on_read: bool = False):
        self.accessed_files = set()
        # The top-level package of the code being cached, when *user_ns* is a
        # module's globals (the decorator): an installed tool's own files are
        # its data. A notebook's namespace is `__main__` -- no package.
        name = user_ns.get("__name__") if isinstance(user_ns, dict) else None
        self._own_package = (name.split(".")[0] if isinstance(name, str)
                             and name != "__main__" else None)
        # The content hash of each regular file WHEN IT WAS FIRST READ, for a
        # caller that stores what the block read (the decorator). Taken at
        # store time instead, a file changed mid-call by a writer that moves no
        # timestamp -- an np.memmap write on Windows -- was fingerprinted as
        # the NEW file next to a result computed from the old one, and served
        # to every later process (round 20). Moving the hash here costs
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
        # what the body read -- and served, stale, forever after (CAS-109).
        # Comparing against this is what lets the store step refuse instead.
        self.read_stats: dict[str, tuple[int, int, int]] = {}
        self.user_ns = user_ns or {}
        self.registry = FileDependencyRegistry()
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
        # `_credit_read_to_stack`): its recorded reads are live, not remembered.
        self.reading_codes: set[Any] = set()
        # Files a memo handed this block data from that was read from an
        # EARLIER version of the file (see `Cash._credit_remembered_reads`).
        self.stale_memo_reads: set[str] = set()

    def __enter__(self):
        # Install permanent dispatcher patches. Each (module/dict, name)
        # target is patched only once for the lifetime of the process —
        # ``_apply_patches`` self-skips targets whose current attribute
        # already carries the ``_is_file_tracker_patch`` sentinel. This
        # is cheap on the hot path (a single ``getattr`` per registered
        # target) and lets newly-imported modules and newly-registered
        # handlers be picked up on the next ``__enter__``.
        with _install_lock:
            self._apply_patches()
        _ensure_import_hook_installed()
        # Capture the enclosing tracker (if any) BEFORE we become active, so a
        # read inside this block also registers with the outer tracker(s).
        self._parent_stack.append(_active_tracker.get())
        self._token_stack.append(_active_tracker.set(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token_stack:
            _active_tracker.reset(self._token_stack.pop())
        if self._parent_stack:
            self._parent_stack.pop()

    def suspend(self):
        """Stop tracking until :meth:`resume`, restoring the enclosing tracker.

        For a streaming cached generator: production is tracked, the caller's
        loop body is not. `__enter__` cannot be used per item -- it reinstalls
        patches and re-checks the import hook every time, measured at 5.1us
        against 0.15us for the ContextVar swap alone, which on a 200k-item
        iterator is over a second of pure bookkeeping.
        """
        parent = self._parent_stack[-1] if self._parent_stack else None
        return _active_tracker.set(parent)

    def resume(self, token) -> None:
        """Undo :meth:`suspend`."""
        _active_tracker.reset(token)

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
            if _regular_file_stat(path) != before:
                moved.append(path)
        return moved

    def get_accessed_remote_urls(self) -> set[str]:
        """Remote URLs read in this block, tracked by store validator instead."""
        return self.accessed_remote

    def get_absent_files(self) -> set[str]:
        """Paths this block looked for and did not find."""
        return self.absent_files

    def _track_path(self, path):
        started = time.perf_counter()
        try:
            self._track_path_untimed(path)
        finally:
            _TRACKING_SECONDS[0] += time.perf_counter() - started

    def _track_path_untimed(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            # ``open(3)`` opens a file DESCRIPTOR: joblib and loky do, and
            # ``str(3)`` was recorded as a read of ``<cwd>/3`` -- a directory
            # every file under the cwd sits in, so every write there read as
            # an input (round 25, r25s1).
            return
        raw_path = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if _is_pseudo_fs(raw_path):
            # See _PSEUDO_FS_PREFIXES. Checked BEFORE realpath, which on
            # Windows rewrites /proc/... to C:/proc/... and would slip past.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", raw_path)
            return
        if is_remote_url(raw_path):
            # A remote URL is a real dependency, just not a stat-able one:
            # ``realpath`` would mangle it into a nonexistent local path and the
            # dependency would vanish. Record it on the remote channel, where it
            # is tracked by the store's own validator. See CAS-236.
            self._add_tracked_remote(raw_path)
            return
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls. Resolved once per cell run
            # (``realpath_this_run``): a loop reads the same files again.
            from cash.notebook.file_dep_snapshot import realpath_of_read_this_run
            resolved, read_lstat = realpath_of_read_this_run(raw_path)
            abs_path = normalize_path(resolved)
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)
            return
        if _is_pseudo_fs(abs_path):
            # See _PSEUDO_FS_PREFIXES: recording one of these makes the entry
            # permanently unfreshenable. Return before the relative-path arm
            # too — these paths are always absolute.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", abs_path)
            return
        if _SCRATCH_MEMMAP in abs_path:
            # joblib's memmaps of a parallel call's arrays: deleted when the
            # call returns, so recorded, every entry that read them was stale
            # for ever -- r25s1's ``cross_val_predict(n_jobs=4)`` loop re-ran
            # on every run of the report cell (round 25).
            logger.debug("[TRACKER] Ignoring joblib scratch read %r", abs_path)
            return
        if _RUNTIME_CACHE_SEGMENT in abs_path or abs_path.endswith(_RUNTIME_CACHE_SUFFIXES):
            logger.debug("[TRACKER] Ignoring runtime-cache read %r", abs_path)
            return
        if _is_cash_internal(abs_path):
            # See _CASH_INTERNAL_SEGMENTS. Checked after realpath so a relative
            # or symlinked cache path is caught too.
            logger.debug("[TRACKER] Ignoring cash-internal read %r", abs_path)
            return
        why = incidental_read(abs_path, self._own_package)
        if why is not None:
            logger.debug("[TRACKER] Ignoring %s read %r", why, abs_path)
            return
        self._add_tracked(abs_path, lstat=read_lstat)
        try:
            _credit_read_to_stack(abs_path, self)
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
                    self._add_tracked(rel)
            elif raw:
                # An ABSOLUTE path through a junction or symlink gets the same
                # treatment: its unresolved form is recorded too. The realpath
                # above resolved the link at WRITE time, so after the link is
                # re-pointed every check stats the old target -- which still
                # exists and has not changed. Round 17 lost eleven nightly
                # reports to that, including a rollback that returned the NEWER
                # release's answer (CAS-108). Checked through the link as it
                # points NOW, the switch is seen; the realpath entry keeps
                # catching an edit to the target itself.
                link = normalize_path(os.path.abspath(raw))
                if os.path.normcase(link) != os.path.normcase(abs_path):
                    self._add_tracked(link)
        except (TypeError, ValueError, OSError):
            logger.debug("[TRACKER] Could not record unresolved path for %r", path)

    def _add_tracked(self, abs_path: str, digest: str | None = None, lstat: Any = None) -> None:
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
            st = (_regular_file_stat(abs_path) if lstat is None
                  else (lstat.st_size, lstat.st_mtime_ns, getattr(lstat, "st_ctime_ns", 0)))
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
        if not self._propagate_to_parent:
            return
        parent = self._parent_stack[-1] if self._parent_stack else None
        if parent is not None and parent is not self:
            parent._add_tracked(abs_path, digest)

    def _digest_now(self, abs_path: str, size: int) -> str | None:
        """The file's content hash as the body is about to read it."""
        from cash.notebook.file_dep_snapshot import file_content_hash
        t0 = time.perf_counter()
        try:
            return file_content_hash(abs_path, size)
        finally:
            self.read_hash_seconds += time.perf_counter() - t0

    def _note_reading_code(self, code: Any) -> None:
        self.reading_codes.add(code)
        if not self._propagate_to_parent:
            return
        parent = self._parent_stack[-1] if self._parent_stack else None
        if parent is not None and parent is not self:
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
        if not raw or _is_pseudo_fs(raw):
            return
        try:
            normalized = normalize_path(raw)
        except (TypeError, ValueError):
            return
        if _is_cash_internal(normalized):
            return
        if os.path.isabs(raw):
            # An absolute probe is about one fixed file, so record it resolved
            # the way a read of it would be -- otherwise the two spellings of
            # the same path would not match.
            try:
                normalized = normalize_path(os.path.realpath(raw))
            except (TypeError, ValueError, OSError):
                pass
            if _is_pseudo_fs(normalized) or _is_cash_internal(normalized):
                return
        # A library probing for an optional file while it is imported
        # (matplotlib looks for a `matplotlibrc` in the working directory) or
        # inside its own package is not the user's question either.
        if incidental_read(os.path.abspath(raw), self._own_package) is not None:
            return
        self._add_tracked_absent(normalized)

    def _add_tracked_absent(self, path: str) -> None:
        """Record an absent path here and, when propagating, on the parents."""
        self.absent_files.add(path)
        if not self._propagate_to_parent:
            return
        parent = self._parent_stack[-1] if self._parent_stack else None
        if parent is not None and parent is not self:
            parent._add_tracked_absent(path)

    def _add_tracked_remote(self, url: str) -> None:
        """Record a remote *url* read, propagating to the enclosing tracker."""
        self.accessed_remote.add(url)
        if not self._propagate_to_parent:
            return
        parent = self._parent_stack[-1] if self._parent_stack else None
        if parent is not None and parent is not self:
            parent._add_tracked_remote(url)

    def _apply_patches(self):
        # 1. Patch Builtins
        _install_module_patches('builtins', builtins)

        # 2. Patch User Namespace (for interactive sessions showing 'open')
        self._patch_user_ns()

        # 2b. Python 3.10 only: pathlib captured io.open at import time, so the
        # io.open patch above misses every pathlib read. See the function.
        _patch_pathlib_accessor()
        _patch_pathlib_listing()
        _patch_pathlib_stat()

        # 2c. Work handed to a thread pool runs under the submitter's tracker,
        # and work handed to a process pool reports what it read back to it.
        _patch_thread_pool_submit()
        _patch_process_pool_submit()

        # 3. Patch Loaded Modules
        # Iterate over registered modules
        for mod_name in self.registry.handlers:
            if mod_name == 'builtins':
                continue

            if mod_name in sys.modules:
                module = sys.modules[mod_name]
                _install_module_patches(mod_name, module)

    def _patch_user_ns(self):
        """Patch open in user namespace (IPython specific). Self-heals
        by walking past any leaked wrappers to the real callable."""
        # Handle user_ns['open'] — skip if dispatcher already installed.
        if 'open' in self.user_ns and not getattr(
            self.user_ns['open'], '_is_file_tracker_patch', False
        ):
            real_open = _unwrap_to_real(self.user_ns['open'])
            factory = self.registry._create_open_handler
            wrapper = factory(real_open, _dispatch_track)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_open

            self.user_ns['open'] = wrapper

        # Handle user_ns['__builtins__']['open'] (if dict)
        if '__builtins__' in self.user_ns:
            bs = self.user_ns['__builtins__']
            if isinstance(bs, dict) and 'open' in bs and not getattr(
                bs['open'], '_is_file_tracker_patch', False
            ):
                real_open = _unwrap_to_real(bs['open'])
                factory = self.registry._create_open_handler
                wrapper = factory(real_open, _dispatch_track)
                wrapper._is_file_tracker_patch = True
                wrapper._original_func = real_open

                bs['open'] = wrapper

