"""Which reads are the user's data, and which are not.

The tracker sees every read in the process, so it also sees reads that are
machine state (kernel pseudo-filesystems), a runtime's own caches, cash's own
storage, and what the interpreter and libraries read for themselves. The
checks here tell those apart from the user's data. `regular_file_stat` is the
stat the tracker keeps of each file it records.
"""

from __future__ import annotations

import functools
import os
import stat
import sys
import threading
import zoneinfo

from cash._memo import SOURCE_FILES, LruMemo
from cash.install_paths import installed_roots, interpreter_roots, norm_dir, normcase_path, site_roots

__all__ = [
    "RUNTIME_CACHE_SEGMENT",
    "RUNTIME_CACHE_SUFFIXES",
    "SCRATCH_MEMMAP",
    "incidental_read",
    "is_cash_internal",
    "is_pseudo_fs",
    "register_cache_dir",
    "regular_file_stat",
]

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
# Same shape as cash's own ``open`` shim poisoning its cache key: the tracker cannot tell cash's internal reads from the user's, so paths that
# are definitionally not data get excluded here. Linux-only in effect; on
# Windows these prefixes never match.
_PSEUDO_FS_PREFIXES: tuple[str, ...] = ("/proc/", "/sys/", "/dev/")


def regular_file_stat(path: str) -> tuple[int, int, int] | None:
    """``(size, mtime_ns, ctime_ns)`` for a regular file, None otherwise.

    ``ctime_ns`` is the inode change time on POSIX, so an edit that restores
    the mtime still moves this tuple there; on Windows it is the creation
    time and adds nothing.
    """

    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", 0))


#: The folder joblib memory-maps a parallel call's large arguments into.
SCRATCH_MEMMAP = "joblib_memmapping_folder_"

#: Caches a runtime keeps for ITSELF: the interpreter's bytecode and numba's
#: JIT index/data (``.nbi`` / ``.nbc``), which live in a ``__pycache__`` next to
#: the code or wherever ``NUMBA_CACHE_DIR`` points. Never the user's data, and
#: rewritten by any other process that runs the same function -- recorded,
#: scanpy's normalize made every step after it re-run after a restart.
RUNTIME_CACHE_SEGMENT = "/__pycache__/"
RUNTIME_CACHE_SUFFIXES = (".nbi", ".nbc", ".pyc")


def is_pseudo_fs(path: str) -> bool:
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
# HTML display. Each was behind a "code or state changed" with no change to
# them: they happen only the FIRST time (the second run finds everything
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
    "importlib.metadata",
    "importlib_metadata",
    "importlib.resources",
    "importlib_resources",
    "pkg_resources",
    "pkgutil",
)

#: Modules a read passes through between the code that asked for it and the OS.
_READ_PLUMBING: tuple[str, ...] = (
    "io",
    "_io",
    "codecs",
    "pathlib",
    "contextlib",
    "zipfile",
    "shutil",
    "tempfile",
    "os",
    "posixpath",
    "ntpath",
    "genericpath",
    "fnmatch",
    "glob",
    "cash",
)


def _in_modules(module: str, names: tuple[str, ...]) -> bool:
    """Is *module* one of *names* or inside one of them (``os`` but not ``osgeo``)?"""
    return any(module == n or module.startswith(n + ".") for n in names)


def _under(path_nc: str, roots: tuple[str, ...]) -> bool:
    return any(path_nc.startswith(root) for root in roots)


@functools.lru_cache(maxsize=1)
def _tz_roots() -> tuple[str, ...]:
    """The system time zone database directories ``zoneinfo`` searches."""
    return tuple(sorted({norm_dir(p) for p in zoneinfo.TZPATH if os.path.isdir(p)}))


def _installed_data_file(path_nc: str, own_package: str | None) -> bool:
    """Is *path_nc* a file of an installed package other than *own_package*,
    or of the system time zone database?

    Whoever reads it, it is library data, not the user's. For example
    ``zoneinfo`` -- the standard library, so not "a library reading its own
    package" -- loaded ``tzdata/zoneinfo/UTC`` on the first load in a process
    and kept the zone for the rest of it. The load's lineage carried that file
    after a restart and not on a re-run in the same session, and everything
    below it missed once.
    """
    if _under(path_nc, _tz_roots()):
        return True
    for root in site_roots():
        if path_nc.startswith(root):
            top = path_nc[len(root) :].split("/", 1)[0]
            name = top.split(".", 1)[0].split("-", 1)[0]
            return "/" in path_nc[len(root) :] and name != own_package
    return False


def _module_package_dir(module_name: str) -> str | None:
    """The directory of *module_name*'s top-level package, or None."""
    top = sys.modules.get(module_name.split(".")[0])
    if top is None:
        return None
    paths = getattr(top, "__path__", None)
    if paths:
        try:
            return norm_dir(list(paths)[0])
        except (TypeError, IndexError):
            return None
    file = getattr(top, "__file__", None)
    return norm_dir(os.path.dirname(file)) if file else None


#: module name -> (metadata module?, read plumbing?, top-level name). A read
#: walks the whole stack, ~30 frames in a kernel, and a folder read does it for
#: every file, so each module is classified once.
_module_kinds: LruMemo[str, tuple[bool, bool, str]] = LruMemo(SOURCE_FILES)


def incidental_read(path: str, own_package: str | None = None) -> str | None:
    """Why the read of *path* happening now is not the user's data, or None.

    Four cases, each seen in practice: a file of the interpreter itself;
    a package metadata or resource lookup; a library reading files while it is
    being imported; and a library reading a file inside its own installed
    package directory. *own_package* is the top-level package of the code
    being cached -- its own files are its data, even when it is installed.
    """
    path_nc = normcase_path(path)
    if _under(path_nc, interpreter_roots()) and not _under(path_nc, site_roots()):
        return "interpreter"
    if _installed_data_file(path_nc, own_package):
        return "installed package data"
    installed = installed_roots()
    frame = sys._getframe(1)
    reader_seen = False
    while frame is not None:
        module = frame.f_globals.get("__name__") or ""
        kind = _module_kinds.get(module)
        if kind is None:
            kind = (_in_modules(module, _METADATA_MODULES), _in_modules(module, _READ_PLUMBING), module.split(".")[0])
            _module_kinds[module] = kind
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
                if (
                    getattr(spec, "_initializing", False)
                    and isinstance(origin, str)
                    and _under(normcase_path(origin), installed)
                ):
                    return "library import"
            if not reader_seen and not is_plumbing:
                reader_seen = True
                package_dir = _module_package_dir(module) if module else None
                if package_dir and _under(package_dir, installed) and path_nc.startswith(package_dir):
                    return "library resource"
        frame = frame.f_back
    return None


#: Path segments that belong to cash's OWN storage, never to the user's data.
#:
#: A cache HIT reads the ``.entry`` file to deserialise it, and that read
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


def register_cache_dir(path: str) -> None:
    """Declare *path* as cash's own storage, whatever it is called."""
    try:
        resolved = os.path.realpath(path)
    except OSError:
        resolved = os.path.abspath(path)
    with _CASH_CACHE_DIRS_LOCK:
        _CASH_CACHE_DIRS.add(resolved.replace("\\", "/").rstrip("/") + "/")


def is_cash_internal(path: str) -> bool:
    """True for a read or write of cash's own cache storage.

    Being in a registered directory is NOT enough on its own: ``cache_dir``
    can legitimately point at a directory that also holds the user's data --
    ``Cash(cache_dir=".")`` is enough to do it -- and swallowing a real
    dependency is far worse than the bug this guard exists to prevent. A
    missed dependency serves a stale value silently; an extra one only costs a
    recompute. So the file must also be one cash writes there
    (`cache_dir.is_cash_file`, the list ``cash clear`` uses too).
    """
    p = str(path).replace("\\", "/")
    if any(seg in p for seg in _CASH_INTERNAL_SEGMENTS):
        return True

    with _CASH_CACHE_DIRS_LOCK:
        dirs = tuple(_CASH_CACHE_DIRS)
    if not dirs:
        return False
    # The recorded path may be relative while the registered one is absolute.
    absolute = p if os.path.isabs(p) else os.path.abspath(p).replace("\\", "/")
    inside = [absolute[len(d) :] for d in dirs if absolute.startswith(d)]
    if not inside:
        return False
    # Imported here: cash.backends imports this module.
    from cash.backends.cache_dir import is_cash_file

    return any(is_cash_file(rel) for rel in inside)
