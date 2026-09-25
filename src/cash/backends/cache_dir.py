"""The file cache directory itself: its format stamp, its `.gitignore`, temp files.

Everything here is about the directory as a whole rather than about one entry:
`CacheDirStamp` keeps the format stamp that says which layout the entries use
and doubles as the directory's generation token, `recreate_cache_dir` builds a
missing directory the way the file backend would, and `create_temp_file` is the
temp-file primitive the entry writes and the writability probe share.

It also holds the one list of what cash writes into a cache directory
(`is_cash_file`): the name of every sidecar store is defined here, so ``cash
clear`` (which removes nothing else) and the file tracker (which never makes
these a dependency of user code) cannot drift from the writers.
"""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Callable
from typing import Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheStoreFailedWarning
from .entry_format import ENTRY_SUFFIX, MAGIC

logger = logging.getLogger(__name__)

__all__ = [
    "ANALYTICS_DB_FILENAME",
    "CACHE_FORMAT_VERSION",
    "COMPUTE_BASELINES_FILENAME",
    "DB_FILENAME",
    "ENTRY_GLOB",
    "EVICTIONS_FILENAME",
    "KEYS_DIRNAME",
    "LOOP_SPLIT_FILENAME",
    "MISS_GUARD_FILENAME",
    "RANK_INDEX_FILENAME",
    "VERSIONS_INDEX_FILENAME",
    "VERSION_FILENAME",
    "CacheDirStamp",
    "create_temp_file",
    "entry_totals",
    "is_cash_file",
    "recreate_cache_dir",
    "warn_if_unwritable",
    "write_all",
]

#: Version of the on-disk cache format (the ``*.entry`` layout in
#: ``entry_format``). Bump it, together with ``entry_format.MAGIC``, whenever a
#: change makes entries written by an older build undecodable or liable to be
#: misread. A directory stamped with another version is cleared on open.
CACHE_FORMAT_VERSION = 2

#: The per-directory format stamp. No entry suffix, so entry globs skip it.
VERSION_FILENAME = "CACHE_VERSION"

# Everything else cash writes into a cache directory. None of these has the
# entry suffix, so every entry glob (listing, sizing, clearing, format
# migration) passes them by. A new sidecar gets its name here, or `cash clear`
# refuses to remove a directory holding it.

#: What keeps the directory out of version control.
GITIGNORE_FILENAME = ".gitignore"
#: SQLiteBackend's database, when a tier gives a cache directory, not a path.
DB_FILENAME = "cache.db"
#: The file tier's eviction priorities (`rank_index`).
RANK_INDEX_FILENAME = "_rank.log"
#: The file tier's superseded versions (`versions`).
VERSIONS_INDEX_FILENAME = "_versions.log"
#: The file tier's notes of what its cap evicted (`eviction_log`).
EVICTIONS_FILENAME = "_evicted.log"
#: The notebook's statement miss guard.
MISS_GUARD_FILENAME = "_miss_guard.json"
#: The notebook's loop-split verdicts.
LOOP_SPLIT_FILENAME = "_loop_split.json"
#: The notebook's measured compute costs, for ``%cash_stats``.
COMPUTE_BASELINES_FILENAME = "_compute_baselines.json"
#: The analytics database, in the per-user cache root.
ANALYTICS_DB_FILENAME = "analytics.db"
#: The decorator's stored-key records: one ``<function>.json`` each.
KEYS_DIRNAME = ".keys"

#: Files SQLite creates beside a database: the WAL, its shared-memory index,
#: and the rollback journal of a database not in WAL mode.
_SQLITE_COMPANIONS = ("-wal", "-shm", "-journal")
_CASH_FILE_NAMES = frozenset(
    {
        VERSION_FILENAME,
        GITIGNORE_FILENAME,
        RANK_INDEX_FILENAME,
        VERSIONS_INDEX_FILENAME,
        EVICTIONS_FILENAME,
        MISS_GUARD_FILENAME,
        LOOP_SPLIT_FILENAME,
        COMPUTE_BASELINES_FILENAME,
        *(db + companion for db in (DB_FILENAME, ANALYTICS_DB_FILENAME) for companion in ("", *_SQLITE_COMPANIONS)),
    }
)

#: Every live entry in a cache directory.
ENTRY_GLOB = f"*{ENTRY_SUFFIX}"

GITIGNORE_TEXT = "# Created by cash: this directory is a cache.\n*\n"

#: How many names `create_temp_file` tries. Only a name we just created can
#: collide with a fresh 48-bit random one, so more than a handful means
#: something retrying cannot fix.
_TEMP_NAME_ATTEMPTS = 8


def entry_totals(cache_dir: str) -> tuple[int, int] | None:
    """``(entries, bytes)`` of the entry files in *cache_dir*, or None when the
    directory cannot be listed.

    One ``scandir`` and a ``stat`` per entry, no file opened. The eviction cap
    and ``cash info`` both count with this, so the size a cap is compared with
    is the size the CLI reports.
    """
    count = size = 0
    try:
        with os.scandir(cache_dir) as found:
            for entry in found:
                if not entry.name.endswith(ENTRY_SUFFIX):
                    continue
                try:
                    size += entry.stat().st_size
                except OSError:
                    continue  # removed while we looked
                count += 1
    except OSError:
        return None
    return count, size


def _is_temp_of(name: str, owner: Callable[[str], bool]) -> bool:
    """Is *name* the temp file of a write that replaces a file *owner* accepts?

    The stores write ``<name>.tmp``, ``<name>.<pid>.tmp`` or
    ``<name>.<pid>.<thread>.tmp`` beside the file and rename it into place.
    """
    if not name.endswith(".tmp"):
        return False
    base = name[: -len(".tmp")]
    while True:
        head, dot, tail = base.rpartition(".")
        if not dot or not tail.isdigit():
            break
        base = head
    return owner(base)


def _is_cash_name(name: str) -> bool:
    return name in _CASH_FILE_NAMES or name.endswith(ENTRY_SUFFIX)


def is_cash_file(relpath: str) -> bool:
    """Did cash write *relpath*, a path relative to a cache directory?

    An entry, the format stamp, the ``.gitignore``, a sidecar store, a
    stored-key record, an SQLite database with its companions, or the temp
    file of a write to one of these (``create_temp_file``'s ``.tmp-*.part``
    and ``.probe-*.tmp`` included). Anything else is the user's.
    """
    parts = relpath.replace("\\", "/").split("/")
    if len(parts) == 2 and parts[0] == KEYS_DIRNAME:
        name = parts[1]
        return name.endswith(".json") or _is_temp_of(name, lambda base: base.endswith(".json"))
    if len(parts) != 1:
        return False
    name = parts[0]
    if _is_cash_name(name) or _is_temp_of(name, _is_cash_name):
        return True
    return (name.startswith(".tmp-") and name.endswith(".part")) or (
        name.startswith(".probe-") and name.endswith(".tmp")
    )


def create_temp_file(directory: str, prefix: str = ".tmp-", suffix: str = ".part") -> tuple[int, str]:
    """Create a new file in *directory* and return ``(fd, path)``.

    ``tempfile.mkstemp`` without its retry of ``PermissionError``: CPython
    retries that up to 10,000 times behind an ``os.access`` check that on
    Windows knows nothing about ACLs, so a directory denied by ACL turned every
    write into a long hang. A cache write is best effort, so a permission error
    propagates at once and takes the ordinary failed-write path.

    Only ``FileExistsError`` is retried. ``os.urandom`` rather than ``random``:
    cash watches the process RNG, and must not draw from it itself. Six bytes
    keep the name as short as ``mkstemp``'s, for Windows' path-length limit.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    last: OSError | None = None
    for _ in range(_TEMP_NAME_ATTEMPTS):
        candidate = os.path.join(directory, f"{prefix}{os.urandom(6).hex()}{suffix}")
        try:
            return os.open(candidate, flags, 0o600), candidate
        except FileExistsError as exc:
            last = exc
            continue
    raise FileExistsError(
        f"could not find an unused temporary name in {directory!r} after {_TEMP_NAME_ATTEMPTS} attempts"
    ) from last


def write_all(fd: int, data: bytes) -> None:
    """``os.write`` until every byte is out; it may write fewer than asked."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _write_gitignore(cache_dir: str) -> None:
    """Keep a directory cash created out of version control, as ``.pytest_cache``
    does: a ``.gitignore`` of ``*`` inside it. Never over an existing file."""
    path = os.path.join(cache_dir, GITIGNORE_FILENAME)
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(GITIGNORE_TEXT)
    except OSError:
        logger.debug("Could not write %s", path, exc_info=True)


def recreate_cache_dir(cache_dir: str) -> bool:
    """Create *cache_dir* as the file backend would, if it is missing.

    For anything that writes a sidecar into the cache directory: a bare
    ``os.makedirs`` after ``cash clear`` left the next entry in a directory
    with no ``.gitignore`` and no format stamp. Returns whether it created it.
    Raises what ``os.makedirs`` raises.
    """
    if os.path.isdir(cache_dir):
        return False
    os.makedirs(cache_dir, exist_ok=True)
    _write_gitignore(cache_dir)
    path = os.path.join(cache_dir, VERSION_FILENAME)
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(str(CACHE_FORMAT_VERSION))
    except OSError:
        logger.debug("Could not write %s", path, exc_info=True)
    return True


def warn_if_unwritable(cache_dir: str) -> None:
    """Say at once, naming the path, if *cache_dir* cannot be written.

    Writes are asynchronous and best effort, so an unwritable directory
    otherwise has no symptom but a cache that never hits. One create and
    delete, on the first cache operation.
    """
    try:
        fd, probe = create_temp_file(cache_dir, prefix=".probe-", suffix=".tmp")
    except OSError as exc:
        warn_diagnostic(
            CashCacheStoreFailedWarning,
            "CACHE-DIR-UNWRITABLE",
            f"cash cannot write to its cache directory {cache_dir} "
            f"({type(exc).__name__}: {exc}). Nothing will be cached to disk "
            f"this run, so every call recomputes.",
            "point cash somewhere writable -- cash.configure(cache_dir=...), "
            "CASH_CACHE_DIR, or the cache_dir= argument -- or grant this "
            "user write permission on that path.",
        )
        return
    os.close(fd)
    try:
        os.remove(probe)
    except OSError:
        logger.debug("Could not remove writability probe %s", probe, exc_info=True)


class CacheDirStamp:
    """The format stamp of one cache directory, and what it tells other processes.

    The stamp says which entry layout the directory holds. Its file identity is
    also the directory's generation: ``cash clear --all`` removes it with the
    directory, and a partial clear rewrites it, so a process holding
    results in RAM can tell the cache was cleared under it.
    """

    #: How many unstamped entries `_entries_are_current` reads. A sample, so
    #: opening stays O(1) in the entry count; an old-format cache fails on its
    #: first entry.
    FORMAT_SAMPLE = 16

    def __init__(self, cache_dir: str, untracked: Callable[[], Any]) -> None:
        self.cache_dir = cache_dir
        self.path = os.path.join(cache_dir, VERSION_FILENAME)
        self._untracked = untracked
        #: The token this process's last stamp write left, and how many it has
        #: written. A process that started cold saw no stamp and then wrote
        #: one; without these it could not tell a later clear from "still new".
        self.written: tuple | None = None
        self.writes = 0

    def entry_files(self) -> list[str]:
        """Every entry file in the directory, without it becoming a dependency."""
        with self._untracked():
            return glob.glob(os.path.join(self.cache_dir, ENTRY_GLOB))

    def token(self) -> tuple | None:
        """The stamp's file identity, or None when there is no stamp."""
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def write(self) -> None:
        """Write the current format stamp."""
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(str(CACHE_FORMAT_VERSION))
        except OSError:
            logger.debug("Could not write cache format marker at %s", self.path, exc_info=True)
            return
        self.written = self.token()
        self.writes += 1

    def bump(self) -> None:
        """Give the stamp a new identity without changing what it says, so a
        running process notices that entries were removed under it. A missing
        stamp stays missing: the next write stamps the directory afresh."""
        if not os.path.exists(self.path):
            return
        tmp = self.path + ".tmp"
        try:
            with open(self.path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            os.replace(tmp, self.path)
        except OSError:
            logger.debug("Could not refresh %s", self.path, exc_info=True)

    def ignore_in_git(self) -> None:
        """Write the ``.gitignore`` of a directory cash just created."""
        _write_gitignore(self.cache_dir)

    def check(self) -> None:
        """Clear the entries if the stamp names another format, then stamp.

        A missing or unreadable stamp counts as a mismatch, unless the entries
        themselves carry the current format's magic: a directory cleared under
        a live process and refilled by it looks exactly like that.
        """
        stored: int | None = None
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    stored = int(fh.read().strip())
            except (OSError, ValueError):
                stored = None

        if stored == CACHE_FORMAT_VERSION:
            return
        entry_files = self.entry_files()
        if stored is None and self._entries_are_current(entry_files):
            entry_files = []
        if entry_files:
            logger.warning(
                "Cash cache at %s was written in format v%s but this build "
                "expects v%s; clearing %d stale file(s). Cache formats are not "
                "compatible across this change.",
                self.cache_dir,
                "<unstamped>" if stored is None else stored,
                CACHE_FORMAT_VERSION,
                len(entry_files),
            )
            for f in entry_files:
                try:
                    os.remove(f)
                except OSError:
                    logger.debug("Could not remove stale cache file %s", f, exc_info=True)
        self.write()

    @classmethod
    def _entries_are_current(cls, entry_files: list[str]) -> bool:
        """Do these unstamped entries carry the current format's magic?"""
        if not entry_files:
            return False
        for path in entry_files[: cls.FORMAT_SAMPLE]:
            try:
                with open(path, "rb") as fh:
                    if fh.read(len(MAGIC)) != MAGIC:
                        return False
            except OSError:
                return False
        return True
