"""Folder change notifications: a file's stat holds until its folder reports a change.

Before every cell the notebook checks each file the cached statements above
read, and a check is a stat per file: 19,000 photos in 300 folders cost
seconds a cell, on Windows most of all, where a stat opens the file. Most
cells change no file. So cash asks the operating system to report changes
to each folder it took a stat in, and a stat taken while its folder was
watched stands for the file (`FileWatch.current`) until the folder reports
anything at all: a write, a new or deleted name, a rename, a change of
attributes. On Linux each folder has an inotify watch, and only which
folder sent an event counts, not what it says. On Windows a handle on a
folder blocks renaming it and every folder above it, so each volume is
watched once from its root (`_VolumeChanges`), and the reported names say
which folders changed; a name that may be a short alias, or a lost report,
counts as a change of every folder it could name.

The rules that keep it safe:

* A watch is armed before the stat it vouches for is taken
  (`FileWatch.before_stat`), and the stat is kept only if no change was
  reported between the two (the folder's ``version``). A change before the
  watch started is in the stat itself.
* A folder's own stat (a listed folder) is kept only once a watch on the
  folder itself is armed and a second stat taken after it agrees.
* Events are consumed (`FileWatch.drain`) before any stat is handed out as
  current: the check before a cell drains them first, and each pass of
  the cell run's stats drains them again. On Linux an event is queued by the
  write itself, so a drain after a write sees it.
* A lost watch, a queue overflow, a folder whose path now names another
  folder (an ancestor renamed or re-linked: checked by a stat of each
  watched folder before each cell) drops every stat under it.
* No stat is kept of a link, of a file with a second hard link (written
  through the other name, its folder reports nothing), or in a folder that
  cannot be watched: a network or FUSE file system, past the watch limit,
  on macOS (not implemented: FSEvents needs a run loop), or after any error.
  Those are stat-ed every time, as without the watch.

``CASH_FILE_WATCH=0`` turns it off.
"""

from __future__ import annotations

import logging
import ntpath
import os
import re
import stat as _stat
import struct
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

from .read_classification import own_lstat, own_stat

__all__ = ["FileWatch", "watcher", "reset"]

logger = logging.getLogger(__name__)

#: At most this many folders are watched; past it, folders are stat-ed.
DIR_LIMIT = 8192

_S_IFMT = 0o170000
_S_IFDIR = 0o040000
_S_IFREG = 0o100000


class _Dir:
    """One watched folder: the stats its watch vouches for (``records``),
    the names it vouches are not there (``missing``), moved on by every
    change it reports (``version``)."""

    __slots__ = ("alive", "ident", "key", "listed", "missing", "path", "records", "version")

    def __init__(self, path: str, key: Any, ident: tuple[int, int]) -> None:
        self.path = path
        self.key = key
        self.ident = ident
        self.records: dict[str, os.stat_result] = {}
        self.listed: set[str] = set()
        self.missing: set[str] = set()
        self.version = 0
        self.alive = True


def _ident(st: os.stat_result) -> tuple[int, int]:
    return (st.st_dev, st.st_ino)


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (
        a.st_ino == b.st_ino
        and a.st_dev == b.st_dev
        and a.st_mode == b.st_mode
        and a.st_size == b.st_size
        and a.st_mtime_ns == b.st_mtime_ns
        and a.st_ctime_ns == b.st_ctime_ns
    )


class FileWatch:
    """The stats cash took while their folders were watched, each kept until
    its folder reports a change. *backend* is the operating system's side:
    ``add(path) -> key | None``, ``remove(key)``, ``poll() -> [(key, dead)]``
    (a key of None: events were lost, every folder changed). ``add`` finds
    out once whether *path* can be watched: None when it cannot."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._dirs: dict[str, _Dir] = {}
        self._by_key: dict[Any, list[_Dir]] = {}
        #: path -> the folder whose records hold its stat.
        self._home: dict[str, _Dir] = {}
        #: folder path -> its own watch, which also holds its stat.
        self._self: dict[str, _Dir] = {}
        #: Folders found unwatchable since the last check before a cell.
        self._refused: set[str] = set()
        self._full = False
        self._lock = threading.Lock()
        self.pid = os.getpid()
        #: How many times a folder's stats were dropped, ever.
        self.changes = 0
        #: Why stats were not kept, by reason (`report`).
        self.unkept: dict[str, int] = {}
        #: When the last drain began (``time.monotonic()``): a stat still
        #: current is as good as one taken then.
        self.confirmed_at = 0.0

    # -- events ------------------------------------------------------------

    def drain(self, check_dirs: bool = False) -> None:
        """Take every change reported so far, dropping the stats each one
        touches. With *check_dirs*, also stat each watched folder by its path:
        one that is gone or now names another folder (an ancestor renamed,
        a link re-pointed) is dropped, and folders refused earlier are
        tried again."""
        with self._lock:
            at = time.monotonic()
            try:
                backend = self._backend
                # A run's first drain, or the first after a statement wrote
                # files, holds every change made before it (`_Volume.sync`).
                reported = backend.poll(True) if check_dirs and getattr(backend, "syncs", False) else backend.poll()
            except Exception:  # any doubt: everything changed
                logger.debug("file watch: poll failed", exc_info=True)
                reported = [(None, True)]
            for key, dead in reported:
                if key is None:
                    for d in list(self._dirs.values()):
                        if dead:
                            self._kill(d)
                        else:
                            self._invalidate(d)
                    continue
                for d in list(self._by_key.get(key, ())):
                    if dead:
                        self._kill(d)
                    else:
                        self._invalidate(d)
            if check_dirs:
                self._refused.clear()
                for d in list(self._dirs.values()):
                    try:
                        same = _ident(own_stat(d.path)) == d.ident
                    except (OSError, ValueError):
                        same = False
                    if not same:
                        self._kill(d)
            self.confirmed_at = at

    def _invalidate(self, d: _Dir) -> None:
        if d.records or d.listed or d.missing:
            d.records = {}
            d.listed = set()
            d.missing = set()
        d.version += 1
        self.changes += 1

    def _kill(self, d: _Dir) -> None:
        self._invalidate(d)
        if not d.alive:
            return
        d.alive = False
        if self._dirs.get(d.path) is d:
            del self._dirs[d.path]
        sharing = self._by_key.get(d.key)
        if sharing is not None:
            if d in sharing:
                sharing.remove(d)
            if not sharing:
                del self._by_key[d.key]
                try:
                    self._backend.remove(d.key)
                except Exception:
                    logger.debug("file watch: remove failed", exc_info=True)

    # -- watching ----------------------------------------------------------

    def _arm(self, folder: str) -> _Dir | None:
        d = self._dirs.get(folder)
        if d is not None:
            return d
        if self._full or folder in self._refused:
            return None
        if len(self._dirs) >= DIR_LIMIT:
            self._full = True
            return None
        self._refused.add(folder)  # until armed below
        try:
            before = own_stat(folder)
            if not _stat.S_ISDIR(before.st_mode):
                return None
            key = self._backend.add(folder)
            if key is None:
                return None
            after = own_stat(folder)
        except _WatchesFull:
            self._full = True
            return None
        except Exception:  # unwatchable: stat-ed instead
            logger.debug("file watch: cannot watch %s", folder, exc_info=True)
            return None
        sharing = self._by_key.setdefault(key, [])
        if _ident(before) != _ident(after):
            # Replaced while the watch was armed: which one it watches is unknown.
            if not sharing:
                del self._by_key[key]
                try:
                    self._backend.remove(key)
                except Exception:  # noqa: BLE001 - dropped either way
                    pass
            return None
        self._refused.discard(folder)
        d = self._dirs[folder] = _Dir(folder, key, _ident(after))
        sharing.append(d)
        return d

    def before_stat(self, path: str) -> tuple[_Dir, int] | None:
        """Arm the watch on *path*'s folder, if it can be watched, before a
        stat of *path* is taken: the token `record` keeps that stat by."""
        if not os.path.isabs(path):
            return None
        folder = os.path.dirname(path)
        if not folder or folder == path:
            return None
        d = self._dirs.get(folder)
        if d is None:
            with self._lock:
                d = self._arm(folder)
            if d is None:
                return None
        return (d, d.version)

    def record(self, token: tuple[_Dir, int] | None, path: str, st: os.stat_result, listed: bool = False) -> None:
        """Keep *st*, the stat of *path* taken after `before_stat` gave
        *token*, for as long as its folder reports no change: only a regular
        file with no second link, or a folder whose own watch is armed and
        whose stat agrees with one taken after that."""
        if token is None:
            return
        d, version = token
        kind = st.st_mode & _S_IFMT
        if kind == _S_IFREG:
            if st.st_nlink > 1:
                self._unkept("a second hard link")
                return
        elif kind != _S_IFDIR:
            self._unkept("neither a file nor a folder")
            return
        with self._lock:
            if not d.alive or d.version != version:
                self._unkept("its folder changed while it was stat-ed")
                return
            if kind == _S_IFDIR:
                own = self._arm(path)
                if own is None:
                    return
                own_version = own.version
                try:
                    again = own_lstat(path)
                except (OSError, ValueError):
                    return
                if not _same(st, again) or own.version != own_version or d.version != version:
                    return
                own.records[path] = st
                self._self[path] = own
            d.records[path] = st
            d.missing.discard(path)
            self._home[path] = d
            if listed:
                d.listed.add(path)
            else:
                d.listed.discard(path)

    def record_absent(self, token: tuple[_Dir, int] | None, path: str) -> None:
        """Keep that *path* was not there (its ``lstat`` found no such file)
        after `before_stat` gave *token*, for as long as its folder reports
        no change: a name made there is a change of the folder."""
        if token is None:
            return
        d, version = token
        with self._lock:
            if not d.alive or d.version != version:
                return
            d.records.pop(path, None)
            d.listed.discard(path)
            d.missing.add(path)
            self._home[path] = d

    def _unkept(self, why: str) -> None:
        self.unkept[why] = self.unkept.get(why, 0) + 1

    def report(self) -> dict[str, Any]:
        """What the watch holds, and why it holds no more: for finding out
        on a machine why a cell still checks files one by one."""
        with self._lock:
            return {
                "folders watched": len(self._dirs),
                "folders that cannot be watched": len(self._refused),
                "some of those": sorted(self._refused)[:10],
                "watch limit reached": self._full,
                "stats kept": sum(len(d.records) for d in self._dirs.values()),
                "paths kept as not there": sum(len(d.missing) for d in self._dirs.values()),
                "folder changes taken": self.changes,
                "stats not kept, by why": dict(self.unkept),
                "reports": self._backend_report(),
            }

    def _backend_report(self) -> dict[str, int]:
        report = getattr(self._backend, "report", None)
        try:
            return dict(report()) if report is not None else {}
        except Exception:  # noqa: BLE001 - a diagnostic only
            return {}

    # -- answers -----------------------------------------------------------

    def absent(self, path: str) -> bool:
        """Was *path* found not there while its folder was watched, with no
        change reported since? As good as a look at the last `drain`."""
        d = self._home.get(path)
        return d is not None and path in d.missing

    def current(self, path: str) -> os.stat_result | None:
        """The stat of *path* kept while its folder was watched, if no change
        was reported since: as good as a stat taken at the last `drain`."""
        d = self._home.get(path)
        if d is None:
            return None
        st = d.records.get(path)
        if st is not None and st.st_mode & _S_IFMT == _S_IFDIR:
            own = self._self.get(path)
            if own is None or own.records.get(path) is not st:
                return None
        return st

    def is_listed(self, path: str) -> bool:
        """Was *path*'s kept stat read from a folder listing?"""
        d = self._home.get(path)
        return d is not None and path in d.listed

    def cover(self, paths: Iterable[str]) -> tuple | None:
        """A token that `still` accepts while no folder holding a stat of any
        of *paths*, or vouching one is not there, reported a change; None
        when one of them has neither. The watched folders of *paths* and their versions."""
        dirs: dict[int, tuple[_Dir, int]] = {}
        home = self._home
        own_of = self._self
        for p in paths:
            d = home.get(p)
            if d is None:
                return None
            st = d.records.get(p)
            if st is None:
                if p not in d.missing:
                    return None
                if id(d) not in dirs:
                    dirs[id(d)] = (d, d.version)
                continue
            if id(d) not in dirs:
                dirs[id(d)] = (d, d.version)
            if st.st_mode & _S_IFMT == _S_IFDIR:
                own = own_of.get(p)
                if own is None or own.records.get(p) is not st:
                    return None
                if id(own) not in dirs:
                    dirs[id(own)] = (own, own.version)
        return tuple(dirs.values())

    def cover_by_folder(self, paths: Mapping[str, str]) -> tuple[list[tuple[_Dir, int, list[str]]], list[str]]:
        """``(parts, loose)`` for *paths* (``{key: path}``): each watched
        folder vouching for some of them, with its version and their keys,
        and the keys of those none vouches for. A change in one folder then
        asks again only of the files in it (`changed_in`), where a `cover`
        lapsed whole: a snapshot of 8,000 files was checked file by file
        before every cell because one of its folders (the notebook's own,
        written by every save) reported a change, or one file could not be
        vouched for."""
        parts: dict[int, tuple[_Dir, int, list[str]]] = {}
        loose: list[str] = []
        home = self._home
        own_of = self._self
        for key, p in paths.items():
            d = home.get(p)
            st = d.records.get(p) if d is not None else None
            if st is None:
                if d is None or p not in d.missing:
                    loose.append(key)
                    continue
            elif st.st_mode & _S_IFMT == _S_IFDIR:
                own = own_of.get(p)
                if own is None or own.records.get(p) is not st:
                    loose.append(key)
                    continue
                part = parts.get(id(own))
                if part is None:
                    part = parts[id(own)] = (own, own.version, [])
                part[2].append(key)
            part = parts.get(id(d))
            if part is None:
                part = parts[id(d)] = (d, d.version, [])
            part[2].append(key)
        return list(parts.values()), loose

    @staticmethod
    def changed_in(parts: list[tuple[_Dir, int, list[str]]]) -> list[str]:
        """The keys of `cover_by_folder`'s *parts* whose folder reported a
        change since (or whose watch ended)."""
        changed: list[str] = []
        for d, version, keys in parts:
            if not d.alive or d.version != version:
                changed.extend(keys)
        return changed

    @staticmethod
    def still(cover: tuple) -> bool:
        """Did no folder of *cover* (`cover`) report a change since?"""
        return all(d.alive and d.version == version for d, version in cover)

    def watched(self) -> int:
        """How many folders are watched now."""
        return len(self._dirs)


class _WatchesFull(Exception):
    """The operating system's limit on watches is reached."""


# -- Linux: inotify -----------------------------------------------------------

_IN_MODIFY = 0x2
_IN_ATTRIB = 0x4
_IN_CLOSE_WRITE = 0x8
_IN_MOVED_FROM = 0x40
_IN_MOVED_TO = 0x80
_IN_CREATE = 0x100
_IN_DELETE = 0x200
_IN_DELETE_SELF = 0x400
_IN_MOVE_SELF = 0x800
_IN_UNMOUNT = 0x2000
_IN_Q_OVERFLOW = 0x4000
_IN_IGNORED = 0x8000
_IN_ONLYDIR = 0x01000000
_IN_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_ONLYDIR
)
_IN_GONE = _IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT | _IN_IGNORED
_EVENT = struct.Struct("iIII")

#: Local file systems whose changes inotify reports, by ``statfs`` magic:
#: ext2/3/4, xfs, btrfs, tmpfs, ramfs, f2fs, zfs, overlayfs, jfs, reiserfs,
#: bcachefs, ntfs3, vfat, exfat. Anything else (nfs, cifs, fuse, 9p, ...)
#: may change without a local event, and is stat-ed instead.
LOCAL_FILE_SYSTEMS = frozenset(
    {
        0xEF53,
        0x58465342,
        0x9123683E,
        0x01021994,
        0x858458F6,
        0xF2F52010,
        0x2FC12FC1,
        0x794C7630,
        0x3153464A,
        0x52654973,
        0xCA451A4E,
        0x7366746E,
        0x4D44,
        0x2011BAB0,
    }
)


class _Inotify:
    """One inotify instance, read without blocking."""

    def __init__(self) -> None:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        self._ctypes = ctypes
        self._add = libc.inotify_add_watch
        self._add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._add.restype = ctypes.c_int
        self._rm = libc.inotify_rm_watch
        self._rm.argtypes = [ctypes.c_int, ctypes.c_int]
        self._rm.restype = ctypes.c_int
        self._statfs = libc.statfs
        self._statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self._statfs.restype = ctypes.c_int
        init1 = libc.inotify_init1
        init1.argtypes = [ctypes.c_int]
        init1.restype = ctypes.c_int
        fd = init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.fd = fd
        self._buffer = ctypes.create_string_buffer(512)

    def _local(self, path: str) -> bool:
        if self._statfs(os.fsencode(path), self._buffer) != 0:
            return False
        # ``f_type`` is the struct's first field, a native long.
        f_type = self._ctypes.c_long.from_buffer(self._buffer).value & 0xFFFFFFFF
        return f_type in LOCAL_FILE_SYSTEMS

    def add(self, path: str) -> int | None:
        """The watch descriptor of *path*, None when it is not on a local
        file system or cannot be watched."""
        if not self._local(path):
            return None
        wd = self._add(self.fd, os.fsencode(path), _IN_MASK)
        if wd < 0:
            import errno

            if self._ctypes.get_errno() == errno.ENOSPC:
                raise _WatchesFull
            return None
        return wd

    def remove(self, wd: int) -> None:
        self._rm(self.fd, wd)

    def close(self) -> None:
        os.close(self.fd)

    def poll(self) -> list[tuple[int | None, bool]]:
        reported: list[tuple[int | None, bool]] = []
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return reported
            if not data:
                return reported
            reported.extend(parse_inotify_events(data))


def parse_inotify_events(data: bytes) -> list[tuple[int | None, bool]]:
    """``[(wd, dead)]`` for a buffer of inotify events; ``(None, False)``
    for an overflow (events were lost)."""
    out: list[tuple[int | None, bool]] = []
    offset = 0
    size = _EVENT.size
    while offset + size <= len(data):
        wd, mask, _cookie, length = _EVENT.unpack_from(data, offset)
        if offset + size + length > len(data):
            break
        offset += size + length
        if mask & _IN_Q_OVERFLOW or wd < 0:
            out.append((None, False))
        else:
            out.append((wd, bool(mask & _IN_GONE)))
    if offset != len(data):
        out.append((None, False))  # a torn read: trust nothing
    return out


# -- Windows: one ReadDirectoryChangesW per volume ---------------------------------
#
# A handle open on a folder, or on any folder below it, makes Windows refuse
# to rename or move that folder: a watch per folder blocked a user renaming
# the project folder in Explorer while a kernel ran. So each volume is
# watched once, from its root, over the whole tree (``bWatchSubtree``), and
# the names it reports say which watched folders changed. No handle is held
# on any folder a user can rename. A thread per volume takes the reports as
# they come, so a busy system drive rarely fills the buffer; a full buffer,
# a report that cannot be read, or a name that may be a short (8.3) alias
# counts as a change of every watched folder it could name.

_FILE_LIST_DIRECTORY = 0x0001
_FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OVERLAPPED = 0x40000000
_ERROR_IO_INCOMPLETE = 996
_ERROR_NOTIFY_ENUM_DIR = 1022
_WAIT_OBJECT_0 = 0
#: How long stopping a volume's watch waits for its cancelled read to end.
_CANCEL_WAIT_SECONDS = 1.0
#: How long a poll waits for the volume's thread to pass through a wait (`_Volume.sync`).
_SYNC_SECONDS = 1.0
#: How long the volume's thread waits for a report before looking at its stop flag.
_THREAD_WAIT_MS = 250
#: How many short names a volume keeps the long forms of (`_Volume._resolve`).
_LONGS_KEPT = 4096
#: The report buffer of a volume, in bytes.
_VOLUME_BUFFER = 256 * 1024
#: FILE_NOTIFY_CHANGE_ FILE_NAME | DIR_NAME | ATTRIBUTES | SIZE | LAST_WRITE |
#: CREATION | SECURITY.
_NOTIFY_FILTER = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x40 | 0x100
_DRIVE_FIXED = 3
_DRIVE_RAMDISK = 6
_NOTIFY_HEADER = struct.Struct("<III")
#: FILE_ACTION_REMOVED and FILE_ACTION_RENAMED_OLD_NAME: the name is gone.
_ACTIONS_GONE = frozenset((2, 4))
#: FILE_ACTION_MODIFIED: what is at the name changed in place (`_FolderIndex.touched`).
_ACTION_MODIFIED = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
#: A folder name whose real long path is its parent's and this name: none
#: that may be a short alias (any ``~``), that Windows trims (a trailing dot
#: or space), or that is not one plain component.
_PLAIN_NAME = re.compile(r"[^~:/\\]*[^~:/\\. ]")

#: A name a change report gives that may be a short (8.3) alias: a ``~``
#: and a digit, as in ``PROGRA~1``.
_SHORT_NAME = re.compile(r"~\d")

#: Every watched folder of a volume (`_FolderIndex.touched`).
EVERY_FOLDER = None


def parse_notify_information(data: bytes) -> list[tuple[int, str]] | None:
    """The ``(action, name)`` (a name below the watched root) of each record
    in a buffer of ``FILE_NOTIFY_INFORMATION`` records; None when it cannot
    be read whole."""
    names: list[tuple[int, str]] = []
    offset = 0
    size = _NOTIFY_HEADER.size
    while True:
        if offset + size > len(data):
            return None
        following, action, length = _NOTIFY_HEADER.unpack_from(data, offset)
        start = offset + size
        if length % 2 or start + length > len(data):
            return None
        try:
            names.append((action, data[start : start + length].decode("utf-16-le")))
        except UnicodeDecodeError:
            return None
        if following == 0:
            return names
        if following < size or following % 4:
            return None
        offset += following


class _FolderIndex:
    """The watched folders of one volume, by their path below its root
    (case folded, ``\\``-separated, no leading separator), and which of them
    a change at a reported name touches (`touched`)."""

    def __init__(self) -> None:
        self.keys: dict[str, Any] = {}
        #: Every folder above a watched one, with how many watched ones it holds.
        self.above: dict[str, int] = {}

    @staticmethod
    def spelled(name: str) -> str:
        return name.replace("/", "\\").strip("\\").lower()

    def add(self, below_root: str, key: Any) -> None:
        below_root = self.spelled(below_root)
        if below_root in self.keys:
            return
        self.keys[below_root] = key
        for above in self._parents(below_root):
            self.above[above] = self.above.get(above, 0) + 1

    def remove(self, below_root: str) -> None:
        below_root = self.spelled(below_root)
        if self.keys.pop(below_root, None) is None:
            return
        for above in self._parents(below_root):
            left = self.above[above] - 1
            if left:
                self.above[above] = left
            else:
                del self.above[above]

    @staticmethod
    def _parents(path: str) -> list[str]:
        parts = path.split("\\") if path else []
        return ["\\".join(parts[:i]) for i in range(len(parts))]

    def under(self, prefix: str) -> list[Any]:
        """The watched folders at or below *prefix* ("" is the root)."""
        if not prefix:
            return list(self.keys.values())
        lead = prefix + "\\"
        return [key for path, key in self.keys.items() if path == prefix or path.startswith(lead)]

    def _long_parts(self, name: str, resolve: Any) -> tuple[list[str], int | None]:
        """*name*'s components, case folded, with each that may be a short
        alias put in its long form by *resolve* (the long name below the
        root of a name below it, or None); and the index of the first one
        still left as it was reported (None when there is none)."""
        parts = self.spelled(name).split("\\")
        i = 0
        while i < len(parts):
            if not _SHORT_NAME.search(parts[i]):
                i += 1
                continue
            long = resolve("\\".join(parts[: i + 1]), i == len(parts) - 1) if resolve is not None else None
            if long is None:
                return parts, i
            # The long form of the components up to here, not looked at again:
            # a long name may hold a ``~1`` of its own.
            lead = self.spelled(long).split("\\")
            parts = lead + parts[i + 1 :]
            i = len(lead)
        return parts, None

    def moved(self, name: str, resolve: Any = None) -> list[Any]:
        """The watched folders no longer where they were if *name* was
        removed or renamed: those at or below it, or, when a component of
        it may be a short alias that *resolve* cannot put in its long form,
        everything below the folder above that component."""
        parts, cut = self._long_parts(name, resolve)
        if cut is not None:
            return self.under("\\".join(parts[:cut]))
        name = "\\".join(parts)
        if name in self.keys or name in self.above:
            return self.under(name)
        return []

    def touched(self, name: str, resolve: Any = None, in_place: bool = False) -> list[Any]:
        """The watched folders a change at *name* may have changed: the
        folder holding it, and, when it is a watched folder or holds one,
        every watched folder at or below it (a folder renamed, removed or
        replaced). A name with a component that may be a short alias, and
        that *resolve* cannot put in its long form, stands for anything
        below the folder above that component.

        A change *in_place* (``FILE_ACTION_MODIFIED``) is one of the folder
        holding the name only: a folder modified in place changed its own
        times, attributes or rights, not a file below it, and it is still
        where it was. Windows reports a folder modified when its parent's
        entry for it is brought up to date, which may be only when the
        folder is next opened: one file written was reported again, by the
        listing the next check took of its folder, as a change of that
        folder and each one below it, and all their files were checked once
        more before the cell after. A folder that now leads elsewhere (made
        a junction or a mount point) is found by the stat of each watched
        folder before each cell (`FileWatch.drain`)."""
        parts, cut = self._long_parts(name, resolve)
        if cut is not None:
            return self.under("\\".join(parts[:cut]))
        name = "\\".join(parts)
        touched = []
        holder = self.keys.get("\\".join(parts[:-1]))
        if holder is not None:
            touched.append(holder)
        if in_place:
            return touched
        if name in self.keys or name in self.above:
            touched.extend(self.under(name))
        return touched


class _Win32:
    """The kernel32 calls a volume's watch makes, through ctypes."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_void_p),
                ("InternalHigh", ctypes.c_void_p),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ctypes = ctypes
        self._wintypes = wintypes
        self.Overlapped = Overlapped

        def fn(name, restype, *argtypes):
            f = getattr(k, name)
            f.argtypes = list(argtypes)
            f.restype = restype
            return f

        H, D, B, P = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.c_void_p
        self._create = fn("CreateFileW", H, wintypes.LPCWSTR, D, D, P, D, D, H)
        self._read = fn("ReadDirectoryChangesW", B, H, P, D, B, D, P, P, P)
        self._result = fn("GetOverlappedResult", B, H, P, ctypes.POINTER(D), B)
        self._cancel = fn("CancelIoEx", B, H, P)
        self._close = fn("CloseHandle", B, H)
        self._event = fn("CreateEventW", H, P, B, B, wintypes.LPCWSTR)
        self._wait = fn("WaitForSingleObject", D, H, D)
        self._wait_many = fn("WaitForMultipleObjects", D, D, P, B, D)
        self._set = fn("SetEvent", B, H)
        self._reset = fn("ResetEvent", B, H)
        self._drive_type = fn("GetDriveTypeW", wintypes.UINT, wintypes.LPCWSTR)
        self._volume_path = fn("GetVolumePathNameW", B, wintypes.LPCWSTR, wintypes.LPWSTR, D)
        self._long_path = fn("GetLongPathNameW", D, wintypes.LPCWSTR, wintypes.LPWSTR, D)
        self._attributes = fn("GetFileAttributesW", D, wintypes.LPCWSTR)
        self._invalid = wintypes.HANDLE(-1).value

    def real_path(self, path: str) -> str:
        return os.path.realpath(path)

    def attributes(self, path: str) -> int | None:
        """*path*'s attributes, of the entry itself (a junction's, not its
        target's); None when it cannot be read."""
        found = self._attributes(path)
        return None if found == _INVALID_FILE_ATTRIBUTES else found

    def volume_root(self, path: str) -> str | None:
        buffer = self._ctypes.create_unicode_buffer(1024)
        if not self._volume_path(path, buffer, 1024):
            return None
        return buffer.value

    def long_path(self, path: str) -> str | None:
        buffer = self._ctypes.create_unicode_buffer(32768)
        n = self._long_path(path, buffer, 32768)
        return buffer.value if 0 < n < 32768 else None

    def local(self, root: str) -> bool:
        return self._drive_type(root) in (_DRIVE_FIXED, _DRIVE_RAMDISK)

    def open_root(self, root: str) -> Any:
        handle = self._create(
            root,
            _FILE_LIST_DIRECTORY,
            _FILE_SHARE_ALL,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OVERLAPPED,
            None,
        )
        return None if handle is None or handle == self._invalid else handle

    def new_event(self) -> Any:
        return self._event(None, True, False, None)  # manual reset, not set

    def new_read(self) -> tuple[Any, Any]:
        return self.Overlapped(), (self._ctypes.c_uint32 * (_VOLUME_BUFFER // 4))()

    def issue(self, handle: Any, overlapped: Any, buffer: Any, event: Any) -> bool:
        ctypes = self._ctypes
        ctypes.memset(ctypes.byref(overlapped), 0, ctypes.sizeof(overlapped))
        overlapped.hEvent = event
        self._reset(event)  # set again only when this read ends
        return bool(
            self._read(
                handle,
                ctypes.byref(buffer),
                ctypes.sizeof(buffer),
                True,
                _NOTIFY_FILTER,
                None,
                ctypes.byref(overlapped),
                None,
            )
        )

    def result(self, handle: Any, overlapped: Any) -> tuple[bool, int, int]:
        """``(ended, bytes, error)`` of the read, asked without waiting."""
        moved = self._wintypes.DWORD()
        ctypes = self._ctypes
        if self._result(handle, ctypes.byref(overlapped), ctypes.byref(moved), False):
            return True, moved.value, 0
        error = ctypes.get_last_error()
        return error != _ERROR_IO_INCOMPLETE, 0, error

    def data(self, buffer: Any, n: int) -> bytes:
        return self._ctypes.string_at(self._ctypes.addressof(buffer), n)

    def wait(self, event: Any, ms: int) -> bool:
        return self._wait(event, ms) == _WAIT_OBJECT_0

    def new_poke(self) -> Any:
        return self._event(None, False, False, None)  # auto reset, not set

    def poke(self, event: Any) -> None:
        self._set(event)

    def wait_either(self, event: Any, poke: Any, ms: int) -> int | None:
        """0 when *event* is set, 1 when *poke* is, None when *ms* passed."""
        handles = (self._wintypes.HANDLE * 2)(event, poke)
        found = self._wait_many(2, handles, False, ms)
        return found - _WAIT_OBJECT_0 if found in (_WAIT_OBJECT_0, _WAIT_OBJECT_0 + 1) else None

    def cancel(self, handle: Any, overlapped: Any) -> None:
        self._cancel(handle, self._ctypes.byref(overlapped))

    def close(self, handle: Any) -> None:
        self._close(handle)


class _Volume:
    """One volume's watch: a handle on its root, one read pending over the
    whole tree, and the watched folders below it (`_FolderIndex`). Reports
    are taken (`take`) by its thread as they come and by every poll, under
    one lock: a poll sees every change reported before it."""

    def __init__(self, kernel: Any, root: str, handle: Any) -> None:
        self.kernel = kernel
        self.root = root
        self.handle = handle
        self.index = _FolderIndex()
        #: Folders whose real long path `_VolumeChanges` derives its
        #: children's from, kept only while no report says one of them or a
        #: folder above it may have been removed or renamed.
        self.located = _FolderIndex()
        self.lock = threading.Lock()
        self.event = kernel.new_event()
        self.overlapped, self.buffer = kernel.new_read()
        #: Keys of folders changed since the last poll; `everything` when all may have.
        self.changed: set[Any] = set()
        #: Keys of folders removed or renamed (or below one that was) since the last poll.
        self.gone: set[Any] = set()
        self.everything = False
        #: A folder's name below the root ending in a short (8.3) alias -> its
        #: long form (`_resolve`), both case folded; dropped by a report that
        #: it, or a folder above it, went, which may free the alias.
        self.longs: dict[str, str] = {}
        #: What the reports were (`FileWatch.report`).
        self.counts: dict[str, int] = {}
        self.pending = False
        self.dead = False
        self.stopping = False
        #: Reads that did not end within the wait after a cancel, kept alive.
        self.kept: list[tuple[Any, Any]] = []
        self.thread: threading.Thread | None = None
        #: `sync`: how many polls asked the thread to pass through a wait,
        #: and how many of those it did; None when the kernel cannot be poked.
        self.poke = kernel.new_poke() if hasattr(kernel, "wait_either") else None
        self.asked = 0
        self.answered = 0
        self.passed = threading.Condition(threading.Lock())

    def start(self) -> bool:
        """Issue the first read: changes from now on are reported."""
        with self.lock:
            self.pending = self.kernel.issue(self.handle, self.overlapped, self.buffer, self.event)
            if not self.pending:
                return False
        self.thread = threading.Thread(target=self._run, name="cash-file-watch", daemon=True)
        self.thread.start()
        return True

    def _run(self) -> None:
        try:
            while not self.stopping:
                seen = self.asked  # before the wait: see `sync`
                # A poll asked after the last wait began woke it, and that
                # wait can answer only what was asked before it: the answer
                # the poll waits for is this wait's, so it must not sit out
                # the whole timeout. A wait of no time still enters the
                # kernel after the ask, which is all `sync` needs.
                ms = 0 if seen != self.answered and self.answered >= 0 else _THREAD_WAIT_MS
                try:
                    if self.poke is not None:
                        which = self.kernel.wait_either(self.event, self.poke, ms)
                    else:
                        which = 0 if self.kernel.wait(self.event, ms) else None
                except Exception:  # noqa: BLE001 - the polls still take every report
                    return
                if self.stopping:
                    return
                with self.lock:
                    took = self.take()
                if self.asked != self.answered:
                    with self.passed:
                        self.answered = max(self.answered, seen)
                        self.passed.notify_all()
                if which == 0 and not took:
                    time.sleep(0.01)  # signalled with nothing to take: do not spin
                if self.dead:
                    return
        finally:
            with self.passed:
                self.answered = -1  # no more answers: `sync` waits for none
                self.passed.notify_all()

    def sync(self) -> bool:
        """Have the thread pass through a wait, so that every report of a
        change made before this call is taken by the next `poll`; False when
        it did not within ``_SYNC_SECONDS`` (or cannot be asked).

        The read a report ends is the thread's: Windows ends it by an APC
        queued to the thread that issued it, which runs when that thread
        next runs in the kernel. A file written and closed by the cell
        itself, its report already made, could so still show as not ended
        to a poll right after: the cell would take the file's kept stat,
        from before the write. The thread reads `asked` before each wait
        and passes the value on after it; an APC queued before the poll
        asked runs before that wait returns."""
        thread = self.thread
        if self.poke is None or thread is None or thread is threading.current_thread() or self.dead:
            return thread is threading.current_thread()
        with self.passed:
            if self.answered < 0:
                return False
            self.asked += 1
            mine = self.asked
        try:
            self.kernel.poke(self.poke)
        except Exception:  # noqa: BLE001 - not synced: taken as anything changed
            return False
        with self.passed:
            return self.passed.wait_for(lambda: self.answered >= mine or self.answered < 0, _SYNC_SECONDS) and (
                self.answered >= mine
            )

    def take(self) -> bool:
        """With the lock held: take the read's report if it ended, and issue
        the next. False when no read had ended."""
        if not self.pending or self.dead:
            return False
        ended, n, error = self.kernel.result(self.handle, self.overlapped)
        if not ended:
            return False
        self.pending = False
        records = None if error or n == 0 else parse_notify_information(self.kernel.data(self.buffer, n))
        if records is None:
            # An overflow (none read), an error, or a report not read whole:
            # anything changed, and any folder may have moved.
            self._count("whole-volume reports (an overflow, or one not read)")
            self.everything = True
            self.located = _FolderIndex()
            self.longs = {}
        else:
            self._count("names reported", len(records))
            touched = self.index.touched
            resolve = self._resolve
            for action, name in records:
                if not self.everything:
                    self.changed.update(touched(name, resolve, action == _ACTION_MODIFIED))
                if action in _ACTIONS_GONE:
                    self._went(name)
        if error and error != _ERROR_NOTIFY_ENUM_DIR:
            self.dead = True  # the root went (a volume unmounted): stop
            return True
        self.pending = self.kernel.issue(self.handle, self.overlapped, self.buffer, self.event)
        if not self.pending:
            self.dead = True
        return True

    def _went(self, name: str) -> None:
        """*name* was removed or renamed: a watched folder at or below it is
        no longer where its key says, and a path derived through it may be
        wrong."""
        resolve = self._resolve
        if self.located.keys and self.located.moved(name, resolve):
            self.located = _FolderIndex()
        if self.index.keys:
            self.gone.update(self.index.moved(name, resolve))
        if self.longs:
            # An alias freed for another entry is one of a name gone, or below it.
            gone = _FolderIndex.spelled(name)
            below = gone + "\\"
            self.longs = {
                short: long
                for short, long in self.longs.items()
                if not (short == gone or short.startswith(below) or long == gone or long.startswith(below))
            }

    def _count(self, what: str, n: int = 1) -> None:
        self.counts[what] = self.counts.get(what, 0) + n

    def _resolve(self, below: str, last: bool = False) -> str | None:
        """With the lock held: the long form of *below*, a name below the
        root ending in a component that may be a short alias, while it is
        there; None when it is not, or cannot be read. Kept for a folder
        above the name reported (not for the *last* component, most often a
        file come and gone), until a name at or above it is reported gone.

        A change is reported by the name it was made through, and a program
        writing through a short path (``C:\\Users\\PHILIP~1\\AppData\\Local\\Temp``,
        as ``%TEMP%`` often is) names its files so. Such a name counted as a
        change of every watched folder below the one above the alias: each
        temporary file any program wrote dropped every stat kept below
        ``C:\\Users``, so a cell checked every file again."""
        if not last and below in self.longs:
            found = self.longs[below]
        else:
            found = None
            long_path = getattr(self.kernel, "long_path", None)
            root = self.root.rstrip("\\")
            if long_path is not None:
                try:
                    long = long_path(root + "\\" + below)
                except Exception:  # noqa: BLE001 - taken as not resolved
                    long = None
                if long is not None and ntpath.normcase(long).startswith(ntpath.normcase(root) + "\\"):
                    found = _FolderIndex.spelled(long[len(root) + 1 :])
            if found is not None and not last:
                if len(self.longs) >= _LONGS_KEPT:
                    self.longs = {}
                self.longs[below] = found
        self._count("short names put in their long form" if found is not None else "short names not found")
        return found

    def poll(self, sync: bool = False) -> tuple[list[Any], list[Any], bool]:
        """``(changed keys, gone keys, dead)`` since the last poll: a gone
        folder was removed or renamed, or one above it was. With *sync*,
        every change made before the poll is in it (`sync`); when that
        cannot be made sure of, every folder counts as changed."""
        synced = self.sync() if sync else True
        with self.lock:
            if not synced and self.poke is not None:
                self._count("polls the thread did not answer")
                self.everything = True
            self.take()
            if self.everything or self.dead:
                keys = list(self.index.keys.values())
            else:
                keys = list(self.changed)
            gone = list(self.gone)
            self.changed = set()
            self.gone = set()
            self.everything = False
            return keys, gone, self.dead

    def stop(self) -> None:
        """End the watch, never waiting for good: a cancelled read is waited
        on for at most ``_CANCEL_WAIT_SECONDS``, and one still running then
        keeps its buffer alive for the process's life."""
        self.stopping = True
        if self.poke is not None:
            try:
                self.kernel.poke(self.poke)  # out of its wait now
            except Exception:  # noqa: BLE001 - it ends at the wait's end
                pass
        with self.lock:
            kernel = self.kernel
            if self.pending:
                kernel.cancel(self.handle, self.overlapped)
                deadline = time.monotonic() + _CANCEL_WAIT_SECONDS
                while True:
                    ended, _n, _error = kernel.result(self.handle, self.overlapped)
                    if ended:
                        break
                    if time.monotonic() >= deadline:
                        self.kept.append((self.overlapped, self.buffer))
                        break
                    time.sleep(0.001)
                self.pending = False
            kernel.close(self.handle)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2 * _THREAD_WAIT_MS / 1000)
        kernel.close(self.event)
        if self.poke is not None:
            kernel.close(self.poke)


class _VolumeChanges:
    """The Windows backend of `FileWatch`: a folder is watched through its
    volume's watch (`_Volume`), keyed by its real long path, so no handle is
    open on it or on any folder above it but the volume's root.

    Finding a folder's real long path and volume costs milliseconds
    (``realpath``, ``GetLongPathNameW``, ``GetVolumePathNameW``), and a run
    watches thousands of folders, most of them installed packages'. So a
    folder whose name is plain (`_PLAIN_NAME`) and that is no reparse point
    (no junction, link or mounted volume) takes its parent's place and its
    own name: one ``GetFileAttributesW``. A parent's place is found once and
    kept (`_located`) while its volume's watch runs and reports no removal
    or rename of it or of a folder above it (`_Volume.located`); one found
    before the volume was watched is not kept."""

    def __init__(self, kernel: Any = None) -> None:
        self.kernel = kernel if kernel is not None else _Win32()
        self._volumes: dict[str, _Volume] = {}
        self._keys: set[tuple[str, str]] = set()
        #: Volumes kept alive after a stop that did not end in time.
        self._stopped: list[_Volume] = []
        #: A folder's path, case folded, that is its real long path -> its
        #: volume's watch and its path below the root.
        self._located: dict[str, tuple[_Volume, str]] = {}

    def _locate(self, path: str) -> tuple[str, str] | None:
        """``(volume root, path below it)`` of *path*'s real, long form;
        None for a network or removable drive, or a path not understood.
        By the real path: a change is reported under it, not under a
        junction or a link that leads there."""
        real = self.kernel.real_path(path)
        if real.startswith("\\\\?\\") and not real.upper().startswith("\\\\?\\UNC\\"):
            real = real[4:]
        if real.startswith(("\\\\", "//")):
            return None  # a network share
        long = self.kernel.long_path(real)
        if long is None:
            return None
        root = self.kernel.volume_root(long)
        if not root or root.startswith("\\\\") or not self.kernel.local(root):
            return None
        if not ntpath.normcase(long).startswith(ntpath.normcase(root.rstrip("\\"))):
            return None
        return root, long[len(root.rstrip("\\")) :]

    def _known(self, path: str) -> tuple[_Volume, str] | None:
        """*path*'s kept place, if its volume's watch still vouches for it."""
        found = self._located.get(ntpath.normcase(path))
        if found is None:
            return None
        volume, below = found
        if volume.dead or self._volumes.get(ntpath.normcase(volume.root)) is not volume:
            del self._located[ntpath.normcase(path)]
            return None
        if _FolderIndex.spelled(below) not in volume.located.keys:
            return None
        return found

    def _place(self, path: str, live: dict[str, tuple[_Volume, _FolderIndex]]) -> tuple[str, str, Any] | None:
        """``(volume root, path below it, kept places it was derived from)``
        of the folder *path*; the last is None when any of it was found
        afresh. A plain-named folder that is no reparse point is its
        parent's place and its name, and so on up to a kept place; the
        topmost folder that is not is looked up."""
        known = self._known(path)
        if known is not None:
            return known[0].root, known[1], known[0].located
        parent, name = ntpath.split(path)
        if parent and parent != path and _PLAIN_NAME.fullmatch(name):
            attributes = self.kernel.attributes(path)
            if (
                attributes is None
                or not attributes & _FILE_ATTRIBUTE_DIRECTORY
                or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                return self._found(path, live)
            above = self._place(parent, live)
            if above is None:
                return None
            root, parent_below, basis = above
            below = parent_below.rstrip("\\") + "\\" + name
            self._keep(path, root, below, live, basis)
            return root, below, basis
        return self._found(path, live)

    def _found(self, path: str, live: dict[str, tuple[_Volume, _FolderIndex]]) -> tuple[str, str, Any] | None:
        located = self._locate(path)
        if located is None:
            return None
        self._keep(path, *located, live, None)
        return (*located, None)

    def _live(self) -> dict[str, tuple[_Volume, _FolderIndex]]:
        """Each watched volume, with its kept places as they are now."""
        return {root: (volume, volume.located) for root, volume in self._volumes.items() if not volume.dead}

    def _keep(
        self,
        path: str,
        root: str,
        below: str,
        live: dict[str, tuple[_Volume, _FolderIndex]],
        basis: Any,
    ) -> None:
        """Keep *path*'s place when *path* is its real long path, its volume
        was watched before the place was found, and no report since said a
        kept place moved: any removal or rename after that is reported."""
        was = live.get(ntpath.normcase(root))
        if was is None:
            return
        volume, located = was
        if volume.dead or self._volumes.get(ntpath.normcase(root)) is not volume:
            return
        if ntpath.normcase(path).rstrip("\\") != ntpath.normcase(root.rstrip("\\") + below).rstrip("\\"):
            return  # reached through a junction, a link or a short alias
        with volume.lock:
            if volume.located is not located or (basis is not None and basis is not located):
                return
            located.add(below, True)
        self._located[ntpath.normcase(path)] = (volume, below)

    def add(self, path: str) -> Any:
        """The key *path* is watched by; None when it cannot be watched."""
        live = self._live()
        placed = self._place(path, live)
        while True:
            if placed is None:
                return None
            root, below, basis = placed
            key = (ntpath.normcase(root), _FolderIndex.spelled(below))
            volume = self._volumes.get(key[0])
            if volume is None or volume.dead:
                handle = self.kernel.open_root(root)
                if handle is None:
                    return None
                volume = _Volume(self.kernel, root, handle)
                if not volume.start():
                    volume.stop()
                    return None
                self._volumes[key[0]] = volume
            with volume.lock:
                # A place derived from kept ones holds only while none of
                # them was reported moved, up to the moment it is watched.
                if basis is None or volume.located is basis:
                    volume.index.add(below, key)
                    break
            located = self._locate(path)
            placed = None if located is None else (*located, None)
        self._keys.add(key)
        return key

    def remove(self, key: Any) -> None:
        if key not in self._keys:
            return
        self._keys.discard(key)
        volume = self._volumes.get(key[0])
        if volume is None:
            return
        with volume.lock:
            volume.index.remove(key[1])
            empty = not volume.index.keys
        if empty:
            self._drop(key[0], volume)

    def _drop(self, root: str, volume: _Volume) -> None:
        del self._volumes[root]
        volume.stop()
        if volume.kept:
            self._stopped.append(volume)
        self._located = {path: place for path, place in self._located.items() if place[0] is not volume}

    #: `poll` takes *sync* (`FileWatch.drain`).
    syncs = True

    def poll(self, sync: bool = False) -> list[tuple[Any, bool]]:
        reported: list[tuple[Any, bool]] = []
        for root, volume in list(self._volumes.items()):
            keys, gone, dead = volume.poll(sync)
            if dead:
                reported.extend((key, True) for key in keys)
                self._keys.difference_update(keys)
                self._drop(root, volume)
                continue
            reported.extend((key, True) for key in gone)
            reported.extend((key, False) for key in keys)
        return reported

    def report(self) -> dict[str, int]:
        """What the volumes' reports were, summed (`FileWatch.report`)."""
        counts: dict[str, int] = {"volumes watched": len(self._volumes)}
        for volume in list(self._volumes.values()):
            with volume.lock:
                for what, n in volume.counts.items():
                    counts[what] = counts.get(what, 0) + n
        return counts

    def close(self) -> None:
        for volume in list(self._volumes.values()):
            volume.stop()
        self._volumes.clear()
        self._keys.clear()
        self._located.clear()


# -- the process's watch ---------------------------------------------------------

_WATCH: FileWatch | None = None
_TRIED_IN: int | None = None


def _switched_off() -> bool:
    return os.environ.get("CASH_FILE_WATCH", "").strip().lower() in ("0", "false", "no", "off")


def _backend() -> Any:
    if sys.platform.startswith("linux"):
        return _Inotify()
    if sys.platform == "win32":
        return _VolumeChanges()
    return None  # macOS and others: stat-ed every time


def watcher() -> FileWatch | None:
    """The process's `FileWatch`, started on first use; None where folders
    cannot be watched here (or ``CASH_FILE_WATCH=0``). A forked child starts
    its own: the parent's events are the parent's to read."""
    global _WATCH, _TRIED_IN
    pid = os.getpid()
    if _WATCH is not None and _WATCH.pid == pid:
        return _WATCH
    if _TRIED_IN == pid:
        return None
    _TRIED_IN = pid
    _WATCH = None
    if _switched_off():
        return None
    try:
        backend = _backend()
    except Exception:  # no watch: stat-ed every time
        logger.debug("file watch unavailable", exc_info=True)
        backend = None
    if backend is not None:
        _WATCH = FileWatch(backend)
    return _WATCH


def reset() -> None:
    """Close the process's watch, so the next `watcher` starts another
    (and reads ``CASH_FILE_WATCH`` again)."""
    global _WATCH, _TRIED_IN
    old = _WATCH
    _WATCH = None
    _TRIED_IN = None
    if old is not None:
        _close(old)


def _close(watch: FileWatch) -> None:
    # Every folder's watch ends: a verdict held by one of them (`FileWatch.still`,
    # `FileWatch.changed_in`) lapses, as no change of it is reported any more.
    with watch._lock:
        for d in list(watch._dirs.values()):
            d.alive = False
            d.version += 1
        watch._dirs.clear()
    backend = watch._backend
    try:
        for key in list(watch._by_key):
            backend.remove(key)
        backend.close()
    except Exception:  # noqa: BLE001 - closing a watch no longer used
        pass
