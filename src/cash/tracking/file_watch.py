"""Folder change notifications: a file's stat holds until its folder reports a change.

Before every cell the notebook checks each file the cached statements above
read, and a check is a stat per file: 19,000 photos in 300 folders cost
seconds a cell, on Windows most of all, where a stat opens the file. Most
cells change no file. So cash asks the operating system to report changes
to each folder it took a stat in (inotify on Linux, ReadDirectoryChangesW on
Windows), and a stat taken while its folder was watched stands for the file
(`FileWatch.current`) until the folder reports anything at all: a write, a
new or deleted name, a rename, a change of attributes. Nothing is decided
from what an event says, only from which folder sent it, so a short name,
a dropped detail or a coalesced event cannot hide a change.

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
import os
import stat as _stat
import struct
import sys
import threading
import time
from collections.abc import Iterable
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
    moved on by every change it reports (``version``)."""

    __slots__ = ("alive", "ident", "key", "listed", "path", "records", "version")

    def __init__(self, path: str, key: Any, ident: tuple[int, int]) -> None:
        self.path = path
        self.key = key
        self.ident = ident
        self.records: dict[str, os.stat_result] = {}
        self.listed: set[str] = set()
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
    (a key of None: events were lost, every folder changed) and
    ``usable(path) -> bool``."""

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
                reported = self._backend.poll()
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
        if d.records or d.listed:
            d.records = {}
            d.listed = set()
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
            if not self._backend.usable(folder):
                return None
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
                return
        elif kind != _S_IFDIR:
            return
        with self._lock:
            if not d.alive or d.version != version:
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
            self._home[path] = d
            if listed:
                d.listed.add(path)
            else:
                d.listed.discard(path)

    # -- answers -----------------------------------------------------------

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
        of *paths* reported a change; None when one of them has no current
        stat. The watched folders of *paths* and their versions."""
        dirs: dict[int, tuple[_Dir, int]] = {}
        home = self._home
        own_of = self._self
        for p in paths:
            d = home.get(p)
            if d is None:
                return None
            st = d.records.get(p)
            if st is None:
                return None
            if id(d) not in dirs:
                dirs[id(d)] = (d, d.version)
            if st.st_mode & _S_IFMT == _S_IFDIR:
                own = own_of.get(p)
                if own is None or own.records.get(p) is not st:
                    return None
                if id(own) not in dirs:
                    dirs[id(own)] = (own, own.version)
        return tuple(dirs.values())

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

    def usable(self, path: str) -> bool:
        if self._statfs(os.fsencode(path), self._buffer) != 0:
            return False
        # ``f_type`` is the struct's first field, a native long.
        f_type = self._ctypes.c_long.from_buffer(self._buffer).value & 0xFFFFFFFF
        return f_type in LOCAL_FILE_SYSTEMS

    def add(self, path: str) -> int | None:
        wd = self._add(self.fd, os.fsencode(path), _IN_MASK)
        if wd < 0:
            import errno

            if self._ctypes.get_errno() == errno.ENOSPC:
                raise _WatchesFull
            return None
        return wd

    def remove(self, wd: int) -> None:
        self._rm(self.fd, wd)

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


# -- Windows: ReadDirectoryChangesW ---------------------------------------------

_FILE_LIST_DIRECTORY = 0x0001
_FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OVERLAPPED = 0x40000000
_ERROR_IO_INCOMPLETE = 996
#: How long removing a folder waits for its cancelled read to end.
_CANCEL_WAIT_SECONDS = 1.0
#: FILE_NOTIFY_CHANGE_ FILE_NAME | DIR_NAME | ATTRIBUTES | SIZE | LAST_WRITE |
#: CREATION | SECURITY.
_NOTIFY_FILTER = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x40 | 0x100
_DRIVE_FIXED = 3
_DRIVE_RAMDISK = 6


class _DirectoryChanges:
    """One overlapped ``ReadDirectoryChangesW`` per folder, polled without
    waiting. Any completion -- changes, an overflow (no bytes), an error --
    counts as a change; the read is issued again, or the folder is dropped."""

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
        self._Overlapped = Overlapped
        self._create = k.CreateFileW
        self._create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._create.restype = wintypes.HANDLE
        self._read = k.ReadDirectoryChangesW
        self._read.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._read.restype = wintypes.BOOL
        self._result = k.GetOverlappedResult
        self._result.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        self._result.restype = wintypes.BOOL
        self._cancel = k.CancelIoEx
        self._cancel.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self._cancel.restype = wintypes.BOOL
        self._close = k.CloseHandle
        self._close.argtypes = [wintypes.HANDLE]
        self._close.restype = wintypes.BOOL
        self._drive_type = k.GetDriveTypeW
        self._drive_type.argtypes = [wintypes.LPCWSTR]
        self._drive_type.restype = wintypes.UINT
        self._invalid = wintypes.HANDLE(-1).value
        self._open: dict[int, tuple[Any, Any, Any]] = {}
        #: Folders whose read ended without being issued again.
        self._idle: set[int] = set()
        #: Reads that did not end within the wait after a cancel.
        self._kept: list[tuple[Any, Any, Any]] = []
        self._next = 0
        self._drives: dict[str, bool] = {}

    def usable(self, path: str) -> bool:
        drive, _rest = os.path.splitdrive(path)
        if not drive or drive.startswith(("\\\\", "//")):
            return False  # a network share
        drive = drive.upper()
        known = self._drives.get(drive)
        if known is None:
            known = self._drives[drive] = self._drive_type(drive + "\\") in (_DRIVE_FIXED, _DRIVE_RAMDISK)
        return known

    def _issue(self, handle: Any, overlapped: Any, buffer: Any) -> bool:
        ctypes = self._ctypes
        ctypes.memset(ctypes.byref(overlapped), 0, ctypes.sizeof(overlapped))
        return bool(
            self._read(
                handle,
                ctypes.byref(buffer),
                ctypes.sizeof(buffer),
                False,
                _NOTIFY_FILTER,
                None,
                ctypes.byref(overlapped),
                None,
            )
        )

    def add(self, path: str) -> int | None:
        handle = self._create(
            path,
            _FILE_LIST_DIRECTORY,
            _FILE_SHARE_ALL,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OVERLAPPED,
            None,
        )
        if handle is None or handle == self._invalid:
            return None
        overlapped = self._Overlapped()
        buffer = (self._ctypes.c_uint32 * 1024)()
        if not self._issue(handle, overlapped, buffer):
            self._close(handle)
            return None
        self._next += 1
        self._open[self._next] = (handle, overlapped, buffer)
        return self._next

    def remove(self, key: int) -> None:
        held = self._open.pop(key, None)
        idle = key in self._idle
        self._idle.discard(key)
        if held is None:
            return
        handle, overlapped, _buffer = held
        if idle:
            # No read is pending (it ended with an error, or was not issued
            # again after the folder went): nothing to cancel or wait for.
            # Waiting here hung the kernel for good once a watched folder was
            # deleted and made again.
            self._close(handle)
            return
        ctypes = self._ctypes
        moved = self._wintypes.DWORD()
        self._cancel(handle, ctypes.byref(overlapped))
        # Wait for the cancelled read to end -- until then the system may
        # still write into the buffer and the overlapped block -- but never
        # for good: asked without waiting, a bounded number of times.
        deadline = time.monotonic() + _CANCEL_WAIT_SECONDS
        while not self._result(handle, ctypes.byref(overlapped), ctypes.byref(moved), False):
            if ctypes.get_last_error() != _ERROR_IO_INCOMPLETE:
                break  # ended (cancelled or failed): the buffer is free
            if time.monotonic() >= deadline:
                # Still not ended: keep the buffer and the block alive for
                # the process's life rather than free memory the system may
                # still write into.
                self._kept.append(held)
                break
            time.sleep(0.001)
        self._close(handle)

    def poll(self) -> list[tuple[int | None, bool]]:
        ctypes = self._ctypes
        reported: list[tuple[int | None, bool]] = []
        moved = self._wintypes.DWORD()
        for key, (handle, overlapped, buffer) in list(self._open.items()):
            if self._result(handle, ctypes.byref(overlapped), ctypes.byref(moved), False):
                if self._issue(handle, overlapped, buffer):
                    reported.append((key, False))
                    continue
            elif ctypes.get_last_error() == _ERROR_IO_INCOMPLETE:
                continue
            self._idle.add(key)  # no read pending any more: never waited on
            reported.append((key, True))  # an error, or no read pending: dropped
        return reported


# -- the process's watch ---------------------------------------------------------

_WATCH: FileWatch | None = None
_TRIED_IN: int | None = None


def _switched_off() -> bool:
    return os.environ.get("CASH_FILE_WATCH", "").strip().lower() in ("0", "false", "no", "off")


def _backend() -> Any:
    if sys.platform.startswith("linux"):
        return _Inotify()
    if sys.platform == "win32":
        return _DirectoryChanges()
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
    backend = watch._backend
    try:
        for key in list(watch._by_key):
            backend.remove(key)
        fd = getattr(backend, "fd", None)
        if fd is not None:
            os.close(fd)
    except Exception:  # noqa: BLE001 - closing a watch no longer used
        pass
