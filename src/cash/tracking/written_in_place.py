"""Files mapped to write in place, whose metadata cannot be trusted.

A file's digest is taken from its stat when the stat is as it was when the
digest was taken (`file_dep_snapshot._unchanged_since_hashed`, the digest
memo and `digest_table`): any write moves the size, the modification time
or, on Linux and macOS, the inode change time. A write through a shared
memory mapping is the exception: on Windows it can leave all of them as they
were, so after ``np.memmap(path, mode="r+")`` changed a file in place, the
old digest was trusted and the result computed from the old content served.

A file this process maps writable (``mmap.mmap`` with write access, which
``np.memmap`` in ``r+`` or ``w+`` mode and ``np.load(mmap_mode="r+")`` use)
is noted here by its identity (device and inode) from the ``mmap.__new__``
audit event, and from then on is read at every check instead. The marks are
kept in this process and in the cache directory (`use_written_in_place`), so
a later process checks the file by content too. A write through a mapping
made in a process without cash, or before ``import cash``, is not seen.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading

from ..backends.cache_dir import WRITTEN_IN_PLACE_FILENAME
from . import io_watch

logger = logging.getLogger(__name__)

__all__ = ["is_written_in_place", "use_written_in_place"]

Identity = tuple[int, int]

#: Marked in this process.
_MARKED: set[Identity] = set()
#: Marked in the cache directory in use, by any process.
_DIR_MARKS: set[Identity] = set()
_DIR: str | None = None
_lock = threading.Lock()
_WRITABLE = frozenset({mmap.ACCESS_DEFAULT, mmap.ACCESS_WRITE})


def _on_mmap(args: tuple) -> None:
    fileno, _length, access, _offset = args
    if fileno < 0 or access not in _WRITABLE:
        return
    try:
        st = os.fstat(fileno)
    except OSError:
        return
    if st.st_ino:
        _mark((st.st_dev, st.st_ino))


def _mark(identity: Identity) -> None:
    with _lock:
        if identity in _MARKED:
            return
        _MARKED.add(identity)
        if _DIR is not None and identity not in _DIR_MARKS:
            _DIR_MARKS.add(identity)
            _append(_DIR, [identity])


def _append(cache_dir: str, identities: list[Identity]) -> None:
    try:
        with open(os.path.join(cache_dir, WRITTEN_IN_PLACE_FILENAME), "a", encoding="utf-8") as fh:
            fh.writelines(f"{dev} {ino}\n" for dev, ino in identities)
    except OSError:
        logger.debug("[FILE_DEP] could not note a file written in place", exc_info=True)


def _load(cache_dir: str) -> set[Identity]:
    marks: set[Identity] = set()
    try:
        with open(os.path.join(cache_dir, WRITTEN_IN_PLACE_FILENAME), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    marks.add((int(parts[0]), int(parts[1])))
    except OSError:
        pass
    return marks


def use_written_in_place(cache_dir: str | None) -> None:
    """Keep the marks in *cache_dir* from now on, and take those it holds
    (the cache directory of the `Cash` that last built its backend). None
    (a cache with no directory) changes nothing."""
    global _DIR, _DIR_MARKS
    if not cache_dir or cache_dir == _DIR:
        return
    marks = _load(cache_dir)
    with _lock:
        new = sorted(_MARKED - marks)
        if new:
            _append(cache_dir, new)
        _DIR, _DIR_MARKS = cache_dir, marks | _MARKED


def is_written_in_place(st: os.stat_result, path: str | None = None) -> bool:
    """Was the file *st* describes mapped to write, here or by a process
    using the same cache directory? A stat with no identity (a directory
    listing's) is re-taken from *path*."""
    if not _DIR_MARKS and not _MARKED:
        return False
    if not st.st_ino:
        if path is None:
            return True
        try:
            st = os.stat(path)
        except OSError:
            return True
    identity = (st.st_dev, st.st_ino)
    return identity in _MARKED or identity in _DIR_MARKS


def reset_for_tests() -> None:
    global _DIR, _DIR_MARKS
    with _lock:
        _MARKED.clear()
        _DIR, _DIR_MARKS = None, set()


io_watch.subscribe_always("mmap.__new__", _on_mmap)
