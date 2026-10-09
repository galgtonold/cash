"""File digests remembered across processes, in the cache directory.

`file_content_hash` reads every byte of a file. Within a process it
remembers the digest of a file that had settled, keyed by the file's stat
identity (`file_dep_snapshot._HASH_MEMO`); this table carries those digests
to the next process: a ``FileDataSource`` token, a ``file_depends_on=``
check or a first read of an unchanged input then costs one ``stat``.

A digest is written here only under the rule that lets a stored entry's
file dependency skip the read (`file_dep_snapshot._unchanged_since_hashed`):
the file had been left alone for a while before it was hashed, so it cannot
have been written again within its timestamp's tick. It is looked up only
under the exact stat identity it was taken at -- path, device, inode, size,
modification and change time to the nanosecond. Any write moves one of
them, except one that puts the timestamps back: on Linux and macOS the
inode change time still moves, on Windows nothing does (see
known-limitations: an edit that keeps size and timestamps).

The file is an append-only log of JSON lines, one per digest, each carrying
the digest scheme's version (`cash.bulk_digest.TAG`): a line from another
scheme, a torn or corrupt line is skipped, and an empty table means only
that the file is read. When it grows past `_COMPACT_AT` lines it is
rewritten with the newest `_KEEP` entries.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from .._paths import replace_with_retry
from ..backends.cache_dir import FILE_DIGESTS_FILENAME
from ..bulk_digest import TAG

logger = logging.getLogger(__name__)

__all__ = ["DigestTable", "current_table", "use_digest_table"]

#: The scheme a line's digest was taken with.
_SCHEME = TAG.rstrip(b"\x00").decode("ascii")
_COMPACT_AT = 50_000
_KEEP = 20_000

StatKey = tuple[str, int, int, int, int, int]


class DigestTable:
    """``stat identity -> (digest, when it was taken)`` for one cache dir."""

    def __init__(self, cache_dir: str) -> None:
        self.path = os.path.join(cache_dir, FILE_DIGESTS_FILENAME)
        self._items: dict[StatKey, tuple[str, float]] = {}
        self._lines = 0
        self._loaded = False
        self._lock = threading.Lock()

    def get(self, key: StatKey) -> tuple[str, float] | None:
        self._ensure_loaded()
        return self._items.get(key)

    def put(self, key: StatKey, digest: str, hashed_at: float) -> None:
        self._ensure_loaded()
        if self._items.get(key, (None,))[0] == digest:
            return
        line = json.dumps([_SCHEME, *key, digest, hashed_at], separators=(",", ":")) + "\n"
        with self._lock:
            self._items[key] = (digest, hashed_at)
            if not os.path.isdir(os.path.dirname(self.path)):
                return  # a cache dir that is gone: never recreated from here
            try:
                # One write of one short line, in append mode: lines from two
                # processes interleave whole.
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError:
                logger.debug("[FILE_DEP] could not note a digest in %s", self.path, exc_info=True)
                return
            self._lines += 1
            if self._lines > _COMPACT_AT:
                self._compact()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                with open(self.path, encoding="utf-8", errors="replace") as fh:
                    for raw in fh:
                        self._lines += 1
                        parsed = _parse(raw)
                        if parsed is not None:
                            self._items[parsed[0]] = parsed[1]
            except OSError:
                pass
            self._loaded = True

    def _compact(self) -> None:
        """Rewrite the log with the newest `_KEEP` entries (lock held)."""
        newest = sorted(self._items.items(), key=lambda kv: kv[1][1])[-_KEEP:]
        self._items = dict(newest)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for key, (digest, hashed_at) in newest:
                    fh.write(json.dumps([_SCHEME, *key, digest, hashed_at], separators=(",", ":")) + "\n")
            replace_with_retry(tmp, self.path)
            self._lines = len(newest)
        except OSError:
            logger.debug("[FILE_DEP] could not compact %s", self.path, exc_info=True)
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _parse(raw: str) -> tuple[StatKey, tuple[str, float]] | None:
    """One line as ``(key, (digest, hashed_at))``, or None for a line of
    another scheme, a torn one or anything else that is not one of ours."""
    try:
        row = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(row, list) or len(row) != 9 or row[0] != _SCHEME:
        return None
    _, path, dev, ino, size, mtime_ns, ctime_ns, digest, hashed_at = row
    if not isinstance(path, str) or not all(type(v) is int for v in (dev, ino, size, mtime_ns, ctime_ns)):
        return None
    if not isinstance(digest, str) or len(digest) != 64 or not _is_hex(digest):
        return None
    if type(hashed_at) not in (int, float):
        return None
    return (path, dev, ino, size, mtime_ns, ctime_ns), (digest, float(hashed_at))


def _is_hex(text: str) -> bool:
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


_TABLE: DigestTable | None = None
_TABLES: dict[str, DigestTable] = {}


def use_digest_table(cache_dir: str | None) -> None:
    """Keep this process's file digests in *cache_dir* from now on (the
    cache directory of the `Cash` that last built its backend). None (a
    cache with no directory) changes nothing."""
    global _TABLE
    if not cache_dir:
        return
    table = _TABLES.get(cache_dir)
    if table is None:
        table = _TABLES[cache_dir] = DigestTable(cache_dir)
    _TABLE = table


def current_table() -> DigestTable | None:
    """The table file digests are kept in, or None when no cache has a directory."""
    return _TABLE


def reset_for_tests() -> None:
    """Forget every table (a test that must start with none)."""
    global _TABLE
    _TABLE = None
    _TABLES.clear()
