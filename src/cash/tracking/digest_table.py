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

The file is an append-only log of JSON lines, one per digest, written a
batch at a time (`cash.tracking.log_lines`), each carrying
the digest scheme's algorithm and version (`cash.bulk_digest.SCHEME`): a
line from another scheme (an older cash, or an install with or without
``blake3``), a torn or corrupt line is skipped, and an empty table means
only that the file is read. When it grows past `_COMPACT_AT` lines (or
twice what it kept the last time) it is rewritten with every entry this
process put or found, and the newest others up to `_KEEP`: a statement that
read 50,000 files kept only the newest 20,000, so each restart read the
other 30,000 again.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading

from .._paths import replace_with_retry
from ..backends.cache_dir import FILE_DIGESTS_FILENAME
from ..bulk_digest import SCHEME
from .log_lines import LogLines, flush_all

logger = logging.getLogger(__name__)

__all__ = ["DigestTable", "current_table", "use_digest_table"]

#: The scheme a line's digest was taken with (its algorithm and version).
_SCHEME = SCHEME
_COMPACT_AT = 50_000
_KEEP = 20_000
#: Entries this process put or found are kept past `_KEEP`, up to this many.
_KEEP_USED = 500_000

StatKey = tuple[str, int, int, int, int, int]


class DigestTable:
    """``stat identity -> (digest, when it was taken)`` for one cache dir."""

    def __init__(self, cache_dir: str) -> None:
        self.path = os.path.join(cache_dir, FILE_DIGESTS_FILENAME)
        self._items: dict[StatKey, tuple[str, float]] = {}
        self._lines = 0
        self._compact_at = _COMPACT_AT
        self._used: set[StatKey] = set()  # put or found by this process: kept by `_compact`
        self._loaded = False
        self._lock = threading.Lock()
        self._log = LogLines(self.path, self._lock, "FILE_DEP")

    def get(self, key: StatKey) -> tuple[str, float] | None:
        self._ensure_loaded()
        found = self._items.get(key)
        if found is not None:
            self._used.add(key)
        return found

    def put(self, key: StatKey, digest: str, hashed_at: float) -> None:
        self._ensure_loaded()
        self._used.add(key)
        if self._items.get(key, (None,))[0] == digest:
            return
        line = _line(key, digest, hashed_at)
        with self._lock:
            self._items[key] = (digest, hashed_at)
            self._log.add(line)
            self._lines += 1
            if self._lines > self._compact_at:
                self._compact()

    def flush(self) -> None:
        """Write the lines still waiting (`cash.tracking.log_lines`)."""
        self._log.flush()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        flush_all()  # this process's own lines, still waiting, are read too
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
        """Rewrite the log with the entries this process put or found and
        the newest others up to `_KEEP` (lock held)."""
        by_age = sorted(self._items.items(), key=lambda kv: kv[1][1])
        used = [kv for kv in by_age if kv[0] in self._used][-_KEEP_USED:]
        room = _KEEP - len(used)
        others = [kv for kv in by_age if kv[0] not in self._used][-room:] if room > 0 else []
        newest = sorted(others + used, key=lambda kv: kv[1][1])
        self._items = dict(newest)
        self._used &= self._items.keys()
        self._compact_at = max(_COMPACT_AT, 2 * len(newest))
        self._log.drop()  # in the rewrite, if among the newest
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for key, (digest, hashed_at) in newest:
                    fh.write(_line(key, digest, hashed_at))
            replace_with_retry(tmp, self.path)
            self._lines = len(newest)
        except OSError:
            logger.debug("[FILE_DEP] could not compact %s", self.path, exc_info=True)
            with contextlib.suppress(OSError):
                os.unlink(tmp)


_text = json.encoder.encode_basestring_ascii  # what ``json.dumps`` writes a str as


def _line(key: StatKey, digest: str, hashed_at: float) -> str:
    """``json.dumps([_SCHEME, *key, digest, hashed_at], separators=(",", ":"))``
    and a newline, byte for byte, without the encoder's set-up per line
    (~10 us of each of 30,000 puts)."""
    path, dev, ino, size, mtime_ns, ctime_ns = key
    if (
        type(path) is not str
        or type(digest) is not str
        or not all(type(v) is int for v in (dev, ino, size, mtime_ns, ctime_ns))
        or not (type(hashed_at) is int or (type(hashed_at) is float and math.isfinite(hashed_at)))
    ):
        return json.dumps([_SCHEME, *key, digest, hashed_at], separators=(",", ":")) + "\n"
    when = int.__repr__(hashed_at) if type(hashed_at) is int else float.__repr__(hashed_at)
    return (
        f"[{_text(_SCHEME)},{_text(path)},{int.__repr__(dev)},{int.__repr__(ino)},{int.__repr__(size)},"
        f"{int.__repr__(mtime_ns)},{int.__repr__(ctime_ns)},{_text(digest)},{when}]\n"
    )


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
