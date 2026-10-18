"""Which earlier version of a file a later one holds byte for byte.

A notebook statement that reads a file folds the file's stat (modification
and change time, size) into its outputs' lineage
(``notebook.statement.file_deps.compute_file_hash_component``). A file
rewritten with the same bytes (a data export run again) gets a new stat. A
stored entry that read it is still fresh, since its content check passes,
and is restored with the lineage it was stored with. A statement that runs
again instead (one too cheap to keep on disk, after a restart) took the new
stat, so everything built on it keyed differently than before the restart
and ran again with its inputs unchanged.

Where a content check found the bytes unchanged under a new stat
(`note_same_content`), the new stat is recorded as a version of the old
one, and `file_version` answers the old stat for it from then on. The
record is kept in the cache directory (`use_file_versions`), one line per
version appended (a batch at a time, `cash.tracking.log_lines`), so that a
restarted kernel answers the same.

A file nothing stored checks by content is linked the same way when it was
written while this process ran (`content_version`): ``tarfile.extractall``
in the statement that then reads the file rewrites it with the same bytes on
every run, and the statement, which writes a file, is never stored.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .._memo import FILE_DIGESTS, LruMemo
from .._paths import normalize_path, replace_with_retry
from ..backends.cache_dir import FILE_VERSIONS_FILENAME
from .log_lines import LogLines, flush_all

logger = logging.getLogger(__name__)

__all__ = ["content_version", "file_version", "note_same_content", "use_file_versions"]

#: ``(modification time ns, size, change time ns)``.
Version = tuple[int, int, int]

#: Lines the log holds before it is rewritten with its newest `_KEEP` versions.
_COMPACT_AT = 50_000
_KEEP = 20_000

#: ``(canonical path, version)`` -> the earliest version found to hold the same bytes.
_EARLIER: LruMemo[tuple[str, Version], Version] = LruMemo(FILE_DIGESTS)

#: ``(canonical path, content digest)`` -> the earliest version seen holding those bytes.
_BY_CONTENT: LruMemo[tuple[str, str], Version] = LruMemo(FILE_DIGESTS)

#: When this process began to watch: a file last written before it was not
#: rewritten by this process, and is never hashed here (`content_version`).
_STARTED_NS = time.time_ns()


def _version(st: os.stat_result) -> Version:
    return (st.st_mtime_ns, st.st_size, getattr(st, "st_ctime_ns", 0))


def file_version(canonical_path: str, st: os.stat_result) -> Version:
    """The version of the file at *canonical_path* (resolved and normalised),
    stat'ed as *st*: the earliest one found to hold the same bytes, else its
    own stat."""
    own = _version(st)
    if not len(_EARLIER):
        return own
    return _EARLIER.get((canonical_path, own)) or own


def content_version(canonical_path: str, st: os.stat_result, digest: Callable[[], str | None]) -> Version:
    """`file_version`, and for a file written since this process started,
    the earliest version seen holding the same bytes (*digest* hashes the
    file, asked once per version).

    A file that was not written while this process ran is answered by its
    stat alone: its readers' lineage needs no hashing of data nobody
    touched. One rewritten unchanged keeps its readers' lineage; a version
    with new bytes is its own."""
    own = _version(st)
    known = _EARLIER.get((canonical_path, own))
    if known is not None:
        return known
    if max(st.st_mtime_ns, own[2]) < _STARTED_NS:
        return own
    content = digest()
    if content is None:
        return own
    root = _BY_CONTENT.get((canonical_path, content))
    if root is None:
        _BY_CONTENT[(canonical_path, content)] = own
        if _LOG is not None:
            _LOG.append_content(canonical_path, content, own)
        root = own
    elif _LOG is not None:
        _LOG.append(canonical_path, own, root)
    _EARLIER[(canonical_path, own)] = root
    return root


def note_same_content(path: str, st: os.stat_result, recorded: Mapping[str, Any]) -> None:
    """The file at *path*, now stat'ed as *st*, was found to hold the bytes
    of the snapshot *recorded* (``file_dep_snapshot.snapshot_file_deps``)."""
    try:
        earlier: Version = (int(recorded["mtime_ns"]), int(recorded["size"]), int(recorded.get("ctime_ns") or 0))
    except (KeyError, TypeError, ValueError):
        return
    own = _version(st)
    if own == earlier:
        return
    try:
        canonical = normalize_path(os.path.realpath(path))
    except (OSError, ValueError):
        return
    root = _EARLIER.get((canonical, earlier)) or earlier
    if _EARLIER.get((canonical, own)) == root:
        return
    _EARLIER[(canonical, own)] = root
    if _LOG is not None:
        _LOG.append(canonical, own, root)


class _VersionLog:
    """The versions, one JSON line each, in a cache directory."""

    def __init__(self, cache_dir: str) -> None:
        self.path = os.path.join(cache_dir, FILE_VERSIONS_FILENAME)
        self._lines = 0
        self._lock = threading.Lock()
        self._log = LogLines(self.path, self._lock, "FILE_VERSIONS")

    def load(self) -> None:
        """Take the versions the log holds into this process's."""
        flush_all()  # this process's own lines, still waiting, are read too
        try:
            with open(self.path, encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    self._lines += 1
                    parsed = _parse(raw)
                    if parsed is None:
                        content = _parse_content(raw)
                        if content is not None and content[0] not in _BY_CONTENT:
                            _BY_CONTENT[content[0]] = content[1]
                    elif parsed[0] not in _EARLIER:
                        _EARLIER[parsed[0]] = parsed[1]
        except OSError:
            pass

    def append(self, path: str, own: Version, root: Version) -> None:
        self._write(json.dumps([path, *own, *root], separators=(",", ":")) + "\n")

    def append_content(self, path: str, digest: str, version: Version) -> None:
        self._write(json.dumps([path, digest, *version], separators=(",", ":")) + "\n")

    def _write(self, line: str) -> None:
        with self._lock:
            # Written with the next lines, by the end of the cell at the
            # latest: a restart kills the kernel without an exit hook.
            self._log.add(line)
            self._lines += 1
            if self._lines > _COMPACT_AT:
                self._compact()

    def flush(self) -> None:
        """Write the lines still waiting (`cash.tracking.log_lines`)."""
        self._log.flush()

    def _compact(self) -> None:
        """Rewrite the log with the newest `_KEEP` versions (lock held)."""
        newest = [(key, root) for key in _EARLIER.keys()[-_KEEP:] if (root := _EARLIER.get(key)) is not None]
        contents = [(key, v) for key in _BY_CONTENT.keys()[-_KEEP:] if (v := _BY_CONTENT.get(key)) is not None]
        self._log.drop()  # in the rewrite
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for (path, digest), version in contents:
                    fh.write(json.dumps([path, digest, *version], separators=(",", ":")) + "\n")
                for (path, own), root in newest:
                    if own != root:
                        fh.write(json.dumps([path, *own, *root], separators=(",", ":")) + "\n")
            replace_with_retry(tmp, self.path)
            self._lines = len(newest) + len(contents)
        except OSError:
            logger.debug("[FILE_VERSIONS] could not compact %s", self.path, exc_info=True)
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _parse(raw: str) -> tuple[tuple[str, Version], Version] | None:
    """One line as ``((path, version), earlier)``, or None for a torn line
    or anything else that is not one of ours."""
    try:
        row = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(row, list) or len(row) != 7 or not isinstance(row[0], str):
        return None
    if not all(type(v) is int for v in row[1:]):
        return None
    return (row[0], (row[1], row[2], row[3])), (row[4], row[5], row[6])


def _parse_content(raw: str) -> tuple[tuple[str, str], Version] | None:
    """One content line as ``((path, digest), version)``, or None."""
    try:
        row = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(row, list) or len(row) != 5 or not isinstance(row[0], str) or not isinstance(row[1], str):
        return None
    if not all(type(v) is int for v in row[2:]):
        return None
    return (row[0], row[1]), (row[2], row[3], row[4])


_LOG: _VersionLog | None = None
_LOGS: dict[str, _VersionLog] = {}


def use_file_versions(cache_dir: str | None) -> None:
    """Keep the versions this process finds in *cache_dir* from now on, and
    take those it holds (the cache directory of the `Cash` that last built
    its backend). None (a cache with no directory) changes nothing."""
    global _LOG
    if not cache_dir:
        return
    log = _LOGS.get(cache_dir)
    if log is None:
        log = _LOGS[cache_dir] = _VersionLog(cache_dir)
        log.load()
    _LOG = log
