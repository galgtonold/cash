"""File-dependency snapshots: capturing what a computation read, and
checking later whether it still matches.

A snapshot is ``{path: {'mtime': float, 'size': int, 'hash': str}}`` (plus
remote and absent entries). These are not pure functions: they stat and hash
files, keep short-lived memos of both (``begin_file_state_epoch``,
:class:`FreshnessMemo`), and do their own reads outside every tracker
(``untracked``). A file's content is always hashed in full, never sampled.

Used by the decorator (``decorator/file_deps.py``, ``explain``) and by the
notebook, so it lives outside the notebook package and the decorator never
imports the notebook for it.

**Content-hash freshness.** ``(mtime, size)`` alone is an
ambiguous freshness signal and fails two opposite ways: a touch-only change
(identical content + size, only the mtime bumped) spuriously invalidates,
and a same-size edit under a mtime the coarse check can't tell apart
(sub-resolution / same-second write) is missed. We therefore record a
content hash at snapshot time and treat CONTENT as authoritative whenever the
size matches: the cheap size check runs first (and never hashes on the
size-differs path), and only when the size is equal do we hash to decide --
unless nothing else moved either: a file whose timestamps and identity are as
recorded, and that had settled before it was hashed, is not read again
(``_unchanged_since_hashed``).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import stat as _stat
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, NamedTuple

from cash._memo import FILE_DIGESTS, LruMemo
from cash._paths import MAIN_MODULE_NAMES, main_namespace_name, normalize_path, resolve_file_dep_path
from cash.bulk_digest import BulkHasher
from cash.remote_source import RemoteFileDataSource, addressing_options, read_options
from cash.tracking.digest_table import current_table
from cash.tracking.file_versions import note_same_content
from cash.tracking.pending_digest import MOVED, DeferredDigest
from cash.tracking.read_classification import own_exists, own_lstat, own_stat, stat_key
from cash.tracking.tracker_context import untracked
from cash.tracking.written_in_place import is_written_in_place

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_file_deps",
    "walk_dataset_dir",
    "snapshot_remote_deps",
    "snapshot_absent_deps",
    "snapshot_present_deps",
    "probe_answers",
    "probe_state_parts",
    "snapshot_dependencies",
    "existing_file_deps",
    "file_content_hash",
    "file_dep_is_fresh",
    "dep_is_fresh",
    "snapshot_is_fresh",
    "FreshnessMemo",
    "StaleDep",
    "ALSO_SPELLED",
    "fold_relative_twins",
    "unfold_relative_twins",
]

# Marks a snapshot entry as a REMOTE object rather than a local path. Remote
# entries ride in the same ``auto_file_deps`` dict as files - they answer the
# same question ("did what this call read change since?") and every consumer
# already routes that question through ``file_dep_is_fresh``, so one dict with
# one branch beats a parallel channel each consumer would have to learn.
_REMOTE_MARKER = "remote"

# Marks a snapshot entry as a path that was NOT there when the call ran. The
# absence of a file is an input like any other -- an optional config that is
# missing means "use the defaults" -- and it was the one input cash could not
# see, because a file that is never opened produces no read to track. Without
# it, a cached function reading `cfg.txt` by relative name in directory A, then
# in B (which has no such file), then in A again, is served B's answer in A as
# a hit: the B run recorded NO dependencies, so its entry looks valid
# everywhere.
_ABSENT_MARKER = "absent"

# Marks a snapshot entry as a path that WAS there and was not read: a flag file
# or a folder the call only checked for. Its value is what the probe asked for
# (``file``, ``dir`` or ``any``), and the entry is fresh while that still holds.
_PRESENT_MARKER = "present"

# Marks a snapshot entry for a file that was read but could not be stat'ed as
# recorded. Such an entry is never fresh: failing closed costs a recompute,
# failing open (dropping the dependency) served every later edit stale.
_UNRESOLVED_MARKER = "unresolved"

_HASH_READ_CHUNK = 1024 * 1024  # 1 MiB streaming chunk
#: Bytes, not text, on Windows (no newline translation); 0 elsewhere.
_O_BINARY = getattr(os, "O_BINARY", 0)


#: Digests already computed this process, keyed by the file's identity AND its
#: stat fields: ``(path, st_dev, st_ino, size, mtime_ns, ctime_ns)``.
#:
#: File dependencies propagate, so one burst of cached calls checks the same
#: inputs many times; with the memo later checks are one ``stat`` each. Only
#: a file UNTOUCHED for ``_HASH_MEMO_MIN_AGE_SECONDS`` before it is hashed is
#: memoized -- the rule a stored entry's dependency skips its read by
#: (``_unchanged_since_hashed``): a file written moments ago may still be
#: being written, within one timestamp tick. Any later write moves ``mtime``,
#: and on POSIX ``ctime`` too, which no tool puts back. On Windows a
#: same-size edit that RESTORES the mtime leaves every key field identical
#: and is not seen -- a documented limitation ("an edit that keeps size and
#: timestamps"), the same one a stored entry's check has.
#:
#: The digests are kept for the next process too, in the cache directory
#: (`cash.tracking.digest_table`), under the same key and rule. A stat with
#: no file identity is the exception (`_LISTED_STAT_TTL_SECONDS`). In a
#: notebook a digest so bounded also holds for the rest of the CELL RUN
#: (``begin_file_state_epoch`` .. ``end_file_state_epoch``): a cell over
#: thousands of files outlasts the bound on its own, and must not re-hash
#: them for every derived statement.
_HASH_MEMO: LruMemo[tuple[str, int, int, int, int, int], tuple[float, str, int | None]] = LruMemo(FILE_DIGESTS)
#: How long a remembered digest is reused, or None for as long as its key
#: holds. A number turns the cross-process table off and bounds the memo
#: (a test that must see every read sets 0).
_HASH_MEMO_TTL_SECONDS: float | None = None
_HASH_MEMO_MIN_AGE_SECONDS = 10.0
#: The current cell run's number, or None between runs.
HASH_EPOCH: int | None = None
_EPOCH_COUNT = 0
#: A cell run started inside another (a cell that calls
#: `get_ipython().run_cell(...)`) is the same run.
_EPOCH_DEPTH = 0


#: Digests of files too young for ``_HASH_MEMO``, by its key, kept while a
#: statement is processed (`young_digests_kept`): a file the statement
#: read was hashed when it was read and again for its key, ~2x the bytes.
#: Within the statement its stat is the whole key -- size, both times and
#: the file's identity -- as ``_HASH_MEMO``'s is for a settled file; across
#: statements a young file is read again, as ever. A write cash sees in the
#: statement forgets them all (`forget_young_digests`).
_YOUNG_DIGESTS: dict[tuple[str, int, int, int, int, int], str] = {}
#: How many `young_digests_kept` blocks are open.
_YOUNG_DEPTH = 0


@contextlib.contextmanager
def young_digests_kept() -> Iterator[None]:
    """Keep young files' digests for the block, one statement's processing;
    forgotten as it starts and as it ends (a nested one included)."""
    global _YOUNG_DEPTH
    _YOUNG_DIGESTS.clear()
    _YOUNG_DEPTH += 1
    try:
        yield
    finally:
        _YOUNG_DEPTH -= 1
        _YOUNG_DIGESTS.clear()


def forget_young_digests() -> None:
    """A file may have been written: no young file's digest is reused."""
    if _YOUNG_DIGESTS:
        _YOUNG_DIGESTS.clear()


def begin_file_state_epoch() -> None:
    """A cell run starts: digests from earlier runs are looked at again."""
    global HASH_EPOCH, _EPOCH_COUNT, _EPOCH_DEPTH
    _EPOCH_DEPTH += 1
    if _EPOCH_DEPTH == 1:
        _EPOCH_COUNT += 1
        HASH_EPOCH = _EPOCH_COUNT


def file_state_epoch() -> int | None:
    """The current cell run's number, or None between runs.

    The cell run's own identity, for memos that must not outlive it. The
    shell's ``execution_count`` is not one: it only moves for a cell run
    with ``store_history=True``, so ``shell.run_cell(code)`` and a frontend's
    history-less execute leave it where it was, cell after cell.
    """
    return HASH_EPOCH


def end_file_state_epoch() -> None:
    """The cell run that `begin_file_state_epoch` started is over."""
    global HASH_EPOCH, _EPOCH_DEPTH
    _EPOCH_DEPTH = max(0, _EPOCH_DEPTH - 1)
    if _EPOCH_DEPTH == 0:
        HASH_EPOCH = None


#: ``realpath`` answers for the current cell run, keyed on the path as given
#: (and the working directory, for a relative one).
_REALPATH_MEMO: dict[tuple[str, str], str] = {}

#: How many answers `_REALPATH_MEMO` holds before it starts over.
_REALPATH_KEPT = 65536
_REALPATH_MEMO_EPOCH: int | None = None


def _remember_realpath(key: tuple[str, str], resolved: str) -> None:
    if len(_REALPATH_MEMO) >= _REALPATH_KEPT:
        # Full: start over, so a folder resolved from now on is remembered
        # again. Kept full, a statement unpacking 50,000 files resolved the
        # folder of each one past the first ~20,000 in full, a stat of every
        # component of its path, three times a file.
        _REALPATH_MEMO.clear()
    _REALPATH_MEMO[key] = resolved
    # A resolved path resolves to itself, and callers hand it back in
    # both spellings: the tracker records ``normalize_path`` of it, and
    # the lineage component resolves that record again.
    _REALPATH_MEMO[("", resolved)] = resolved
    _REALPATH_MEMO[("", normalize_path(resolved))] = resolved


def realpath_of_read_this_run(
    path: str, folders: dict[tuple[str, str], str] | None = None, lstat: os.stat_result | None = None
) -> tuple[str, os.stat_result | None]:
    """:func:`realpath_this_run` for a file about to be read, and its stat when
    that is what answered.

    ``realpath`` costs two ``_getfinalpathname`` calls per file on Windows, and
    a folder read paid them for all 5,030 files in one directory: a quarter of
    what cash added to the read. A regular file that is not
    itself a link or reparse point resolves to its directory's real path plus
    its name, so the directory is resolved once; the ``lstat`` that shows it is
    such a file is also the stat the read needs. So does a missing one, and a
    folder that is not a link: a statement listing 5,750 folders, one per
    person, resolved each in full, ~30 us a listing. Anything else -- a link,
    a short ``~`` name -- is resolved in full.

    Outside a cell run (a cached function called from a script) there is no
    run to remember answers for; *folders*, when given, is the calling
    block's own memo of its folders' real paths, used the same way for as
    long as the block runs. Without either, the path is resolved in full.

    *lstat* is the caller's ``lstat`` of *path*, taken in this cell run, in
    place of one taken here: the same answer from the same facts.
    """

    epoch = HASH_EPOCH
    if epoch is None:
        if folders is None:
            return os.path.realpath(path), None

        def resolve_folder(folder: str) -> str:
            key = ("" if os.path.isabs(folder) else os.getcwd(), folder)
            resolved = folders.get(key)
            if resolved is None:
                resolved = folders[key] = os.path.realpath(folder)
            return resolved
    else:
        resolve_folder = realpath_this_run
        key = ("" if os.path.isabs(path) else os.getcwd(), path)
        if _REALPATH_MEMO_EPOCH == epoch:
            resolved = _REALPATH_MEMO.get(key)
            if resolved is not None:
                return resolved, None
    absolute = os.path.abspath(path)
    parent, name = os.path.split(absolute)
    # ``abspath`` drops a ``..`` by its spelling, but on POSIX the open goes
    # up from where a linked folder POINTS: ``/w/link/../data.csv`` is the
    # target's sibling, not ``/w/data.csv``, which a hit would have checked.
    if (
        name
        and name not in (".", "..")
        and "~" not in name
        and parent != absolute
        and ".." not in path.replace("\\", "/").split("/")
    ):
        try:
            st = lstat if lstat is not None else own_lstat(absolute)
        except FileNotFoundError:
            # Not there yet, as at an open for writing: ``realpath`` answers
            # its directory's real path plus the name too. Resolved in full,
            # a folder of 13,000 images unpacked by ``tarfile`` paid a walk
            # of every path component per file.
            resolved = os.path.join(resolve_folder(parent), name)
            if epoch is not None:
                _remember_realpath(key, resolved)
            return resolved, None
        except (OSError, ValueError):
            st = None
        if (
            st is not None
            and (_stat.S_ISREG(st.st_mode) or _stat.S_ISDIR(st.st_mode))
            and not getattr(st, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            resolved = os.path.join(resolve_folder(parent), name)
            if epoch is not None:
                _remember_realpath(key, resolved)
            if _NOTE_LINKS:
                # For a listing of its folder later on, which gives no link
                # count: the read's records are taken at a store below, if
                # at all (`_PendingRecords`), and a listing before took a
                # stat of each file and folder read to learn it.
                note_links(absolute, st)
            return resolved, (st if _stat.S_ISREG(st.st_mode) else None)
    return (realpath_this_run(path) if epoch is not None else os.path.realpath(path)), None


#: ``FILE_ATTRIBUTE_REPARSE_POINT``: a symlink or junction on Windows.
_REPARSE_POINT = 0x400


def realpath_this_run(path: str) -> str:
    """``os.path.realpath(path)``, remembered for the rest of the cell run.

    ``realpath`` is a handful of ``_getfinalpathname`` calls on Windows, ~60us,
    and a statement's files were resolved again at every read, every lineage
    component and every snapshot: 13% of a cell reading 3,000 files.
    Within one run a link is not re-pointed under the same statement's feet;
    the next run resolves afresh, and so does anything outside a run. A
    relative path is keyed on the working directory too, so an ``os.chdir``
    mid-cell resolves anew.
    """
    global _REALPATH_MEMO_EPOCH
    epoch = HASH_EPOCH
    if epoch is None:
        return os.path.realpath(path)
    if _REALPATH_MEMO_EPOCH != epoch:
        _REALPATH_MEMO.clear()
        _REALPATH_MEMO_EPOCH = epoch
    key = ("" if os.path.isabs(path) else os.getcwd(), path)
    resolved = _REALPATH_MEMO.get(key)
    if resolved is None:
        resolved = os.path.realpath(path)
        _remember_realpath(key, resolved)
    return resolved


def file_content_hash(
    path: str,
    size: int | None = None,
    st: os.stat_result | None = None,
) -> str | None:
    """Return a stable content hash for *path*, or ``None`` if unreadable.

    Every byte, whatever the size: a hash of a few regions of a big file read
    an edit between them as no change. The byte length is folded in too.
    The digest is `cash.bulk_digest`'s, so a big file is hashed on several
    threads.

    Determinism is the contract: given the same bytes and size, this returns
    the same digest at snapshot time and at every later freshness check.

    Remembered, in this process and the next, on the file's stat fields --
    see ``_HASH_MEMO`` for when.

    *st* is the caller's stat of *path*, when it has just taken one.
    """
    memo_key = young_key = None
    hashed_at = time.time()
    try:
        if st is None:
            st = own_stat(path)
        if _stat.S_ISDIR(st.st_mode):
            # ``io.FileIO`` refuses a folder: no digest, without the open
            # (a statement that unpacked 3,400 folders had each one tried).
            return None
        if size is None:
            size = st.st_size
        if (hashed_at - st.st_mtime) > _HASH_MEMO_MIN_AGE_SECONDS and not is_written_in_place(st, path):
            memo_key = _stat_memo_key(path, st, size)
            remembered = _remembered_digest(memo_key)
            if remembered is not None:
                return remembered
        elif (
            _YOUNG_DEPTH
            and _HASH_MEMO_TTL_SECONDS is None
            and st.st_ino
            and _stat.S_ISREG(st.st_mode)
            and not is_written_in_place(st, path)
        ):
            young_key = _stat_memo_key(path, st, size)
            kept = _YOUNG_DIGESTS.get(young_key)
            if kept is not None:
                return kept
    except OSError:
        logger.debug("[FILE_DEP] Could not stat file for freshness: %s", path)
        return None
    try:
        digest = _read_digest(path, size)
    except OSError:
        logger.debug("[FILE_DEP] Could not hash file for freshness: %s", path)
        return None
    if young_key is not None:
        _YOUNG_DIGESTS[young_key] = digest
    if memo_key is not None:
        _HASH_MEMO[memo_key] = (time.monotonic(), digest, HASH_EPOCH)
        table = current_table() if _reuse_seconds(memo_key) is None else None
        if table is not None:
            table.put(memo_key, digest, hashed_at)
    return digest


#: How long a digest is reused under a stat with no file identity (``st_ino``
#: 0): a Windows directory listing's (`stats_from_listings`), which can lag a
#: file edited through another hard link. Bounded, and never kept on disk.
_LISTED_STAT_TTL_SECONDS = 5.0


def _reuse_seconds(memo_key: tuple[str, int, int, int, int, int]) -> float | None:
    """How long a digest remembered under *memo_key* is reused; None: while the key holds."""
    ttl = _HASH_MEMO_TTL_SECONDS
    if not memo_key[2]:
        return _LISTED_STAT_TTL_SECONDS if ttl is None else min(ttl, _LISTED_STAT_TTL_SECONDS)
    return ttl


def digest_later(path: str, size: int, st: os.stat_result) -> str | DeferredDigest | None:
    """The digest of a small file being read, or a `DeferredDigest` that
    takes it when the entry is stored; None when it must be taken now.

    Put off only for a file the digest memo itself would trust (``_HASH_MEMO``):
    settled (last written over ``_HASH_MEMO_MIN_AGE_SECONDS`` ago), with a
    file identity, never mapped to write (`written_in_place`), the memo on
    its stat in use, *st* the stat the read just took. The deferred digest is
    taken only while the file's stat is still *st* -- the memo's whole key,
    change time and identity included -- and is `MOVED` otherwise.
    Remembered already, it is given at once.
    """
    if _HASH_MEMO_TTL_SECONDS is not None or not st.st_ino or not _stat.S_ISREG(st.st_mode):
        return None
    if (time.time() - st.st_mtime) <= _HASH_MEMO_MIN_AGE_SECONDS or is_written_in_place(st, path):
        return None
    key = _stat_memo_key(path, st, size)
    remembered = _remembered_digest(key)
    if remembered is not None:
        return remembered

    def at_store(deferred: DeferredDigest) -> str | None:
        # Untracked: cash's own read and its digest table's write, not the
        # block's (whose write would take the deferred digests again).
        with untracked():
            try:
                now = own_stat(path)
            except OSError:
                return MOVED
            if _stat_memo_key(path, now, now.st_size) != key or is_written_in_place(now, path):
                return MOVED
            deferred.end_stat = now
            return file_content_hash(path, size, now)

    return DeferredDigest(at_store)


def remembered_file_digest(path: str) -> str | None:
    """The digest `file_content_hash` would return for *path* without reading
    it -- remembered in this process or the cache directory -- else None."""
    try:
        st = own_stat(path)
    except OSError:
        return None
    if (time.time() - st.st_mtime) <= _HASH_MEMO_MIN_AGE_SECONDS or is_written_in_place(st, path):
        return None
    return _remembered_digest(_stat_memo_key(path, st, st.st_size))


def _stat_memo_key(path: str, st: os.stat_result, size: int) -> tuple[str, int, int, int, int, int]:
    """A settled file's digest is remembered under ``_HASH_MEMO``'s key.

    st_dev/st_ino: the FILE's identity, not only the path's. A path through
    a re-pointed junction names a different file with the same path, and two
    release copies laid down by one deploy can share size and timestamps
    exactly. Where the filesystem gives an identity, it is the whole key: a
    relative read is recorded under both spellings
    (``FileAccessTracker.track_path``) and was hashed once for each. Where it
    gives none (st_ino 0 -- including every stat a Windows directory listing
    returns), the absolute path stands in for it, so both spellings still
    share.
    """
    return (
        os.path.normcase(os.path.abspath(path)) if not st.st_ino else "",
        st.st_dev,
        st.st_ino,
        size,
        st.st_mtime_ns,
        getattr(st, "st_ctime_ns", 0),
    )


def _remembered_digest(memo_key: tuple[str, int, int, int, int, int]) -> str | None:
    """The digest remembered for a settled file under *memo_key*: this
    process's, else the cache directory's (`digest_table`)."""
    cached = _HASH_MEMO.get(memo_key)
    ttl = _reuse_seconds(memo_key)
    if cached is not None and (
        ttl is None or (cached[2] is not None and cached[2] == HASH_EPOCH) or time.monotonic() - cached[0] < ttl
    ):
        return cached[1]
    table = current_table() if ttl is None else None
    found = table.get(memo_key) if table is not None else None
    if found is None:
        return None
    _HASH_MEMO[memo_key] = (time.monotonic(), found[0], HASH_EPOCH)
    return found[0]


#: How much of a big file one read takes: enough for every hashing thread
#: to get leaves (`bulk_digest.PARALLEL_MIN_BYTES` each).
_HASH_READ_BLOCK = 32 * 1024 * 1024


def _read_digest(path: str, size: int) -> str:
    """`bulk_digest` of the file's bytes as they are now (`file_content_hash`)."""
    hasher = BulkHasher()
    if size < _HASH_READ_CHUNK:
        # A descriptor, not a file object: ``io.FileIO`` raises the ``open``
        # audit event as a read, which went through cash's own read handlers
        # to be dropped there, ~15us of a 2 KB file's digest in a statement
        # reading 6,000. ``os.open`` reports flags, which name no read.
        # ``read(n)`` allocates n bytes before it reads, so a 2 KB file read
        # in 1 MiB chunks cost two 1 MiB allocations: ~400us a file against
        # ~60us reading size + 1 (the +1 finds EOF in the first read; a file
        # that grew is still read to its end).
        with untracked():
            fd = os.open(path, os.O_RDONLY | _O_BINARY)
            try:
                while chunk := os.read(fd, size + 1):
                    hasher.update(chunk)
            finally:
                os.close(fd)
        return hasher.hexdigest()
    # Untracked: cash's own read of a file must not be tracked as a read
    # by the cached call it is checking on behalf of (which then hashed
    # the file a second time to fingerprint that "read").
    with untracked(), io.FileIO(path, "rb") as f:
        block = bytearray(min(size + 1, _HASH_READ_BLOCK))
        view = memoryview(block)
        while True:
            n = f.readinto(block)
            if not n:
                break
            hasher.update(view[:n])
    return hasher.hexdigest()


def walk_dataset_dir(path: str, keep_dir: Any = None) -> Iterable[tuple[str, list[str], list[str]]]:
    """``os.walk(path)`` as a dataset reader walks a folder: into linked
    sub-folders too, each real folder once.

    pandas, pyarrow and polars follow a symlinked (or junctioned) sub-folder
    of a dataset folder; ``os.walk`` by default does not, so the files under
    one were never recorded and an edit there kept serving the old result.
    A link that leads back up the tree is entered once, not forever.
    *keep_dir* filters sub-folder names (hidden ones, for a dataset reader).
    """
    seen: set[Any] = set()
    top = _folder_identity(path)
    if top is not None:
        seen.add(top)
    for root, dirs, files in os.walk(path, followlinks=True):
        kept = []
        for name in sorted(dirs):
            if keep_dir is not None and not keep_dir(name):
                continue
            identity = _folder_identity(os.path.join(root, name))
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            kept.append(name)
        dirs[:] = kept
        yield root, dirs, files


def _folder_identity(path: str) -> Any:
    """Which folder *path* is, through links, or None when it cannot be
    stat'ed. A file system that reports no inode number is told apart by the
    resolved path."""
    try:
        st = own_stat(path)
    except (OSError, ValueError):
        return None
    if st.st_ino:
        return (st.st_dev, st.st_ino)
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return None


def snapshot_file_deps(
    paths: set[str],
    known: dict[str, tuple[Any, str]] | None = None,
    stats: dict[str, tuple[int, int, int] | None] | None = None,
    checked: Mapping[str, os.stat_result] | None = None,
    as_read: Mapping[str, tuple[int, int, int]] | None = None,
    taken: Mapping[str, os.stat_result] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'mtime', 'size', 'hash'}}`` for paths that exist.

    ``hash`` is a content hash (see :func:`file_content_hash`) used as the
    authoritative freshness signal when the size is ambiguous. It is omitted
    only when the file cannot be read at snapshot time.

    *known* maps a path to ``(stat when read, content hash when read[, time
    it was hashed])``: that
    hash is used while the stat is still the same, so the entry describes the
    file as the body read it (see ``FileAccessTracker.read_digests``).

    *stats*, when given, gets each path's ``regular_file_stat`` as this
    snapshot found it: the check that no input moved during the call
    (`FileAccessTracker.inputs_changed_since_read`) compares with it rather
    than taking a second stat of every file.

    *checked* maps a path to the stat that check took just before, when it
    found the file as read (``FileAccessTracker.checked_stats``): taken in
    place of a stat of its own when the digest from the read is still the
    file's (*known*). The entry then describes the file as read, stat and
    digest alike. A file the snapshot must hash itself gets a stat of its
    own, taken right before the hash.

    *as_read* maps a path to the `stat_key` its read took, when the
    snapshot is taken a while after the read (the records of a statement not
    stored, taken when a statement below stores them). A file that moved
    since, whose content is not the one the read took (*known*) or cannot be
    told, is recorded as never fresh, not as it is now, which would vouch
    for a value built from what the read found.

    *taken* maps a path to a stat of the file itself its owner took a
    moment before (the cell run's, ``run_memo.own_stats_this_run``): taken
    in place of a stat of its own where no digest is taken now, which an
    older stat would not describe: a folder, or a file whose digest from
    the read is still its own (*known*). Older than a stat taken here, it
    can only find the file moved since sooner.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    # A relative read is recorded under both spellings
    # (``FileAccessTracker.track_path``). The relative one, when it names the
    # same path from the cwd as an absolute one, takes that one's entry: a
    # cell reading 5,000 files by relative path stat'ed and hashed each again.
    absolute: list[str] = []
    relative: list[str] = []
    isabs = os.path.isabs
    for f in paths:  # each path's spelling looked at once
        (absolute if isabs(f) else relative).append(f)
    todo = absolute + relative if relative else paths
    first_relative = len(absolute)
    for i, f in enumerate(todo):
        if relative and i >= first_relative:
            same = snapshot.get(spelled_absolute(f))
            if same is not None and ("size" in same or (as_read is not None and same.get(_UNRESOLVED_MARKER))):
                # Its twin's moved since read too (*as_read*).
                snapshot[f] = dict(same)
                continue
        st = checked.get(f) if checked else None
        if st is not None and not (known and f in known and known[f][0] == stat_key(st)):
            st = None
        if st is None and taken is not None:
            st = taken.get(f)
            if st is not None and not (
                _stat.S_ISDIR(st.st_mode) or (known and f in known and known[f][0] == stat_key(st))
            ):
                st = None
        try:
            if st is None:
                st = own_stat(f)
                note_links(f, st)
        except (FileNotFoundError, NotADirectoryError):
            # Not there: a file the call deleted, or a read that failed (the
            # absence is recorded on its own channel).
            if stats is not None:
                stats[f] = None
            continue
        except OSError:
            # There, but it cannot be stat'ed (permissions, a path too long):
            # a dependency nobody can check is never fresh. Dropping it would
            # leave the entry with no dependency on the file at all.
            snapshot[f] = {_UNRESOLVED_MARKER: True}
            if stats is not None:
                stats[f] = None
            continue
        if stats is not None:
            stats[f] = stat_key(st) if _stat.S_ISREG(st.st_mode) else None
        entry: dict[str, Any] = {"mtime": st.st_mtime, "size": st.st_size}
        read = known.get(f) if known else None
        hashed_at = None
        if read is not None and read[0] == stat_key(st):
            # Hashed when the body read it; when, if the tracker noted it.
            content_hash = read[1]
            hashed_at = read[2] if len(read) > 2 else None
        else:
            digest_read = read[1] if read is not None and isinstance(read[1], str) else None
            moved = as_read is not None and as_read.get(f, stat_key(st)) != stat_key(st)
            if moved and digest_read is None:
                # No digest of what the read found: nothing tells the content apart.
                snapshot[f] = {_UNRESOLVED_MARKER: True}
                if stats is not None:
                    stats.pop(f, None)
                continue
            hashed_at = time.time()  # before the read: an edit after it is not in the digest
            content_hash = file_content_hash(f, st.st_size, st)
            if as_read is not None and digest_read is not None and content_hash != digest_read:
                snapshot[f] = {_UNRESOLVED_MARKER: True}
                if stats is not None:
                    stats.pop(f, None)
                continue
            if stats is not None:
                # Hashed now, after the stat: an edit landing between the two
                # is in the digest but not in this stat. The check stats it again.
                stats.pop(f, None)
        if content_hash is not None:
            entry["hash"] = content_hash
            if hashed_at is not None:
                entry["hashed_at"] = hashed_at
        # The integer nanoseconds alongside the float. ``st_mtime`` is derived
        # FROM this by CPython, not the other way round, so the float is the
        # lossy one -- and `_unchanged_since_hashed` is an equality test
        # where every lost digit is a window an edit can hide in.
        entry["mtime_ns"] = st.st_mtime_ns
        # On POSIX ``st_ctime`` is the inode CHANGE time: it moves on any write
        # and no ordinary tool restores it, so it catches the edit that `cp -p`,
        # `rsync -a` or `tar -x` hides by putting mtime back. On Windows it is
        # the creation time and buys nothing, which is why it is an extra
        # signal rather than one relied on.
        entry["ctime_ns"] = getattr(st, "st_ctime_ns", 0)
        # Which file it was: two releases laid down by one deploy can share size
        # and timestamps exactly, and a re-pointed junction swaps one for the
        # other under the same path.
        entry["dev"], entry["ino"] = st.st_dev, st.st_ino
        snapshot[f] = entry
    if relative:
        snapshot = {f: snapshot[f] for f in paths if f in snapshot}  # in the order of *paths*, as before
    return snapshot


class _Missing:
    """`MISSING`: a caller's answer that a path is not there."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


#: In place of a stat in `FreshnessMemo.taken`: the path was looked for and
#: is not there (``run_memo`` hands it on only while the folder watch vouches
#: for it). An entry recorded as absent is then fresh with no look of its own.
MISSING: Any = _Missing()


def is_absent_entry(recorded: Any) -> bool:
    """Is *recorded* a snapshot entry for a path found not there?"""
    return isinstance(recorded, Mapping) and bool(recorded.get(_ABSENT_MARKER))


def spelled_absolute(path: str) -> str | None:
    """*path* made absolute from the cwd by its spelling, as a read records
    it, or None when its spelling does not say which file it is: through a
    ``..``, the open goes up from where a linked folder points."""
    cwd = os.getcwd()
    memo = _SPELLED
    if memo.get(_SPELLED_IN) != cwd:
        memo.clear()
        memo[_SPELLED_IN] = cwd
    elif len(memo) > _SPELLED_KEPT:
        memo.clear()
        memo[_SPELLED_IN] = cwd
    try:
        return memo[path]
    except KeyError:
        pass
    spelled = None if ".." in path.replace("\\", "/").split("/") else normalize_path(os.path.abspath(path))
    memo[path] = spelled
    return spelled


#: `spelled_absolute`'s answers in the working directory they were given in
#: (under the key `_SPELLED_IN`): asked of each relative dependency by each
#: check, 361,000 times before one cell.
_SPELLED: dict[Any, str | None] = {}
_SPELLED_IN = object()
_SPELLED_KEPT = 200_000


def snapshot_absent_deps(paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'absent': True}}`` for paths that were looked for and
    were not there.

    Recorded for paths that STILL do not exist at snapshot time: one that
    appeared between the probe and the snapshot is about to be recorded
    properly by whatever read it, and claiming it absent would invalidate the
    entry on every later run.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            if own_exists(path):
                continue
        except (OSError, ValueError):
            continue
        snapshot[path] = {_ABSENT_MARKER: True}
    return snapshot


def _is_present(path: str, kind: Any) -> bool:
    """Is *path* there as *kind* (``file``, ``dir``, anything else: any)?"""
    if kind == "file":
        return os.path.isfile(path)
    if kind == "dir":
        return os.path.isdir(path)
    return os.path.lexists(path)


def snapshot_present_deps(paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'present': kind}}`` for paths that were probed and
    found, and are still there as *kind* at snapshot time (one the call itself
    removed is not an input it found)."""
    snapshot: dict[str, dict[str, Any]] = {}
    for path, kind in paths.items():
        try:
            if not _is_present(path, kind):
                continue
        except (OSError, ValueError):
            continue
        snapshot[path] = {_PRESENT_MARKER: kind}
    return snapshot


def probe_answers(tracker: Any) -> dict[str, str]:
    """The existence checks *tracker* saw, as ``{path: answer}``: ``absent``
    for a path looked for and missing, else what the probe asked the found
    path to be (``file``, ``dir`` or ``any``). Empty when it saw none."""
    if tracker is None:
        return {}
    answers: dict[str, str] = dict(getattr(tracker, "present_files", None) or {})
    for path in getattr(tracker, "absent_files", None) or ():
        answers.setdefault(path, _ABSENT_MARKER)
    return answers


def probe_state_parts(probes: Mapping[str, str] | None) -> list[str]:
    """What the existence checks in *probes* (:func:`probe_answers`) answer
    NOW, one ``?path:0|1`` part each, sorted: the part of a lineage that moves
    when a file a statement only checked for appears or vanishes."""
    parts: list[str] = []
    for path, asked in sorted((probes or {}).items()):
        try:
            there = own_exists(path) if asked == _ABSENT_MARKER else _is_present(path, asked)
        except (OSError, ValueError):
            there = False
        parts.append(f"?{path}:{int(there)}")
    return parts


def snapshot_remote_deps(urls: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return ``{url: {'remote': True, 'hash': token}}`` for remote reads.

    The token is whatever the object's store maintains to describe its content -
    an ETag, a version id, a GCS generation - read with a single metadata
    request (see :class:`cash.remote_source.RemoteFileDataSource`). It stands in
    for the content hash a local file gets, and slots into the same snapshot
    dict, so lineage and freshness treat a remote read exactly like a local one.

    A token that could not be read is recorded as ``unresolved``. That is
    deliberate and fail-**closed**: the entry stays checkable and reports itself
    stale forever, so the call recomputes. Dropping it instead would leave the
    entry with no dependency at all - a permanent silent stale hit, the bug this
    whole mechanism exists to prevent.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for url in urls:
        entry: dict[str, Any] = {_REMOTE_MARKER: True}
        # Checked against the store the read went to: the reader's
        # ``storage_options``, of which the part that names the store is
        # written into the entry and the credentials are not.
        options = read_options(url)
        named = addressing_options(options)
        if named:
            entry["options"] = named
        token = RemoteFileDataSource(url, storage_options=options).state_token()
        if token.startswith("unresolved:"):
            entry["unresolved"] = True
        else:
            entry["hash"] = token
        snapshot[url] = entry
    return snapshot


def snapshot_dependencies(
    paths: Iterable[str],
    urls: Iterable[str] | None = None,
    absent: Iterable[str] | None = None,
    known: dict[str, tuple[Any, str]] | None = None,
    unresolved: Iterable[str] | None = None,
    present: Mapping[str, str] | None = None,
    stats: dict[str, tuple[int, int, int] | None] | None = None,
    checked: Mapping[str, os.stat_result] | None = None,
    as_read: Mapping[str, tuple[int, int, int]] | None = None,
    taken: Mapping[str, os.stat_result] | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot everything a call read — local files and remote objects — as one dict.

    *unresolved* are paths the call read that cannot be stat'ed as recorded
    (``FileAccessTracker.get_unresolved_files``): each is recorded as never
    fresh. *present* are paths it probed and found without reading
    (``FileAccessTracker.get_present_files``). *stats*, *checked*,
    *as_read* and *taken* are `snapshot_file_deps`'s.

    The single entry point both caching subsystems use, so neither has to
    remember to merge two helpers. Local and remote entries answer the same
    question ("did what this call read change since?") and are re-checked
    through the same :func:`file_dep_is_fresh`, which dispatches on the entry
    shape; keeping the *capture* side unified too means the discriminator is
    written in exactly one place.
    """
    if not paths:
        snapshot = {}
    else:
        extra: dict[str, Any] = {}
        if stats is not None:
            extra["stats"] = stats
        if checked:
            extra["checked"] = checked
        if as_read is not None:
            extra["as_read"] = as_read
        if taken:
            extra["taken"] = taken
        snapshot = snapshot_file_deps(set(paths), known, **extra)
    if urls:
        snapshot.update(snapshot_remote_deps(urls))
    if absent:
        # After the present ones, and never over them: a path both probed and
        # read is present, and the read is the stronger record.
        for path, entry in snapshot_absent_deps(absent).items():
            snapshot.setdefault(path, entry)
    if present:
        # Weaker than a read of the same path, which already says it is there.
        for path, entry in snapshot_present_deps(present).items():
            snapshot.setdefault(path, entry)
    for path in unresolved or ():
        snapshot[path] = {_UNRESOLVED_MARKER: True}
    return snapshot


def remote_dep_is_fresh(url: str, stored: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(is_fresh, stale_reason)`` for a remote dependency.

    Fresh exactly when the store reports the same validator it reported when
    the entry was written. Anything else - a moved ETag, an unreadable object,
    a token that could not be resolved in the first place - is stale, so the
    call recomputes rather than serving a result nobody could verify.
    """
    stored_token = stored.get("hash")
    if stored_token is None:
        return False, "remote-unresolved"
    options = read_options(url, stored.get("options") or {})
    current = RemoteFileDataSource(url, storage_options=options).state_token()
    if current != stored_token:
        return False, "remote-changed"
    return True, None


def existing_file_deps(paths: Iterable[str]) -> list[str]:
    """Filter tracked paths down to the ones that actually exist, sorted.

    ``executed_file_deps`` is a strict SUPERSET of the persisted snapshot: the
    tracker records an access *attempt*, so ``importlib.metadata``'s probes for
    optional metadata that legitimately does not exist (``entry_points.txt``,
    ``direct_url.json``, ``pythonXY.zip`` on ``sys.path``) land in it. Those are
    already correctly ignored for freshness, but they leaked into
    the ``%cash_provenance`` display as 100+ phantom venv paths, making a
    variable look like it depended on all of site-packages.

    Call this at DISPLAY time only — never per statement. Filtering at record
    time would put one ``stat`` per tracked path on the hot path to fix what is
    purely a readability problem.
    """
    return sorted({p for p in paths if os.path.exists(p)})


def _timestamps_match(st: os.stat_result, stored: dict[str, Any], field: str) -> bool:
    """Did *field* (``mtime`` / ``ctime``) stay put since the snapshot?

    Exact on the integer nanoseconds: a tolerance is a window an edit can sit
    inside. A snapshot without the nanoseconds cannot prove the file
    unchanged, so it does not match.
    """
    stored_ns = stored.get(f"{field}_ns")
    return stored_ns is not None and getattr(st, f"st_{field}_ns", None) == stored_ns


#: A snapshot of at least this many files shares its verdict
#: (`FreshnessMemo.verdicts`).
VERDICT_MIN_FILES = 64

#: A directory holding at least this many of one lookup's dependencies is read
#: with one listing rather than a stat per file.
LISTING_MIN_FILES = 16


def stats_from_listings(paths: Iterable[str]) -> dict[str, os.stat_result]:
    """``{path: stat}`` from one directory listing per crowded directory. Windows only.

    A stat on Windows opens the file, ~90 us here; re-running statements
    derived from 3,000 files made one per file per statement lookup, most of
    the cell. A listing returns every entry's size and timestamps
    from the directory itself, a few milliseconds for the lot. Elsewhere a
    listing entry's stat IS a stat, so there is nothing to gain.

    What the listing reports can lag the file in one case: a file with
    a second hard link, edited through the other name, until something opens
    this one. ``file_dep_is_fresh`` takes a listed stat only for a file it
    hashes in full, where content decides and a lagging size or time can at
    most reuse a digest for ``_LISTED_STAT_TTL_SECONDS``: a listed stat has
    no file identity (``st_ino`` 0), and a digest taken under one is never
    reused longer, nor kept on disk.

    Listed untracked: the file tracker records a directory listed while it is
    active as a read, and this one is cash's, not the user's.
    """
    if os.name != "nt":
        return {}
    return listing_stats(paths)


#: Who is told of each listing `listing_stats` takes (`listings_told`):
#: ``(before, after)``, innermost last.
_LISTING_WATCHERS: list[tuple[Callable[[str], Any], Callable[[Any, dict[str, os.stat_result]], None]]] = []


@contextlib.contextmanager
def listings_told(before: Callable[[str], Any], after: Callable[[Any, dict[str, os.stat_result]], None]):
    """Within the block, `listing_stats` calls ``before(directory)`` just
    before it lists a directory, and ``after(told, entries)`` once it has,
    with what ``before`` returned and every entry it found
    (``{normcase(name): stat}``, links and junctions left out); with None
    from ``before``, nothing more. The folder watch keeps a crowded
    folder's listing whole this way (`file_watch.FileWatch.keep_listing`):
    a cell asking for 100 more of its 13,000 files listed it again."""
    _LISTING_WATCHERS.append((before, after))
    try:
        yield
    finally:
        _LISTING_WATCHERS.pop()


def listing_begins(directory: str) -> Any:
    """Tell whoever `listings_told` names that *directory* is about to be
    listed; what they answer goes to `listing_ended` (None: nothing to
    tell). Every listing of files whose stats are handed on calls it
    first, before the directory is read."""
    return _LISTING_WATCHERS[-1][0](directory) if _LISTING_WATCHERS else None


def listing_ended(told: Any, entries: dict[str, os.stat_result]) -> None:
    """Tell them every entry the listing `listing_begins` answered *told*
    for found (``{normcase(name): stat}``, links and junctions left out)."""
    if told is not None and _LISTING_WATCHERS:
        _LISTING_WATCHERS[-1][1](told, entries)


def listing_stats(paths: Iterable[str]) -> dict[str, os.stat_result]:
    """`stats_from_listings` on any system: what it takes on Windows.

    A symbolic link or junction is left out, to be stat-ed by itself
    (`run_memo` notes it as followed): its entry's stat is that of what it
    points to, which may lie in a folder the folder watch does not see. It
    was kept as the link's for as long as the link's own folder reported no
    change, and an edit of the file it pointed to was not seen."""
    by_dir: dict[str, dict[str, str]] = {}
    for path in paths:
        directory, name = os.path.split(path)
        by_dir.setdefault(directory, {})[os.path.normcase(name)] = path
    found: dict[str, os.stat_result] = {}
    for directory, wanted in by_dir.items():
        if len(wanted) < LISTING_MIN_FILES:
            continue
        base = None
        told = listing_begins(directory)
        every: dict[str, os.stat_result] | None = {} if told is not None else None
        try:
            with untracked(), os.scandir(directory or ".") as entries:
                for entry in entries:
                    name = os.path.normcase(entry.name)
                    path = wanted.get(name)
                    if (path is None and every is None) or _links_elsewhere(entry):
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    if every is not None:
                        every[name] = st
                    if path is None:
                        continue
                    if not st.st_nlink:
                        if base is None:
                            base = os.path.normcase(os.path.abspath(directory or "."))
                        if _links_of(entry.path, os.path.join(base, name)) != 1:
                            continue  # stat-ed by itself
                    found[path] = st
        except OSError:
            continue  # nothing kept of a listing cut short
        if every is not None:
            listing_ended(told, every)
    return found


#: How many links each file a listing found has, by `_link_key`: a Windows
#: listing gives no link count (``st_nlink`` 0), and its size and times of
#: a file with a second link are those of the last write through this link
#: alone (NTFS keeps them per folder entry). Such a file is stat-ed by
#: itself each time; one with a single link takes one stat, once, unless a
#: snapshot's stat of it already told (`note_links`). A link made after
#: that stat is the documented hard-link gap.
_LISTED_LINKS: dict[str, int] = {}
_LISTED_LINKS_MAX = 1_000_000
#: Only a Windows listing lacks the count; elsewhere nothing is noted.
_NOTE_LINKS = os.name == "nt"


def _link_key(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return os.path.normcase(path)


def _keep_links(key: str, n: int) -> None:
    if len(_LISTED_LINKS) >= _LISTED_LINKS_MAX:
        _LISTED_LINKS.clear()
    _LISTED_LINKS[key] = n


def note_links(path: str, st: os.stat_result) -> None:
    """Keep the link count a stat of *path* by itself gave, for a listing of
    its folder later on (Windows: a listing reports none)."""
    if _NOTE_LINKS and st.st_nlink:
        _keep_links(_link_key(path), st.st_nlink)


def _links_of(path: str, key: str) -> int:
    """The link count of the file a listing found at *path*, kept under
    *key* (0: no stat)."""
    n = _LISTED_LINKS.get(key)
    if n is None:
        try:
            n = own_stat(path).st_nlink
        except (OSError, ValueError):
            return 0
        _keep_links(key, n)
    return n


def listed_links(path: str) -> int | None:
    """The link count a listing learnt for *path*, or None."""
    return _LISTED_LINKS.get(_link_key(path))


def _links_elsewhere(entry: os.DirEntry) -> bool:
    """Is the listing's *entry* a symbolic link or a junction? Read from the
    listing, with no stat of its own."""
    try:
        if entry.is_symlink():
            return True
        is_junction = getattr(entry, "is_junction", None)  # 3.12 on
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _unchanged_since_hashed(st: os.stat_result, stored: dict[str, Any]) -> bool:
    """Is the file as it was when its recorded digest was taken, by its metadata alone?

    Size and modification time to the nanosecond, and on Linux and macOS the
    inode change time, which no tool puts back -- all as recorded -- and the
    file had been left alone for ``_HASH_MEMO_MIN_AGE_SECONDS`` before it was
    hashed. That last condition is what keeps a coarse clock honest: a file
    written moments before its digest was taken can be written again within
    the same timestamp tick, so it is read, as before. One that had settled
    cannot be changed afterwards without its timestamp moving -- unless
    something puts the timestamp back, or writes without moving it (a Windows
    ``np.memmap`` write). A file mapped to write in a process with cash is
    read instead (`written_in_place`); other such writes are not seen: see
    known-limitations.

    Re-reading every input at every check is what this saves -- 1,312 exports
    re-hashed before each cell, about 5 s a cell, and
    again after every restart. The digest still decides whenever the metadata
    moved: a ``touch`` or a byte-identical re-download stays fresh.
    """
    hashed_at = stored.get("hashed_at")
    if hashed_at is None:
        return False  # the tracker did not note when it hashed: the digest decides
    if st.st_mtime_ns != stored.get("mtime_ns"):
        return False
    # The same file, where the stat says which: a directory listing's does not
    # (``st_ino`` 0), and that is the cost of taking one listing for thousands
    # of files rather than a stat each (see ``stats_from_listings``).
    if (
        st.st_ino
        and stored.get("ino") is not None
        and ((st.st_dev, st.st_ino) != (stored.get("dev"), stored.get("ino")))
    ):
        return False
    if os.name != "nt" and stored.get("ctime_ns") != getattr(st, "st_ctime_ns", None):
        return False
    return hashed_at - st.st_mtime > _HASH_MEMO_MIN_AGE_SECONDS


def _note_settled(st: os.stat_result, stored: dict[str, Any], checked_at: float) -> None:
    """Move *stored*'s ``hashed_at`` to *checked_at* when the digest just
    re-taken shows what `_unchanged_since_hashed` needs: the same content,
    under the same metadata, of a file that had settled by then.

    Otherwise a file read within ``_HASH_MEMO_MIN_AGE_SECONDS`` of being
    written would keep the ``hashed_at`` of that first read for the life of
    the entry, and be hashed in full on every check, in every process -- an
    intermediate one pipeline stage writes just before the next reads it,
    say. The snapshot is the entry's own metadata, held by the backend:
    the file backend writes it back with the entry's access stamps, so later
    processes take the stat-only path too.
    """
    if stored.get("hashed_at") is not None and stored["hashed_at"] >= checked_at:
        return
    if _unchanged_since_hashed(st, {**stored, "hashed_at": checked_at}):
        try:
            stored["hashed_at"] = checked_at
        except TypeError:  # a read-only mapping: nothing to remember it in
            pass


def file_dep_is_fresh(
    resolved_path: str,
    stored: dict[str, Any],
    listed: os.stat_result | None = None,
    taken: os.stat_result | None = None,
) -> tuple[bool, str | None]:
    """Return ``(is_fresh, stale_reason)`` for a resolved file dependency.

    *stored* is a snapshot entry (``{'mtime', 'size'[, 'hash']}``). The size is
    checked first — it proves staleness cheaply and we never hash on the
    size-differs path. When nothing else moved either, the file is not read
    (``_unchanged_since_hashed``). Otherwise, when a content hash was recorded,
    the content hash is authoritative: equal content is FRESH even if the mtime
    moved (touch), and differing content is STALE even if the mtime
    is indistinguishable (same-size quick edit). A snapshot with no ``hash``
    (the file could not be read when it was taken) is fresh only while its
    mtime is unchanged to the nanosecond.

    ``stale_reason`` is ``None`` when fresh, else one of
    ``'unreadable' | 'size' | 'content' | 'mtime' |
    'appeared' | 'vanished' | 'unresolved' | 'remote-changed' |
    'remote-unresolved'`` for debug attribution.

    *taken* is a stat of *resolved_path* the caller took already, in place
    of one of its own (`FreshnessMemo.taken`). One with no file identity
    (``st_ino`` 0) came from a directory listing, and stands in as *listed*
    does. `MISSING` says the caller found nothing there: it answers an
    entry recorded as absent.

    **Remote dependencies** short-circuit to :func:`remote_dep_is_fresh`: the
    "path" is a URL, so there is nothing to stat, and the store's own validator
    answers the question instead.
    """
    if taken is MISSING:
        if stored.get(_ABSENT_MARKER):
            return True, None  # still not there, by the caller's look
        taken = None
    if stored.get(_REMOTE_MARKER):
        return remote_dep_is_fresh(resolved_path, stored)
    if stored.get(_ABSENT_MARKER):
        # The call ran with this path missing. It is fresh for exactly as long
        # as the path is still missing; a file that has appeared is a changed
        # input, whether it appeared because someone created it or because the
        # same relative name now resolves into a different directory.
        try:
            return (not own_exists(resolved_path)), "appeared"
        except (OSError, ValueError):
            return False, "appeared"
    if _PRESENT_MARKER in stored:
        # The call found this path and did not read it: fresh while it is
        # still there, as what the probe asked for.
        try:
            return _is_present(resolved_path, stored[_PRESENT_MARKER]), "vanished"
        except (OSError, ValueError):
            return False, "vanished"
    if stored.get(_UNRESOLVED_MARKER):
        return False, "unresolved"
    stored_size = stored.get("size")
    stored_hash = stored.get("hash")
    if taken is not None:
        if taken.st_ino:
            listed = None
            st = taken
        elif listed is None:
            listed = taken
    # A listed stat (``stats_from_listings``) stands in for one where content
    # is the authority, not the size or the time the listing reports.
    if listed is not None and stored_hash is not None:
        st = listed
    elif taken is None or not taken.st_ino:
        try:
            st = own_stat(resolved_path)
        except OSError:
            return False, "unreadable"
    # Cheap size check first — never hash when the size already proves staleness.
    if stored_size is not None and st.st_size != stored_size:
        return False, "size"
    if stored_hash is not None:
        if _unchanged_since_hashed(st, stored) and not is_written_in_place(st, resolved_path):
            return True, None
        checked_at = time.time()  # after the stat, before the read
        cur_hash = file_content_hash(resolved_path, st.st_size, st)
        if cur_hash != stored_hash:
            return False, "content"
        _note_settled(st, stored, checked_at)
        if listed is None:
            note_same_content(resolved_path, st, stored)
        return True, None
    # No content hash: the file was unreadable when the snapshot was taken.
    if not _timestamps_match(st, stored, "mtime"):
        return False, "mtime"
    return True, None


# ---------------------------------------------------------------------------
# Files that belong to the code, not to the data
# ---------------------------------------------------------------------------
#
# A file a function reads from beside ITS OWN CODE -- package data through
# ``importlib.resources``, ``Path(__file__).parent / "ref.csv"``, a release's
# own config -- is part of that install, not a fixed location on disk. Two
# installs of one tool on one machine (a checkout and a wheel), or two releases
# of one job side by side, run byte-identical code, so they share cache keys;
# and the entry's file dependency is recorded at the WRITER's path. Checked
# there, the other install's lookup would validate the writer's file, not its
# own, and take the writer's answer.
#
# So such a dependency is also recorded relative to the code's root, and each
# process checks it against ITS OWN copy. Same bytes in both installs: a hit,
# as before. Different: a miss. Missing: a miss, and the call raises as it
# should instead of being served a value it could never have computed.

_CODE_REL = "code_rel"
_CODE_MOD = "code_mod"


def _loaded_code_module(module_name: str) -> Any:
    """The module of THIS process that *module_name* names, or None.

    *module_name* is an import name (``fxpkg.core``, ``job``). The script this
    process runs answers to the name an import would give it, as its functions'
    cache keys do (`main_namespace_name`): ``python job.py`` is ``job``, and
    ``python other.py`` or ``python -m pytest`` is not.
    """
    mod = sys.modules.get(module_name)
    if mod is not None and module_name not in MAIN_MODULE_NAMES:
        return mod
    for main_name in MAIN_MODULE_NAMES:
        main = sys.modules.get(main_name)
        if main is not None and main_namespace_name(vars(main)) == module_name:
            return main
    return None


_CODE_ROOTS: dict[tuple[str, str], str | None] = {}


def _code_root(module_name: str, mod: Any) -> str | None:
    """The directory holding the top-level package of *module_name*, whose
    file is *mod*'s, resolved.

    Remembered per module name and file for the life of the process: a worker
    checks thousands of dependencies against the same loaded code, and
    resolving its path each time cost a ``realpath`` per check."""
    path = getattr(mod, "__file__", None)
    if not isinstance(path, str) or not path:
        return None
    key = (module_name, path)
    try:
        return _CODE_ROOTS[key]
    except KeyError:
        pass
    root = _resolve_code_root(module_name, path)
    _CODE_ROOTS[key] = root
    return root


def _resolve_code_root(module_name: str, path: str) -> str | None:
    try:
        root = os.path.dirname(os.path.realpath(path))
    except (OSError, ValueError):
        return None
    depth = module_name.count(".")
    if os.path.basename(path) != "__init__.py":
        depth = max(depth - 1, 0)
    for _ in range(depth):
        root = os.path.dirname(root)
    return root


def code_identity_of(module_name: str | None) -> tuple[str, str] | None:
    """``(import name, code root)`` of *module_name* as loaded in this process.

    ``fxpkg.core`` -> ``("fxpkg.core", "…/fxpkg")``. The script run as
    ``__main__`` is named as an import would name it (``python job.py`` ->
    ``("job", <its directory>)``): ``__main__`` names a different script in
    every process, so a dependency recorded against it would be looked for
    beside whatever another process happens to run. None when the module is
    not loaded or has no file.
    """
    if not module_name:
        return None
    mod = sys.modules.get(module_name)
    if mod is None:
        return None
    if module_name in MAIN_MODULE_NAMES:
        module_name = main_namespace_name(vars(mod))
        if module_name in MAIN_MODULE_NAMES:
            return None  # a REPL, ``python -c``, a notebook: no code on disk
    root = _code_root(module_name, mod)
    return (module_name, root) if root else None


def attach_code_relative(
    snapshot: dict[str, dict[str, Any]] | None, module_name: str | None
) -> dict[str, dict[str, Any]] | None:
    """Mark the snapshot entries that live under the writer's code root."""
    if not snapshot or not module_name:
        return snapshot
    identity = code_identity_of(module_name)
    if not identity:
        return snapshot
    name, root = identity
    root_key = os.path.normcase(os.path.normpath(root)).rstrip(os.sep) + os.sep
    for path, entry in snapshot.items():
        if not isinstance(entry, dict) or entry.get(_REMOTE_MARKER):
            continue
        if not os.path.isabs(path):
            continue  # relative twins already re-resolve
        native = os.path.normcase(os.path.normpath(path))
        if not native.startswith(root_key):
            continue
        entry[_CODE_REL] = os.path.relpath(os.path.normpath(path), root).replace(os.sep, "/")
        entry[_CODE_MOD] = name
    return snapshot


def dep_path_for_this_process(path: str, recorded: Any) -> str:
    """Where THIS process would find the file *recorded* describes.

    For a dependency that lives beside the code, that is the same relative
    location under this process's own copy of that code; for everything else,
    and when this process has not loaded that code (``cash inspect``, another
    script, a notebook), the recorded path.
    """
    if not isinstance(recorded, dict):
        return path
    rel = recorded.get(_CODE_REL)
    name = recorded.get(_CODE_MOD)
    if not rel or not isinstance(name, str) or name in MAIN_MODULE_NAMES:
        # An entry written before the script was named (``code_mod:
        # "__main__"``) does not say which script: relocated, it was looked
        # for beside any other process's ``__main__``.
        return path
    mod = _loaded_code_module(name)
    root = _code_root(name, mod) if mod is not None else None
    if not root:
        return path
    here = normalize_path(os.path.join(root, *rel.split("/")))
    if here != path and _same_place(here, path):
        # The recorded path in another spelling (``C:/`` for ``C:\``): the
        # run's stat of the recorded path, and the folder watch's, stand.
        # Spelled anew, every file of an installed package was stat-ed
        # again by each check of each cell on Windows.
        return path
    return here


def code_places(names: Iterable[str]) -> tuple:
    """Where this process's copy of each module of *names* lies
    (`dep_path_for_this_process`): an answer about a file beside the code
    holds while these do."""
    places = []
    for name in sorted(names):
        mod = _loaded_code_module(name)
        places.append((name, _code_root(name, mod) if mod is not None else None))
    return tuple(places)


def _same_place(here: str, path: str) -> bool:
    normcase, normpath = os.path.normcase, os.path.normpath
    return normcase(normpath(here)) == normcase(normpath(path))


# ---------------------------------------------------------------------------
# Is a whole snapshot still fresh?
# ---------------------------------------------------------------------------
#
# The one place that decides where a recorded dependency is looked for and
# whether it still matches. The decorator, call units, restore, the statement
# freshness check, the upstream simulation and the re-execution planner all
# ask through here, so a fix to either half (the relocated install above, the
# moved-project fallbacks in ``resolve_file_dep_path``) reaches all of them.


class FreshnessMemo:
    """Answers reused across :func:`snapshot_is_fresh` calls.

    The caller owns its lifetime and must drop it whenever a file could have
    changed (a statement ran, a new cell started). ``answers`` holds one
    ``(resolved, is_fresh, reason)`` per (path, recorded snapshot): under the
    path for the first snapshot of a path, with a copy of that snapshot under
    the path in ``recorded``, and under ``_memo_key`` for any other
    (:func:`dep_is_fresh`); ``listed`` holds stats taken from directory
    listings (``stats_from_listings``).

    ``taken``, when set, gives for a snapshot's paths the stats its owner
    already took of them and may stand in for a stat of each
    (``{path: stat}``, a path missing from it is stat-ed): the cell run's
    own stats (`cash.notebook.run_memo`), shared by every check of the run.

    The copy is kept apart from the answer: a tuple holding a dict stays
    tracked by the garbage collector, and one per file of a 24,000-file check
    was promoted to the oldest generation every cell, setting off a full
    collection every few cells. An answer of strings and booleans, and the
    copy as one tuple of its keys and values (`_snapshot_copy`), are not
    tracked after a collection; a dict is tracked for good from
    Python 3.14, however few numbers it holds.

    With *lasts* (seconds), an answer holds only while the stat it rests on
    is at most that old when the owner's check began (`begin_check`):
    ``dated`` holds when that stat was taken (``time.monotonic()``), per
    answer, and an older answer is taken again. *taken* is then called with
    the oldest date it may hand a stat of (``taken(paths, not_before)``),
    and *taken_dates* (``{path: when}``) says when each was taken. The
    answers a check gives from stats of its own are dated by when it began,
    and each `snapshot_is_fresh` leaves in ``oldest`` the date of the oldest
    stat its verdict rested on. A whole-memo date by its oldest stat made a
    lookup after a slow pass of 20,000 stats take every one of them again,
    the ones a moment old too.

    ``verdicts``, when set, holds verdicts on whole snapshots shared with
    the owner's other memos (the cell run's, `cash.notebook.run_memo`):
    ``verdicts.known(snap, not_before)`` gives the date of the stats under
    a fresh verdict on an equal snapshot no older than *not_before* (any,
    for None), or None; ``verdicts.note(snap, at)`` keeps one. For a
    snapshot of at least ``VERDICT_MIN_FILES`` files: thirteen checks of
    one cell went through the same 19,000 files.
    """

    __slots__ = (
        "answers",
        "checked_at",
        "dated",
        "lasts",
        "listed",
        "not_before",
        "oldest",
        "recorded",
        "taken",
        "taken_dates",
        "verdicts",
    )

    def __init__(
        self,
        taken: Callable[..., Mapping[str, os.stat_result]] | None = None,
        lasts: float | None = None,
        taken_dates: Callable[[], Mapping[str, float]] | None = None,
        verdicts: Any = None,
    ) -> None:
        self.answers: dict[Any, Any] = {}
        self.recorded: dict[str, Any] = {}
        self.listed: dict[str, os.stat_result] = {}
        self.taken = taken
        self.lasts = lasts
        #: When the stat each answer rests on was taken, by its ``answers`` key.
        self.dated: dict[Any, float] | None = {} if lasts is not None else None
        self.taken_dates = taken_dates
        self.verdicts = verdicts
        #: Set by each check of a dated memo: answers dated before
        #: ``not_before`` are taken again; ``checked_at`` dates the stats it takes.
        self.not_before = self.checked_at = self.oldest = 0.0
        if lasts is not None:
            self.begin_check()
            self.oldest = self.checked_at

    def begin_check(self) -> None:
        """A check of a dated memo begins: it gives answers resting on stats
        taken at most ``lasts`` before now, and dates its own stats by now.
        A listing an earlier check took is not one of them: it is taken
        again, as the stats are."""
        self.checked_at = time.monotonic()
        self.not_before = self.checked_at - (self.lasts or 0.0)
        if self.listed:
            self.listed = {}

    def keep_since(self, not_before: float) -> None:
        """Drop the answers of a dated memo resting on a stat taken before
        *not_before*: they are never given again, and a cell checking files
        for minutes kept every one."""
        dated = self.dated
        if dated is None:
            return
        keep = [k for k, at in dated.items() if at >= not_before]
        if len(keep) == len(dated):
            return
        answers, recorded = self.answers, self.recorded
        self.answers = {k: answers[k] for k in keep}
        self.recorded = {k: recorded[k] for k in keep if k in recorded}
        self.dated = {k: dated[k] for k in keep}


class StaleDep(NamedTuple):
    """The first dependency of a snapshot that is not fresh."""

    path: str
    """The path as recorded."""
    resolved: str | None
    """Where this process looked for it; None when it is nowhere to be found."""
    reason: str
    """A :func:`file_dep_is_fresh` reason code, ``'missing'`` or ``'unrecorded'``."""

    def __str__(self) -> str:
        return f"{self.reason}: {self.resolved or self.path}"


def _memo_key(path: str, recorded: Any) -> Any:
    # A tuple of the snapshot's items, not its repr: the key is built on every
    # check, answered or not, and a sorted repr was 1 s of 120,000 lookups
    # against 5,000 real checks.
    try:
        key = (path, tuple(recorded.items()) if isinstance(recorded, dict) else recorded)
        hash(key)
        return key
    except TypeError:
        try:
            return (path, repr(sorted(recorded.items())) if isinstance(recorded, dict) else repr(recorded))
        except TypeError:
            return None


def _check_dep(
    path: str, recorded: Any, listed: os.stat_result | None, taken: os.stat_result | None = None
) -> tuple[str | None, bool, str | None]:
    if not isinstance(recorded, Mapping):
        return path, False, "unrecorded"
    here = dep_path_for_this_process(path, recorded)
    # Checked where it was recorded first: the stat that decides freshness
    # also says the file is there, so a dependency costs one syscall, not an
    # ``exists`` and then a stat (a re-run of statements derived from 3,000
    # files made 72,000).
    if here != path:
        listed = taken = None
    is_fresh, reason = file_dep_is_fresh(here, recorded, listed, taken)
    if reason != "unreadable" or here != path or recorded.get(_REMOTE_MARKER):
        # A dependency beside the code is checked in THIS install's copy and
        # nowhere else: missing there is a miss.
        return here, is_fresh, reason
    # Not where it was recorded: the project may have moved.
    moved = resolve_file_dep_path(path)
    if moved is None:
        return None, False, "missing"
    if moved == path:
        return path, False, "unreadable"
    return (moved, *file_dep_is_fresh(moved, recorded))


def _snapshot_copy(recorded: Mapping[str, Any]) -> tuple[Any, ...]:
    """*recorded* as one flat tuple, its keys and then its values (one length
    per key count, so two snapshots give equal tuples only when equal, item
    by item in order). Of numbers and strings, the first collection that sees
    it leaves it untracked. A tuple of tuples took a second: a collection
    looks at a tuple before the tuples it holds, which are still tracked."""
    return (*recorded.keys(), *recorded.values())


def dep_is_fresh(
    path: str,
    recorded: Any,
    memo: FreshnessMemo | None = None,
    taken: os.stat_result | None = None,
    taken_at: float | None = None,
) -> tuple[str | None, bool, str | None]:
    """``(resolved, is_fresh, stale_reason)`` for one recorded dependency.

    Looked for where THIS process would read it: this install's own copy for a
    file beside the code (:func:`dep_path_for_this_process`), otherwise the
    recorded path, and when that is gone, the moved-project fallbacks of
    :func:`cash._paths.resolve_file_dep_path`. ``resolved`` is None when the
    file is nowhere to be found (reason ``'missing'``). A recorded value that
    is not a snapshot entry is stale (``'unrecorded'``). *taken* is a stat
    of *path* the caller took already (`FreshnessMemo.taken`), at
    *taken_at* (``time.monotonic()``) when a dated memo asks.
    """
    if memo is None:
        return _check_dep(path, recorded, None, taken)
    # Keyed on the path, with a copy of the snapshot it was answered for: a
    # key holding the snapshot's items was nine tuples per file, kept for the
    # run, and the collections they set off were a large part of a check of
    # 24,000 files.
    # A second snapshot of the same path (an entry recorded at another time)
    # is answered under the full key.
    answers = memo.answers
    dated = memo.dated
    known = memo.recorded.get(path)
    if known is not None:
        if isinstance(recorded, Mapping) and known == _snapshot_copy(recorded):
            if dated is None:
                return answers[path]
            at = dated[path]
            if at >= memo.not_before:
                if at < memo.oldest:
                    memo.oldest = at
                return answers[path]
            key = path  # resting on a stat too old: taken again, in its place
        else:
            key = _memo_key(path, recorded)
            found = answers.get(key) if key is not None else None
            if found is not None:
                if dated is None:
                    return found
                at = dated[key]
                if at >= memo.not_before:
                    if at < memo.oldest:
                        memo.oldest = at
                    return found
    else:
        key = path
    copied = _snapshot_copy(recorded) if key is path and isinstance(recorded, Mapping) else None
    answer = _check_dep(path, recorded, memo.listed.get(path), taken)
    if dated is not None:
        # The stat the answer rests on: the one handed in, or one this
        # check took (its own, or a listing's) since it began.
        at = memo.checked_at if taken is None else (taken_at if taken_at is not None else memo.not_before)
        if at < memo.oldest:
            memo.oldest = at
    if key is path:
        if copied is not None:
            memo.recorded[path] = copied
            answers[path] = answer
            if dated is not None:
                dated[path] = at
    elif key is not None:
        answers[key] = answer
        if dated is not None:
            dated[key] = at
    return answer


def snapshot_is_fresh(
    snap: Mapping[str, Any] | None, memo: FreshnessMemo | None = None
) -> tuple[bool, StaleDep | None]:
    """``(True, None)`` when every dependency in *snap* still matches, else
    ``(False, StaleDep)`` naming the first one that does not.

    *snap* is an ``auto_file_deps`` / ``file_dependencies`` snapshot
    (``{path: entry}``); an empty or missing one is vacuously fresh. Each entry
    is judged by :func:`dep_is_fresh`. With a *memo*, answers and directory
    listings are shared with the caller's other checks for as long as it
    keeps the memo.
    """
    dated = memo is not None and memo.dated is not None
    if dated:
        memo.oldest = memo.checked_at  # lowered by each answer given
    if not snap:
        return True, None
    verdicts = memo.verdicts if memo is not None and len(snap) >= VERDICT_MIN_FILES else None
    checking = snap
    if verdicts is not None:
        at = verdicts.known(snap, memo.not_before if dated else None)
        if at is not None:
            if dated and at < memo.oldest:
                memo.oldest = at
            return True, None
        partly = getattr(verdicts, "partly", None)
        held = partly(snap, memo.not_before if dated else None) if partly is not None else None
        if held is not None:
            # Fresh but for the files the watch no longer vouches for: only
            # those are checked (`run_memo._RunVerdicts.partly`).
            at, unsure = held
            if dated and at < memo.oldest:
                memo.oldest = at
            checking = {p: snap[p] for p in unsure}
    taken: Mapping[str, os.stat_result] = {}
    taken_at: Mapping[str, float] = {}
    if memo is not None and memo.taken is not None:
        # The stats the owner's run took already: each file once, whichever
        # check asks first.
        if dated:
            # Only the ones as recent as the answers may rest on.
            taken = memo.taken(checking, memo.not_before)
            if memo.taken_dates is not None:
                taken_at = memo.taken_dates()
        else:
            taken = memo.taken(checking)
    if memo is not None and not taken and len(checking) >= LISTING_MIN_FILES:
        # Many files, none of them stat-ed by the owner (a check outside a
        # cell run, whose owner takes nothing): read their directories once
        # rather than stat each.
        unlisted = [p for p, s in checking.items() if isinstance(s, dict) and "size" in s and p not in memo.listed]
        if len(unlisted) >= LISTING_MIN_FILES:
            memo.listed.update(stats_from_listings(unlisted))
    # A relative read is recorded under both spellings (`snapshot_file_deps`);
    # its absolute twin's answer, kept here, is the relative one's too.
    answered: dict[str, tuple[str | None, bool, str | None]] = {}
    isabs = os.path.isabs
    for path, recorded in checking.items():
        if isabs(path):
            answer = answered.get(path)
            if answer is None:
                answer = answered[path] = dep_is_fresh(path, recorded, memo, taken.get(path), taken_at.get(path))
            resolved, is_fresh, reason = answer
        elif _fresh_as_absolute_twin(snap, path, recorded, memo, answered, taken, taken_at):
            continue
        else:
            resolved, is_fresh, reason = dep_is_fresh(path, recorded, memo, taken.get(path), taken_at.get(path))
        if not is_fresh:
            return False, StaleDep(path, resolved, reason or "changed")
    if verdicts is not None:
        verdicts.note(snap, memo.oldest if dated else None)
    return True, None


def _recorded_alike(absolute: Any, relative: Mapping[str, Any], name: str | None = None) -> bool:
    """Do the two spellings' entries describe the file alike? The absolute
    one may also say where it lies beside the code (`_CODE_REL`), which
    only decides where this process looks for it: the twin's answer counts
    only when it was found at its own path. One stored with its relative
    spellings folded in (`ALSO_SPELLED`) vouches for relative *name* when
    it lists it: the entry of that name was made from it
    (`unfold_relative_twins`)."""
    if absolute == relative:
        return True
    if not isinstance(absolute, Mapping):
        return False
    spelled = absolute.get(ALSO_SPELLED)
    if spelled is not None:
        return name is not None and name in spelled
    if _CODE_REL not in absolute:
        return False
    return {k: v for k, v in absolute.items() if k not in (_CODE_REL, _CODE_MOD)} == relative


#: In an absolute path's entry of a stored snapshot: the relative spellings
#: the same read was recorded under, each with this entry for its own
#: (`fold_relative_twins`). Read back as entries of their own
#: (`unfold_relative_twins`).
ALSO_SPELLED = "also_spelled"

_NOT_FOR_RELATIVE = (ALSO_SPELLED, _CODE_REL, _CODE_MOD)


def fold_relative_twins(snap: Mapping[str, Any]) -> dict[str, Any]:
    """*snap* with each relative spelling whose absolute twin it records
    alike (`_recorded_alike`) folded into that twin's entry, as a name it
    lists under `ALSO_SPELLED`: a relative read is recorded under both
    spellings (``FileAccessTracker.track_path``), and stored so every
    entry of a folder read by relative path was written twice.

    The twin is the relative spelling made absolute from the working
    directory now, as a check finds it (`spelled_absolute`); a spelling
    with no such twin keeps its own entry. `unfold_relative_twins` gives
    back each folded spelling's own entry, as recorded, so a check under
    another working directory looks for it where its name leads now.
    """
    isabs = os.path.isabs
    folded_into: dict[str, list[str]] = {}
    kept: dict[str, Any] = {}
    refolded = False
    for path, recorded in snap.items():
        if not isabs(path):
            if isinstance(recorded, Mapping):
                twin = spelled_absolute(path)
                if twin is not None and twin != path and _same_entry(snap.get(twin), recorded):
                    folded_into.setdefault(twin, []).append(path)
                    continue
        elif type(recorded) is dict and ALSO_SPELLED in recorded:
            refolded = True  # taken from a stored form: its list is made again
        kept[path] = recorded
    if not folded_into and not refolded:
        return kept
    for path, recorded in kept.items():
        names = folded_into.get(path)
        if names is None and not (type(recorded) is dict and ALSO_SPELLED in recorded):
            continue
        entry = {k: v for k, v in recorded.items() if k != ALSO_SPELLED}
        if names:
            entry[ALSO_SPELLED] = names
        kept[path] = entry
    return kept


def _same_entry(absolute: Any, relative: Mapping[str, Any]) -> bool:
    """Does relative spelling's entry *relative* say what *absolute*, its
    twin's, does but for what only an absolute path's entry says?"""
    if absolute == relative:
        return True
    if not isinstance(absolute, Mapping):
        return False
    return {k: v for k, v in absolute.items() if k not in _NOT_FOR_RELATIVE} == relative


def unfold_relative_twins(snap: Mapping[str, Any]) -> Mapping[str, Any]:
    """*snap* as recorded, from its stored form (`fold_relative_twins`):
    each spelling folded into an absolute path's entry an entry of its own
    again, that entry less what only the absolute one says. *snap* itself
    when nothing was folded into it. The absolute entries are *snap*'s
    own objects (`_note_settled` notes in them)."""
    found = False
    for recorded in snap.values():
        if type(recorded) is dict and ALSO_SPELLED in recorded:
            found = True
            break
    if not found:
        return snap
    out: dict[str, Any] = {}
    for path, recorded in snap.items():
        out[path] = recorded
        names = recorded.get(ALSO_SPELLED) if type(recorded) is dict else None
        if names:
            plain = {k: v for k, v in recorded.items() if k not in _NOT_FOR_RELATIVE}
            for name in names:
                if name not in snap:
                    out[name] = dict(plain)
    return out


def _fresh_as_absolute_twin(
    snap: Mapping[str, Any],
    path: str,
    recorded: Any,
    memo: FreshnessMemo | None,
    answered: dict[str, tuple[str | None, bool, str | None]],
    taken: Mapping[str, os.stat_result],
    taken_at: Mapping[str, float],
) -> bool:
    """Is relative *path* fresh by its absolute twin's answer?

    A relative read is recorded under its name and under that name made
    absolute from the working directory, with the same entry; the check
    stat'ed (and, past a moved stat, hashed) the file once under each:
    48,000 stats of p25's all-hit run for 24,000 documents, where on
    Windows a stat opens the file. While the name still spells the twin's
    path from this working directory and both entries are alike
    (`_recorded_alike`), one file answers both -- when it was found fresh
    at that very path. Any other
    answer leaves *path* to its own check, as before. *answered* holds the
    answers for absolute paths this check has asked.
    """
    if not isinstance(recorded, Mapping):
        return False
    twin = spelled_absolute(path)
    if twin is None or twin == path or not _recorded_alike(snap.get(twin), recorded, path):
        return False
    answer = answered.get(twin)
    if answer is None:
        answer = answered[twin] = dep_is_fresh(twin, snap[twin], memo, taken.get(twin), taken_at.get(twin))
    return answer[0] == twin and answer[1]
