"""File-dependency snapshots: capturing what a computation read, and
checking later whether it still matches.

A snapshot is ``{path: {'mtime': float, 'size': int, 'hash': str}}`` (plus
remote and absent entries). These are not pure functions: they stat and hash
files, keep short-lived memos of both (``begin_file_state_epoch``,
:class:`FreshnessMemo`), and do their own reads outside every tracker
(``untracked``). One setting shapes the answers, the size above which a file
is hashed by sampling; every entry point takes it as ``full_hash_max``, and
resolves it from the running ``Cash``'s config (:func:`full_hash_max_bytes`)
only when a caller does not pass it. Consumed by:

- ``src/cash/core.py`` — the decorator subsystem, when recording file deps for
  a cached function call.
- :class:`cash.notebook.Restorer` (``restore.py``) — when validating that
  cached file deps still match.
- :class:`cash.notebook.upstream.VirtualLineage` — when checking file
  freshness during upstream simulation.
- :class:`cash.notebook.statement.CacheFreshnessChecker` — the post-execution
  freshness check for statement-level caching.

They live here, outside the notebook's ``statement/`` package, so the
decorator and other callers never import
``cash.notebook.statement.freshness`` for them.

**Content-hash freshness.** ``(mtime, size)`` alone is an
ambiguous freshness signal and fails two opposite ways: a touch-only change
(identical content + size, only the mtime bumped) spuriously invalidates
, and a same-size edit under a mtime the coarse check can't tell apart
(sub-resolution / same-second write) is missed. We therefore record a
content hash at snapshot time and treat CONTENT as authoritative whenever the
size matches: the cheap size check runs first (and never hashes on the
size-differs path), and only when the size is equal do we hash to decide --
unless nothing else moved either: a file whose timestamps and identity are as
recorded, and that had settled before it was hashed, is not read again
(``_unchanged_since_hashed``).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import stat as _stat
import sys
import time
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from cash._active import active_config
from cash._memo import FILE_DIGESTS, LruMemo
from cash._paths import normalize_path, resolve_file_dep_path
from cash.remote_source import RemoteFileDataSource, addressing_options, read_options
from cash.tracking.tracker_context import untracked

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_file_deps",
    "snapshot_remote_deps",
    "snapshot_absent_deps",
    "snapshot_present_deps",
    "snapshot_dependencies",
    "existing_file_deps",
    "file_content_hash",
    "file_dep_is_fresh",
    "dep_is_fresh",
    "snapshot_is_fresh",
    "FreshnessMemo",
    "StaleDep",
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

# Files up to this size are hashed in full; larger files are sampled
# deterministically (head / middle / tail) so hashing a multi-GB parquet on
# every freshness check stays cheap. The sample is a function of the file size
# only, so snapshot-time and check-time hashes are computed identically.
#
# 256 MiB, not the 8 MiB this shipped with: the sampled regime has a hole (see
# ``file_dep_is_fresh``) that serves wrong answers, and the memo below makes the full hash a once-per-window cost rather than a
# per-check one -- which is what makes covering the ordinary CSV affordable.
# Not 64 MiB either: an 80 MiB .npy written through np.memmap on Windows
# changes neither its size nor any timestamp, so above the cap only content
# can see it. Content is now read
# only when the metadata moved (``_unchanged_since_hashed``), so that write is
# not seen below the cap either, until the file is touched -- a documented
# limitation; the cap still decides how a file whose metadata moved is hashed.
_HASH_FULL_MAX_BYTES_DEFAULT = 256 * 1024 * 1024  # 256 MiB


def full_hash_max_bytes() -> int:
    """Largest file hashed IN FULL rather than sampled, for a caller that did
    not pass its own ``full_hash_max``.

    Configurable (``file_hash_full_max_bytes``) because the sampled regime has
    a hole that serves wrong answers: a same-size
    interior edit with the mtime restored is invisible to both the sample and
    the mtime backstop. Raising this closes it, and the price is real and
    measurable -- a full hash costs about 0.72 ms per MiB, on every freshness
    check, i.e. on every cache HIT that depends on the file.

    Resolved per call rather than at import so ``cash.configure(...)`` takes
    effect; falls back to the default when the config layer is unavailable.
    """
    try:
        value = int(active_config().file_hash_full_max_bytes)
    except Exception:  # teardown, or a config that cannot load
        logger.debug("[SNAPSHOT] no config for the full-hash threshold; using the default", exc_info=True)
        return _HASH_FULL_MAX_BYTES_DEFAULT
    return value if value > 0 else _HASH_FULL_MAX_BYTES_DEFAULT


_HASH_SAMPLE_REGION_BYTES = 256 * 1024  # 256 KiB per sampled region
_HASH_READ_CHUNK = 1024 * 1024  # 1 MiB streaming chunk


#: Digests already computed this process, keyed by the file's identity AND its
#: stat fields: ``(path, st_dev, st_ino, size, mtime_ns, ctime_ns)``.
#:
#: File dependencies propagate, so one burst of cached calls checks the same
#: inputs many times (50 files x 2 MiB cost 168 ms per hit without it); with
#: the memo later checks are one ``stat`` each. Any write moves ``mtime``, and
#: on POSIX ``ctime`` too; on Windows a same-size edit that RESTORES the mtime
#: leaves every key field identical. Three rules bound that (a fully-hashed
#: file must still catch it: ``test_same_size_edit_under_identical_mtime_
#: invalidates``):
#:
#: 1. Only a file UNTOUCHED for ``_HASH_MEMO_MIN_AGE_SECONDS`` is memoized --
#:    a file written moments ago may still be being written.
#: 2. A digest is reused for ``_HASH_MEMO_TTL_SECONDS`` from when it was
#:    computed (not refreshed on use; one second expired entries mid-pass).
#:    Within that window such an edit is not seen -- a documented limitation
#:    ("an edit that keeps size and timestamps, in a running process"), and
#:    one that never reaches a stored entry, whose fingerprint is taken when
#:    the body reads the file. Re-hashing on every call cost ~144 ms per
#:    iteration of a loop over a 200 MB input.
#: 3. In a notebook a digest also holds for the rest of the CELL RUN
#:    (``begin_file_state_epoch`` .. ``end_file_state_epoch``): a cell over
#:    thousands of files outlasts the window on its own, and re-hashing them
#:    for every derived statement cost 19-108 s a cell. The next cell run
#:    falls back to the window; between cells only the window applies.
_HASH_MEMO: LruMemo[tuple[str, int, int, int, int, int], tuple[float, str, int | None]] = LruMemo(FILE_DIGESTS)
_HASH_MEMO_TTL_SECONDS = 5.0
_HASH_MEMO_MIN_AGE_SECONDS = 10.0
#: The current cell run's number, or None between runs.
HASH_EPOCH: int | None = None
_EPOCH_COUNT = 0
#: A cell run started inside another (a cell that calls
#: `get_ipython().run_cell(...)`) is the same run.
_EPOCH_DEPTH = 0


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
_REALPATH_MEMO_EPOCH: int | None = None


def _remember_realpath(key: tuple[str, str], resolved: str) -> None:
    if len(_REALPATH_MEMO) < 65536:
        _REALPATH_MEMO[key] = resolved
        # A resolved path resolves to itself, and callers hand it back in
        # both spellings: the tracker records ``normalize_path`` of it, and
        # the lineage component resolves that record again.
        _REALPATH_MEMO[("", resolved)] = resolved
        _REALPATH_MEMO[("", normalize_path(resolved))] = resolved


def realpath_of_read_this_run(path: str) -> tuple[str, os.stat_result | None]:
    """:func:`realpath_this_run` for a file about to be read, and its stat when
    that is what answered.

    ``realpath`` costs two ``_getfinalpathname`` calls per file on Windows, and
    a folder read paid them for all 5,030 files in one directory: a quarter of
    what cash added to the read. A regular file that is not
    itself a link or reparse point resolves to its directory's real path plus
    its name, so the directory is resolved once; the ``lstat`` that shows it is
    such a file is also the stat the read needs. Anything else -- a link, a
    missing file, a short ``~`` name -- is resolved in full.
    """

    if HASH_EPOCH is None:
        return os.path.realpath(path), None
    key = ("" if os.path.isabs(path) else os.getcwd(), path)
    if _REALPATH_MEMO_EPOCH == HASH_EPOCH:
        resolved = _REALPATH_MEMO.get(key)
        if resolved is not None:
            return resolved, None
    absolute = os.path.abspath(path)
    parent, name = os.path.split(absolute)
    if name and name not in (".", "..") and "~" not in name and parent != absolute:
        try:
            st = os.lstat(absolute)
        except (OSError, ValueError):
            st = None
        if st is not None and _stat.S_ISREG(st.st_mode) and not getattr(st, "st_file_attributes", 0) & _REPARSE_POINT:
            resolved = os.path.join(realpath_this_run(parent), name)
            _remember_realpath(key, resolved)
            return resolved, st
    return realpath_this_run(path), None


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
    full_hash_max: int | None = None,
    st: os.stat_result | None = None,
) -> str | None:
    """Return a stable content hash for *path*, or ``None`` if unreadable.

    Small files (``<= full_hash_max_bytes()``) are hashed in full. Larger files
    are sampled at three deterministic, size-derived offsets (head, middle,
    tail) so the cost is bounded while still catching the overwhelming majority
    of edits. The byte length is folded into the digest so a change that leaves
    every sampled region untouched but alters the size still differs (the size
    check catches that first anyway — this is belt-and-suspenders).

    Determinism is the contract: given the same bytes and size, this returns
    the same digest at snapshot time and at every later freshness check.

    Memoized per process on the file's stat fields — see ``_HASH_MEMO`` for what
    that costs and what it saves.

    *full_hash_max* lets a caller that checks many files resolve the threshold
    once instead of per file. That is not a micro-optimisation: reading it from
    the config costs a full config merge, which walks the directory tree looking
    for a project marker, and profiling a 50-dependency hit found 7,000
    ``os.path.exists`` calls and 130 ms spent there -- three times the hashing
    it was guarding.

    *st* is the caller's stat of *path*, when it has just taken one.
    """
    memo_key = None
    try:
        if st is None:
            st = os.stat(path)
        if size is None:
            size = st.st_size
        memoizable = (time.time() - st.st_mtime) > _HASH_MEMO_MIN_AGE_SECONDS
        if memoizable:
            # st_dev/st_ino: the FILE's identity, not only the path's. A path
            # through a re-pointed junction names a different file with the same
            # path, and two release copies laid down by one deploy can share
            # size and timestamps exactly. Where
            # the filesystem gives an identity, it is the whole key: a relative
            # read is recorded under both spellings (``FileAccessTracker.track_path``)
            # and was hashed once for each. Where it gives none (st_ino 0 --
            # including every stat a Windows directory listing returns), the
            # absolute path stands in for it, so both spellings still share.
            memo_key = (
                os.path.normcase(os.path.abspath(path)) if not st.st_ino else "",
                st.st_dev,
                st.st_ino,
                size,
                st.st_mtime_ns,
                getattr(st, "st_ctime_ns", 0),
            )
            cached = _HASH_MEMO.get(memo_key)
            if cached is not None and (
                (cached[2] is not None and cached[2] == HASH_EPOCH)
                or time.monotonic() - cached[0] < _HASH_MEMO_TTL_SECONDS
            ):
                return cached[1]
    except OSError:
        logger.debug("[FILE_DEP] Could not stat file for freshness: %s", path)
        return None
    try:
        if full_hash_max is None:
            full_hash_max = full_hash_max_bytes()
        h = hashlib.sha256()
        h.update(str(size).encode("ascii"))
        # Untracked: cash's own read of a file must not be tracked as a read
        # by the cached call it is checking on behalf of (which then hashed
        # the file a second time to fingerprint that "read").
        with untracked(), io.FileIO(path, "rb") as f:
            if size <= full_hash_max:
                # ``FileIO.read(n)`` allocates n bytes before it reads, so a
                # 2 KB file read in 1 MiB chunks cost two 1 MiB allocations:
                # ~400us a file against ~60us reading size + 1 (the +1 finds EOF
                # in the first read; a file that grew is still read to its end).
                want = min(size + 1, _HASH_READ_CHUNK)
                for chunk in iter(lambda: f.read(want), b""):
                    h.update(chunk)
            else:
                half = _HASH_SAMPLE_REGION_BYTES // 2
                offsets = (
                    0,
                    max(0, size // 2 - half),
                    max(0, size - _HASH_SAMPLE_REGION_BYTES),
                )
                for off in offsets:
                    f.seek(off)
                    h.update(f.read(_HASH_SAMPLE_REGION_BYTES))
        digest = h.hexdigest()
        if memo_key is not None:
            _HASH_MEMO[memo_key] = (time.monotonic(), digest, HASH_EPOCH)
        return digest
    except OSError:
        logger.debug("[FILE_DEP] Could not hash file for freshness: %s", path)
        return None


def snapshot_file_deps(
    paths: set[str],
    known: dict[str, tuple[Any, str]] | None = None,
    full_hash_max: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'mtime', 'size', 'hash'}}`` for paths that exist.

    ``hash`` is a content hash (see :func:`file_content_hash`) used as the
    authoritative freshness signal when the size is ambiguous. It is omitted
    only when the file cannot be read at snapshot time.

    *known* maps a path to ``(stat when read, content hash when read[, time
    it was hashed])``: that
    hash is used while the stat is still the same, so the entry describes the
    file as the body read it (see ``FileAccessTracker.read_digests``).

    *full_hash_max* is the caller's ``file_hash_full_max_bytes``; resolved
    from the running config when omitted (:func:`full_hash_max_bytes`).
    """
    snapshot: dict[str, dict[str, Any]] = {}
    if full_hash_max is None:
        full_hash_max = full_hash_max_bytes()
    for f in paths:
        try:
            st = os.stat(f)
        except (FileNotFoundError, NotADirectoryError):
            # Not there: a file the call deleted, or a read that failed (the
            # absence is recorded on its own channel).
            continue
        except OSError:
            # There, but it cannot be stat'ed (permissions, a path too long):
            # a dependency nobody can check is never fresh. Dropping it would
            # leave the entry with no dependency on the file at all.
            snapshot[f] = {_UNRESOLVED_MARKER: True}
            continue
        entry: dict[str, Any] = {"mtime": st.st_mtime, "size": st.st_size}
        read = known.get(f) if known else None
        hashed_at = None
        if read is not None and read[0] == (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", 0)):
            # Hashed when the body read it; when, if the tracker noted it.
            content_hash = read[1]
            hashed_at = read[2] if len(read) > 2 else None
        else:
            hashed_at = time.time()  # before the read: an edit after it is not in the digest
            content_hash = file_content_hash(f, st.st_size, full_hash_max, st)
        if content_hash is not None:
            entry["hash"] = content_hash
            if hashed_at is not None:
                entry["hashed_at"] = hashed_at
        # The integer nanoseconds alongside the float. ``st_mtime`` is derived
        # FROM this by CPython, not the other way round, so the float is the
        # lossy one -- and the sampled comparison below is an equality test
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
        entry["sampled"] = st.st_size > full_hash_max
        snapshot[f] = entry
    return snapshot


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
            if os.path.exists(path):
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
    full_hash_max: int | None = None,
    unresolved: Iterable[str] | None = None,
    present: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot everything a call read — local files and remote objects — as one dict.

    *unresolved* are paths the call read that cannot be stat'ed as recorded
    (``FileAccessTracker.get_unresolved_files``): each is recorded as never
    fresh. *present* are paths it probed and found without reading
    (``FileAccessTracker.get_present_files``).

    The single entry point both caching subsystems use, so neither has to
    remember to merge two helpers. Local and remote entries answer the same
    question ("did what this call read change since?") and are re-checked
    through the same :func:`file_dep_is_fresh`, which dispatches on the entry
    shape; keeping the *capture* side unified too means the discriminator is
    written in exactly one place.
    """
    snapshot = snapshot_file_deps(set(paths), known, full_hash_max) if paths else {}
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

    Exact on the integer nanoseconds, because
    this is the SAMPLED regime's backstop: the hash covers three regions of
    the file, so an interior edit is caught by the timestamp or not at all,
    and a tolerance is a window the edit can sit inside. Measured on a 65 MiB
    file, one byte rewritten in place outside every sampled region: an edit
    landing 8.02 ms after the recorded mtime was served FRESH under the old
    0.01 s tolerance, and the boundary sat exactly where the constant says
    (9.50 ms missed, 11.03 ms caught).

    The tolerance is unreachable in practice on any filesystem worth the name.
    Measured, 400 one-byte appends: ext4 and tmpfs gave all 400 writes a
    distinct timestamp (smallest gap 3.2 us), NTFS 54 distinct (0.389 ms), a
    9p translated mount 1.6 ms. Only the 1-2 second tier (FAT32, ext3, HFS+)
    is coarser than 10 ms, and there no comparison at any resolution helps --
    the answer there is to stay under ``file_hash_full_max_bytes`` so the
    content hash decides and timestamps are never consulted.

    A snapshot without the nanoseconds cannot prove the file unchanged, so it
    does not match.
    """
    stored_ns = stored.get(f"{field}_ns")
    return stored_ns is not None and getattr(st, f"st_{field}_ns", None) == stored_ns


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

    What the listing reports can lag the file in one case measured: a file with
    a second hard link, edited through the other name, until something opens
    this one. ``file_dep_is_fresh`` takes a listed stat only for a file it
    hashes in full, where content decides and a lagging size or time can at
    most reuse a digest within the window it already reuses one.

    Listed untracked: the file tracker records a directory listed while it is
    active as a read, and this one is cash's, not the user's.
    """
    if os.name != "nt":
        return {}
    by_dir: dict[str, dict[str, str]] = {}
    for path in paths:
        directory, name = os.path.split(path)
        by_dir.setdefault(directory, {})[os.path.normcase(name)] = path
    found: dict[str, os.stat_result] = {}
    for directory, wanted in by_dir.items():
        if len(wanted) < LISTING_MIN_FILES:
            continue
        try:
            with untracked(), os.scandir(directory or ".") as entries:
                for entry in entries:
                    path = wanted.get(os.path.normcase(entry.name))
                    if path is not None:
                        try:
                            found[path] = entry.stat()
                        except OSError:
                            pass
        except OSError:
            continue
    return found


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
    ``np.memmap`` write). Those are not seen: see known-limitations.

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


def file_dep_is_fresh(
    resolved_path: str,
    stored: dict[str, Any],
    full_hash_max: int | None = None,
    listed: os.stat_result | None = None,
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

    **Sampled-file backstop.** For files larger than ``full_hash_max_bytes()``
    the content hash only covers three fixed head/middle/tail regions (see
    :func:`file_content_hash`), so a same-size edit *outside* those regions
    produces an identical hash and would silently pass as FRESH — serving stale
    data. For that regime only, mtime is re-instated as an additional signal:
    a matching sampled hash is trusted only when the mtime also matches. Any
    real in-place edit bumps mtime, so the stale read is caught; the sole cost
    is that merely *touching* a large file forces a (safe) spurious recompute.
    "ignore mtime" reasoning holds only when the hash is authoritative,
    i.e. for fully-hashed (<= cap) files, which keep the touch-tolerant path.

    ``stale_reason`` is ``None`` when fresh, else one of
    ``'unreadable' | 'size' | 'content' | 'hash-mode' | 'mtime' |
    'mtime-sampled' | 'ctime-sampled' | 'appeared' | 'vanished' |
    'unresolved' | 'remote-changed' | 'remote-unresolved'`` for debug
    attribution.

    **Remote dependencies** short-circuit to :func:`remote_dep_is_fresh`: the
    "path" is a URL, so there is nothing to stat, and the store's own validator
    answers the question instead.
    """
    if stored.get(_REMOTE_MARKER):
        return remote_dep_is_fresh(resolved_path, stored)
    if stored.get(_ABSENT_MARKER):
        # The call ran with this path missing. It is fresh for exactly as long
        # as the path is still missing; a file that has appeared is a changed
        # input, whether it appeared because someone created it or because the
        # same relative name now resolves into a different directory.
        try:
            return (not os.path.exists(resolved_path)), "appeared"
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
    if full_hash_max is None and listed is not None:
        full_hash_max = full_hash_max_bytes()
    # A listed stat (``stats_from_listings``) stands in for one only where the
    # file is hashed in full: there content is the authority, not the size or
    # the time the listing reports. A sampled file keeps its timestamps as a
    # backstop, so it gets a stat of its own.
    if (
        listed is not None
        and stored_hash is not None
        and listed.st_size <= full_hash_max
        and (stored_size is None or stored_size <= full_hash_max)
    ):
        st = listed
    else:
        try:
            st = os.stat(resolved_path)
        except OSError:
            return False, "unreadable"
    # Cheap size check first — never hash when the size already proves staleness.
    if stored_size is not None and st.st_size != stored_size:
        return False, "size"
    if stored_hash is not None:
        if _unchanged_since_hashed(st, stored):
            return True, None
        if full_hash_max is None:
            full_hash_max = full_hash_max_bytes()
        cur_hash = file_content_hash(resolved_path, st.st_size, full_hash_max, st)
        if cur_hash != stored_hash:
            # Recorded in one regime and checked in the other -- the size is
            # the same, so `file_hash_full_max_bytes` moved across it. The
            # two digests are not comparable, and "content changed" blamed
            # the data for a setting. The snapshot says which.
            if stored.get("sampled", False) != (st.st_size > full_hash_max):
                return False, "hash-mode"
            return False, "content"
        # Full-hashed file: content is authoritative, mtime ignored.
        if st.st_size <= full_hash_max:
            return True, None
        # Sampled file: the hash only covers head/middle/tail, so trust it only
        # when the timestamps also match — otherwise a same-size edit outside
        # the sampled regions would be served stale: a 9 MiB CSV with one
        # amount field rewritten in place and the mtime restored.
        #
        # mtime alone is restorable -- that is exactly what `cp -p`, `rsync -a`
        # and `tar -x` do. On POSIX ``st_ctime`` is not: it is the inode change
        # time, it moves on any write, and no ordinary tool puts it back. On
        # Windows it is the creation time and does not move, so this closes the
        # hole on Linux and macOS and narrows nothing there -- nor does NTFS move
        # LastWriteTime or ChangeTime for a write through np.memmap. The remaining
        # Windows case is documented, and ``file_hash_full_max_bytes`` closes it
        # on any platform at the cost of hashing the whole file on every check
        # (measured ~0.72 ms/MiB).
        if not _timestamps_match(st, stored, "mtime"):
            return False, "mtime-sampled"
        if not _timestamps_match(st, stored, "ctime"):
            return False, "ctime-sampled"
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
# and the entry's file dependency was recorded at the WRITER's path. The other
# install's lookup validated the writer's file, found it unchanged, and served
# the writer's answer: a tool served another install's exchange rates, and a
# rollback served the newer release's report.
#
# So such a dependency is also recorded relative to the code's root, and each
# process checks it against ITS OWN copy. Same bytes in both installs: a hit,
# as before. Different: a miss. Missing: a miss, and the call raises as it
# should instead of being served a value it could never have computed.

_CODE_REL = "code_rel"
_CODE_MOD = "code_mod"


def code_root_of(module_name: str | None) -> str | None:
    """The directory holding the top-level package of *module_name*, resolved.

    ``fxpkg.core`` -> ``…/fxpkg``; a script run as ``__main__`` -> its own
    directory. None when the module is not loaded or has no file.
    """

    if not module_name:
        return None
    mod = sys.modules.get(module_name)
    path = getattr(mod, "__file__", None)
    if not path:
        return None
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


def attach_code_relative(
    snapshot: dict[str, dict[str, Any]] | None, module_name: str | None
) -> dict[str, dict[str, Any]] | None:
    """Mark the snapshot entries that live under the writer's code root."""
    if not snapshot or not module_name:
        return snapshot
    root = code_root_of(module_name)
    if not root:
        return snapshot
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
        entry[_CODE_MOD] = module_name
    return snapshot


def dep_path_for_this_process(path: str, recorded: Any) -> str:
    """Where THIS process would find the file *recorded* describes.

    For a dependency that lives beside the code, that is the same relative
    location under this process's own copy of the code; for everything else,
    the recorded path.
    """
    if not isinstance(recorded, dict):
        return path
    rel = recorded.get(_CODE_REL)
    if not rel:
        return path
    root = code_root_of(recorded.get(_CODE_MOD))
    if not root:
        return path
    return normalize_path(os.path.join(root, *rel.split("/")))


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
    ``(resolved, is_fresh, reason)`` per (path, recorded snapshot);
    ``listed`` holds stats taken from directory listings (``stats_from_listings``).
    """

    __slots__ = ("answers", "listed")

    def __init__(self) -> None:
        self.answers: dict[Any, tuple[str | None, bool, str | None]] = {}
        self.listed: dict[str, os.stat_result] = {}


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
    path: str, recorded: Any, full_hash_max: int | None, listed: os.stat_result | None
) -> tuple[str | None, bool, str | None]:
    if not isinstance(recorded, Mapping):
        return path, False, "unrecorded"
    here = dep_path_for_this_process(path, recorded)
    # Checked where it was recorded first: the stat that decides freshness
    # also says the file is there, so a dependency costs one syscall, not an
    # ``exists`` and then a stat (a re-run of statements derived from 3,000
    # files made 72,000).
    is_fresh, reason = file_dep_is_fresh(here, recorded, full_hash_max, listed if here == path else None)
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
    return (moved, *file_dep_is_fresh(moved, recorded, full_hash_max))


def dep_is_fresh(
    path: str,
    recorded: Any,
    full_hash_max: int | None = None,
    memo: FreshnessMemo | None = None,
) -> tuple[str | None, bool, str | None]:
    """``(resolved, is_fresh, stale_reason)`` for one recorded dependency.

    Looked for where THIS process would read it: this install's own copy for a
    file beside the code (:func:`dep_path_for_this_process`), otherwise the
    recorded path, and when that is gone, the moved-project fallbacks of
    :func:`cash._paths.resolve_file_dep_path`. ``resolved`` is None when the
    file is nowhere to be found (reason ``'missing'``). A recorded value that
    is not a snapshot entry is stale (``'unrecorded'``).
    """
    key = _memo_key(path, recorded) if memo is not None else None
    if key is not None:
        known = memo.answers.get(key)
        if known is not None:
            return known
    answer = _check_dep(path, recorded, full_hash_max, memo.listed.get(path) if memo is not None else None)
    if key is not None:
        memo.answers[key] = answer
    return answer


def snapshot_is_fresh(
    snap: Mapping[str, Any] | None, memo: FreshnessMemo | None = None, full_hash_max: int | None = None
) -> tuple[bool, StaleDep | None]:
    """``(True, None)`` when every dependency in *snap* still matches, else
    ``(False, StaleDep)`` naming the first one that does not.

    *snap* is an ``auto_file_deps`` / ``file_dependencies`` snapshot
    (``{path: entry}``); an empty or missing one is vacuously fresh. Each entry
    is judged by :func:`dep_is_fresh`. With a *memo*, answers and directory
    listings are shared with the caller's other checks for as long as it
    keeps the memo. *full_hash_max* as for :func:`snapshot_file_deps`.
    """
    if not snap:
        return True, None
    if full_hash_max is None:
        full_hash_max = full_hash_max_bytes()
    if memo is not None and len(snap) >= LISTING_MIN_FILES:
        # Many files: read their directories once rather than stat each.
        unlisted = [p for p, s in snap.items() if isinstance(s, dict) and "size" in s and p not in memo.listed]
        if len(unlisted) >= LISTING_MIN_FILES:
            memo.listed.update(stats_from_listings(unlisted))
    for path, recorded in snap.items():
        resolved, is_fresh, reason = dep_is_fresh(path, recorded, full_hash_max, memo)
        if not is_fresh:
            return False, StaleDep(path, resolved, reason or "changed")
    return True, None
