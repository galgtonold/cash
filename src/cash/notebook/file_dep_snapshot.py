"""File-dependency snapshot utilities used across cache subsystems.

Pure helpers for capturing and unpacking file metadata snapshots
(``{path: {'mtime': float, 'size': int, 'hash': str}}``). Consumed by:

- ``src/cash/core.py`` — the decorator subsystem, when recording file deps for
  a cached function call.
- :class:`cash.notebook.Restorer` (``restore.py``) — when validating that
  cached file deps still match.
- :class:`cash.notebook.upstream.VirtualLineage` — when checking file
  freshness during upstream simulation.
- :class:`cash.notebook.statement.CacheFreshnessChecker` — the post-execution
  freshness check for statement-level caching.

These helpers used to live alongside ``CacheFreshnessChecker`` in
``cache_freshness.py``. They were extracted before the ``statement/`` package
was formed (ADR-011) so callers outside the statement subsystem don't end up
reaching into ``cash.notebook.statement.freshness`` for what is really a
pure utility.

**Content-hash freshness.** ``(mtime, size)`` alone is an
ambiguous freshness signal and fails two opposite ways: a touch-only change
(identical content + size, only the mtime bumped) spuriously invalidates
, and a same-size edit under a mtime the coarse check can't tell apart
(sub-resolution / same-second write) is missed. We therefore record a
content hash at snapshot time and treat CONTENT as authoritative whenever the
size matches: the cheap size check runs first (and never hashes on the
size-differs path), and only when the size is equal do we hash to decide.
"""

from __future__ import annotations

import contextvars
import hashlib
import io
import logging
import os
import time
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_file_deps",
    "snapshot_remote_deps",
    "snapshot_absent_deps",
    "snapshot_dependencies",
    "existing_file_deps",
    "split_file_dep_value",
    "file_content_hash",
    "file_dep_is_fresh",
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
# see, because a file that is never opened produces no read to track. A
# round-16 tester found what that costs: a cached function reading `cfg.txt`
# by relative name in directory A, then in B (which has no such file), then in
# A again, was served B's answer in A as a hit, with no warning. The B run
# recorded NO dependencies at all, so its entry looked valid everywhere.
_ABSENT_MARKER = "absent"

# Files up to this size are hashed in full; larger files are sampled
# deterministically (head / middle / tail) so hashing a multi-GB parquet on
# every freshness check stays cheap. The sample is a function of the file size
# only, so snapshot-time and check-time hashes are computed identically.
#
# 256 MiB, not the 8 MiB this shipped with: the sampled regime has a hole (see
# ``file_dep_is_fresh``) that cost two round-16 testers a wrong answer each,
# and the memo below makes the full hash a once-per-window cost rather than a
# per-check one -- which is what makes covering the ordinary CSV affordable.
# Raised from 64 MiB in round 19: an 80 MiB .npy written through np.memmap on
# Windows changes neither its size nor any timestamp, so above the cap only
# content can see it, and a stale answer came back 3 of 3.
_HASH_FULL_MAX_BYTES_DEFAULT = 256 * 1024 * 1024      # 256 MiB


#: The config of the `Cash` instance whose call is running, set by its
#: wrapper. The threshold used to come from the process-wide singleton, or a
#: fresh `get_config()` when there was none, so `Cash(file_hash_full_max_bytes=
#: ...)` on an instance of your own was silently ignored.
ACTIVE_CONFIG: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "cash_active_config", default=None)


def _full_hash_max_bytes() -> int:
    """Largest file hashed IN FULL rather than sampled.

    Configurable (``file_hash_full_max_bytes``) because the sampled regime has
    a hole that cost two round-16 testers a wrong answer each: a same-size
    interior edit with the mtime restored is invisible to both the sample and
    the mtime backstop. Raising this closes it, and the price is real and
    measurable -- a full hash costs about 0.72 ms per MiB, on every freshness
    check, i.e. on every cache HIT that depends on the file.

    Resolved per call rather than at import so ``cash.configure(...)`` takes
    effect; falls back to the default when the config layer is unavailable.
    """
    try:
        # The LIVE config first: ``cash.configure(...)`` updates the singleton's
        # config in place, while ``get_config()`` re-merges env and TOML from
        # disk and would not see it. Falls through to the merged one for a
        # process that has not built a Cash yet.
        config = ACTIVE_CONFIG.get()
        if config is None:
            import cash
            config = getattr(getattr(cash, "_global_cash", None), "config", None)
        if config is None:
            from cash.config import get_config
            config = get_config()
        value = int(config.file_hash_full_max_bytes)
    except Exception:  # noqa: BLE001 - teardown, or a config that cannot load
        return _HASH_FULL_MAX_BYTES_DEFAULT
    return value if value > 0 else _HASH_FULL_MAX_BYTES_DEFAULT
_HASH_SAMPLE_REGION_BYTES = 256 * 1024        # 256 KiB per sampled region
_HASH_READ_CHUNK = 1024 * 1024                # 1 MiB streaming chunk


#: Digests already computed this process, keyed by the file's identity AND its
#: stat fields: ``(path, st_dev, st_ino, size, mtime_ns, ctime_ns)``.
#:
#: Freshness is checked once per cached call, and file dependencies PROPAGATE --
#: an aggregate that calls ten cached functions inherits their inputs -- so a
#: pipeline over fifty files re-read and re-hashed all fifty on every one of
#: those calls. Measured before this memo, warm page cache: 50 files x 2 MiB
#: cost 168 ms per hit, and 50 x 32 MiB cost 156 ms (per-file overhead
#: dominates once you have that many). Ten such hits in a run paid it ten times.
#: With the memo the second and later checks are one ``stat`` each.
#:
#: What the key buys, and what it does not. Any write moves ``mtime``, so an
#: ordinary edit re-hashes immediately. On Linux and macOS ``ctime`` moves on
#: any write whatever the tool does, so the memo cannot be fooled at all. On
#: Windows a same-size edit that RESTORES the mtime leaves every key field
#: identical, and the memo would repeat the digest it already has -- the
#: sampled regime's blind spot (see ``file_dep_is_fresh``), extended to
#: fully-hashed files.
#:
#: Two rules keep that from mattering, and an existing regression test is what
#: forced them: ``test_same_size_edit_under_identical_mtime_invalidates`` pins
#: that a fully-hashed file catches exactly this edit, and a memo keyed on stat
#: fields alone broke it.
#:
#: 1. A file is memoized only once it has been UNTOUCHED for a while
#:    (``_HASH_MEMO_MIN_AGE_SECONDS``). A file written moments ago is the one
#:    plausibly still being written; an input from this morning is not. This is
#:    what keeps "write it, then read it twice in one run" honest.
#: 2. A digest is reused for a few seconds only (``_HASH_MEMO_TTL_SECONDS``),
#:    which is what the memo is actually for: one burst of related calls -- ten
#:    aggregates hitting the same fifty inputs, milliseconds apart. A long-lived
#:    worker re-hashes each file at most once per window, so what the memo
#:    borrows is bounded to that window rather than the life of the process.
#:
#: Five seconds, not one: the timestamp is when the digest was COMPUTED and is
#: not refreshed on use, so a window shorter than the pass itself expires
#: entries mid-pass and re-hashes them. Measured with a one-second window, a
#: 50-file 400 MiB pass fell back to 151 ms from 49 ms.
#:
#: The window is a documented limitation, not an oversight (known-limitations:
#: "an edit that keeps size and timestamps, in a running process"). Round 20
#: found it -- on Windows an ``np.memmap`` write, or a write with the mtime put
#: back, leaves every key field alone, and a call within the window got the
#: old result in that process -- and closing it was tried: re-hashing on every
#: call made a loop over a 200 MB input pay ~144 ms per iteration, minutes per
#: thousand calls, to catch an edit that is seen five seconds later anyway and
#: never reaches a stored entry (the fingerprint an entry is stored with is
#: taken when the body reads the file). Reverted.
_HASH_MEMO: dict[tuple[str, int, int, int, int, int], tuple[float, str]] = {}
_HASH_MEMO_MAX = 4096
_HASH_MEMO_TTL_SECONDS = 5.0
_HASH_MEMO_MIN_AGE_SECONDS = 10.0


def file_content_hash(
    path: str, size: int | None = None, full_hash_max: int | None = None,
) -> str | None:
    """Return a stable content hash for *path*, or ``None`` if unreadable.

    Small files (``<= _full_hash_max_bytes()``) are hashed in full. Larger files
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
    """
    memo_key = None
    try:
        st = os.stat(path)
        if size is None:
            size = st.st_size
        memoizable = (time.time() - st.st_mtime) > _HASH_MEMO_MIN_AGE_SECONDS
        if memoizable:
            # st_dev/st_ino: the FILE's identity, not only the path's. A path
            # through a re-pointed junction names a different file with the same
            # path, and two release copies laid down by one deploy can share
            # size and timestamps exactly (CAS-108's reproduction did).
            memo_key = (path, st.st_dev, st.st_ino, size, st.st_mtime_ns,
                        getattr(st, "st_ctime_ns", 0))
            cached = _HASH_MEMO.get(memo_key)
            if cached is not None and (
                time.monotonic() - cached[0]
            ) < _HASH_MEMO_TTL_SECONDS:
                return cached[1]
    except OSError:
        logger.debug("[FILE_DEP] Could not stat file for freshness: %s", path)
        return None
    try:
        if full_hash_max is None:
            full_hash_max = _full_hash_max_bytes()
        h = hashlib.sha256()
        h.update(str(size).encode("ascii"))
        # FileIO, not `open`: cash's own read of a file must not be tracked as
        # a read by the cached call it is checking on behalf of (which then
        # hashed the file a second time to fingerprint that "read").
        with io.FileIO(path, "rb") as f:
            if size <= full_hash_max:
                for chunk in iter(lambda: f.read(_HASH_READ_CHUNK), b""):
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
        if memo_key is not None and len(_HASH_MEMO) < _HASH_MEMO_MAX:
            _HASH_MEMO[memo_key] = (time.monotonic(), digest)
        return digest
    except OSError:
        logger.debug("[FILE_DEP] Could not hash file for freshness: %s", path)
        return None


def snapshot_file_deps(
    paths: set[str], known: dict[str, tuple[Any, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'mtime', 'size', 'hash'}}`` for paths that exist.

    ``hash`` is a content hash (see :func:`file_content_hash`) used as the
    authoritative freshness signal when the size is ambiguous. It is omitted
    only when the file cannot be read at snapshot time.

    *known* maps a path to ``(stat when read, content hash when read)``: that
    hash is used while the stat is still the same, so the entry describes the
    file as the body read it (see ``FileAccessTracker.read_digests``).
    """
    snapshot: dict[str, dict[str, Any]] = {}
    full_hash_max = _full_hash_max_bytes()
    for f in paths:
        try:
            st = os.stat(f)
        except OSError:
            continue
        entry: dict[str, Any] = {"mtime": st.st_mtime, "size": st.st_size}
        read = known.get(f) if known else None
        if read is not None and read[0] == (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", 0)):
            content_hash = read[1]
        else:
            content_hash = file_content_hash(f, st.st_size, full_hash_max)
        if content_hash is not None:
            entry["hash"] = content_hash
        # The integer nanoseconds alongside the float. ``st_mtime`` is derived
        # FROM this by CPython, not the other way round, so the float is the
        # lossy one -- and the sampled comparison below is an equality test
        # where every lost digit is a window an edit can hide in.
        entry["mtime_ns"] = st.st_mtime_ns
        if st.st_size > full_hash_max:
            # Sampled regime only, where mtime is load-bearing rather than a
            # convenience -- see ``file_dep_is_fresh``. On POSIX ``st_ctime``
            # is the inode CHANGE time: it moves on any write and no ordinary
            # tool restores it, so it catches the edit that `cp -p`, `rsync -a`
            # or `tar -x` hides by putting mtime back. On Windows it is the
            # creation time and this buys nothing, which is why it is recorded
            # as an extra signal rather than relied on.
            entry["ctime"] = st.st_ctime
            entry["ctime_ns"] = getattr(st, "st_ctime_ns", 0)
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
    from cash.remote_source import RemoteFileDataSource

    snapshot: dict[str, dict[str, Any]] = {}
    for url in urls:
        entry: dict[str, Any] = {_REMOTE_MARKER: True}
        token = RemoteFileDataSource(url).state_token()
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
) -> dict[str, dict[str, Any]]:
    """Snapshot everything a call read — local files and remote objects — as one dict.

    The single entry point both caching subsystems use, so neither has to
    remember to merge two helpers. Local and remote entries answer the same
    question ("did what this call read change since?") and are re-checked
    through the same :func:`file_dep_is_fresh`, which dispatches on the entry
    shape; keeping the *capture* side unified too means the discriminator is
    written in exactly one place.
    """
    snapshot = snapshot_file_deps(set(paths), known) if paths else {}
    if urls:
        snapshot.update(snapshot_remote_deps(urls))
    if absent:
        # After the present ones, and never over them: a path both probed and
        # read is present, and the read is the stronger record.
        for path, entry in snapshot_absent_deps(absent).items():
            snapshot.setdefault(path, entry)
    return snapshot


def remote_dep_is_fresh(url: str, stored: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(is_fresh, stale_reason)`` for a remote dependency.

    Fresh exactly when the store reports the same validator it reported when
    the entry was written. Anything else - a moved ETag, an unreadable object,
    a token that could not be resolved in the first place - is stale, so the
    call recomputes rather than serving a result nobody could verify.
    """
    from cash.remote_source import RemoteFileDataSource

    stored_token = stored.get("hash")
    if stored_token is None:
        return False, "remote-unresolved"
    current = RemoteFileDataSource(url).state_token()
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


def split_file_dep_value(value: dict[str, Any]) -> tuple[float, int | None]:
    """Return ``(mtime, size_or_None)`` from a file-dep snapshot dict.

    Snapshots are written as ``{'mtime': float, 'size': int, 'hash': str}``;
    ``size`` may be absent for callers that only record mtime, in which case
    the size check is skipped downstream. ``hash`` is read separately by
    :func:`file_dep_is_fresh`.
    """
    return float(value.get('mtime', 0.0)), value.get('size')


#: How far two timestamps may differ and still count as the same one, for a
#: snapshot that recorded only the float seconds. Inherited from the
#: pre-content-hash check, where it absorbed storage jitter across a whole
#: comparison; it is FOUR ORDERS OF MAGNITUDE wider than any filesystem's
#: resolution, so a snapshot carrying integer nanoseconds does not use it.
_LEGACY_TIMESTAMP_TOLERANCE_SECONDS = 0.01


def _timestamps_match(st: os.stat_result, stored: Any, field: str) -> bool:
    """Did *field* (``mtime`` / ``ctime``) stay put since the snapshot?

    Exact on the integer nanoseconds when the snapshot recorded them, because
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

    Falls back to the tolerance for a snapshot written before this was
    recorded, so existing entries keep their meaning rather than invalidating
    en masse on upgrade.
    """
    if not isinstance(stored, dict):
        return True
    stored_ns = stored.get(f"{field}_ns")
    if stored_ns is not None:
        live_ns = getattr(st, f"st_{field}_ns", None)
        if live_ns is not None:
            return live_ns == stored_ns
    stored_seconds = stored.get(field)
    if stored_seconds is None:
        return True
    live = getattr(st, f"st_{field}")
    return abs(live - stored_seconds) <= _LEGACY_TIMESTAMP_TOLERANCE_SECONDS


def file_dep_is_fresh(
    resolved_path: str, stored: dict[str, Any], full_hash_max: int | None = None,
) -> tuple[bool, str | None]:
    """Return ``(is_fresh, stale_reason)`` for a resolved file dependency.

    *stored* is a snapshot entry (``{'mtime', 'size'[, 'hash']}``). The size is
    checked first — it proves staleness cheaply and we never hash on the
    size-differs path. When the size matches and a content hash was recorded,
    the content hash is authoritative: equal content is FRESH even if the mtime
    moved (touch), and differing content is STALE even if the mtime
    is indistinguishable (same-size quick edit). Snapshots written
    before content hashing (no ``hash`` key) fall back to the old mtime
    tolerance so pre-existing cache entries keep working.

    **Sampled-file backstop.** For files larger than ``_full_hash_max_bytes()``
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
    'mtime-sampled' | 'ctime-sampled' | 'remote-changed' |
    'remote-unresolved'`` for debug attribution.

    **Remote dependencies** short-circuit to :func:`remote_dep_is_fresh`: the
    "path" is a URL, so there is nothing to stat, and the store's own validator
    answers the question instead.
    """
    if isinstance(stored, dict) and stored.get(_REMOTE_MARKER):
        return remote_dep_is_fresh(resolved_path, stored)
    if isinstance(stored, dict) and stored.get(_ABSENT_MARKER):
        # The call ran with this path missing. It is fresh for exactly as long
        # as the path is still missing; a file that has appeared is a changed
        # input, whether it appeared because someone created it or because the
        # same relative name now resolves into a different directory.
        try:
            return (not os.path.exists(resolved_path)), "appeared"
        except (OSError, ValueError):
            return False, "appeared"
    stored_mtime, stored_size = split_file_dep_value(stored)
    stored_hash = stored.get("hash") if isinstance(stored, dict) else None
    try:
        st = os.stat(resolved_path)
    except OSError:
        return False, "unreadable"
    # Cheap size check first — never hash when the size already proves staleness.
    if stored_size is not None and st.st_size != stored_size:
        return False, "size"
    if stored_hash is not None:
        if full_hash_max is None:
            full_hash_max = _full_hash_max_bytes()
        cur_hash = file_content_hash(resolved_path, st.st_size, full_hash_max)
        if cur_hash != stored_hash:
            # Recorded in one regime and checked in the other -- the size is
            # the same, so `file_hash_full_max_bytes` moved across it. The
            # two digests are not comparable, and "content changed" blamed
            # the data for a setting (round 20). Only sampled snapshots
            # record a ctime.
            recorded_sampled = "ctime_ns" in stored or "ctime" in stored
            if recorded_sampled != (st.st_size > full_hash_max):
                return False, "hash-mode"
            return False, "content"
        # Full-hashed file: content is authoritative, mtime ignored.
        if st.st_size <= full_hash_max:
            return True, None
        # Sampled file: the hash only covers head/middle/tail, so trust it only
        # when the timestamps also match — otherwise a same-size edit outside
        # the sampled regions would be served stale. Measured, twice, by two
        # round-16 testers independently: a 9 MiB CSV with one amount field
        # rewritten in place and the mtime restored was served from cache with
        # the old total, 5/5 and 3/3.
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
    # Legacy snapshot with no content hash: fall back to the mtime tolerance.
    if abs(st.st_mtime - stored_mtime) > _LEGACY_TIMESTAMP_TOLERANCE_SECONDS:
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
# the writer's answer. Round 17 measured both: a tool served another install's
# exchange rates, and a rollback served the newer release's report (CAS-108).
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
    import sys

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


def attach_code_relative(snapshot: dict[str, dict[str, Any]] | None,
                         module_name: str | None) -> dict[str, dict[str, Any]] | None:
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
            continue                      # relative twins already re-resolve
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
    return os.path.join(root, *rel.split("/")).replace(os.sep, "/")
