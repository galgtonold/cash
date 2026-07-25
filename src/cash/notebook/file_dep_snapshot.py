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

import hashlib
import logging
import os
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_file_deps",
    "snapshot_remote_deps",
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

# Files up to this size are hashed in full; larger files are sampled
# deterministically (head / middle / tail) so hashing a multi-GB parquet on
# every freshness check stays cheap. The sample is a function of the file size
# only, so snapshot-time and check-time hashes are computed identically.
_HASH_FULL_MAX_BYTES = 8 * 1024 * 1024        # 8 MiB
_HASH_SAMPLE_REGION_BYTES = 256 * 1024        # 256 KiB per sampled region
_HASH_READ_CHUNK = 1024 * 1024                # 1 MiB streaming chunk


def file_content_hash(path: str, size: int | None = None) -> str | None:
    """Return a stable content hash for *path*, or ``None`` if unreadable.

    Small files (``<= _HASH_FULL_MAX_BYTES``) are hashed in full. Larger files
    are sampled at three deterministic, size-derived offsets (head, middle,
    tail) so the cost is bounded while still catching the overwhelming majority
    of edits. The byte length is folded into the digest so a change that leaves
    every sampled region untouched but alters the size still differs (the size
    check catches that first anyway — this is belt-and-suspenders).

    Determinism is the contract: given the same bytes and size, this returns
    the same digest at snapshot time and at every later freshness check.
    """
    try:
        if size is None:
            size = os.path.getsize(path)
        h = hashlib.sha256()
        h.update(str(size).encode("ascii"))
        with open(path, "rb") as f:
            if size <= _HASH_FULL_MAX_BYTES:
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
        return h.hexdigest()
    except OSError:
        logger.debug("[FILE_DEP] Could not hash file for freshness: %s", path)
        return None


def snapshot_file_deps(paths: set[str]) -> dict[str, dict[str, Any]]:
    """Return ``{path: {'mtime', 'size', 'hash'}}`` for paths that exist.

    ``hash`` is a content hash (see :func:`file_content_hash`) used as the
    authoritative freshness signal when the size is ambiguous. It is omitted
    only when the file cannot be read at snapshot time.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for f in paths:
        try:
            st = os.stat(f)
        except OSError:
            continue
        entry: dict[str, Any] = {"mtime": st.st_mtime, "size": st.st_size}
        content_hash = file_content_hash(f, st.st_size)
        if content_hash is not None:
            entry["hash"] = content_hash
        snapshot[f] = entry
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


def file_dep_is_fresh(resolved_path: str, stored: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(is_fresh, stale_reason)`` for a resolved file dependency.

    *stored* is a snapshot entry (``{'mtime', 'size'[, 'hash']}``). The size is
    checked first — it proves staleness cheaply and we never hash on the
    size-differs path. When the size matches and a content hash was recorded,
    the content hash is authoritative: equal content is FRESH even if the mtime
    moved (touch), and differing content is STALE even if the mtime
    is indistinguishable (same-size quick edit). Snapshots written
    before content hashing (no ``hash`` key) fall back to the old mtime
    tolerance so pre-existing cache entries keep working.

    **Sampled-file backstop.** For files larger than ``_HASH_FULL_MAX_BYTES``
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
    ``'unreadable' | 'size' | 'content' | 'mtime' | 'mtime-sampled' |
    'remote-changed' | 'remote-unresolved'`` for debug attribution.

    **Remote dependencies** short-circuit to :func:`remote_dep_is_fresh`: the
    "path" is a URL, so there is nothing to stat, and the store's own validator
    answers the question instead.
    """
    if isinstance(stored, dict) and stored.get(_REMOTE_MARKER):
        return remote_dep_is_fresh(resolved_path, stored)
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
        cur_hash = file_content_hash(resolved_path, st.st_size)
        if cur_hash != stored_hash:
            return False, "content"
        # Full-hashed file: content is authoritative, mtime ignored.
        if st.st_size <= _HASH_FULL_MAX_BYTES:
            return True, None
        # Sampled file: the hash only covers head/middle/tail, so trust it only
        # when the mtime also matches — otherwise a same-size edit outside the
        # sampled regions would be served stale. A real edit bumps mtime.
        if abs(st.st_mtime - stored_mtime) > 0.01:
            return False, "mtime-sampled"
        return True, None
    # Legacy snapshot with no content hash: fall back to the mtime tolerance.
    if abs(st.st_mtime - stored_mtime) > 0.01:
        return False, "mtime"
    return True, None
