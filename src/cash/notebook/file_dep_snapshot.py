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

**Content-hash freshness (CAS-98, CAS-10).** ``(mtime, size)`` alone is an
ambiguous freshness signal and fails two opposite ways: a touch-only change
(identical content + size, only the mtime bumped) spuriously invalidates
(CAS-98), and a same-size edit under a mtime the coarse check can't tell apart
(sub-resolution / same-second write) is missed (CAS-10). We therefore record a
content hash at snapshot time and treat CONTENT as authoritative whenever the
size matches: the cheap size check runs first (and never hashes on the
size-differs path), and only when the size is equal do we hash to decide.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_file_deps",
    "split_file_dep_value",
    "file_content_hash",
    "file_dep_is_fresh",
]

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
    moved (fixes CAS-98 touch), and differing content is STALE even if the mtime
    is indistinguishable (fixes CAS-10 same-size quick edit). Snapshots written
    before content hashing (no ``hash`` key) fall back to the old mtime
    tolerance so pre-existing cache entries keep working.

    ``stale_reason`` is ``None`` when fresh, else one of
    ``'unreadable' | 'size' | 'content' | 'mtime'`` for debug attribution.
    """
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
        # Size matches: content is the authoritative signal (mtime ignored).
        cur_hash = file_content_hash(resolved_path, st.st_size)
        if cur_hash == stored_hash:
            return True, None
        return False, "content"
    # Legacy snapshot with no content hash: fall back to the mtime tolerance.
    if abs(st.st_mtime - stored_mtime) > 0.01:
        return False, "mtime"
    return True, None
