"""The digest of big raw data: array buffers, table columns, file contents.

The leaves are hashed with BLAKE3 from the ``blake3`` package, a dependency
of cash, or with SHA-256 from ``hashlib`` where it cannot load. Both
are collision resistant. SHA-256 is always there and runs on the CPU's SHA
instructions where it has them (x86 SHA-NI, ARMv8 SHA2), about 2 GB/s a
core; on a CPU without them about 0.4 GB/s. BLAKE3 runs on the vector
units every x86-64 and ARM64 CPU has, 4-6 GB/s a core, so on a machine
without SHA instructions it hashes a big array ten times faster. MD5 and
SHA-1 are faster than SHA-256, and broken.

What makes it fast on big data is the shape as much as the algorithm: the
data is cut into fixed 1 MiB leaves, each leaf is hashed on its own, and
the leaf digests are hashed together with the total length (`BulkHasher`).
The leaves of one big buffer are hashed on several threads at once --
hashlib and blake3 both release the GIL while they hash -- so a 1 GB array
takes a fraction of the time one core needs. Each leaf is one call on one
thread: blake3's own threads (``max_threads``) are never asked for, so the
threads a digest uses are `hashing_threads`, never that times another
pool. The result depends only on the bytes and the algorithm, never on the
number of threads or how the data was handed over in pieces.

`fold_buffer` folds one buffer into a running SHA-256 (a key, a frame's
digest): a small buffer goes in as it is, a bigger one as its tree digest,
each behind a tag and its length so the two can never read alike. Where
small ends is `DIRECT_MAX_BYTES`: a whole leaf under SHA-256, 16 KiB under
BLAKE3, past which a BLAKE3 tree digest costs less than SHA-256 over the
bytes themselves.

The tags carry this scheme's algorithm and version (``TAG``). Every digest
built on it differs from one taken with the other algorithm or an older
cash, so an entry written by an install with another scheme is never
matched -- it misses and is computed again, never served for other data.
Which algorithm is fixed when this module is imported, once per process.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

__all__ = [
    "ALGORITHM",
    "BulkHasher",
    "LEAF_BYTES",
    "SCHEME",
    "TAG",
    "bulk_digest",
    "fold_buffer",
    "hashing_threads",
    "leaf_hasher",
]

try:  # a dependency; a build that does not load here falls back to SHA-256
    from blake3 import blake3 as _blake3
except ImportError:  # not installed, or a build that does not load here
    _blake3 = None

#: The leaf size. Fixed: it is part of what a digest means.
LEAF_BYTES = 1_048_576  # 1 MiB


def _scheme() -> tuple[str, bytes, int, Any]:
    """``(ALGORITHM, TAG, DIRECT_MAX_BYTES, leaf_hasher)`` for this
    install: BLAKE3 when ``blake3`` imports, else SHA-256."""
    if _blake3 is not None:
        return "blake3", b"cash-bulk-blake3-tree-1\x00", 16 * 1024, _blake3
    return "sha256", b"cash-bulk-sha256-tree-1\x00", LEAF_BYTES, hashlib.sha256


#: ``ALGORITHM``, the algorithm leaves are hashed with (``"blake3"`` or
#: ``"sha256"``, fixed for the process); ``TAG``, the scheme's algorithm
#: and version at the head of every tree digest; ``DIRECT_MAX_BYTES``, the
#: biggest buffer `fold_buffer` folds in whole rather than as its tree
#: digest; ``leaf_hasher``, a new hasher of the leaf algorithm (``update``,
#: ``digest``, ``hexdigest``), single-threaded: blake3's own threads
#: (``max_threads``) are never asked for.
ALGORITHM, TAG, DIRECT_MAX_BYTES, leaf_hasher = _scheme()
#: ``TAG`` as text, for records that note which scheme their digests are.
SCHEME = TAG.rstrip(b"\x00").decode("ascii")
#: Below this many bytes in one go, the leaves are hashed on the calling
#: thread: starting a thread costs about what hashing 100 KB does.
PARALLEL_MIN_BYTES = 8 << 20
#: The most threads one digest uses: past this, memory bandwidth, not
#: cores, is what limits it.
MAX_THREADS = 8
#: The same under BLAKE3, which reads 4-6 GB/s a core: four cores already
#: read memory about as fast as it comes.
BLAKE3_MAX_THREADS = 4


def hashing_threads() -> int:
    """How many threads a big digest may use: the CPUs this process may
    run on, at most `MAX_THREADS` (`BLAKE3_MAX_THREADS` under BLAKE3)."""
    count = getattr(os, "process_cpu_count", None)  # 3.13+
    if count is not None:
        n = count()
    else:
        try:
            n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            n = os.cpu_count()
    return max(1, min(n or 1, BLAKE3_MAX_THREADS if ALGORITHM == "blake3" else MAX_THREADS))


_THREADS = hashing_threads()


def _leaf_digests(view: memoryview, count: int) -> list[bytes]:
    """The digests of the first *count* whole leaves of *view*, in order,
    on several threads when there is enough to share out."""
    threads = min(_THREADS, count * LEAF_BYTES // PARALLEL_MIN_BYTES)
    if threads < 2:
        return [leaf_hasher(view[i * LEAF_BYTES : (i + 1) * LEAF_BYTES]).digest() for i in range(count)]
    out: list[Any] = [None] * count
    errors: list[BaseException] = []

    def run(first: int, stop: int) -> None:
        try:
            for i in range(first, stop):
                out[i] = leaf_hasher(view[i * LEAF_BYTES : (i + 1) * LEAF_BYTES]).digest()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            errors.append(exc)

    bounds = [count * k // threads for k in range(threads + 1)]
    workers = []
    for k in range(1, threads):
        worker = threading.Thread(target=run, args=(bounds[k], bounds[k + 1]), name="cash-digest", daemon=True)
        try:
            worker.start()
        except RuntimeError:  # no new threads (interpreter shutting down): do it here
            run(bounds[k], bounds[k + 1])
            continue
        workers.append(worker)
    run(bounds[0], bounds[1])
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]
    return out


def _as_bytes_view(data: Any) -> memoryview:
    view = data if isinstance(data, memoryview) else memoryview(data)
    if view.ndim != 1 or view.itemsize != 1 or view.format not in ("B", "b", "c"):
        view = view.cast("B") if view.c_contiguous else memoryview(view.tobytes())
    return view


class BulkHasher:
    """The tree digest of a stream of bytes handed over in any pieces.

    ``update`` as often as needed, then ``digest``. The digest is
    ``SHA-256(TAG, total length, digest of each 1 MiB leaf in order)``,
    whatever the pieces were; the leaves' digests are `ALGORITHM`'s.
    """

    __slots__ = ("_leaves", "_partial", "_total")

    def __init__(self) -> None:
        self._leaves: list[bytes] = []
        self._partial = bytearray()
        self._total = 0

    def update(self, data: Any) -> None:
        view = _as_bytes_view(data)
        n = len(view)
        if not n:
            return
        self._total += n
        start = 0
        if self._partial:
            start = min(n, LEAF_BYTES - len(self._partial))
            self._partial += view[:start]
            if len(self._partial) < LEAF_BYTES:
                return
            self._leaves.append(leaf_hasher(self._partial).digest())
            self._partial = bytearray()
        whole = (n - start) // LEAF_BYTES
        if whole:
            self._leaves.extend(_leaf_digests(view[start:], whole))
        rest = start + whole * LEAF_BYTES
        if rest < n:
            self._partial += view[rest:]

    def digest(self) -> bytes:
        root = hashlib.sha256(TAG)
        root.update(self._total.to_bytes(8, "little"))
        root.update(b"".join(self._leaves))
        if self._partial:
            root.update(leaf_hasher(self._partial).digest())
        return root.digest()

    def hexdigest(self) -> str:
        return self.digest().hex()


def bulk_digest(data: Any) -> bytes:
    """`BulkHasher`'s digest of one buffer, without copying its last,
    partial leaf aside first."""
    view = _as_bytes_view(data)
    n = len(view)
    whole = n // LEAF_BYTES
    root = hashlib.sha256(TAG)
    root.update(n.to_bytes(8, "little"))
    if whole:
        root.update(b"".join(_leaf_digests(view, whole)))
    if n > whole * LEAF_BYTES:
        root.update(leaf_hasher(view[whole * LEAF_BYTES :]).digest())
    return root.digest()


def fold_buffer(h: Any, data: Any) -> None:
    """Fold the bytes of one buffer into the running hash *h*.

    Up to `DIRECT_MAX_BYTES` they go in as they are, behind ``r`` and their
    length; a bigger buffer goes in as its tree digest (`bulk_digest`),
    behind ``t`` and its length. *data* is anything with the buffer
    protocol; one that is not contiguous is copied first.
    """
    view = _as_bytes_view(data)
    n = len(view)
    if n <= DIRECT_MAX_BYTES:
        h.update(b"r" + n.to_bytes(8, "little"))
        h.update(view)
    else:
        h.update(b"t" + n.to_bytes(8, "little"))
        h.update(bulk_digest(view))
