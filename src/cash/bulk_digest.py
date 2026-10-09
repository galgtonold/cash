"""The digest of big raw data: array buffers, table columns, file contents.

One algorithm everywhere: SHA-256 from ``hashlib``. It is collision
resistant, the same on every Python cash supports, and on most machines
the fastest of the strong digests the standard library has: OpenSSL runs
it on the CPU's SHA instructions (x86 SHA-NI, ARMv8 SHA2), about 2 GB/s a
core. On a CPU without them it runs at about 0.4 GB/s a core, and BLAKE2b
or SHA-512 would be some 1.5x faster there -- and 2-3x slower on the
machines that have them. MD5 and SHA-1 are faster still, and broken.

What makes it fast on big data is the shape, not the algorithm: the data
is cut into fixed 1 MiB leaves, each leaf is hashed on its own, and the
leaf digests are hashed together with the total length (`BulkHasher`). The
leaves of one big buffer are hashed on several threads at once -- hashlib
releases the GIL while it hashes -- so a 1 GB array takes a fraction of
the time one core needs. The result depends only on the bytes, never on
the number of threads or how the data was handed over in pieces.

`fold_buffer` folds one buffer into a running SHA-256 (a key, a frame's
digest): a small buffer goes in as it is, a big one as its tree digest,
each behind a tag and its length so the two can never read alike.

The tags carry this scheme's version (``TAG``). Every digest built on it
differs from the one an older cash took of the same data, so an entry
written by a build with another scheme is never matched -- it misses and
is computed again.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

__all__ = ["BulkHasher", "LEAF_BYTES", "TAG", "bulk_digest", "fold_buffer", "hashing_threads"]

#: The scheme's name and version, at the head of every tree digest.
TAG = b"cash-bulk-sha256-tree-1\x00"
#: The leaf size. Fixed: it is part of what a digest means.
LEAF_BYTES = 1_048_576  # 1 MiB
#: A buffer `fold_buffer` folds in whole rather than as a tree digest.
DIRECT_MAX_BYTES = LEAF_BYTES
#: Below this many bytes in one go, the leaves are hashed on the calling
#: thread: starting a thread costs about what hashing 100 KB does.
PARALLEL_MIN_BYTES = 8 << 20
#: The most threads one digest uses: past this, memory bandwidth, not
#: cores, is what limits it.
MAX_THREADS = 8

def hashing_threads() -> int:
    """How many threads a big digest may use: the CPUs this process may
    run on, at most `MAX_THREADS`."""
    count = getattr(os, "process_cpu_count", None)  # 3.13+
    if count is not None:
        n = count()
    else:
        try:
            n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            n = os.cpu_count()
    return max(1, min(n or 1, MAX_THREADS))


_THREADS = hashing_threads()


def _leaf_digests(view: memoryview, count: int) -> list[bytes]:
    """The digests of the first *count* whole leaves of *view*, in order,
    on several threads when there is enough to share out."""
    threads = min(_THREADS, count * LEAF_BYTES // PARALLEL_MIN_BYTES)
    if threads < 2:
        return [hashlib.sha256(view[i * LEAF_BYTES : (i + 1) * LEAF_BYTES]).digest() for i in range(count)]
    out: list[Any] = [None] * count
    errors: list[BaseException] = []

    def run(first: int, stop: int) -> None:
        try:
            for i in range(first, stop):
                out[i] = hashlib.sha256(view[i * LEAF_BYTES : (i + 1) * LEAF_BYTES]).digest()
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
    ``SHA-256(TAG, total length, SHA-256 of each 1 MiB leaf in order)``,
    whatever the pieces were.
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
            self._leaves.append(hashlib.sha256(self._partial).digest())
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
            root.update(hashlib.sha256(self._partial).digest())
        return root.digest()

    def hexdigest(self) -> str:
        return self.digest().hex()


def bulk_digest(data: Any) -> bytes:
    """`BulkHasher`'s digest of one buffer."""
    hasher = BulkHasher()
    hasher.update(data)
    return hasher.digest()


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
