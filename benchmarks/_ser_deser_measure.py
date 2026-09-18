"""Single-cell measurement primitive for the cost-model bench matrix.

``measure_one()`` produces one row of the matrix: for a given
(family, target_bytes, backend_kind) combination, generates an object
and measures the wall time of ``backend.set()`` and ``backend.get()``.

3+ repeats per cell, first discarded as warmup, median of the rest
returned. Errors (missing deps, OOM, backend failures) are captured
in the result rather than raised, so the matrix run keeps going.

Two things about the read, both of which moved the numbers by more than the
fit ever did:

**The write is drained first.** A file backend hands the disk write to a
background worker and ``set()`` returns once the bytes are serialized, so a
``get()`` of that same key waits for the write to land -- and without the drain,
every "deserialize" number carries part of a write. That is what made the May
matrix non-monotonic (11.8 KB at 0.47 ms, 115 KB at 14.94 ms in one family), and
those steps are what pushed the fitted intercept to ~10 ms.

**A disk read is timed in a fresh process; a RAM read is not.** Every
in-process attempt to time a read of a just-written entry measured the harness
instead of cash. The same 1 KB entry came back as 4.5 ms through the backend
that wrote it, 0.7 ms through a second backend made after the write, 4.9 ms
through one made before it, and 3.5-5 ms in any of those arrangements once a
dozen backends were alive -- each one runs about eight threads, and a matrix
that leaks one per cell ends up timing thread contention. From a fresh process:
1.5 ms, stable.

So disk rows delegate to `_cold_read.cold_read_seconds`, one process per
sample. That is also the regime the number is FOR: a persistent tier exists to
serve later sessions, because within a session the RAM tier answers first. RAM
rows stay in-process, where the same reasoning says they belong -- a RAM restore
is by definition in-session.

Backends are shut down per cell either way, so a long matrix run measures the
same thing at the end as at the start.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks._cold_read import cold_read_seconds
from benchmarks._object_generators import estimate_in_memory_size, make_object


@dataclass
class MeasureResult:
    family: str
    target_bytes: int
    backend_kind: str            # "ram" or "disk"
    repeats: int
    actual_size_bytes: int
    serialize_seconds: float
    deserialize_seconds: float
    error: str | None = None


def _build_backend(backend_kind: str, cache_root: Path):
    """Construct a fresh backend instance. cache_root is used only for disk."""
    if backend_kind == "ram":
        from cash.backends.memory_backend import InMemoryBackend
        return InMemoryBackend()
    if backend_kind == "disk":
        from cash.backends.file_backend import FileBackend
        cache_root.mkdir(parents=True, exist_ok=True)
        return FileBackend(str(cache_root))
    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def measure_one(
    family: str,
    target_bytes: int,
    backend_kind: str,
    repeats: int,
    cache_root: Path,
) -> MeasureResult:
    """Run the (family, target_bytes, backend_kind) cell and return one
    measurement row. Errors are captured into ``result.error``."""
    try:
        obj = make_object(family, target_bytes)
    except Exception as e:  # noqa: BLE001 — capture, don't propagate
        return MeasureResult(
            family=family,
            target_bytes=target_bytes,
            backend_kind=backend_kind,
            repeats=repeats,
            actual_size_bytes=0,
            serialize_seconds=0.0,
            deserialize_seconds=0.0,
            error=str(e),
        )

    actual_size = estimate_in_memory_size(obj)
    backend = _build_backend(backend_kind, cache_root)
    # A RAM restore is read back through the SAME backend that holds it. An
    # in-memory tier has no later session to model -- it is gone with the
    # process -- and a second InMemoryBackend is simply an empty one, so reading
    # through it measured a miss and reported a 100 MB frame restoring in 0.00 ms.
    # A DISK restore is read from a fresh process instead (see the docstring).
    reader = backend if backend_kind != "disk" else None

    ser_samples: list[float] = []
    deser_samples: list[float] = []
    keys = [f"bench:{family}:{target_bytes}:{i}" for i in range(repeats + 1)]
    try:
        writes = getattr(backend, "_writes", None)
        for i, key in enumerate(keys):  # index 0 is the warmup
            t0 = time.perf_counter()
            backend.set(key, {"variables": {"v": obj}}, {"timestamp": 0.0})
            if writes is not None:
                writes.wait_all()     # the write is part of storing, not of reading
            ser_samples.append(time.perf_counter() - t0)

            if backend_kind != "disk":
                t0 = time.perf_counter()
                reader.get(key)
                deser_samples.append(time.perf_counter() - t0)

        if backend_kind == "disk":
            # Every write first, then close the writer, THEN read. A live
            # backend is still flushing and still holding what it wrote, and a
            # cold read taken across that measures the flush: the same 1 KB
            # entry reads in 6 ms while its writer is open and 1.1 ms once it
            # has closed. A later session -- what this number is for -- never
            # sees the writing process at all.
            backend.shutdown()
            for key in keys:
                deser_samples.append(cold_read_seconds(cache_root, key, repeats=1))
    except Exception as e:  # noqa: BLE001
        return MeasureResult(
            family=family,
            target_bytes=target_bytes,
            backend_kind=backend_kind,
            repeats=repeats,
            actual_size_bytes=actual_size,
            serialize_seconds=0.0,
            deserialize_seconds=0.0,
            error=str(e),
        )

    for b in ({id(backend): backend, id(reader): reader}.values()):
        if b is None:
            continue                  # disk cells never build a second backend
        try:
            b.shutdown()              # ~8 threads per backend; see the docstring
        except Exception:  # noqa: BLE001 - teardown must not lose a measurement
            pass

    # Discard repeat 0 as warmup; median of the rest.
    return MeasureResult(
        family=family,
        target_bytes=target_bytes,
        backend_kind=backend_kind,
        repeats=repeats,
        actual_size_bytes=actual_size,
        serialize_seconds=statistics.median(ser_samples[1:]),
        deserialize_seconds=statistics.median(deser_samples[1:]),
        error=None,
    )
