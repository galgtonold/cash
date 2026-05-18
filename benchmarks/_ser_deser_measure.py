"""Single-cell measurement primitive for the cost-model bench matrix.

``measure_one()`` produces one row of the matrix: for a given
(family, target_bytes, backend_kind) combination, generates an object
and measures the wall time of ``backend.set()`` and ``backend.get()``.

3+ repeats per cell, first discarded as warmup, median of the rest
returned. Errors (missing deps, OOM, backend failures) are captured
in the result rather than raised, so the matrix run keeps going.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

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

    ser_samples: list[float] = []
    deser_samples: list[float] = []
    try:
        for i in range(repeats + 1):  # +1 for warmup
            key = f"bench:{family}:{target_bytes}:{i}"
            t0 = time.perf_counter()
            backend.set(key, {"variables": {"v": obj}}, {"timestamp": 0.0})
            ser_samples.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            backend.get(key)
            deser_samples.append(time.perf_counter() - t0)
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
