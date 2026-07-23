"""Machine-scaled cache-size caps.

A single 1 GiB ``max_cache_size`` used to cap *every* tier, including the
disk tier — whose whole job is to hold big frames persistently. Persisting
two ~680 MB DataFrames on that default put a real user into a
write-and-evict treadmill: each frame was written, then immediately
LRU-evicted to stay under 1 GiB, so cash ended up **slower** than parsing
from scratch, with no warning. "Persisting fewer things made it faster."

The fix is to stop sharing one number. This module derives a *separate*,
machine-scaled cap for each tier:

* **disk** — a generous fraction of the *free* space on the cache volume,
  so it can actually retain what the user persists;
* **RAM**  — a modest fraction of *total* system memory.

Both are clamped to sane floors/ceilings. The arithmetic lives in the pure
``adaptive_disk_cap`` / ``adaptive_ram_cap`` functions (they take the
already-measured numbers), so every clamp branch is unit-testable without
touching the real machine; the ``resolve_*`` wrappers read the machine and
delegate.

``psutil`` is an *optional* dependency (CAS-129 bare-install guard), so the
RAM source guards its import and falls back to a fixed cap when it is
missing — importing psutil unconditionally here would make a bare
``pip install cash-lib`` unimportable.
"""
from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

__all__ = [
    "adaptive_disk_cap",
    "adaptive_ram_cap",
    "resolve_disk_cap",
    "resolve_ram_cap",
    "human_bytes",
]


def human_bytes(n: int | None) -> str:
    """Format a byte count with a sensible unit (e.g. ``7.8 KiB``, ``8.0 GiB``).

    Used in the cap warnings so a message never reads ``~0 MiB`` for a small
    cap or an unwieldy raw byte count for a large one.
    """
    size = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # unreachable; keeps type-checkers happy

_MIB = 1024 ** 2
_GIB = 1024 ** 3

# --- Disk-tier policy ------------------------------------------------------
# A quarter of free space, but never less than 8 GiB (so a laptop still gets
# room for a couple of frames) and never more than 100 GiB (a cache has
# diminishing returns past that). Finally clamped to 80% of what the volume
# actually has free, so a small disk never gets a cap it cannot honour.
DISK_FRACTION = 0.25
DISK_FLOOR = 8 * _GIB
DISK_CEILING = 100 * _GIB
DISK_SAFETY = 0.8

# --- RAM-tier policy -------------------------------------------------------
# A fifth of total RAM, clamped to [512 MiB, 4 GiB]. The RAM tier is a hot
# cache in front of disk, not the system of record, so it stays modest.
RAM_FRACTION = 0.20
RAM_FLOOR = 512 * _MIB
RAM_CEILING = 4 * _GIB
# Used only when psutil is unavailable (bare install) — a sane middle value
# that never imports psutil just to size the RAM tier.
RAM_FALLBACK = _GIB


def adaptive_disk_cap(free_bytes: int) -> int:
    """Disk-tier cap in bytes, derived from *free_bytes* on the cache volume.

    Pure and deterministic — the caller supplies the measured free space, so
    each clamp branch is directly unit-testable::

        cap = min(CEILING, max(FLOOR, FRACTION * free))
        cap = min(cap, SAFETY * free)          # never claim more than the disk has

    Worked examples:
      * 20 GiB free  → 8 GiB   (FLOOR wins: 0.25·20 = 5 GiB < 8 GiB)
      * 2 TiB free   → 100 GiB (CEILING wins)
      * 4 GiB free   → ~3.2 GiB (SAFETY wins: 0.8·4, below the 8 GiB floor)

    ``free_bytes <= 0`` means the volume's free space could not be measured;
    we return the floor (treated as "be generous") rather than a 0 cap, which
    would evict everything on the first write.
    """
    if free_bytes <= 0:
        return DISK_FLOOR
    cap = min(DISK_CEILING, max(DISK_FLOOR, int(DISK_FRACTION * free_bytes)))
    return min(cap, int(DISK_SAFETY * free_bytes))


def adaptive_ram_cap(total_ram_bytes: int | None) -> int:
    """RAM-tier cap in bytes, derived from *total_ram_bytes*.

    Pure and deterministic. ``None`` (psutil unavailable) yields the fixed
    :data:`RAM_FALLBACK`, so a bare install never imports psutil to size the
    RAM tier.

    Worked example: 16 GiB RAM → ~3.2 GiB (0.20·16, within [512 MiB, 4 GiB]).
    """
    if not total_ram_bytes or total_ram_bytes <= 0:
        return RAM_FALLBACK
    return min(RAM_CEILING, max(RAM_FLOOR, int(RAM_FRACTION * total_ram_bytes)))


# ---------------------------------------------------------------------------
# Machine-reading resolvers — thin wrappers that measure, then delegate.
# ---------------------------------------------------------------------------

def _free_bytes_on_volume(path: str) -> int:
    """Free bytes on the volume holding *path*.

    The cache directory may not exist yet at resolution time (the file
    backend creates it lazily), so walk up to the nearest existing ancestor.
    Returns 0 if nothing along the chain can be measured — callers treat that
    as "unknown, be generous".
    """
    probe = os.path.abspath(path)
    while True:
        try:
            return shutil.disk_usage(probe).free
        except (OSError, ValueError):
            parent = os.path.dirname(probe)
            if not parent or parent == probe:
                return 0
            probe = parent


def _total_system_ram() -> int | None:
    """Total physical RAM in bytes, or ``None`` when psutil is unavailable.

    Guarded import: psutil is optional. A bare install falls back
    to the fixed RAM cap rather than crashing on the missing dependency.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 — any psutil failure → fixed fallback
        logger.debug("psutil.virtual_memory() failed; using RAM fallback", exc_info=True)
        return None


def resolve_disk_cap(cache_dir: str) -> int:
    """Adaptive disk-tier cap for the volume that holds *cache_dir*."""
    cap = adaptive_disk_cap(_free_bytes_on_volume(cache_dir))
    logger.debug("[CAPS] adaptive disk cap for %s: %d bytes (%.1f GiB)",
                 cache_dir, cap, cap / _GIB)
    return cap


def resolve_ram_cap() -> int:
    """Adaptive RAM-tier cap for this machine (psutil-guarded)."""
    cap = adaptive_ram_cap(_total_system_ram())
    logger.debug("[CAPS] adaptive RAM cap: %d bytes (%.0f MiB)", cap, cap / _MIB)
    return cap
