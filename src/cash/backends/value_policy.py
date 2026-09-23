"""Whether a value is worth the disk it would occupy.

A cache trades bytes for seconds, and the trade is only good while the bytes
buy enough seconds back. Version pruning (:mod:`cash.backends.versions`)
rations superseded copies by that rate, but the large populations are ones it
never sees: call results (``call:`` entries carry no ``version_slot``), loop
iterations (each in its own slot, so nothing is ever superseded), and the
newest superseded version, which is always kept. This module is the same
yardstick as a predicate the write path applies to every entry.

**Both numbers are measured, not chosen.** Over a sample of real caches the
rates separate hard (p75 46 MiB/s, p90 430,000 MiB/s), so a ceiling anywhere in
the gap reclaims most of the bytes for ~1% of the compute. 128 MiB/s keeps a
45.8 MiB array computed in 0.6 s (75 MiB/s), the large-frame case the
integration suite requires to stay on disk, while still refusing the 1.3 GB
frames rebuilt in 5 s (263 MiB/s). An 8 MiB floor gives up 0.05% of what the
rule reclaims and keeps it off the ordinary case entirely.

What this never overrides is an explicit decision. ``@cash.cache`` and
``@cash:persist`` mean the caller has already decided, and cash's job is to
honour that -- the same exemption they have from the compute floor and the
cost model. The caller-facing half lives in ``TieredBackend.set``.
"""

from __future__ import annotations

__all__ = ["BYTES_PER_COMPUTE_SECOND", "WORTH_CEILING_BYTES_PER_SECOND", "WORTH_FLOOR_BYTES", "worth_its_bytes"]

#: Bytes of cache one second of compute pays for a SUPERSEDED copy -- a spare
#: kept in case you undo. Used by :mod:`cash.backends.versions`.
BYTES_PER_COMPUTE_SECOND = 64 * 1024 * 1024

#: ...and for a LIVE entry, which is twice as generous on purpose. A spare copy
#: is speculative; the live entry is the one that will actually be restored, so
#: it is worth more per byte. At 64 MiB/s the ceiling would refuse the 75 MiB/s
#: large-frame case the module docstring names.
#: Spelled out rather than written as ``2 * BYTES_PER_COMPUTE_SECOND`` so the
#: docs' claim-anchor checker can read the value it documents.
WORTH_CEILING_BYTES_PER_SECOND = 128 * 1024 * 1024

#: Below this, an entry is not worth refusing whatever its rate: it costs no
#: disk anyone would miss, and refusing it only risks turning a working cache
#: into a recompute.
WORTH_FLOOR_BYTES = 8 * 1024 * 1024


def worth_its_bytes(size_bytes: int, compute_seconds: float) -> bool:
    """Whether *size_bytes* of cache is worth *compute_seconds* of rebuild.

    Pure and deterministic, so every branch is unit-testable without a
    backend::

        size <= WORTH_FLOOR_BYTES                             -> always worth it
        size <= WORTH_CEILING_BYTES_PER_SECOND * seconds      -> worth it

    A negative or missing cost is treated as zero -- unmeasured compute is not
    evidence of expensive compute (a loop iteration's large entry often
    records none).

    Worked examples:
      * 1.3 GB rebuilt in 5.0 s   -> False (263 MiB/s)
      * 48 MiB rebuilt in 0.00 s  -> False
      * 45.8 MiB rebuilt in 0.6 s -> True  (75 MiB/s)
      * 1 GiB rebuilt in 60 s     -> True  (17 MiB/s)
      * 4 MiB of anything         -> True  (under the floor)
    """
    if size_bytes <= WORTH_FLOOR_BYTES:
        return True
    return size_bytes <= WORTH_CEILING_BYTES_PER_SECOND * max(compute_seconds, 0.0)
