"""Whether a value is worth the disk it would occupy.

A cache trades bytes for seconds. The trade is only good while the bytes buy
enough seconds back, and nothing was checking the rate. Round 26's five
caches held 58 GiB between them for input data of 61-360 MB, and reading the
entries found three separate ways to get there -- every one of them a
population :mod:`cash.backends.versions` has no way to ration:

* r26s5 kept seven ``build_features(lags=...)`` frames, 1.3 GB each for 5.0 s
  of compute. They are ``call:`` entries, which carry no ``version_slot``, so
  version pruning never saw them. **263 MiB per second saved.**
* r26s4 kept 72 loop-iteration entries of 48 MiB at ``execution_time 0.00s``.
  A loop body's source carries ``# __iteration_context__: <hash>`` and
  ``version_slot = f(source_hash)``, so every iteration lands in its own slot
  with one entry and there is never anything to supersede. **188 MiB/s.**
* r26s3 kept 113 spare ~1 GB frames because ``superseded_to_drop`` keeps the
  newest superseded version unconditionally, budget or not.

So the 64 MiB-per-compute-second yardstick :mod:`versions` rations superseded
copies with was the right idea applied to the one population that was not the
problem. This module is that same yardstick, as a predicate the write path
applies to every entry.

**Both numbers are measured, not chosen.** Swept over all 3120 entries in
those five caches, the population separates hard -- p75 is 46 MiB/s and p90 is
430,000 MiB/s -- so a ceiling anywhere in the gap reclaims most of the bytes
for ~1% of the compute. What fixes it within that gap is the bottom end: CAS-141
(``tests/test_notebook_integration/test_cas141_large_frame_disk.py``) is a
45.8 MiB array computed in 0.6 s, **75 MiB/s**, and it must stay cached -- it
is the regression test for a policy that once left big frames RAM-only and gave
a user 393 KB of cache for a 0.68 GB workload. 128 MiB/s clears it with margin
while still refusing r26s5's 263 MiB/s and r26s4's 188: 72% of the bytes for 1%
of the compute, against 76%/1% at 64 MiB/s. Four points of reclamation is worth
not re-creating a bug this project already fixed once.

Sweeping the floor with the ceiling fixed, 8 MiB drops the number of entries
the rule fires on from 739 to 277 while costing 0.05% of what it reclaims:
below that there is nothing worth refusing, and leaving small entries alone
keeps the rule off the ordinary case entirely.

What this never overrides is an explicit decision. ``@cash.cache`` and
``@cash:persist`` mean the caller has already decided, and cash's job is to
honour that -- the same exemption they have from the compute floor and the
cost model. The caller-facing half lives in ``TieredBackend.set``.
"""
from __future__ import annotations

__all__ = ["BYTES_PER_COMPUTE_SECOND", "WORTH_CEILING_BYTES_PER_SECOND",
           "WORTH_FLOOR_BYTES", "worth_its_bytes"]

#: Bytes of cache one second of compute pays for a SUPERSEDED copy -- a spare
#: kept in case you undo. Used by :mod:`cash.backends.versions`.
BYTES_PER_COMPUTE_SECOND = 64 * 1024 * 1024

#: ...and for a LIVE entry, which is twice as generous on purpose. A spare copy
#: is speculative; the live entry is the one that will actually be restored, so
#: it is worth more per byte. Keeping them at one number put the ceiling at
#: 64 MiB/s, which refuses CAS-141's 75 MiB/s case (see the module docstring).
#: Spelled out rather than written as ``2 * BYTES_PER_COMPUTE_SECOND`` so the
#: docs' claim-anchor checker can read the value it documents.
WORTH_CEILING_BYTES_PER_SECOND = 128 * 1024 * 1024

#: Below this, an entry is not worth refusing whatever its rate: it costs no
#: disk anyone would miss, and refusing it only risks turning a working cache
#: into a recompute. Measured: raising the floor from 0 to 8 MiB gives up
#: 0.05% of the bytes reclaimed and stops the rule firing on 462 entries.
WORTH_FLOOR_BYTES = 8 * 1024 * 1024


def worth_its_bytes(size_bytes: int, compute_seconds: float) -> bool:
    """Whether *size_bytes* of cache is worth *compute_seconds* of rebuild.

    Pure and deterministic, so every branch is unit-testable without a
    backend::

        size <= WORTH_FLOOR_BYTES                             -> always worth it
        size <= WORTH_CEILING_BYTES_PER_SECOND * seconds      -> worth it

    A negative or missing cost is treated as zero -- unmeasured compute is not
    evidence of expensive compute, and an entry large enough to be over the
    floor with nothing recorded against it is exactly r26s4's 48 MiB
    loop iterations.

    Worked examples:
      * 1.3 GB rebuilt in 5.0 s   -> False (263 MiB/s; r26s5)
      * 48 MiB rebuilt in 0.00 s  -> False (r26s4)
      * 45.8 MiB rebuilt in 0.6 s -> True  (75 MiB/s; CAS-141)
      * 1 GiB rebuilt in 60 s     -> True  (17 MiB/s)
      * 4 MiB of anything         -> True  (under the floor)
    """
    if size_bytes <= WORTH_FLOOR_BYTES:
        return True
    return size_bytes <= WORTH_CEILING_BYTES_PER_SECOND * max(compute_seconds, 0.0)
