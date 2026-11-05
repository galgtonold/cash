"""What writing an entry past RAM costs on this machine.

The persistence rule weighs what an entry saves on a re-run (its compute,
less reading it back) against what keeping it costs. Disk space has its
yardstick (`value_policy`); this module is the other cost, the TIME the
first run spends writing the entry, which nothing priced before: a loop
storing fifteen 22 MB frames of text, each computed in 1.4 s and written at
11 MB/s, spent more time writing each pass than a re-run would save by
reading it back.

**Measured here, from the store's own writes.** Each persistent tier's
write queue (`_writes.PendingWrites`) times every write it runs and keeps
a running estimate of seconds per byte (`WriteRates`), one per kind of
value: a frame of numbers pickles as a few buffers, one of Python strings
an object at a time, ten times slower per byte. Until a kind has been
measured, the fitted cost model's serialize prediction stands in
(`cost_model.estimated_serialize_time`, measured as a store plus the wait
for its write to land, on an NVMe disk), or what pickling a sample of the
value took (`sampled_cost`) when that is more.

**What is charged.** Everything writing the entry costs, wherever it runs:
the copy or pickle made on the cell's thread before the store returns, and
the background writer's whole time. The writer is not free time: pickling
holds the interpreter lock, so the cell waits while it runs; writes queue
behind each other on one thread, so a loop writing faster than it computes
piles them up; and a cell ends only once at most `_writes.MAX_BACKLOG_S` of
writing is left. The disk I/O itself can overlap the next statement, which
makes this the cautious side: an entry wrongly judged too costly is
computed again on the next run, never served wrong.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .. import cost_model

__all__ = [
    "MEASURED_FROM_BYTES",
    "OBJECTS_KIND",
    "PIN_ENV",
    "WRITE_KIND",
    "WriteRates",
    "WriteSample",
    "current_sample",
    "on_this_thread",
    "rates_for_queue",
    "sampling",
    "write_kind",
]

#: A write at least this big teaches its kind's rate; a smaller one costs
#: mostly its fixed overhead (opening a file), which the rate does not model.
MEASURED_FROM_BYTES = 1024 * 1024

#: How much the newest write moves its kind's estimate. Recent writes say
#: most about the disk as it is now (a virus scanner, another process
#: writing), and one odd write moves the estimate only part of the way.
_NEWEST_WEIGHT = 0.5

#: The kind of a value `sampled_cost` found to be many Python objects, which
#: pickles at its own, much slower rate whatever its type.
OBJECTS_KIND = "objects"

#: Put on an entry's metadata when its kind is not its cost-model family
#: (`OBJECTS_KIND`), so the end-of-cell write prices it the same way.
WRITE_KIND = "write_kind"


def write_kind(metadata: dict[str, Any], objects: bool = False) -> str:
    """The kind an entry's write is measured and priced as."""
    if objects or metadata.get(WRITE_KIND) == OBJECTS_KIND:
        return OBJECTS_KIND
    return metadata.get("cost_model_family") or cost_model.GENERIC_FAMILY


class WriteRates:
    """Seconds per byte a tier's writes took, per kind, newest weighed most."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_byte: dict[str, float] = {}

    def record(self, kind: str, nbytes: int, seconds: float) -> None:
        """Learn from a write of *nbytes* of *kind* that took *seconds*."""
        if nbytes < MEASURED_FROM_BYTES or seconds <= 0:
            return
        rate = seconds / nbytes
        with self._lock:
            old = self._per_byte.get(kind)
            self._per_byte[kind] = rate if old is None else old + _NEWEST_WEIGHT * (rate - old)

    def per_byte(self, kind: str) -> float | None:
        """The measured seconds per byte of *kind*, or None before any write of it."""
        with self._lock:
            return self._per_byte.get(kind)

    def slowest_per_byte(self) -> float | None:
        """The slowest rate measured for any kind, or None before any write."""
        with self._lock:
            return max(self._per_byte.values(), default=None)

    def seconds(self, kind: str, nbytes: int, assumed: float) -> float:
        """What writing *nbytes* of *kind* is expected to take: by the measured
        rate once there is one, else *assumed* (the cost model's prediction)."""
        per_byte = self.per_byte(kind)
        if per_byte is None or nbytes < MEASURED_FROM_BYTES:
            return assumed
        return nbytes * per_byte


class PinnedRates(WriteRates):
    """A rate fixed by `PIN_ENV`, whatever the writes take: the store
    decisions of a test suite do not then depend on how loaded the machine
    running it is."""

    def __init__(self, per_byte: float) -> None:
        super().__init__()
        self._pinned = per_byte

    def record(self, kind: str, nbytes: int, seconds: float) -> None:
        return None

    def per_byte(self, kind: str) -> float | None:
        return self._pinned

    def slowest_per_byte(self) -> float | None:
        return self._pinned


#: MiB per second every write queue made while it is set takes its writes to
#: run at, instead of measuring them. For test suites (cash's own sets it for
#: every test and test kernel), never needed in use.
PIN_ENV = "CASH_TEST_WRITE_MIB_PER_S"


def rates_for_queue() -> WriteRates:
    """The rates a new write queue keeps: measured, or pinned by `PIN_ENV`."""
    try:
        mib_per_s = float(os.environ.get(PIN_ENV) or 0)
    except ValueError:
        mib_per_s = 0.0
    if mib_per_s > 0:
        return PinnedRates(1.0 / (mib_per_s * 1024 * 1024))
    return WriteRates()


class WriteSample:
    """The entry the writes made on this thread now are for: its kind, the
    bytes it is priced by, and the seconds already spent on it on the cell's
    thread (a copy or a pickle made before the background write)."""

    __slots__ = ("kind", "nbytes", "main_seconds")

    def __init__(self, kind: str, nbytes: int) -> None:
        self.kind = kind
        self.nbytes = nbytes
        self.main_seconds = 0.0

    def take_main_seconds(self) -> float:
        """The cell-thread seconds, charged to the first tier that writes."""
        seconds, self.main_seconds = self.main_seconds, 0.0
        return seconds


_LOCAL = threading.local()


@contextmanager
def sampling(kind: str, nbytes: int) -> Iterator[WriteSample]:
    """Mark the writes this thread makes inside as one entry's, for the
    write queues to time against its kind (`current_sample`)."""
    previous = getattr(_LOCAL, "sample", None)
    sample = WriteSample(kind, nbytes)
    _LOCAL.sample = sample
    try:
        yield sample
    finally:
        _LOCAL.sample = previous


def current_sample() -> WriteSample | None:
    """The entry being written on this thread, if `sampling` marked one."""
    return getattr(_LOCAL, "sample", None)


def on_this_thread(seconds: float) -> None:
    """Charge *seconds* of work on the cell's thread to the entry being written."""
    sample = current_sample()
    if sample is not None and seconds > 0:
        sample.main_seconds += seconds
