"""A cell's end resolves the configuration only when a cache write is queued.

``post_run_cell`` bounds the writes a cell leaves running, by a deadline
read from ``shutdown_write_timeout``: a resolve of every config layer, a
TOML file read included. Read on every cell, it cost a hit of a 0.2 s cell
about 2 ms of its 6 ms. With nothing queued there is nothing to bound.
"""

from __future__ import annotations

from cash.backends import _writes
from tests._cell_driver import run_cash_cell


class _Queue:
    def __init__(self, pending: int) -> None:
        self.pending, self.waits = pending, []

    def pending_count(self) -> int:
        return self.pending

    def wait_for_backlog(self, max_seconds, deadline=None) -> bool:
        self.waits.append(deadline)
        return True


def _count_reads(monkeypatch) -> list:
    reads: list = []
    real = _writes.shutdown_write_timeout

    def counting() -> float:
        reads.append(1)
        return real()

    monkeypatch.setattr(_writes, "shutdown_write_timeout", counting)
    return reads


def test_nothing_queued_reads_no_config(cash_magics, monkeypatch):
    idle = _Queue(0)
    monkeypatch.setattr(_writes, "all_pending_writes", lambda: [idle])
    reads = _count_reads(monkeypatch)
    run_cash_cell(cash_magics, "x = 1")
    cash_magics._after_cell()
    assert reads == [] and idle.waits == []


def test_a_queued_write_is_still_bounded(cash_magics, monkeypatch):
    """The positive control: a queue with a write in flight is waited on, by the configured deadline."""
    idle, busy = _Queue(0), _Queue(1)
    monkeypatch.setattr(_writes, "all_pending_writes", lambda: [idle, busy])
    reads = _count_reads(monkeypatch)
    cash_magics._after_cell()
    assert len(reads) == 1 and len(busy.waits) == 1 and busy.waits[0] is not None
    assert idle.waits == []
