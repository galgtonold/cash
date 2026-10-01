"""Test isolation for the process-wide record of discarded cache writes."""

from __future__ import annotations

from cash.backends import _writes


def reset_discarded_writes(keep: int = 0) -> None:
    """Drop recorded failures past *keep*.

    ``keep`` rather than a bare clear so a test that induces failures on
    purpose can absorb exactly its own: the writes are asynchronous, so its
    failures often land after it has finished.
    """
    with _writes._DISCARDED_LOCK:
        del _writes._DISCARDED_WRITES[keep:]
