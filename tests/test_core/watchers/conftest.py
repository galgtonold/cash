"""Fixtures for the watcher tests: the fast path with no size threshold, and
a namespace that anchors list and tuple values."""

from __future__ import annotations

import pytest

from cash import value_watch


@pytest.fixture
def watching(monkeypatch):
    """The C extension, watching values of any size from their second hash,
    with an empty table; skipped where the fast path is off. Yields the
    namespace values are anchored in (`value_watch._anchors`)."""
    if not value_watch.enabled():
        pytest.skip(f"the fast path is off: {value_watch.REASON}")
    if value_watch.VERIFY:
        pytest.skip("these tests count hashes; CASH_CWATCH_VERIFY=1 hashes behind every reuse")
    cw = value_watch._cw
    before = cw.stats()
    cw.reset()
    cw.configure(min_bytes=0)
    ns: dict = {}
    monkeypatch.setattr(value_watch, "_anchors", lambda: (ns,))
    yield ns
    ns.clear()
    cw.reset()
    value_watch.reset()
    cw.configure(min_bytes=before["min_bytes"], budget=before["budget"])


class Counter:
    """A hash function that counts how often it read the value."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value):
        from cash.value_hash import _compute_hash

        self.calls += 1
        return _compute_hash(value)


@pytest.fixture
def counted():
    return Counter()
