"""A TieredBackend serves no copy of an entry past its ttl, from any tier.

The ttl is stamped on the entry once, for every tier, but the RAM and S3
tiers never check one: inside a TieredBackend the RAM copy was served after
the file tier's copy had expired (``get`` returned the value while the file
tier returned nothing). Only the decorator's own second check hid it.
"""

from __future__ import annotations

import time

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend


@pytest.fixture
def clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


def _stack(tmp_path, default_ttl):
    return TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), default_ttl=default_ttl)])


def test_the_ram_copy_expires_with_the_tier_default(tmp_path, clock):
    stack = _stack(tmp_path, default_ttl=1)
    stack.set("k", 42, {"execution_time": 1.0})
    stack.backends[1]._writes.wait_all()
    assert stack.get("k")[1] == 42, "control: inside the ttl it is served"

    clock[0] += 1.5
    assert stack.get("k") == (None, None)
    assert stack.backends[0].get("k") == (None, None), "the expired RAM copy was kept"
    stack.shutdown()


def test_a_declared_ttl_expires_the_ram_copy(tmp_path, clock):
    stack = _stack(tmp_path, default_ttl=None)
    stack.set("k", 42, {"execution_time": 1.0, "ttl": 2, "ttl_declared": True})
    clock[0] += 2.5
    assert stack.get("k") == (None, None)
    stack.shutdown()


def test_a_newer_copy_in_a_slower_tier_is_still_served(tmp_path, clock):
    """Dropping an expired copy does not end the lookup: another process may
    have rewritten the entry in the slower tier since this one read it."""
    stack = _stack(tmp_path, default_ttl=10)
    stack.set("k", "old", {"execution_time": 1.0})
    stack.backends[1]._writes.wait_all()
    clock[0] += 8
    stack.backends[1].set("k", "new", {"execution_time": 1.0})  # the other process
    stack.backends[1]._writes.wait_all()
    clock[0] += 5  # the RAM copy is 13 s old, the disk copy 5 s
    metadata, value = stack.get("k")
    assert value == "new"
    assert metadata["source"] == "DISK"
    stack.shutdown()
