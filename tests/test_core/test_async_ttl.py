"""TTL on async @cash.cache."""

from __future__ import annotations

import time

import pytest

from cash import Cash


@pytest.fixture
def clock(monkeypatch):
    """A wall clock the test moves by hand -- see ``clock`` in test_ttl.py.

    The entry is stamped when its store begins, and the first store into a
    fresh directory still writes its stored-key record after that, on the
    caller's thread. A loaded Windows runner once let more than a second pass
    between that stamp and the next call's lookup, so the "still inside the
    ttl" call found the entry past ttl=1 and recomputed (`assert 2 == 1`).
    That miss is right -- the value WAS older than its ttl -- so it is the
    window that must not depend on the machine: the clock moves only when
    told to.
    """
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


async def test_async_ttl_expires(tmp_path, clock):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(ttl=1)
    async def f(x):
        n["calls"] += 1
        return x

    await f("k")
    await f("k")
    assert n["calls"] == 1

    clock[0] += 1.4  # past the ttl
    await f("k")
    # FileBackend honours TTL; InMemoryBackend does too via the validate
    # branch in CallRunner._try_get_cached. Tiered (RAM+disk) is the default, which
    # also honours TTL.
    assert n["calls"] == 2
