"""TTL on async @cash.cache."""
from __future__ import annotations

import asyncio
import time

import pytest

from cash import Cash


async def test_async_ttl_expires(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(ttl=1)
    async def f(x):
        n["calls"] += 1
        return x

    await f("k")
    await f("k")
    assert n["calls"] == 1

    time.sleep(1.4)
    await f("k")
    # FileBackend honours TTL; InMemoryBackend does too via the validate
    # branch in _try_get_cached. Tiered (RAM+disk) is the default, which
    # also honours TTL.
    assert n["calls"] == 2
