"""``use_locking`` coalesces concurrent awaits, in one event loop or several.

Found while attacking the decorator before round 26: the in-flight registry
held one ``(loop, asyncio.Event)`` per key. A leader in a second loop
overwrote the first loop's slot, and the first loop's followers -- finding
another loop's event, which they cannot await -- each computed for themselves.
Measured: 4 loops x 4 awaits ran the body 16 times where one loop runs it once.
That is worse than one compute per loop, and the feature exists so an expensive
idempotent call (a paid API request) happens once.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from cash import Cash


@pytest.fixture
def counted(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / "c"), use_locking=True, register_magic=False)
    runs: list[int] = []

    @cash.cache
    async def expensive(n):
        runs.append(n)
        await asyncio.sleep(0.3)
        return n * 2

    return expensive, runs


def _await_many(fn, key, times):
    async def main():
        return await asyncio.gather(*[fn(key) for _ in range(times)])

    return asyncio.run(main())


@pytest.mark.timeout(300)
def test_one_loop_computes_once(counted):
    expensive, runs = counted
    assert _await_many(expensive, 1, 8) == [2] * 8
    assert len(runs) == 1


@pytest.mark.timeout(300)
def test_several_loops_compute_once(counted):
    expensive, runs = counted
    results: list[list[int]] = []
    threads = [threading.Thread(target=lambda: results.append(_await_many(expensive, 2, 4))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [[4] * 4] * 4
    assert len(runs) == 1, f"the body ran {len(runs)} times across 4 loops"


@pytest.mark.timeout(300)
def test_a_cancelled_follower_does_not_cancel_the_computation(counted):
    expensive, runs = counted

    async def main():
        leader = asyncio.create_task(expensive(3))
        await asyncio.sleep(0.05)
        follower = asyncio.create_task(expensive(3))
        await asyncio.sleep(0.05)
        follower.cancel()
        return await leader

    assert asyncio.run(main()) == 6
    assert len(runs) == 1
