"""An async function's cached iterator recomputes a chunk that went missing.

A chunked iterator hit finishes the run from the function itself when a chunk
disappears while the caller is reading (another process cleared the entry, the
RAM tier evicted it). Only the sync wrapper's first lookup handed the hit a way
to recompute; the async wrapper, its single-flight follower and the sync locked
re-read did not, so there the same loss raised instead. The two wrappers were
separate copies and had drifted.
"""

from __future__ import annotations

import asyncio

from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def test_an_async_hit_recomputes_a_chunk_lost_while_reading():
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    runs = []

    @c.cache(chunk_max_items=2)
    async def numbers(n):
        runs.append(n)
        await asyncio.sleep(0)
        return (i for i in range(n))

    async def main():
        assert list(await numbers(6)) == [0, 1, 2, 3, 4, 5]  # computed, stored in 3 chunks
        hit = await numbers(6)
        keys = [e["key"] for e in c.backend.list_entries() if e["key"].endswith(":chunk_1")]
        assert keys, "expected the result to be stored in chunks"
        c.backend.delete(keys[0])
        # Read inside the running loop, as async code would: the rest
        # comes from the function, not an error.
        return list(hit)

    assert asyncio.run(main()) == [0, 1, 2, 3, 4, 5]
    assert len(runs) == 2, "the lost chunk was not recomputed from the function"
