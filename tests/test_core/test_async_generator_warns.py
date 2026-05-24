"""@cash.cache on async generators: warn once, return unwrapped."""
from __future__ import annotations

import warnings

import pytest

from cash import Cash, CashCacheIneffectiveWarning


async def test_async_generator_emits_warning_and_returns_unwrapped(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @c.cache
        async def gen(n):
            for i in range(n):
                yield i

    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1
    msg = str(ineffective[0].message)
    assert "async generator" in msg.lower()

    # The function must still work as an async generator.
    out = []
    async for x in gen(3):
        out.append(x)
    assert out == [0, 1, 2]
