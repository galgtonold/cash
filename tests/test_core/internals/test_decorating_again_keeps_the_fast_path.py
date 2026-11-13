"""A function decorated again on every call, as a cache factory does it
(``cache(func)(args)``), is still served from its memo: the second
decoration is the first one's wrapper, and registers nothing that would
drop remembered hits."""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


def scaled(n):
    RUNS["n"] += 1
    return n * 2


def test_a_function_decorated_on_every_call_is_served_from_the_memo(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    for _ in range(4):
        assert c.cache(scaled)(5) == 10
    before, runs = hit_memo.status()["fast_hits"], RUNS["n"]
    for _ in range(3):
        assert c.cache(scaled)(5) == 10
    assert RUNS["n"] == runs
    assert hit_memo.status()["fast_hits"] >= before + 3
