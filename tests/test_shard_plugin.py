"""The nightly workflow's shards never overlap and together run every test.

``tools/test_selection/shard.py`` keeps a test in shard ``I`` of ``N`` by a hash
of its node id. If two shards could both keep a test, it would run twice; if
none could, it would never run and the nightly job would stay green over it.
"""

from __future__ import annotations

import pytest

from tools.test_selection.shard import parse_shard, shard_of

NODE_IDS = [f"tests/test_notebook_integration/test_{n % 97}.py::test_case_{n}[param-{n % 5}]" for n in range(3000)]


@pytest.mark.parametrize("count", [1, 2, 4, 6, 7])
def test_every_node_id_lands_in_exactly_one_shard(count):
    shards = {i: {nid for nid in NODE_IDS if shard_of(nid, count) == i} for i in range(1, count + 1)}
    assert set().union(*shards.values()) == set(NODE_IDS)
    assert sum(len(s) for s in shards.values()) == len(NODE_IDS)


def test_the_split_is_roughly_even():
    sizes = [sum(shard_of(nid, 6) == i for nid in NODE_IDS) for i in range(1, 7)]
    assert min(sizes) > len(NODE_IDS) / 6 * 0.8, sizes


def test_a_shard_depends_on_the_node_id_alone():
    """xdist workers each collect on their own; they must agree on every test."""
    nid = NODE_IDS[123]
    assert len({shard_of(nid, 6) for _ in range(5)}) == 1
    assert shard_of(nid, 6) == shard_of(str(nid), 6)


@pytest.mark.parametrize("value, expected", [("1/1", (1, 1)), ("2/4", (2, 4)), ("6/6", (6, 6))])
def test_parse_shard_accepts_i_of_n(value, expected):
    assert parse_shard(value) == expected


@pytest.mark.parametrize("value", ["0/4", "5/4", "1/0", "2", "a/b", "1/2/3", ""])
def test_parse_shard_rejects_anything_else(value):
    with pytest.raises(pytest.UsageError):
        parse_shard(value)
