"""A truth test of the run's stats does not count the keys.

`compute_file_hash_component` asks ``if stats`` before each file's stat. With
listed folders pending (`_Identifying`), each test built the union of every
key: 24,745 tests of 38,000 keys cost 26 s a cell on a photo set (E5, PC,
2026-11-03).
"""

from __future__ import annotations

from cash.notebook import run_memo


class _CountingKeys(dict):
    keys_calls = 0

    def keys(self):
        type(self).keys_calls += 1
        return super().keys()


def test_a_truth_test_does_not_build_the_keys():
    by_key = _CountingKeys({f"/d/{i}.png": object() for i in range(1000)})
    run = {"stat": by_key, "listed": {"/d"}}
    stats = run_memo._Identifying(run, {"/d": "/d"}, by_key)
    _CountingKeys.keys_calls = 0
    for _ in range(100):
        assert stats
    assert _CountingKeys.keys_calls == 0


def test_the_length_still_counts_each_key_once():
    by_key = {"/a": 1, "/b": 2}
    stats = run_memo._Identifying({"stat": by_key, "listed": {"/b", "/c"}}, {"/b": "/b", "/c": "/c"}, by_key)
    assert len(stats) == 3
    assert sorted(stats) == ["/a", "/b", "/c"]


def test_an_empty_one_is_false():
    stats = run_memo._Identifying({"stat": {}, "listed": set()}, {}, {})
    assert not stats
