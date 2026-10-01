"""Two streams of one cached generator call never touch each other's chunks.

Two callers that miss before either finishes -- two threads, two requests,
``zip(gen(6), gen(6))`` -- stream the same key at once. Their chunks were
written under the same names:

* the one that stopped early deleted, on its way out, chunks the finished one
  had stored, and every later call recomputed ('entry incomplete');
* two that both finished stored a result mixing chunks from both runs: a
  sequence no call produced, served as a hit.

Each stream now writes chunks under its own id, which its manifest names.
"""

from __future__ import annotations

import time


def test_an_abandoned_stream_leaves_a_finished_one_intact(disk_cash):
    runs = []

    @disk_cash.cache(chunk_max_items=2, assume_safe=True)
    def gen(n):
        runs.append(1)
        yield from range(n)

    first, second = gen(6), gen(6)
    assert list(first) == list(range(6))
    assert [next(second) for _ in range(3)] == [0, 1, 2]  # writes a chunk of its own
    second.close()

    runs.clear()
    assert list(gen(6)) == list(range(6))
    assert runs == [], "the finished stream's entry was damaged by the abandoned one"


def test_two_finished_streams_store_one_whole_run(disk_cash):
    @disk_cash.cache(chunk_max_items=2, assume_safe=True)
    def gen(n):
        run = time.perf_counter_ns()  # which run produced the items
        for i in range(n):
            yield (run, i)

    a, b = gen(4), gen(4)
    got_b = [next(b), next(b), next(b)]  # b has written its first chunk
    got_a = list(a)  # a stores a whole entry
    got_b += list(b)  # b finishes and stores over it

    replay = list(gen(4))
    assert got_a != got_b
    assert replay in (got_a, got_b), f"a mix of two runs: {replay}"
    assert gen.cache_info()["hits"] == 1


def test_a_replaced_entry_leaves_no_chunks_behind(disk_cash):
    @disk_cash.cache(chunk_max_items=2, assume_safe=True)
    def gen(n):
        yield from range(n)

    a, b = gen(6), gen(6)
    list(a)
    list(b)  # replaces a's manifest

    chunks = [e["key"] for e in disk_cash.backend.list_entries() if ":chunk_" in e["key"]]
    assert len(chunks) == 3, chunks
