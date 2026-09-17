"""A cached iterator yields everything, or raises -- never a silent prefix.

Found while attacking the decorator before round 26: ``_chunks_are_intact``
checks every chunk exists at lookup time, then the iterator reads them lazily
and turned a chunk that had since gone into a plain ``StopIteration``. Clearing
or rewriting the entry while a caller iterated served 100 of 1000 items, with
no error -- a sum over the stream was simply wrong. The class's own docstring
and the thread-safety guide both say a truncated answer is worse than a slow
one.
"""
from __future__ import annotations

import pytest

from cash import Cash
from cash.backends.file_backend import FileBackend


@pytest.fixture
def stream(tmp_path):
    cash = Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")), register_magic=False)
    runs = []

    @cash.cache(assume_safe=True, chunk_max_items=100)
    def numbers(k):
        runs.append(k)
        for i in range(1000):
            yield (k, i)

    return numbers, runs


def test_an_entry_cleared_mid_iteration_still_yields_everything(stream):
    numbers, runs = stream
    expected = [(9, i) for i in range(1000)]
    assert list(numbers(9)) == expected

    iterator = iter(numbers(9))
    first = [next(iterator) for _ in range(5)]
    numbers.cache_clear()
    rest = list(iterator)

    assert first + rest == expected, f"served {len(first + rest)} of 1000 items"


def test_a_chunk_deleted_mid_iteration_still_yields_everything(stream, tmp_path):
    numbers, runs = stream
    expected = [(4, i) for i in range(1000)]
    assert list(numbers(4)) == expected

    iterator = iter(numbers(4))
    first = [next(iterator) for _ in range(5)]
    for entry in (tmp_path / "c").glob("*.entry"):
        entry.unlink()
    assert first + list(iterator) == expected


def test_an_untouched_iterator_reads_from_the_cache(stream):
    numbers, runs = stream
    assert len(list(numbers(1))) == 1000
    before = len(runs)
    assert len(list(numbers(1))) == 1000
    assert len(runs) == before, "the second pass recomputed instead of reading chunks"
