"""A cached generator stores each item as it was when it was yielded.

The stream kept a live reference to every item and pickled the chunk only at
its end. By then the caller may have edited the rows it received
(``for row in rows(): row.append(...)``), or the producer refilled the one
buffer it yields again and again (csv-reader style), and every later hit
replayed the edited rows. The caller still receives the object the producer
yielded, as without cash.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("chunk_max_items", [2, 1000])
def test_a_consumer_editing_its_rows_does_not_change_the_hit(disk_cash, chunk_max_items):
    @disk_cash.cache(chunk_max_items=chunk_max_items)
    def rows(n):
        for i in range(n):
            yield [i, i * i]

    for row in rows(3):
        row.append("touched")

    assert list(rows(3)) == [[0, 0], [1, 1], [2, 4]]
    assert rows.cache_info()["hits"] == 1


@pytest.mark.parametrize("chunk_max_items", [3, 1000])
def test_a_producer_reusing_one_buffer_is_stored_item_by_item(disk_cash, chunk_max_items):
    @disk_cash.cache(chunk_max_items=chunk_max_items)
    def buffered(n):
        buf = [0, 0]
        for i in range(n):
            buf[0], buf[1] = i, i * i
            yield buf

    expected = [(0, 0), (1, 1), (2, 4), (3, 9)]
    assert [tuple(b) for b in buffered(4)] == expected
    assert [tuple(b) for b in buffered(4)] == expected
    assert buffered.cache_info()["hits"] == 1


def test_the_caller_receives_the_producers_own_object(disk_cash):
    shared = [1]

    @disk_cash.cache
    def give():
        yield shared

    assert next(iter(give())) is shared
