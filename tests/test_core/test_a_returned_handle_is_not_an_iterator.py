"""A returned file handle is a handle, not a stream to replay.

Found while attacking the decorator before round 26: a file object is its own
iterator, so the streaming path claimed it -- ``open_reader(path)`` returned a
``_StreamingCachedIterator`` with no ``read``/``write``/``name``/``fileno``,
and the handle was drained to build the chunks. A write handle was worse:
``open_writer(p).write("hello")`` raised ``AttributeError``.

Handing the handle back is the only thing cash can do with it; it cannot be
stored (STORE-FAILED says so), so the call recomputes, which for opening a file
is what the user wants anyway.
"""

from __future__ import annotations

import io
import warnings

import pytest

from cash import Cash


@pytest.fixture
def cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / "c"), register_magic=False)


def test_a_read_handle_is_returned_as_a_handle(cash, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello\n", encoding="utf-8")

    @cash.cache
    def reader(p):
        return open(p, encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handle = reader(str(path))
    assert isinstance(handle, io.IOBase), type(handle).__name__
    assert handle.read() == "hello\n"
    handle.close()


def test_a_write_handle_can_still_be_written_to(cash, tmp_path):
    path = tmp_path / "out.txt"

    @cash.cache
    def writer(p):
        return open(p, "w", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handle = writer(str(path))
    handle.write("hello")
    handle.close()
    assert path.read_text(encoding="utf-8") == "hello"


def test_a_generator_is_still_streamed(cash):
    @cash.cache
    def chunks(n):
        yield from range(n)

    assert list(chunks(3)) == [0, 1, 2]
    assert list(chunks(3)) == [0, 1, 2]


def test_a_coroutine_returning_an_async_generator_is_not_exhausted(cash):
    """The second call used to get the FIRST call's generator, already drained:
    ``[]`` where an uncached run yields ``[0, 1, 2]``. Nothing can copy an async
    generator, so the result is not stored and the call recomputes."""
    import asyncio

    @cash.cache
    async def holder(n):
        async def gen():
            for i in range(n):
                yield i

        return gen()

    async def drain():
        first = [v async for v in await holder(3)]
        second = [v async for v in await holder(3)]
        return first, second

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert asyncio.run(drain()) == ([0, 1, 2], [0, 1, 2])
