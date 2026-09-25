"""A cached generator can be advanced from any thread, as an undecorated one can.

On a miss the stream keeps its file tracker and effect observer across
``yield``, suspending them around each item with ContextVar tokens. A token
only resets in the context that made it, and ``asyncio.to_thread`` runs each
``next()`` in a fresh copy of the context; a plain second thread has its own.
So the first miss raised ``ValueError: <Token ...> was created in a different
Context`` into the caller's loop.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from cash import Cash


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_advanced_through_asyncio_to_thread(c):
    @c.cache
    def rows(n):
        yield from range(n)

    async def main():
        g, out = rows(3), []
        while (x := await asyncio.to_thread(next, g, None)) is not None:
            out.append(x)
        return out

    assert asyncio.run(main()) == [0, 1, 2]
    assert list(rows(3)) == [0, 1, 2]
    assert rows.cache_info()["hits"] == 1


def test_started_here_and_finished_on_another_thread(c):
    @c.cache
    def rows(n):
        yield from range(n)

    g = rows(7)
    out = [next(g)]
    errors = []

    def finish():
        try:
            out.extend(g)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=finish)
    worker.start()
    worker.join()
    assert not errors, errors
    assert out == list(range(7))
    assert list(rows(7)) == list(range(7))
    assert rows.cache_info()["hits"] == 1


def test_a_file_read_on_the_other_thread_is_still_a_dependency(c, tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("a\nb\n", encoding="utf-8")

    @c.cache
    def lines(path):
        yield "start"
        with open(path, encoding="utf-8") as f:
            yield from (line.strip() for line in f)

    g = lines(str(data))
    out = [next(g)]
    worker = threading.Thread(target=lambda: out.extend(g))
    worker.start()
    worker.join()
    assert out == ["start", "a", "b"]

    data.write_text("c\n", encoding="utf-8")
    assert list(lines(str(data))) == ["start", "c"]
