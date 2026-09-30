"""A cached generator's return value reaches ``yield from``, miss and hit.

``v = yield from g()`` evaluates to the value ``g`` returns, carried by its
final ``StopIteration``. The cached stream dropped it: ``v`` was ``None`` on
the miss and on every replay. It is kept in the entry's manifest now.
"""

from __future__ import annotations

import pytest

from cash import Cash


@pytest.fixture
def app(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"))


def _drive(gen):
    """The items *gen* yields and the value ``yield from gen`` evaluates to."""
    got = []

    def outer():
        value = yield from gen
        return value

    it = outer()
    try:
        while True:
            got.append(next(it))
    except StopIteration as stop:
        return got, stop.value


@pytest.mark.parametrize("chunk_max_items", [1, 1000])
def test_yield_from_gets_the_return_value_on_miss_and_hit(app, chunk_max_items):
    @app.cache(chunk_max_items=chunk_max_items)
    def g(n):
        for i in range(n):
            yield i
        return {"done": n}

    assert _drive(g(3)) == ([0, 1, 2], {"done": 3})
    assert _drive(g(3)) == ([0, 1, 2], {"done": 3})
    assert g.cache_info()["hits"] == 1


def test_an_empty_generator_keeps_its_return_value(app):
    @app.cache
    def g():
        return "nothing"
        yield  # pragma: no cover - makes this a generator

    assert _drive(g()) == ([], "nothing")
    assert _drive(g()) == ([], "nothing")
    assert g.cache_info()["hits"] == 1
