"""An intercepted call must not store a result that is drained as it is read.

The RAM tier cannot copy an open file or a generator, so it keeps the object
itself: a hit hands back the very handle an earlier reader left at EOF, and
the reader prints ``[]`` where plain Python reads the data again. The
statement path already refuses such an output ("consumed as it is read");
the call path stored it whenever the call cleared the cost floor, which on a
loaded machine even ``open`` does.
"""

from __future__ import annotations

import time

import pytest

import cash
from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import CallCache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site(source):
    return CallSite(source=source, free_names=frozenset({source.split("(")[0], "p"}), occurrence_index=0)


def test_a_call_returning_an_open_file_is_not_served(call_cache, tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    def handle(p):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return open(p, encoding="utf-8")  # the handle is the point

    call_cache.set_sites([_site("handle(p)")])
    cached = call_cache.resolve(handle)
    with cached(str(path)) as first:
        assert first.read() == "a\nb\n"
    with cached(str(path)) as second:
        assert second.read() == "a\nb\n", "the second call was handed the first call's drained handle"
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, False]


def test_a_call_returning_a_generator_is_not_served(call_cache):
    def rows(p):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return (x for x in range(p))

    call_cache.set_sites([_site("rows(p)")])
    cached = call_cache.resolve(rows)
    assert list(cached(3)) == [0, 1, 2]
    assert list(cached(3)) == [0, 1, 2], "the second call was handed the first call's drained generator"


def test_a_call_returning_a_copyable_iterator_is_still_cached(call_cache):
    """Positive control: a list iterator deep-copies, so it restores fresh."""

    def rows(p):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return iter(list(range(p)))

    call_cache.set_sites([_site("rows(p)")])
    cached = call_cache.resolve(rows)
    assert list(cached(3)) == [0, 1, 2]
    assert list(cached(3)) == [0, 1, 2]
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True]
