"""A file read once and memoised is still an input of every call that uses it.

Round 19 (r19s1): a parse memoised with ``functools.lru_cache`` (or a module
dict), aggregated by two cached functions. The first recorded the file; the
second got the memoised rows, read nothing, and stored no file dependency --
after the data changed it kept serving the old total, 2 of 2, while the same
job without the memo was right.
"""
from __future__ import annotations

import functools
import os

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def _parse_plain(path):
    with open(path, encoding="utf-8") as fh:
        return [int(line) for line in fh if line.strip()]


_MEMO: dict = {}


def _parse_dict(path):
    if path not in _MEMO:
        _MEMO[path] = _parse_plain(path)
    return _MEMO[path]


def _write(path, values):
    path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture(params=["lru_cache", "dict"])
def parse(request):
    _MEMO.clear()
    if request.param == "lru_cache":
        fn = functools.lru_cache(maxsize=8)(_parse_plain)
        yield fn
        fn.cache_clear()
    else:
        yield _parse_dict
        _MEMO.clear()


def _consumers(c, parse):
    calls = {"a": 0, "b": 0}

    @c.cache
    def total_a(path):
        calls["a"] += 1
        return sum(parse(path))

    @c.cache
    def total_b(path):
        calls["b"] += 1
        return max(parse(path))

    return total_a, total_b, calls


def test_the_second_consumer_of_a_memoised_parse_invalidates_when_the_file_changes(tmp_path, parse):
    data = tmp_path / "data.txt"
    _write(data, [1, 2, 3])
    total_a, total_b, calls = _consumers(Cash(cache_dir=str(tmp_path / "cache")), parse)

    assert (total_a(str(data)), total_b(str(data))) == (6, 3)
    assert (total_a(str(data)), total_b(str(data))) == (6, 3)
    assert calls == {"a": 1, "b": 1}, "an unchanged file did not hit"

    _write(data, [10, 20, 30, 40])
    if hasattr(parse, "cache_clear"):
        parse.cache_clear()           # what a new process starts with
    _MEMO.clear()
    assert total_b(str(data)) == 40, "the memo's consumer served the old maximum"
    assert calls["b"] == 2


def test_a_memo_keyed_by_path_does_not_tie_one_file_to_another(tmp_path, parse):
    """Control: the memo read two files over the process. The consumer of the
    first must not start depending on the second -- only on the file it was
    given -- or every unrelated edit would recompute it."""
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    _write(first, [1, 2])
    _write(second, [5, 6])
    total_a, total_b, calls = _consumers(Cash(cache_dir=str(tmp_path / "cache")), parse)

    total_a(str(second))                 # the memo reads `second` here
    assert total_b(str(first)) == 2      # reads `first` through... the memo reads it live
    total_a(str(first))                  # memoised: nothing read in this call
    _write(second, [50, 60])
    assert total_a(str(first)) == 3
    assert calls["a"] == 2, "an edit to another file invalidated a memoised consumer"
