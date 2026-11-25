"""Reads outside every cached call are remembered up to a bound, and a
function that read more is known as one whose reads are not all kept.

From ``import cash`` on, a read outside a cached call is credited to the
user code on its stack, so a memo filled then still hands its file to the
cached call that uses it later. A program that only imported cash and walks
a folder of files paid for that on every ``open`` and kept every path it
ever opened, for each function on the stack, for ever.

The record is bounded per function and in total. A function past the bound
is no longer recorded -- once every function on a read's stack is past it,
the read takes no stat at all -- and a cached call that reaches such a
function without reading through it is not stored and says so
(KEY-READS-NOT-KEPT): which file a memo of it handed over can no longer be
named, and an entry that depends on nothing would be served for ever.
"""

from __future__ import annotations

import functools
import gc
import os
import warnings

import pytest

import cash  # noqa: F401 - its open handler, watching outside cached calls
from cash import Cash
from cash.tracking import read_credit
from tests._work_counts import method_calls

pytestmark = [pytest.mark.core]

BOUND = 8


@pytest.fixture(autouse=True)
def small_bound(monkeypatch):
    # A record of this test's own: the process's holds whatever the tests
    # before this one read.
    monkeypatch.setattr(read_credit, "_unresolved_by_code", {}, raising=False)
    monkeypatch.setattr(read_credit, "_idle_total", [0], raising=False)
    monkeypatch.setattr(read_credit, "_idle_twins", {}, raising=False)
    monkeypatch.setattr(read_credit, "IDLE_READS_PER_CODE", BOUND, raising=False)
    monkeypatch.setattr(read_credit, "IDLE_READS_TOTAL", 1000 * BOUND, raising=False)


def _files(tmp_path, n, prefix="f"):
    paths = []
    for i in range(n):
        path = tmp_path / f"{prefix}{i}.txt"
        path.write_text(str(i), encoding="utf-8")
        paths.append(str(path))
    return paths


def _reader(tmp_path, name="reader"):
    """User code of this test's own (reads are kept by code)."""
    source = f"def {name}(paths):\n    {str(tmp_path)!r}\n    for p in paths:\n        with open(p) as fh:\n            fh.read()\n"
    namespace: dict = {}
    exec(compile(source, str(tmp_path / f"{name}.py"), "exec"), namespace)
    return namespace[name]


def _kept() -> int:
    return sum(len(record) for record in read_credit._unresolved_by_code.values())


def _record(fn) -> int:
    return len(read_credit._unresolved_by_code.get(id(fn.__code__), ()))


def test_a_function_reading_without_end_keeps_a_bounded_record(tmp_path):
    reader = _reader(tmp_path)
    paths = _files(tmp_path, 5 * BOUND)
    reader(paths[:BOUND])
    assert not read_credit.reads_not_kept(reader.__code__)
    assert _record(reader) == BOUND  # positive control: under the bound all are kept
    reader(paths[BOUND:])
    assert _record(reader) == BOUND
    assert _kept() <= 2 * BOUND  # the reader's and this test's own, which called it
    assert read_credit.reads_not_kept(reader.__code__)
    assert len(read_credit.credited_reads(reader.__code__)) == BOUND
    # Asked in between, it still keeps no more.
    reader(_files(tmp_path, 3 * BOUND, "again"))
    assert len(read_credit.credited_reads(reader.__code__)) == BOUND
    assert _record(reader) == 0


def test_a_read_by_functions_past_the_bound_takes_no_stat(tmp_path):
    """The work count: once every function on the stack is past the bound, a
    read outside a cached call costs a walk of the stack, no stat and no path
    resolution."""
    reader = _reader(tmp_path)
    paths = _files(tmp_path, 4 * BOUND)
    with method_calls(read_credit, "regular_file_stat") as under:
        reader(paths[:BOUND])
    assert under.calls == BOUND  # positive control: under the bound each read is stat-ed
    reader(paths[BOUND : BOUND + 1])  # one past: the reader is marked
    read_credit._untracked_stats.clear()
    with method_calls(read_credit, "regular_file_stat") as past, method_calls(os.path, "abspath") as resolved:
        reader(paths[BOUND + 1 :])
    assert past.calls == 0
    assert resolved.calls == 0


def test_the_total_over_all_functions_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(read_credit, "IDLE_READS_TOTAL", 4 * BOUND, raising=False)
    readers = [_reader(tmp_path, f"reader{i}") for i in range(8)]
    before = _kept()
    for i, reader in enumerate(readers):
        reader(_files(tmp_path, BOUND - 1, f"r{i}_"))
    assert _kept() - before <= read_credit.IDLE_READS_TOTAL
    assert any(read_credit.reads_not_kept(r.__code__) for r in readers)


def test_the_record_goes_with_the_function(tmp_path):
    reader = _reader(tmp_path)
    reader(_files(tmp_path, 3))
    key = id(reader.__code__)
    assert key in read_credit._unresolved_by_code
    before = _kept()
    del reader
    gc.collect()
    assert key not in read_credit._unresolved_by_code
    assert _kept() == before - 3


def test_two_equal_codes_share_what_they_read_outside_a_cached_call(tmp_path):
    first = _reader(tmp_path)
    second = _reader(tmp_path)
    assert first.__code__ == second.__code__ and first.__code__ is not second.__code__
    paths = _files(tmp_path, 2)
    first(paths[:1])
    second(paths[1:])
    assert len(read_credit.credited_reads(second.__code__)) == 2


def _load_plain(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def load():
    fn = functools.lru_cache(maxsize=None)(_load_plain)
    yield fn
    fn.cache_clear()


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def test_a_call_handed_a_memo_of_a_function_past_the_bound_is_not_stored(tmp_path, load):
    """The guard the bound must not lose: the memo was filled outside every
    cached call, from more files than are remembered, and the cached call is
    given no path that says which."""
    config = tmp_path / "config.txt"
    _write(config, "old")
    load(str(config))  # the memo, filled outside every cached call
    for path in _files(tmp_path, 3 * BOUND):
        load(path)
    assert read_credit.reads_not_kept(_load_plain.__code__)
    where = str(config)
    runs = []
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def setting():
        runs.append(1)
        return load(where)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert setting() == "old"
        assert setting() == "old"
    assert len(runs) == 2, "an entry that names no file was stored and served"
    assert _codes(caught).count("KEY-READS-NOT-KEPT") == 1
    message = str(next(w.message for w in caught if getattr(w.message, "code", None) == "KEY-READS-NOT-KEPT"))
    assert "_load_plain" in message and "setting" in message
    _write(config, "new")
    load.cache_clear()
    assert setting() == "new"


def test_a_path_the_call_is_given_still_names_the_memos_file(tmp_path, load):
    """Past the bound, a memo keyed by a path the call was given is still
    credited that path, when it is among the remembered."""
    data = tmp_path / "data.txt"
    _write(data, "old")
    load(str(data))
    for path in _files(tmp_path, 3 * BOUND):
        load(path)
    assert read_credit.reads_not_kept(_load_plain.__code__)
    runs = []
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def content(path):
        runs.append(1)
        return load(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert content(str(data)) == "old"
        assert content(str(data)) == "old"
    assert len(runs) == 1 and "KEY-READS-NOT-KEPT" not in _codes(caught)
    _write(data, "new")
    load.cache_clear()
    assert content(str(data)) == "new"


def test_a_memo_under_the_bound_is_credited_as_before(tmp_path, load):
    config = tmp_path / "config.txt"
    _write(config, "old")
    load(str(config))
    where = str(config)
    runs = []
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def setting():
        runs.append(1)
        return load(where)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert setting() == "old"
        assert setting() == "old"
    assert len(runs) == 1 and "KEY-READS-NOT-KEPT" not in _codes(caught)
    _write(config, "new")
    load.cache_clear()
    assert setting() == "new"
