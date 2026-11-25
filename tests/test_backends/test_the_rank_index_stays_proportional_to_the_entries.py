"""``_rank.log`` holds about a line per entry, however often the cache is used.

Every write and every flushed read appends a line to the eviction index, and
the file was rewritten to its real size only when the cache reached its cap,
which most caches never do. A script that hit N entries added N lines on
every run, and every process parses the whole file: after a few hundred runs
tens of megabytes for a few thousand entries, read again each time.
"""

from __future__ import annotations

import textwrap

import pytest

from cash.backends import FileBackend
from cash.backends.rank_index import RankIndex
from cash.tracking.tracker_context import untracked
from tests._scripts import run_python

INDEX = "_rank.log"
ENTRIES = 200
SLACK = RankIndex.COMPACT_SLACK


def _lines(cache_dir) -> int:
    return len((cache_dir / INDEX).read_text(encoding="ascii").splitlines())


def _backend(cache_dir):
    # A cap, as every default cache has: only a capped cache ranks.
    return FileBackend(str(cache_dir), max_size_bytes=10**9, flush_interval=0)


def _fill(cache_dir, count=ENTRIES):
    b = _backend(cache_dir)
    for i in range(count):
        b.set(f"k{i}", i, {"execution_time": 1.0})
    b.shutdown()


def _a_run_that_only_reads(cache_dir, count=ENTRIES):
    b = _backend(cache_dir)
    for i in range(count):
        assert b.get(f"k{i}")[1] == i
    b.shutdown()


def test_runs_that_only_read_do_not_make_the_index_longer(tmp_path):
    _fill(tmp_path)
    assert ENTRIES <= _lines(tmp_path) <= ENTRIES + 1, "precondition: one line per entry written"

    sizes = []
    for _ in range(6):
        _a_run_that_only_reads(tmp_path)
        sizes.append(_lines(tmp_path))

    assert max(sizes) <= RankIndex.CLOSE_RATIO * ENTRIES + SLACK + 1, sizes


def test_a_long_running_process_keeps_the_index_bounded(tmp_path):
    """A kernel that reads the same entries for hours never closes: the
    index is rewritten when a flush leaves it past the ratio."""
    _fill(tmp_path)
    b = _backend(tmp_path)
    seen = []
    for _ in range(12):
        for i in range(ENTRIES):
            b.get(f"k{i}")
        b._flush_metadata()
        b.evictor.rank_index.flush()
        seen.append(_lines(tmp_path))
    b.shutdown()

    assert max(seen) <= RankIndex.COMPACT_RATIO * ENTRIES + SLACK + RankIndex.BATCH, seen
    assert max(seen) > ENTRIES + 1, "precondition: the reads did append records"


def test_reading_an_overgrown_index_rewrites_it_for_the_entries_still_there(tmp_path):
    """What an index left by an earlier cash looks like: every entry many
    times over, and entries long gone."""
    _fill(tmp_path, count=20)
    index = tmp_path / INDEX
    records = [line for line in index.read_text(encoding="ascii").splitlines() if not line.startswith("@L")]
    assert len(records) == 20
    stems = [line.split()[0] for line in records]
    gone = [f"{i:064x}" for i in range(50)]
    body = "".join(f"{stem} 1.0\n" for stem in gone)
    body += "".join(f"{stem} {run}.5\n" for run in range(1, 40) for stem in stems)
    body += "@L 7.25\n"
    index.write_text(body, encoding="ascii")

    ranks, clock, count = RankIndex(str(tmp_path), untracked).load()

    assert count == 50 + 39 * 20 + 1
    assert clock == 7.25
    assert [ranks[s] for s in stems] == [39.5] * 20, "the last record of an entry is its priority"
    assert index.read_text(encoding="ascii").splitlines() == [f"{s} 39.5" for s in stems] + ["@L 7.25"]

    again = RankIndex(str(tmp_path), untracked).load()
    assert again == ({s: 39.5 for s in stems}, 7.25, 21)
    assert list(again[0]) == stems, "the order of the records, which breaks ties, is kept"


def test_an_index_near_its_size_is_not_rewritten(tmp_path):
    """The control: a rewrite on every read would pass everything above."""
    _fill(tmp_path, count=20)
    index = tmp_path / INDEX
    with open(index, "a", encoding="ascii") as fh:
        fh.write("".join(f"{i:064x} 1.0\n" for i in range(30)))  # entries gone, but under the ratio
    before = index.read_text(encoding="ascii")

    RankIndex(str(tmp_path), untracked).load()

    assert index.read_text(encoding="ascii") == before


def test_compacting_keeps_what_the_ranking_needs(tmp_path, monkeypatch):
    """An entry read often keeps its standing through the rewrites: the cap
    still evicts the cheap entries nobody reads first."""
    import os

    monkeypatch.setattr(RankIndex, "COMPACT_SLACK", 0)  # so an index of one entry is rewritten at all
    cache = tmp_path / "c"
    b = FileBackend(str(cache), max_size_bytes=10_000_000, flush_interval=0)
    b.set("costly", b"x" * 1_010_000, {"execution_time": 30.0})
    b.shutdown()
    for _ in range(5):
        reader = FileBackend(str(cache), max_size_bytes=10_000_000, flush_interval=0)
        assert reader.get("costly")[1] is not None
        reader.shutdown()
    assert _lines(cache) <= 3, "precondition: the index was rewritten"

    writer = FileBackend(str(cache), max_size_bytes=10_000_000, flush_interval=0)
    for i in range(10):
        writer.set(f"cheap-{i}", b"x" * 1_000_000, {"execution_time": 0.05})
        writer._writes.wait_all()
    assert os.path.exists(writer._get_path("costly"))
    assert not os.path.exists(writer._get_path("cheap-0"))
    writer.shutdown()


@pytest.mark.timeout(120)
def test_a_script_run_again_and_again_leaves_the_index_as_it_was(tmp_path):
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import cash

            @cash.cache
            def f(n):
                return n * 2

            for n in range(300):
                f(n)
            """
        ),
        encoding="utf-8",
    )
    cache = tmp_path / ".cash"
    sizes = []
    for _ in range(4):
        run_python("job.py", cwd=tmp_path)
        sizes.append(_lines(cache))
    assert sizes[0] >= 300, "precondition: the first run recorded its entries"
    assert max(sizes[1:]) <= RankIndex.CLOSE_RATIO * sizes[0] + SLACK, f"the index grew with the runs: {sizes}"
