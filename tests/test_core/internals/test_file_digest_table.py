"""The table of file digests in the cache directory (`digest_table`):
what it keeps and what it refuses to read back."""

from __future__ import annotations

import os

import pytest

from cash.tracking import digest_table, file_dep_snapshot


def test_a_file_written_moments_ago_is_not_kept(tmp_path):
    table = digest_table.DigestTable(str(tmp_path))
    fresh = tmp_path / "fresh.bin"
    fresh.write_bytes(b"x" * 1000)
    old_table = digest_table._TABLE
    digest_table._TABLE = table
    try:
        file_dep_snapshot.file_content_hash(str(fresh))
    finally:
        digest_table._TABLE = old_table
    assert not os.path.exists(table.path) or os.path.getsize(table.path) == 0


@pytest.mark.parametrize(
    "line",
    [
        "not json\n",
        '["cash-bulk-sha256-tree-0","",1,2,3,4,5,"' + "a" * 64 + '",1.0]\n',  # another scheme
        '["cash-bulk-sha256-tree-1","",1,2,3,4,5,"' + "a" * 63 + '",1.0]\n',  # a short digest
        '["cash-bulk-sha256-tree-1","",1,2,3,4,5,"' + "a" * 64,  # torn
    ],
)
def test_a_line_the_table_cannot_trust_is_skipped(tmp_path, line):
    (tmp_path / "_file_digests.log").write_text(line, encoding="utf-8")
    table = digest_table.DigestTable(str(tmp_path))
    assert table.get(("", 1, 2, 3, 4, 5)) is None


def test_a_line_the_table_wrote_is_read_back(tmp_path):
    table = digest_table.DigestTable(str(tmp_path))
    table.put(("", 1, 2, 3, 4, 5), "b" * 64, 12.5)
    assert digest_table.DigestTable(str(tmp_path)).get(("", 1, 2, 3, 4, 5)) == ("b" * 64, 12.5)


def test_lines_are_written_together_and_none_is_lost_at_exit(tmp_path):
    from tests._scripts import run_python

    script = tmp_path / "job.py"
    script.write_text(
        "import sys\n"
        "from cash.tracking import digest_table\n"
        "table = digest_table.DigestTable(sys.argv[1])\n"
        "for i in range(600):\n"
        "    table.put(('p', 1, i, 3, 4, 5), 'c' * 64, 1.0)\n",
        encoding="utf-8",
    )
    run_python(str(script), str(tmp_path), cwd=tmp_path)

    table = digest_table.DigestTable(str(tmp_path))
    assert all(table.get(("p", 1, i, 3, 4, 5)) == ("c" * 64, 1.0) for i in range(600)), "the last batch at exit"


def test_a_table_read_in_the_same_process_sees_lines_still_waiting(tmp_path):
    from cash.tracking import log_lines

    table = digest_table.DigestTable(str(tmp_path))
    table.put(("q", 1, 2, 3, 4, 5), "d" * 64, 2.0)
    assert digest_table.DigestTable(str(tmp_path)).get(("q", 1, 2, 3, 4, 5)) == ("d" * 64, 2.0)
    log_lines.flush_all()
    assert len((tmp_path / "_file_digests.log").read_text(encoding="utf-8").splitlines()) == 1, "written once"


def test_a_compaction_keeps_every_digest_this_process_used(tmp_path, monkeypatch):
    """E10 read 50,000 files in one statement: the table kept the newest
    20,000 at its compaction, and each restart read the other 30,000 again."""
    monkeypatch.setattr(digest_table, "_COMPACT_AT", 50)
    monkeypatch.setattr(digest_table, "_KEEP", 20)
    old = digest_table.DigestTable(str(tmp_path))
    for i in range(10):
        old.put(("old", 1, i, 3, 4, 5), "e" * 64, float(i))
    old.flush()

    table = digest_table.DigestTable(str(tmp_path))
    for i in range(60):
        table.put(("new", 1, i, 3, 4, 5), "f" * 64, 100.0 + i)
    table.flush()

    after = digest_table.DigestTable(str(tmp_path))
    assert all(after.get(("new", 1, i, 3, 4, 5)) == ("f" * 64, 100.0 + i) for i in range(60))
    assert after.get(("old", 1, 0, 3, 4, 5)) is None, "past _KEEP, an entry nobody used goes"


def test_a_compaction_keeps_the_newest_unused_up_to_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(digest_table, "_COMPACT_AT", 30)
    monkeypatch.setattr(digest_table, "_KEEP", 20)
    table = digest_table.DigestTable(str(tmp_path))
    for i in range(25):
        table.put(("a", 1, i, 3, 4, 5), "a" * 64, float(i))
    table.flush()
    table = digest_table.DigestTable(str(tmp_path))
    for i in range(10):
        table.put(("b", 1, i, 3, 4, 5), "b" * 64, 100.0 + i)
    table.flush()
    kept = digest_table.DigestTable(str(tmp_path))
    assert all(kept.get(("b", 1, i, 3, 4, 5)) is not None for i in range(10))
    # Compacted at the 6th put (31 lines): the 6 used, and the newest 14 of the rest.
    assert [i for i in range(25) if kept.get(("a", 1, i, 3, 4, 5)) is not None] == list(range(11, 25))


@pytest.mark.parametrize(
    "key, hashed_at",
    [
        (('C:\\data\\é "q".csv', 2**63, 2**64, 0, -5, 7), 1760000000.123456),
        (("/x/\U0001f600\n", 1, 2, 3, 4, 5), 1760000000),
        (("", 1, 2, 3, 4, 5), 1e-7),
    ],
)
def test_a_line_is_written_as_json_writes_it(key, hashed_at):
    import json

    expected = json.dumps([digest_table._SCHEME, *key, "a" * 64, hashed_at], separators=(",", ":")) + "\n"
    assert digest_table._line(key, "a" * 64, hashed_at) == expected
