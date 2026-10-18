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
