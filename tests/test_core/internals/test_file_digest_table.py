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
