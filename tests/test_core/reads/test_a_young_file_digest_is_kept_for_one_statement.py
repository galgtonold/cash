"""A file too young for the digest memo (written within
``_HASH_MEMO_MIN_AGE_SECONDS``) is hashed once for a statement: its digest
is kept while the statement is processed (``young_digests_kept``), under
its whole stat -- size, modification and change times, the file's
identity -- and forgotten when the statement ends or cash sees it write."""

from __future__ import annotations

import os
import sys

from cash.notebook.write_observer import observe_writes
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import file_content_hash, young_digests_kept
from tests._work_counts import method_calls


def _young(tmp_path, text="a,b\n1,2\n"):
    path = tmp_path / "young.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_hashed_once_within_the_block(tmp_path):
    path = _young(tmp_path)
    with young_digests_kept(), method_calls(file_dep_snapshot, "_read_digest") as reading:
        first = file_content_hash(path)
        assert file_content_hash(path) == first
    assert reading.calls == 1


def test_hashed_each_time_outside_it(tmp_path):
    path = _young(tmp_path)
    with young_digests_kept():
        file_content_hash(path)
    with method_calls(file_dep_snapshot, "_read_digest") as reading:
        file_content_hash(path)
        file_content_hash(path)
    assert reading.calls == 2


def test_a_change_of_stat_reads_it_again(tmp_path):
    path = _young(tmp_path)
    with young_digests_kept():
        before = file_content_hash(path)
        st = os.stat(path)
        # Same size, and the modification time put back: the change time
        # still moves. On Windows ``st_ctime`` is the creation time, so no
        # stat sees such an edit (a documented limit): the mtime moves on.
        with open(path, "r+", encoding="utf-8") as f:
            f.write("x")
        moved = 1_000_000_000 if sys.platform == "win32" else 0
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + moved))
        assert os.stat(path).st_size == st.st_size
        assert file_content_hash(path) != before


def test_a_write_cash_sees_forgets_the_digests(tmp_path):
    path = _young(tmp_path)
    with young_digests_kept(), method_calls(file_dep_snapshot, "_read_digest") as reading:
        file_content_hash(path)
        with observe_writes():
            # Opened to write, nothing written: the stat stays as it was.
            open(path, "a", encoding="utf-8").close()
        file_content_hash(path)
    assert reading.calls == 2
