"""A settled file a block reads is hashed when the block ends, not inside the read.

Each file a statement reads is hashed as it is read. Inside the read hook a
2 KB file's digest cost ~116 us, against ~15 us for the same work done
together when the block ends: a statement reading 6,000 files paid ~0.6 s
for it. A file the digest memo already trusts by its stat -- settled, with
a file identity, never mapped to write -- is hashed when the block ends
now, and only while its stat is still the one the read took
(`file_dep_snapshot.digest_later`). A file that moved in between is never
fresh: the digest of what is there now would vouch for a value computed
from something else. A write in this process takes the digests first.

What must hold: the digest is the one a read-time digest gives; a file
rewritten during the statement, with a new modification time or with the
old one put back, is read again on the next run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from cash.bulk_digest import bulk_digest
from cash.tracking import file_dep_snapshot, pending_digest
from cash.tracking.file_tracker import FileAccessTracker
from tests._cell_driver import run_cash_cell


def _settled(path, text: str):
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    return path


@pytest.fixture
def deferred(monkeypatch):
    """The deferred digests made, by path."""
    made: list[str] = []
    real = file_dep_snapshot.digest_later

    def counting(path, size, st):
        out = real(path, size, st)
        if isinstance(out, pending_digest.DeferredDigest):
            made.append(os.path.basename(path))
        return out

    monkeypatch.setattr("cash.tracking.file_tracker.digest_later", counting)
    file_dep_snapshot._HASH_MEMO.clear()
    return made


@pytest.fixture
def stores_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


def test_the_deferred_digest_is_the_read_time_one(tmp_path, deferred, monkeypatch):
    path = _settled(tmp_path / "doc.txt", "Terms " * 300)
    monkeypatch.setattr(file_dep_snapshot, "current_table", lambda: None)
    with FileAccessTracker(hash_on_read=True) as tracker:
        with open(path, encoding="utf-8") as fh:
            fh.read()
    assert deferred == ["doc.txt"]
    digest = tracker.read_digests[os.path.realpath(path).replace("\\", "/")]
    file_dep_snapshot._HASH_MEMO.clear()
    assert digest == file_dep_snapshot.file_content_hash(str(path)) == bulk_digest(path.read_bytes()).hex()


def test_a_fresh_file_is_hashed_as_it_is_read(tmp_path, deferred):
    path = tmp_path / "doc.txt"
    path.write_text("just written", encoding="utf-8")
    with FileAccessTracker(hash_on_read=True), open(path, encoding="utf-8") as fh:
        fh.read()
    assert deferred == []


def test_a_write_in_the_block_takes_the_digest_first(tmp_path, deferred):
    path = _settled(tmp_path / "doc.txt", "old text")
    with FileAccessTracker(hash_on_read=True) as tracker:
        with open(path, encoding="utf-8") as fh:
            fh.read()
        assert pending_digest.DEFERRED, "nothing was deferred"
        path.write_text("NEW TEXT", encoding="utf-8")
        assert not pending_digest.DEFERRED, "the write did not take the digest first"
    digest = tracker.read_digests[os.path.realpath(path).replace("\\", "/")]
    assert digest == bulk_digest(b"old text").hex()


REWRITE = textwrap.dedent(
    """
    import os, sys
    path, text, keep_mtime = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    before = os.stat(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    if keep_mtime:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    """
)


@pytest.fixture
def rewritten_as_the_block_ends(monkeypatch):
    """Rewrite a file by another process just before the block's deferred
    digests are taken: nothing in this process sees it happen."""
    armed: list = []
    real = FileAccessTracker._settle_deferred

    def rewrite_first(self):
        if armed:
            path, keep_mtime = armed.pop()
            subprocess.run(
                [sys.executable, "-c", REWRITE, str(path), "NEW TEXT", "1" if keep_mtime else "0"], check=True
            )
        return real(self)

    monkeypatch.setattr(FileAccessTracker, "_settle_deferred", rewrite_first)
    return armed


@pytest.mark.parametrize(
    "keep_mtime",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows keeps no inode change time: a documented limitation"
            ),
        ),
    ],
    ids=["new-mtime", "mtime-put-back"],
)
def test_a_file_rewritten_during_its_statement_is_read_again(
    cash_magics, stores_everything, tmp_path, deferred, rewritten_as_the_block_ends, keep_mtime
):
    path = _settled(tmp_path / "doc.txt", "old text")
    cell = f"text = open({str(path).replace(chr(92), '/')!r}, encoding='utf-8').read()"
    rewritten_as_the_block_ends.append((path, keep_mtime))
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"] == "old text"
    assert "doc.txt" in deferred and not rewritten_as_the_block_ends
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"] == "NEW TEXT", "the value read from the old file was served"


OPENED: list[str] = []
sys.addaudithook(lambda event, args: event == "open" and isinstance(args[1], str) and OPENED.append(str(args[0])))


def test_an_unchanged_settled_file_is_served(cash_magics, stores_everything, tmp_path, deferred):
    path = _settled(tmp_path / "doc.txt", "old text")
    spelled = str(path).replace("\\", "/")
    cell = f"text = open({spelled!r}, encoding='utf-8').read()"
    run_cash_cell(cash_magics, cell)
    assert "doc.txt" in deferred
    cash_magics.shell.user_ns["text"] = None
    OPENED.clear()
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"] == "old text"
    assert spelled not in OPENED, "the cell ran again"


def test_the_rewrite_helper_runs(tmp_path):
    path = _settled(tmp_path / "doc.txt", "old text")
    before = os.stat(path)
    subprocess.run([sys.executable, "-c", REWRITE, str(path), "NEW TEXT", "1"], check=True)
    after = os.stat(path)
    assert path.read_text(encoding="utf-8") == "NEW TEXT"
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
