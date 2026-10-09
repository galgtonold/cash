"""A settled input file's digest is taken once, not once per process or per 5 s.

A ``FileDataSource`` token is the file's content digest, taken when the key is
built: before, it was remembered for five seconds in one process, so a
service hit every few minutes and every new process read a 500 MB source in
full (3.45 s a hit for a function that reads it in 0.46 s). Now a settled
file's digest is kept, under its exact stat, in the cache directory
(``_file_digests.log``) and reused while the stat holds. An edit moves the
stat -- on Linux and macOS even one that puts the mtime back -- and the file
is read again.
"""

from __future__ import annotations

import io
import os
import textwrap
import types

import pytest

from cash.tracking import file_dep_snapshot
from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    import cash
    from cash import FileDataSource

    target = os.path.abspath(sys.argv[1])
    reads = []
    sys.addaudithook(lambda event, args: event == "open" and isinstance(args[0], str)
                     and os.path.abspath(args[0]) == target and reads.append(1))
    runs = []

    @cash.cache(dynamic_depends_on=lambda path: FileDataSource(path))
    def load(path):
        runs.append(1)
        with open(path, "rb") as fh:
            return len(fh.read())

    print(load(sys.argv[1]), len(runs), len(reads))
    """
)


def _settled(path, body: bytes):
    path.write_bytes(body)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    return path


def test_a_new_process_does_not_read_an_unchanged_source(tmp_path):
    data = _settled(tmp_path / "data.bin", b"x" * 100_000)
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    first = run_python("job.py", str(data), cwd=tmp_path).stdout.split()
    assert first[:2] == ["100000", "1"], first
    second = run_python("job.py", str(data), cwd=tmp_path).stdout.split()
    assert second == ["100000", "0", "0"], f"a hit in a new process read the source: {second}"
    assert (tmp_path / ".cash" / "_file_digests.log").exists()


def test_an_edited_source_is_read_and_misses(tmp_path):
    data = _settled(tmp_path / "data.bin", b"x" * 100_000)
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    run_python("job.py", str(data), cwd=tmp_path)
    _settled(data, b"y" * 99_999)
    out = run_python("job.py", str(data), cwd=tmp_path).stdout.split()
    assert out[:2] == ["99999", "1"], f"an edited source was served the old result: {out}"


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps no inode change time: a documented limitation")
def test_an_edit_that_puts_the_mtime_back_is_read(tmp_path):
    data = _settled(tmp_path / "data.bin", b"x" * 100_000)
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    run_python("job.py", str(data), cwd=tmp_path)
    before = os.stat(data)
    with open(data, "r+b") as fh:
        fh.write(b"Z")
    os.utime(data, ns=(before.st_atime_ns, before.st_mtime_ns))
    out = run_python("job.py", str(data), cwd=tmp_path).stdout.split()
    assert out[1] == "1", f"a same-size edit with the mtime put back was served the old result: {out}"


def test_a_digest_is_reused_after_five_seconds(tmp_path, monkeypatch):
    """The in-process reuse no longer ends after five seconds."""
    data = str(_settled(tmp_path / "data.bin", b"x" * 10_000))
    file_dep_snapshot._HASH_MEMO.clear()
    _no_table(monkeypatch)
    reads: list[str] = []
    real = io.FileIO
    monkeypatch.setattr(file_dep_snapshot, "io", types.SimpleNamespace(FileIO=lambda p, m: (reads.append(p), real(p, m))[1]))
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    first = file_dep_snapshot.file_content_hash(data)
    clock[0] += 3600
    assert file_dep_snapshot.file_content_hash(data) == first
    assert len(reads) == 1


def _no_table(monkeypatch):
    """Only this process's memo: no cache directory's table from another test."""
    try:
        from cash.tracking import digest_table
    except ImportError:  # a build without one
        return
    monkeypatch.setattr(digest_table, "_TABLE", None)
