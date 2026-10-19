"""A file changed in place through a memory map is read again, not trusted.

An unchanged file's digest is taken from its stat: size, modification time
and, on Linux and macOS, the inode change time. A write through
``np.memmap(path, mode="r+")`` can leave all of them as they were on
Windows, and the result computed from the old content was served. A file
this process maps to write is now read at every check, here and, through
the cache directory, in later processes.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from cash.tracking.file_dep_snapshot import file_content_hash, file_dep_is_fresh
from cash.tracking.written_in_place import is_written_in_place, use_written_in_place
from tests._scripts import run_python

pytestmark = pytest.mark.core

_OLD = time.time_ns() - 3_600 * 10**9


def _settled_file(path, values):
    np.asarray(values, dtype=np.float64).tofile(path)
    os.utime(path, ns=(_OLD, _OLD))
    return file_content_hash(str(path))


def _record_as_it_stands(path, digest):
    """A stored dependency whose metadata matches the file now, digest *digest*."""
    st = os.stat(path)
    return {
        "mtime": st.st_mtime,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": getattr(st, "st_ctime_ns", 0),
        "size": st.st_size,
        "dev": st.st_dev,
        "ino": st.st_ino,
        "hash": digest,
        "hashed_at": time.time(),
    }


@pytest.mark.parametrize(("mode", "writes"), [("r+", True), ("w+", True), ("r", False), ("c", False)])
def test_a_mapping_that_can_write_marks_the_file(tmp_path, mode, writes):
    path = tmp_path / f"a_{mode.replace('+', 'p')}.bin"
    np.zeros(4).tofile(path)
    m = np.memmap(path, dtype=np.float64, mode=mode, shape=(4,))
    del m
    assert is_written_in_place(os.stat(path), str(path)) is writes


def test_a_write_that_puts_the_metadata_back_is_seen(tmp_path):
    path = tmp_path / "data.bin"
    old = _settled_file(path, [1.0, 2.0, 3.0, 4.0])
    m = np.memmap(path, dtype=np.float64, mode="r+", shape=(4,))
    m[:] = [9.0, 9.0, 9.0, 9.0]
    m.flush()
    del m
    # What a Windows mapping write leaves: size and modification time as they
    # were (the record also takes this platform's change time as it is now).
    os.utime(path, ns=(_OLD, _OLD))
    assert file_dep_is_fresh(str(path), _record_as_it_stands(path, old)) == (False, "content")


def test_a_file_never_mapped_keeps_the_stat_shortcut(tmp_path):
    path = tmp_path / "plain.bin"
    old = _settled_file(path, [1.0, 2.0])
    # The digest is not read again: a wrong one is still taken as fresh.
    assert file_dep_is_fresh(str(path), _record_as_it_stands(path, "0" * 64)) == (True, None)
    assert old is not None


def test_a_later_process_reads_the_marks(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path = tmp_path / "shared.bin"
    np.zeros(4).tofile(path)
    script = tmp_path / "writer.py"
    script.write_text(
        "import sys, numpy as np\n"
        "from cash.tracking.written_in_place import use_written_in_place\n"
        "use_written_in_place(sys.argv[1])\n"
        "m = np.memmap(sys.argv[2], dtype=np.float64, mode='r+', shape=(4,))\n"
        "m[0] = 7.0\n"
        "m.flush()\n",
        encoding="utf-8",
    )
    p = run_python(str(script), str(cache_dir), str(path), cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert not is_written_in_place(os.stat(path), str(path)), "marked in this process already"
    use_written_in_place(str(cache_dir))
    assert is_written_in_place(os.stat(path), str(path))


def test_a_cached_call_recomputes_after_a_memmap_write(disk_cash, tmp_path):
    path = tmp_path / "weights.bin"
    np.asarray([1.0, 2.0], dtype=np.float64).tofile(path)
    os.utime(path, ns=(_OLD, _OLD))
    runs = []

    @disk_cash.cache
    def total(p):
        runs.append(1)
        return float(np.fromfile(p, dtype=np.float64).sum())

    assert total(str(path)) == 3.0
    assert total(str(path)) == 3.0
    m = np.memmap(path, dtype=np.float64, mode="r+", shape=(2,))
    m[:] = [5.0, 5.0]
    m.flush()
    del m
    os.utime(path, ns=(_OLD, _OLD))
    assert total(str(path)) == 10.0, "the old content's result was served"
    assert len(runs) == 2
