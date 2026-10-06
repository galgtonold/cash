"""The partial file of a write whose process was killed is removed.

A disk write goes to a ``.tmp-*.part`` file that is renamed into place when
complete. A process killed in between (a kernel restart, an OOM kill during the
background write) left it behind for good: ``cash info`` and the disk cap count
only ``*.entry`` files, so up to an entry's size leaked per crash. The temp
file's name says which process writes it, and the first write of a later
process removes those whose process no longer runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from cash.backends import cache_dir
from cash.backends.file_backend import FileBackend


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _part(directory, owner: str) -> str:
    path = os.path.join(directory, f".tmp-{owner}-0123456789ab.part")
    with open(path, "wb") as fh:
        fh.write(b"x" * 1000)
    return path


def test_a_dead_writers_partial_file_goes_on_the_next_write(tmp_path):
    host = cache_dir.owned_temp_prefix()[len(".tmp-") :].split(".")[0]
    dead = _part(tmp_path, f"{host}.{_dead_pid()}")
    live = _part(tmp_path, f"{host}.{os.getpid()}")
    elsewhere = _part(tmp_path, f"ffff{host}.{_dead_pid()}")  # another machine's: cannot ask it

    backend = FileBackend(cache_dir=str(tmp_path))
    backend.set("k", [1, 2, 3], {})
    backend._writes.wait_all()
    assert backend.get("k")[1] == [1, 2, 3]
    backend.shutdown()

    assert not os.path.exists(dead)
    assert os.path.exists(live), "a write in progress lost its file"
    assert os.path.exists(elsewhere)
    left = {n for n in os.listdir(tmp_path) if n.endswith(".part")}
    assert left == {os.path.basename(live), os.path.basename(elsewhere)}


def test_one_nobody_has_written_for_a_day_goes_too(tmp_path):
    old = _part(tmp_path, "0123456789ab")  # no owner in the name
    recent = _part(tmp_path, "ba9876543210")
    day_ago = time.time() - cache_dir.ORPHAN_TEMP_AGE - 60
    os.utime(old, (day_ago, day_ago))

    assert cache_dir.remove_orphan_temp_files(str(tmp_path)) == 1
    assert not os.path.exists(old)
    assert os.path.exists(recent)
