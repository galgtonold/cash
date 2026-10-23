"""Lines a log keeps waiting are written before this process starts another.

A child may read the log (a cache's notes of the files it read). The flush
rides io_watch's one audit hook: an audit hook of its own was called by
Python for every audited event in the process, every ``id()`` included,
and made each about five times slower under the C extension's hook.
"""

from __future__ import annotations

import subprocess
import sys
import threading

from cash.tracking import io_watch, log_lines


def test_a_waiting_line_is_in_the_file_a_child_reads(tmp_path):
    path = tmp_path / "notes.log"
    lock = threading.Lock()
    log = log_lines.LogLines(str(path), lock, "TEST")
    with lock:
        log.add("a line\n")
    assert not path.exists()
    read = subprocess.run(
        [sys.executable, "-c", "import sys; print(open(sys.argv[1]).read(), end='')", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert read.stdout == "a line\n"


def test_the_flush_is_one_of_io_watch_s_consumers():
    assert all(log_lines._flush_before_start in io_watch._always.get(event, ()) for event in log_lines._STARTS)


def test_new_code_is_counted_by_the_hook_in_place(tmp_path):
    """``exec`` and a function made from code, counted with no hook of
    its own: by the C one, or by io_watch's Python one."""
    from cash.notebook import held_functions

    before = held_functions._new_code()
    exec("pass", {})
    type(test_new_code_is_counted_by_the_hook_in_place)(compile("1", "<t>", "exec"), {})
    assert held_functions._new_code() >= before + 2
