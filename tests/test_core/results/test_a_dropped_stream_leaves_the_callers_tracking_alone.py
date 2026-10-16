"""A cached generator dropped half-read leaves the running call's tracking alone.

A stream keeps its file tracker and effect observer across ``yield``. When
its last reference goes while some other code runs -- the garbage collector,
or a list cleared inside another cached call -- Python closes it right there,
in that code's context. Its cleanup reset its ContextVar tokens there and
rewound the running call's tracker: the file that call read next was left
out of its key, and after the file changed the old result was served.
"""

from __future__ import annotations

import threading

import pytest


def _half_read_stream(cash, started_on_a_thread):
    @cash.cache
    def rows(n):
        yield from range(n)

    held = []

    def start():
        g = rows(5)
        next(g)
        held.append(g)

    if started_on_a_thread:
        worker = threading.Thread(target=start)
        worker.start()
        worker.join()
    else:
        start()
    return held


@pytest.mark.parametrize("started_on_a_thread", [False, True], ids=["here", "on-a-thread"])
def test_a_file_read_after_a_stream_is_dropped_is_still_a_dependency(disk_cash, tmp_path, started_on_a_thread):
    data = tmp_path / "data.txt"
    data.write_text("one", encoding="utf-8")
    pending = _half_read_stream(disk_cash, started_on_a_thread)

    @disk_cash.cache(assume_safe=True)
    def load(path):
        pending.clear()  # the stream's last reference: it is closed here
        with open(path, encoding="utf-8") as f:
            return f.read()

    assert load(str(data)) == "one"
    data.write_text("two", encoding="utf-8")
    assert load(str(data)) == "two", "the edited file was not a dependency"
