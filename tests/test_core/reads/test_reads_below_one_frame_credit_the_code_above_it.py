"""Every read is credited to the user code on the stack above it
(`credit_read_to_stack`). A notebook statement's reads walk the same few dozen
frames above cash's runner each time, so the walk stops at a cash frame it
walked from before: the user code above that frame is credited as before,
every read, and the frame is not kept once the block ends.

Nested trackers also take one stat of a read between them: the enclosing
tracker records the stat the inner one took.
"""

from __future__ import annotations

import os

import cash.tracking.file_tracker as file_tracker
import cash.tracking.read_credit as read_credit
from cash.tracking.file_tracker import FileAccessTracker

_CASH_DIR = os.path.dirname(os.path.abspath(read_credit.__file__))


def _function(source: str, filename: str, name: str, namespace: dict):
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def test_reads_below_a_walked_frame_credit_the_user_code_above_it(tmp_path):
    files = [tmp_path / f"part{i}.txt" for i in range(3)]
    for f in files:
        f.write_text("x", encoding="utf-8")
    ns: dict = {"credit": read_credit.credit_read_to_stack}
    # A library frame does the reading, under a frame of cash's own (a
    # notebook statement's runner), under the user's code.
    ns["read"] = _function("def read(path, tracker):\n    credit(path, tracker)\n", "<library>", "read", ns)
    ns["run"] = _function(
        "def run(paths, tracker):\n    for p in paths:\n        read(p, tracker)\n",
        os.path.join(_CASH_DIR, "runner_in_this_test.py"),
        "run",
        ns,
    )
    user = _function("def user(paths, tracker):\n    run(paths, tracker)\n", str(tmp_path / "user_code.py"), "user", ns)
    paths = [str(f) for f in files]
    tracker = FileAccessTracker()
    with tracker:
        user(paths, tracker)
        anchor = tracker.credit_anchor
    assert anchor is not None and anchor[0].f_code is ns["run"].__code__
    assert user.__code__ in tracker.reading_codes
    assert set(paths) <= set(read_credit._reads_by_code[user.__code__])
    assert tracker.credit_anchor is None


def test_an_enclosing_tracker_takes_the_stat_of_the_inner_read(tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("x" * 10, encoding="utf-8")
    outer = FileAccessTracker(propagate_to_parent=True)
    inner = FileAccessTracker(propagate_to_parent=True)
    lstat = os.lstat(data)  # before the trackers: a stat in one is a read of its own
    stats = []
    real = file_tracker.regular_file_stat
    monkeypatch.setattr(file_tracker, "regular_file_stat", lambda p: stats.append(p) or real(p))
    with outer, inner:
        inner.add_tracked(str(data), lstat=lstat)
    assert stats == []
    assert outer.read_stats[str(data)] == inner.read_stats[str(data)]
