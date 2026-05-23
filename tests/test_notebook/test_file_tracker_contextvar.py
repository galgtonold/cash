"""FileAccessTracker must be isolated across asyncio tasks and threads."""
from __future__ import annotations

import asyncio
import threading

from cash.notebook.file_tracker import FileAccessTracker


def test_two_threads_isolated(tmp_path):
    """Tracker installed in one thread does not capture file reads in another."""
    p_a = tmp_path / "a.txt"; p_a.write_text("a")
    p_b = tmp_path / "b.txt"; p_b.write_text("b")

    result = {}
    barrier = threading.Barrier(2)

    def thread_with_tracker():
        t = FileAccessTracker()
        with t:
            barrier.wait()
            with open(p_a) as f:
                f.read()
            barrier.wait()  # give the other thread time to read p_b
        result["with_tracker"] = {x for x in t.get_accessed_files()}

    def thread_without_tracker():
        barrier.wait()  # sync entry into the tracked block
        with open(p_b) as f:
            f.read()
        barrier.wait()

    t1 = threading.Thread(target=thread_with_tracker)
    t2 = threading.Thread(target=thread_without_tracker)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # The tracker only saw its own thread's read.
    assert any(str(p_a) in r or r.endswith("a.txt") for r in result["with_tracker"])
    assert not any(r.endswith("b.txt") for r in result["with_tracker"])


async def test_two_async_tasks_isolated(tmp_path):
    """asyncio.gather: each task's tracker only sees its own file reads."""
    p_a = tmp_path / "a.txt"; p_a.write_text("a")
    p_b = tmp_path / "b.txt"; p_b.write_text("b")

    async def reader(path):
        t = FileAccessTracker()
        with t:
            await asyncio.sleep(0)  # yield, let the other task interleave
            with open(path) as f:
                f.read()
            await asyncio.sleep(0)
        return t.get_accessed_files()

    files_a, files_b = await asyncio.gather(reader(p_a), reader(p_b))

    assert any(r.endswith("a.txt") for r in files_a)
    assert not any(r.endswith("b.txt") for r in files_a)
    assert any(r.endswith("b.txt") for r in files_b)
    assert not any(r.endswith("a.txt") for r in files_b)


def test_nested_with_blocks(tmp_path):
    """Nested tracker contexts: inner block sees only its own reads;
    outer block resumes capturing after inner exits."""
    p_a = tmp_path / "a.txt"; p_a.write_text("a")
    p_b = tmp_path / "b.txt"; p_b.write_text("b")
    p_c = tmp_path / "c.txt"; p_c.write_text("c")

    outer = FileAccessTracker()
    with outer:
        with open(p_a) as f:
            f.read()
        inner = FileAccessTracker()
        with inner:
            with open(p_b) as f:
                f.read()
        with open(p_c) as f:
            f.read()

    outer_files = outer.get_accessed_files()
    inner_files = inner.get_accessed_files()
    # Outer should see a.txt and c.txt; inner should see b.txt only.
    assert any(r.endswith("a.txt") for r in outer_files)
    assert any(r.endswith("c.txt") for r in outer_files)
    assert any(r.endswith("b.txt") for r in inner_files)
    # And the nesting boundary: b should NOT be in outer.
    assert not any(r.endswith("b.txt") for r in outer_files)
