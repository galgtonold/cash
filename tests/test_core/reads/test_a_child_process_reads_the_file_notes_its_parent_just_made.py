"""A process that a run starts reads the file digests and versions the run
has noted so far, and one that misses some only reads the file again.

The two logs in the cache directory are written a batch at a time
(`cash.tracking.log_lines`). A script a cell runs (``subprocess.run``) or a
worker it starts reads the same cache: the lines still waiting are written
before it starts. A process that ends by ``os._exit`` leaves its last lines
unwritten, and the next one hashes the file instead: never a stale answer.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import os
    import subprocess
    import sys
    import cash
    from cash import FileDataSource

    data, mode = os.path.abspath(sys.argv[1]), sys.argv[2]
    log = os.path.join(os.environ["CASH_CACHE_DIR"], "_file_digests.log")

    @cash.cache(dynamic_depends_on=lambda path: FileDataSource(path))
    def load(path):
        with open(path, "rb") as fh:
            return fh.read().decode()

    def settle(body, hours):
        with open(data, "w") as fh:
            fh.write(body)
        st = os.stat(data)
        os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns - hours * 3600 * 10**9))

    def child():
        done = subprocess.run([sys.executable, __file__, data, "child"], capture_output=True, text=True, check=True)
        return done.stdout.strip()

    if mode == "child":
        lines = len(open(log).read().splitlines()) if os.path.exists(log) else 0
        print(f"{load(data)}:{lines}")
    elif mode == "parent-first":
        settle("aaaa", 1)
        out = [load(data), child()]
        settle("bbbb", 2)
        out += [load(data), child()]
        print(" ".join(out))
    elif mode == "child-first":
        settle("aaaa", 1)
        out = [child(), load(data)]
        settle("bbbb", 2)
        out += [child(), load(data)]
        print(" ".join(out))
    elif mode == "lost-tail":
        settle(sys.argv[3], int(sys.argv[4]))
        print(load(data), flush=True)
        os._exit(0)
    """
)


def _run(tmp_path, *args):
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    return run_python("job.py", str(tmp_path / "data.txt"), *args, cwd=tmp_path).stdout.split()


def test_a_child_started_after_the_parent_read_sees_its_digest_and_the_new_bytes(tmp_path):
    first, child_one, second, child_two = _run(tmp_path, "parent-first")

    assert (first, second) == ("aaaa", "bbbb")
    assert child_one.split(":")[0] == "aaaa" and int(child_one.split(":")[1]) >= 1, "the waiting line was written first"
    assert child_two.split(":")[0] == "bbbb"


def test_a_parent_reading_after_its_child_gets_the_new_bytes(tmp_path):
    child_one, first, child_two, second = _run(tmp_path, "child-first")

    assert (child_one.split(":")[0], first) == ("aaaa", "aaaa")
    assert (child_two.split(":")[0], second) == ("bbbb", "bbbb")


def test_lines_lost_at_os_exit_cost_a_read_not_a_stale_answer(tmp_path):
    (tmp_path / "job.py").write_text(SCRIPT, encoding="utf-8")
    data = str(tmp_path / "data.txt")
    assert run_python("job.py", data, "lost-tail", "aaaa", "1", cwd=tmp_path).stdout.split() == ["aaaa"]
    assert run_python("job.py", data, "lost-tail", "bbbb", "2", cwd=tmp_path).stdout.split() == ["bbbb"]
    assert run_python("job.py", data, "lost-tail", "aaaa", "3", cwd=tmp_path).stdout.split() == ["aaaa"]
