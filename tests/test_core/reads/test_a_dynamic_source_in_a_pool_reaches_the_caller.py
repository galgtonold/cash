"""A ``dynamic_depends_on=`` source of a cached call made in a pool reaches the caller.

A cached orchestrator fanned a cached loader out to ``multiprocessing.Pool``,
``ProcessPoolExecutor`` or ``multiprocessing.pool.ThreadPool``. The files the
tasks read came back to the orchestrator; the loader's sources did not, so
after the source's version moved the orchestrator kept its old result.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

JOB = """
    import multiprocessing as mp, multiprocessing.pool as mpp, os, subprocess, sys
    from concurrent.futures import ProcessPoolExecutor
    import cash
    from cash import DataSource

    # Where the version lives: read in a child process, which cash does not
    # see, so only the source ties a result to it.
    VERSION = os.path.abspath("version")

    def put(version):
        with open(VERSION, "w") as fh:
            fh.write(version)

    class Table(DataSource):
        def __init__(self, name):
            self.name = name

        def get_id(self):
            return "table:" + self.name

        def state_token(self):
            code = f"print(open({VERSION!r}).read())"
            return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout

    @cash.cache(dynamic_depends_on=lambda name: Table(name), assume_safe=True)
    def load(name):
        return name  # reads the table in a way cash does not see

    RUNS = []

    @cash.cache(assume_safe=True)
    def by_thread_pool():
        RUNS.append("thread_pool")
        with mpp.ThreadPool(2) as p:
            return p.map(load, ["a"])

    @cash.cache(assume_safe=True)
    def by_mp_pool():
        RUNS.append("mp_pool")
        with mp.Pool(2) as p:
            return p.map(load, ["a"])

    @cash.cache(assume_safe=True)
    def by_process_pool():
        RUNS.append("process_pool")
        with ProcessPoolExecutor(2) as ex:
            return list(ex.map(load, ["a"]))

    class Locked(Table):
        # Holds a lock, as a source holding a connection would: no pickle.
        def __init__(self, name):
            super().__init__(name)
            import threading
            self.lock = threading.Lock()

    @cash.cache(dynamic_depends_on=lambda name: Locked(name), assume_safe=True)
    def load_locked(name):
        return name

    @cash.cache(assume_safe=True)
    def by_process_pool_unpicklable():
        RUNS.append("unpicklable")
        with ProcessPoolExecutor(2) as ex:
            return list(ex.map(load_locked, ["a"]))

    if __name__ == "__main__":
        put("1")
        by_process_pool_unpicklable()
        by_process_pool_unpicklable()
        print("unpicklable", RUNS.count("unpicklable"))
        fs = [by_thread_pool, by_mp_pool, by_process_pool]
        for f in fs:
            f()
        RUNS.clear()
        for f in fs:
            f()
        print("warm", len(RUNS))
        put("2")
        for f in fs:
            f()
        print("moved", len(RUNS))
"""


def test_a_pool_task_s_dynamic_source_reaches_the_caller(tmp_path):
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    out = run_python("job.py", cwd=tmp_path).stdout.split("\n")
    assert "warm 0" in out, out
    assert "moved 3" in out, out
    # Its source cannot come back from the worker: the caller is not stored.
    assert "unpicklable 2" in out, out
