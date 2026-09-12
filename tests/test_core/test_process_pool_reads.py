"""A file a ProcessPoolExecutor task reads is an input of the cached call that submitted it.

Round 20 (r20s3): a cached orchestrator fanned per-region work out to a
``ProcessPoolExecutor``, and the workers read the CSVs. After a data fix in one
of them the orchestrator served the pre-fix report -- the reads happened in
other processes, where no tracker of the parent's could see them -- while the
thread-pool version beside it invalidated. Fresh processes each run, as a
script would: cold, warm (must hit), and after the edit.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

WORKER = '''\
def read_total(path):
    with open(path) as fh:
        return sum(float(x) for x in fh.read().split())
'''

JOB = '''\
import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import cash
from worker import read_total

PATHS = ("a.txt", "b.txt")


def run(tag):
    print("[RUN]", tag, file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe


@cash.cache
def by_map(paths):
    run("map")
    with ProcessPoolExecutor(2) as ex:
        return sum(ex.map(read_total, paths))


@cash.cache
def by_chunked_map(paths):
    run("chunked-map")
    with ProcessPoolExecutor(2) as ex:
        return sum(ex.map(read_total, paths, chunksize=2))


@cash.cache
def by_submit(paths):
    run("submit")
    with ProcessPoolExecutor(2) as ex:
        futures = [ex.submit(read_total, p) for p in paths]
        return sum(f.result() for f in as_completed(futures))


if __name__ == "__main__":
    print(by_map(PATHS), by_chunked_map(PATHS), by_submit(PATHS))
'''

FORMS = {"map", "chunked-map", "submit"}


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stderr[-2000:]
    ran = {line.split(maxsplit=1)[1] for line in p.stderr.splitlines()
           if line.startswith("[RUN] ")}
    return p.stdout.strip(), ran


def test_a_file_a_process_pool_task_read_invalidates_the_submitter(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "worker.py").write_text(WORKER, encoding="utf-8")
    (tmp_path / "a.txt").write_text("1 2 3", encoding="utf-8")
    (tmp_path / "b.txt").write_text("10 20", encoding="utf-8")

    assert _run(tmp_path) == ("36.0 36.0 36.0", FORMS)
    assert _run(tmp_path) == ("36.0 36.0 36.0", set()), "an unedited run did not hit"
    (tmp_path / "b.txt").write_text("10 20 3000", encoding="utf-8")
    assert _run(tmp_path) == ("3036.0 3036.0 3036.0", FORMS), "the pre-edit total was served"
