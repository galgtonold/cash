"""A file read in a ``multiprocessing.Pool`` or joblib worker is an input of
the cached call that handed out the work.

Only ``ProcessPoolExecutor`` brought its workers' reads back. joblib's
default backend (loky, behind ``Parallel(n_jobs=...)`` and every
scikit-learn ``n_jobs=``) and ``multiprocessing.Pool`` did not, and nothing
said so: after the input was edited, the old result was served. Fresh
processes each run, as a script would: cold, warm (must hit), and after the
edit.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.timeout(300)]

WORKER = """\
def read_total(path):
    with open(path) as fh:
        return sum(float(x) for x in fh.read().split())


def read_scaled(path, factor):
    return read_total(path) * factor
"""

JOB = """\
import sys
# The class, not `multiprocessing.Pool`: that is a method of the default
# context, whose state changes the first time any pool starts, and the key of
# a later function using it moves with it.
from multiprocessing.pool import Pool, ThreadPool
import cash
from worker import read_scaled, read_total

PATHS = ("a.txt", "b.txt")


def run(tag):
    print("[RUN]", tag, file=sys.stderr)  # @cash:assume-safe


@cash.cache
def by_map(paths):
    run("map")
    with Pool(2) as pool:
        return sum(pool.map(read_total, paths))


@cash.cache
def by_imap(paths):
    run("imap")
    with Pool(2) as pool:
        return sum(pool.imap_unordered(read_total, paths, chunksize=2))


@cash.cache
def by_starmap(paths):
    run("starmap")
    with Pool(2) as pool:
        return sum(pool.starmap(read_scaled, [(p, 1) for p in paths]))


@cash.cache
def by_apply(paths):
    run("apply")
    seen = []
    with Pool(2) as pool:
        results = [pool.apply_async(read_total, (p,), callback=seen.append) for p in paths]
        total = sum(r.get() for r in results)
    assert sorted(seen) == sorted(r.get() for r in results), seen
    return total


@cash.cache
def by_thread_pool(paths):
    run("thread-pool")
    with ThreadPool(2) as pool:
        return sum(pool.map(read_total, paths))


@cash.cache
def by_joblib(paths):
    run("joblib")
    from joblib import Parallel, delayed
    return sum(Parallel(n_jobs=2)(delayed(read_total)(p) for p in paths))


if __name__ == "__main__":
    forms = [by_map, by_imap, by_starmap, by_apply, by_thread_pool, by_joblib]
    print(" ".join(str(f(PATHS)) for f in forms))
"""

FORMS = {"map", "imap", "starmap", "apply", "thread-pool", "joblib"}


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=250)
    assert p.returncode == 0, p.stderr[-2000:]
    ran = {line.split(maxsplit=1)[1] for line in p.stderr.splitlines() if line.startswith("[RUN] ")}
    return p.stdout.strip(), ran


def test_a_file_a_pool_or_joblib_worker_read_invalidates_the_caller(tmp_path):
    pytest.importorskip("joblib")
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "worker.py").write_text(WORKER, encoding="utf-8")
    (tmp_path / "a.txt").write_text("1 2 3", encoding="utf-8")
    (tmp_path / "b.txt").write_text("10 20", encoding="utf-8")

    assert _run(tmp_path) == (" ".join(["36.0"] * 6), FORMS)
    assert _run(tmp_path) == (" ".join(["36.0"] * 6), set()), "an unedited run did not hit"
    (tmp_path / "b.txt").write_text("10 20 3000", encoding="utf-8")
    assert _run(tmp_path) == (" ".join(["3036.0"] * 6), FORMS), "the pre-edit total was served"
