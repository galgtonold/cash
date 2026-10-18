"""A pool whose module is first imported inside a cached call still reports its reads.

cash wraps ``ProcessPoolExecutor.submit`` and ``multiprocessing.Pool`` so a
worker's reads reach the cached call that handed it work. It wrapped them as
the first tracker opened, importing ``concurrent.futures.process`` to do it:
multiprocessing and subprocess loaded into every process that cached a call,
9-12 ms each, for a pool most never make. A pool's module is now wrapped when
the tracker opens if it is loaded by then, and as its import finishes if that
is later: ``multiprocessing.Pool()`` imports ``multiprocessing.pool`` inside
the cached call. (Reading ``cf.ProcessPoolExecutor`` in the source, the
decorator loads its module first.) Fresh processes each run, as a script
would: cold, warm (must hit), and after an edit to a file only the workers
read.
"""

from __future__ import annotations

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

WORKER = """\
def read_total(path):
    with open(path) as fh:
        return sum(float(x) for x in fh.read().split())
"""

JOB = """\
import sys, time
import cash
from worker import read_total

PATHS = ("a.txt", "b.txt")


def run(tag):
    print("[RUN]", tag, file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.15)  # @cash:assume-safe


@cash.cache
def plain(x):
    run("plain")
    return x + 1


@cash.cache
def by_executor(paths):
    run("executor")
    import concurrent.futures as cf

    with cf.ProcessPoolExecutor(2) as ex:
        return sum(ex.map(read_total, paths))


@cash.cache
def by_pool(paths):
    run("pool")
    import multiprocessing

    with multiprocessing.Pool(2) as pool:
        return sum(pool.map(read_total, paths))


if __name__ == "__main__":
    plain(1)
    print("[LOADED]", "multiprocessing.pool" in sys.modules, file=sys.stderr)
    print(by_executor(PATHS), by_pool(PATHS))
"""


PLAIN = """\
import sys
import cash


@cash.cache
def plain(x):
    return x + 1


plain(1)
print(sorted(m for m in ("concurrent.futures.process", "multiprocessing") if m in sys.modules))
"""


def test_caching_a_call_loads_no_pool_module(tmp_path):
    (tmp_path / "plain.py").write_text(PLAIN, encoding="utf-8")
    assert run_python("plain.py", cwd=tmp_path).stdout.strip() == "[]"


def _run(proj):
    p = run_python("job.py", cwd=proj, timeout=200)
    ran = {line.split(maxsplit=1)[1] for line in p.stderr.splitlines() if line.startswith("[RUN] ")}
    loaded = [line.split(maxsplit=1)[1] for line in p.stderr.splitlines() if line.startswith("[LOADED] ")]
    return p.stdout.strip(), ran, loaded


def test_a_pool_first_imported_in_the_call_brings_its_reads_back(tmp_path):
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "worker.py").write_text(WORKER, encoding="utf-8")
    (tmp_path / "a.txt").write_text("1 2 3", encoding="utf-8")
    (tmp_path / "b.txt").write_text("10 20", encoding="utf-8")

    out, ran, loaded = _run(tmp_path)
    assert (out, ran) == ("36.0 36.0", {"plain", "executor", "pool"})
    assert loaded == ["False"], "multiprocessing.pool was loaded before the call: the import hook is not tested"
    assert _run(tmp_path)[:2] == ("36.0 36.0", set()), "an unedited run did not hit"
    (tmp_path / "b.txt").write_text("10 20 3000", encoding="utf-8")
    assert _run(tmp_path)[:2] == ("3036.0 3036.0", {"executor", "pool"}), "the pre-edit total was served"
