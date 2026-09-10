"""A multiprocessing worker's cache writes survive the pool that ran it.

CAS-124, round-17 tester r17s4 (F22). `with multiprocessing.Pool() as pool:`
-- the stdlib idiom -- calls `terminate()` on exit. Cache writes run on
daemon threads, so each worker's LAST write was still in flight when it was
killed, and those tasks recomputed on every later run: two of six, forever,
with no warning. (Under the fork start method a worker exits through
`os._exit`, which skips the exit-time drain even with close() + join().)

In a multiprocessing child the write is now part of the task.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


def _project(tmp_path):
    (tmp_path / "kern.py").write_text(textwrap.dedent("""
        import sys, time
        import numpy as np
        import cash

        @cash.cache(assume_safe=True)
        def kernel(seed):
            print(f"@@CALL {seed}", file=sys.stderr)
            time.sleep(0.2)
            # Big enough that its write is still running when the pool exits.
            return np.random.default_rng(seed).standard_normal((2000, 1500))
    """), encoding="utf-8")
    (tmp_path / "job.py").write_text(textwrap.dedent("""
        import multiprocessing as mp, sys
        import kern

        def work(seed):
            return float(kern.kernel(seed)[:, -1].var())   # a small summary back

        if __name__ == "__main__":
            with mp.Pool(2) as pool:                        # __exit__ terminates
                print(pool.map(work, range(4)))
    """), encoding="utf-8")


def _run(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    out = subprocess.run([sys.executable, "-W", "ignore", "job.py"], cwd=str(tmp_path),
                         capture_output=True, text=True, env=env, timeout=240)
    assert out.returncode == 0, out.stderr
    return sorted(int(s) for s in re.findall(r"@@CALL (\d+)", out.stderr))


def test_with_pool_keeps_every_workers_writes(tmp_path):
    """THE LOSS: the second run recomputed each worker's last task."""
    _project(tmp_path)
    assert _run(tmp_path) == [0, 1, 2, 3]
    assert _run(tmp_path) == [], "a worker's write was lost when the pool exited"


def test_the_main_process_is_not_a_child():
    """The control: the parent keeps its background writes."""
    from cash.backends._base import _in_multiprocessing_child
    assert _in_multiprocessing_child() is False


def _report(q):
    from cash.backends._base import _in_multiprocessing_child
    q.put(_in_multiprocessing_child())


def test_a_worker_process_is_one():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_report, args=(q,))
    p.start()
    try:
        assert q.get(timeout=60) is True
    finally:
        p.join(timeout=60)
