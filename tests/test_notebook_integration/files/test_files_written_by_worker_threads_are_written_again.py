"""A helper writing its files from threads it starts (``threading.Thread``,
``multiprocessing.pool.ThreadPool``) runs on every Run All, as in a plain
kernel: the folders cleared at the top get their files back. Worker
processes are covered by the same rule (unit tests); they are slow to tear
down in a kernel."""

import pytest

pytestmark = [pytest.mark.integration]

CELLS = [
    """import os, gzip, shutil, threading, time
from multiprocessing.pool import ThreadPool
def save_one(args):
    d, i = args
    f = gzip.open(f'{d}/part{i}.gz', 'wt'); f.write(str(i)); f.close()
    return i
def by_thread(d, n):
    time.sleep(0.3)
    ts = [threading.Thread(target=save_one, args=((d, i),)) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    return n
def by_threadpool(d, n):
    time.sleep(0.3)
    with ThreadPool(2) as p:
        p.map(save_one, [(d, i) for i in range(n)])
    return n
""",
    "for d in ['e', 'f']:\n    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)",
    "n4 = by_thread('e', 2)",
    "n5 = by_threadpool('f', 2)",
    "print({d: sorted(os.listdir(d)) for d in ['e', 'f']})",
]


def test_the_files_are_there_after_the_second_run_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().endswith("{'e': ['part0.gz', 'part1.gz'], 'f': ['part0.gz', 'part1.gz']}")
