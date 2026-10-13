"""A helper that writes its files from worker processes or threads it starts
runs every time, as a plain kernel does.

The write observer sees the statement's own process and context; files
written in a ``ProcessPoolExecutor``, ``multiprocessing.Pool``,
``joblib.Parallel``, ``ThreadPool`` or ``threading.Thread`` worker were not
seen, the statement was stored, and the next Run All skipped every write.
Whether the work writes cannot be told, so a statement that starts workers is
not stored. ``ThreadPoolExecutor`` runs its tasks in the caller's context, so
a write there is seen as it is, and a statement whose work writes nothing
there is stored.
"""

from __future__ import annotations

import pytest

from cash.analysis.namespace_effects import statement_starting_workers
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def by_thread(d):
    import threading

    t = threading.Thread(target=print, args=(d,))
    t.start()
    t.join()


def by_pool(d):
    import multiprocessing as mp

    with mp.Pool(2) as p:
        p.map(print, [d])


def by_parallel(d):
    from joblib import Parallel, delayed

    Parallel(n_jobs=2)(delayed(print)(d) for _ in range(2))


def outer(d):
    return by_pool(d)


def plain(d):
    return [d] * 2


@pytest.mark.parametrize("fn", [by_thread, by_pool, by_parallel, outer])
def test_a_helper_starting_workers_is_found(fn):
    assert statement_starting_workers(f"n = {fn.__name__}('x')", {fn.__name__: fn, "by_pool": by_pool})


def test_a_pool_in_the_statement_itself_is_found():
    code = "with ProcessPoolExecutor(2) as ex:\n    out = list(ex.map(f, xs))"
    assert statement_starting_workers(code, {}) == "ProcessPoolExecutor"


def test_a_plain_helper_is_not():
    assert statement_starting_workers("n = plain('x')", {"plain": plain}) is None


THREADED = f"""import gzip, os, threading, time
from concurrent.futures import ThreadPoolExecutor
COUNTER = None
def save_one(p):
    f = gzip.open(p, 'wt'); f.write('x'); f.close()
def by_thread(d):
    fd = os.open(COUNTER, os.O_WRONLY | os.O_APPEND | os.O_CREAT); os.write(fd, b'x'); os.close(fd)  # @cash:assume-safe
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    t = threading.Thread(target=save_one, args=(os.path.join(d, 'part.gz'),))
    t.start(); t.join()
    return 1
"""


def test_the_files_a_thread_writes_are_written_on_every_run(cash_magics, tmp_path):
    counter = tmp_path / "runs"
    out = tmp_path / "out"
    out.mkdir()
    cells = [THREADED + f"COUNTER = {str(counter)!r}", f"n = by_thread({str(out)!r})"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        (out / "part.gz").unlink(missing_ok=True)
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (out / "part.gz").exists()
    assert counter.read_bytes() == b"xxx"
