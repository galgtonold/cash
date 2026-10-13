"""``clean = F.preprocess(imgs)`` with a ``@cash.cache`` ``preprocess``
stores the result once.

The decorator stored the result under the call's key, and the statement
then stored the very same object again under its own: a second RAM copy
and a second disk write of a 98 MB image stack (p23), of a 200-300 MB
design matrix (p27), on every compute. When the statement only binds the
call's result to a name, the call tells it which entry the result went
into (`cash.handed_digests.stored_result`), and the statement's entry
refers to that entry instead of holding a copy.

What must still hold: a hit of the statement after a restart of RAM gives
the value without running the body; a statement that changes the result
after binding it keeps its own copy; and a statement whose decorator entry
is gone runs again rather than restoring something else. The body sleeps
past the persistence floor so the statement's own entry reaches the disk.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from cash._active import default_cash, set_default_cash
from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

#: Elements of the result: 4 MiB of float64.
N = 1 << 19
NBYTES = N * 8

LIB = (
    "import cash\n"
    "import time\n"
    "import numpy as np\n"
    "RUNS = []\n"
    "@cash.cache\n"
    "def make(n):\n"
    "    RUNS.append(n)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return np.arange(n, dtype=float)\n"
)


@pytest.fixture
def shared(cash_magics, cash_instance, mock_shell, tmp_path, monkeypatch):
    """The notebook and the decorator over one RAM + disk backend, as in a
    kernel where the notebook's Cash is the default one; and a module
    ``lib`` holding the cached ``make``."""
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    previous = default_cash()
    set_default_cash(cash_instance)
    name = f"_stored_once_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {name} as lib")
    yield cash_magics, ram, disk, sys.modules[name]
    disk._writes.wait_all()
    disk.shutdown()
    set_default_cash(previous)
    sys.modules.pop(name, None)


def _disk_bytes(disk) -> int:
    disk._writes.wait_all()
    return sum(int(meta.get("size") or 0) for meta in disk.list_entries() or ())


def test_the_result_is_written_once(shared):
    magics, _ram, disk, lib = shared
    run_cash_cell(magics, f"x = lib.make({N})")
    assert lib.RUNS == [N]
    written = _disk_bytes(disk)
    assert NBYTES <= written < 2 * NBYTES, f"{written} bytes on disk for one {NBYTES}-byte result"


def test_the_statement_restores_it_from_the_call_entry(shared, mock_shell):
    magics, ram, disk, lib = shared
    run_cash_cell(magics, f"x = lib.make({N})")
    disk._writes.wait_all()
    ram.clear()
    del mock_shell.user_ns["x"]
    run_cash_cell(magics, f"x = lib.make({N})")
    assert lib.RUNS == [N], "the body ran again"
    np.testing.assert_array_equal(mock_shell.user_ns["x"], np.arange(N, dtype=float))


def test_a_result_changed_by_the_statement_keeps_its_own_copy(shared, mock_shell):
    magics, ram, disk, lib = shared
    run_cash_cell(magics, f"x = lib.make({N}); x[0] = -1.0")
    disk._writes.wait_all()
    ram.clear()
    del mock_shell.user_ns["x"]
    run_cash_cell(magics, f"x = lib.make({N}); x[0] = -1.0")
    assert mock_shell.user_ns["x"][0] == -1.0
    assert mock_shell.user_ns["x"][1] == 1.0


def test_a_statement_whose_call_entry_is_gone_runs_again(shared, mock_shell):
    magics, ram, disk, lib = shared
    run_cash_cell(magics, f"x = lib.make({N})")
    disk._writes.wait_all()
    calls = [meta["key"] for meta in disk.list_entries() or () if meta.get("decorator_entry")]
    assert len(calls) == 1
    disk.delete(calls[0])
    ram.clear()
    del mock_shell.user_ns["x"]
    run_cash_cell(magics, f"x = lib.make({N})")
    np.testing.assert_array_equal(mock_shell.user_ns["x"], np.arange(N, dtype=float))
    assert len(lib.RUNS) == 2
