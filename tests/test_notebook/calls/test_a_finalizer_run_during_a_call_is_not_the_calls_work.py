"""What a garbage collection runs while a call runs is not the call's work.

A collection starts wherever an allocation makes it, so the finalizers of
objects long gone run inside whatever call is under way. Run in the call's
context, a ``__del__`` that handed work to a pool or a thread counted as
work the call left running (`work_outlived_block`) whenever that work had
not finished as the call returned -- on a loaded machine, even a short task
-- and a ``__del__`` that wrote a file refused the call for the session. A
cheap, deterministic call then ran again on the next identical cell, in a
full test run and not alone. Collections run in a context of their own
(`cash._gc_context`), where no tracker of the call sees them.

The work here waits on an event, so it is still running when the call
returns, as a short task is on a busy machine.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SETUP = (
    "import gc, os, threading, time\n"
    "import concurrent.futures as cf\n"
    "import numpy as np\n"
    "release = threading.Event()\n"
    "pool = cf.ThreadPoolExecutor(1)\n"
    "class Junk:\n"
    "    def __init__(self, mode, path):\n"
    "        self.me, self.mode, self.path = self, mode, path  # a cycle: only a collection frees it\n"
    "    def __del__(self):\n"
    "        if self.mode == 'pool':\n"
    "            pool.submit(release.wait, 30)\n"
    "        elif self.mode == 'thread':\n"
    "            threading.Thread(target=release.wait, args=(30,), daemon=True).start()\n"
    "        else:\n"
    "            with open(self.path, 'a') as f:\n"
    "                f.write('x')\n"
    "def reads(a, log):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    gc.collect()  # where an allocation might have started one\n"
    "    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'.')\n"
    "    os.close(fd)\n"
    "    return float(a.sum())\n"
)


@pytest.mark.parametrize("mode", ["pool", "thread", "file"])
def test_a_call_a_collection_ran_finalizers_in_is_still_stored(cash_magics, mock_shell, tmp_path, mode):
    mock_shell.user_ns["__name__"] = "__main__"
    log = tmp_path / "runs"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, f"x = np.arange(100_000.0)\nlog = {str(log)!r}")
    ns = mock_shell.user_ns
    try:
        ns["Junk"](mode, str(tmp_path / "finalized"))  # garbage at once: freed by the call's collection
        run_cash_cell(cash_magics, "y = reads(x, log)")
        run_cash_cell(cash_magics, "y = reads(x, log)")
    finally:
        ns["release"].set()
        ns["pool"].shutdown(wait=True)
    assert mock_shell.user_ns["y"] == float(np.arange(100_000.0).sum())
    if mode == "file":
        assert (tmp_path / "finalized").read_text() == "x", "the finalizer never ran"
    assert log.read_bytes() == b".", "the call ran again"
