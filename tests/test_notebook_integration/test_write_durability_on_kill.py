"""Cached results must be on disk by the time a cell reports done (CAS-209).

Cache writes are asynchronous, and nothing drains the queue when a kernel is
*killed* rather than shut down — a crash, an OOM, a force-quit, or a tool that
terminates the process instead of asking it to exit. Anything still queued at
that moment is lost: the badge reported the result as cached, and after the
restart it is not there.

This asserts the invariant directly — every entry the cell cached is readable
from disk while the kernel is still alive — rather than asserting a downstream
symptom. A restart test can pass for the wrong reason (writes happening to win
the race against process death), which is precisely how this went unnoticed.
"""
import os

import pytest
from cash.backends.entry_format import ENTRY_SUFFIX

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _chain_cells(cache_dir: str):
    cdir = cache_dir.replace("\\", "/")
    return [
        "import cash\nfrom cash import Cash, FileBackend\n"
        f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))",
        "import time\n"
        "def base(x):\n    return x + 1\n"
        "@c.cache\n"
        "def load(x):\n"
        "    time.sleep(0.15)\n"
        "    return base(x) * 2\n"
        "@c.cache(depends_on=[load])\n"
        "def mid(x):\n    return load(x) + 10\n"
        "@c.cache(depends_on=[mid])\n"
        "def top(x):\n    return mid(x) + 100",
        "vals = [top(s) for s in (1, 2, 3)]\n"
        "print(f'RESULT vals={vals}')",
    ]


def test_every_cached_entry_is_on_disk_when_the_cell_returns(nb_runner, tmp_path):
    """No entry may still be queued once the cell that produced it is done.

    Three cached functions x three arguments = nine entries, one file each.
    Counted while the kernel is still running, so a shortfall means work was
    left in the write queue where a kill would take it.
    """
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_chain_cells(cache_dir))
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "RESULT vals=[114, 116, 118]" in nb_runner.get_output(3)

    files = os.listdir(cache_dir)
    entries = sorted(f for f in files if f.endswith(ENTRY_SUFFIX))

    assert len(entries) == 9, (
        f"only {len(entries)}/9 entries reached disk before the cell finished; "
        f"a kernel killed now would lose the rest. Files: {sorted(files)}"
    )
    assert not [f for f in files if f.endswith((".meta", ".data"))], (
        f"a half-written entry in the pre-v2 two-file layout: {sorted(files)}"
    )
