"""A call entry that no statement refers to earns its disk on its own.

Every call entry was written as ``referenced``: "the statement holding this
result decides whether it is worth its disk". So it skipped the bytes ceiling
(128 MiB per second of compute) and waited for a statement to judge it. But in
``b = big(n) + 1`` no statement holds the call's result -- ``b`` is a new
value, and a statement whose own work is cheap keeps nothing -- so nothing
ever judged it, and 76 MiB built in 0.4 s went to disk.

Only the call a statement is nothing but (``b = big(n)``) is referred to for
certain; that statement still weighs it.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell

#: 76 MiB in about 0.4 s: ~190 MiB per second, over the ceiling -- but under
#: twice it, so the call is digested (``CallEntries._too_big_to_digest``) and
#: its size is exact, not the estimate that already faced the ceiling.
DEFS = "import time\nimport numpy as np\ndef big(n):\n    time.sleep(0.4)\n    return np.zeros(n)"
N = 10_000_000


@pytest.fixture
def tiers(cash_magics, cash_instance, tmp_path):
    """``cash_magics`` over RAM and a disk tier, as ``%cash_on`` builds it."""
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    yield cash_magics, ram, disk
    disk.shutdown()


def _call_entries(tier):
    return [m for m in tier.list_entries() or () if str(m.get("key", "")).startswith("call:")]


def test_a_call_no_statement_refers_to_faces_the_ceiling(tiers):
    magics, ram, disk = tiers
    run_cash_cell(magics, DEFS)
    with pytest.warns(Warning, match="CACHE-NOT-WORTH-BYTES"):
        run_cash_cell(magics, f"b = big({N}) + 1")
    assert float(magics.shell.user_ns["b"][0]) == 1.0
    disk._writes.wait_all()
    [held] = _call_entries(ram)
    assert not held.get("referenced") and "DISK" not in (held.get("storage") or []), held
    assert not _call_entries(disk), "a call nothing refers to was written to disk past the ceiling"


def test_a_call_worth_its_bytes_still_reaches_disk(tiers):
    """The control: the ceiling does not refuse every unreferenced call."""
    magics, ram, disk = tiers
    run_cash_cell(magics, DEFS)
    run_cash_cell(magics, f"b = big({N // 20}) + 1")
    disk._writes.wait_all()
    assert _call_entries(disk)
