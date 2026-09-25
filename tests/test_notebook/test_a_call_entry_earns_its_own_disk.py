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

#: The call's clock. A slow, loaded machine must not decide these tests: a
#: real 0.4 s sleep once measured 2 s, and 76 MiB built in 2 s is worth its
#: disk. ``big`` spends 0.4 s on this clock instead, and the call unit reads
#: only this clock, so the call's recorded cost is exactly 0.4 s.
_CLOCK = [0.0]


def spend(seconds: float) -> None:
    _CLOCK[0] += seconds


#: 76 MiB in 0.4 s: 190 MiB per second, over the ceiling -- but under twice
#: it, so the call is digested (``CallEntries._too_big_to_digest``) and its
#: size is exact, not the estimate that already faced the ceiling.
DEFS = f"import numpy as np\nfrom {__name__} import spend\ndef big(n):\n    spend(0.4)\n    return np.zeros(n)"
N = 10_000_000


@pytest.fixture
def tiers(cash_magics, cash_instance, tmp_path, monkeypatch):
    """``cash_magics`` over RAM and a disk tier, as ``%cash_on`` builds it,
    with the call unit on the test's clock."""
    _CLOCK[0] = 0.0
    monkeypatch.setattr("cash.notebook.call_unit._perf_counter", lambda: _CLOCK[0])
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
    assert held["execution_time"] == pytest.approx(0.4), "the call was not timed on the test's clock"
    assert not held.get("referenced") and "DISK" not in (held.get("storage") or []), held
    assert not _call_entries(disk), "a call nothing refers to was written to disk past the ceiling"


def test_a_call_worth_its_bytes_still_reaches_disk(tiers):
    """The control: the ceiling does not refuse every unreferenced call."""
    magics, ram, disk = tiers
    run_cash_cell(magics, DEFS)
    run_cash_cell(magics, f"b = big({N // 20}) + 1")
    disk._writes.wait_all()
    assert _call_entries(disk)
