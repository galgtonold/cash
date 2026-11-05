"""Each pass of a loop is written to disk only when a re-run saves more than the write costs.

Writing an entry takes time on the run that makes it. A pass that computes
for a long time and stores little is written whatever the disk: its re-run
saves seconds and the write costs milliseconds. A pass that computes
quickly and stores a lot is not written where the write costs more than the
pass: a loop storing fifteen 22 MB frames of text, 1.4 s each, wrote at
11 MB/s, ~2 s a pass. An accumulator (``acc = step(acc, i)``) is the
extreme: every in-between state is written whole, and only resuming the
loop half way could read one back. Its last state is what the cell leaves,
judged at the cell's end by what rebuilding the whole chain would cost, and
written when that pays for its write.

The disk is made slow here (20 MiB/s, measured) so the verdict does not
depend on the machine; refusing only ever means the pass runs again.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell

MiB = 1 << 20

LIB = (
    "import time\n"
    "import numpy as np\n"
    "def slow_small(i):\n"
    "    time.sleep(0.25)\n"
    "    return [i] * 10\n"
    "def step(acc, i):\n"
    "    time.sleep(0.15)\n"
    "    return acc + i\n"
)

LOOPS = (
    "small = []\n"
    "for i in range(3):\n"
    "    r = lib.slow_small(i)\n"
    "    small.append(r)\n"
    "acc = np.zeros(2_000_000)\n"  # 15.3 MiB, past the bytes ceiling's floor
    "for i in range(8):\n"
    "    acc = lib.step(acc, i)\n"
)


class _SlowDisk:
    """A disk measured at 20 MiB/s for every kind, whatever it really does
    (the write queue's `write_cost.WriteRates`, its rate pinned)."""

    def record(self, kind: str, nbytes: int, seconds: float) -> None:
        return None

    def per_byte(self, kind: str) -> float | None:
        return 1.0 / (20 * MiB)

    def slowest_per_byte(self) -> float | None:
        return self.per_byte("")

    def seconds(self, kind: str, nbytes: int, assumed: float) -> float:
        return nbytes * self.per_byte(kind)


@pytest.fixture
def notebook(cash_magics, cash_instance, tmp_path, monkeypatch):
    disk = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    disk._writes.rates = _SlowDisk()
    cash_instance.backend = TieredBackend([InMemoryBackend(), disk])
    name = f"_pays_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = [f"import {name} as lib\nimport numpy as np", LOOPS]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    yield cash_magics, disk, cells
    disk._writes.wait_all()
    disk.shutdown()
    sys.modules.pop(name, None)


def _on_disk(disk, code: str) -> list:
    disk._writes.wait_all()
    found = []
    for entry in disk.list_entries() or ():
        meta, value = disk.get(entry["key"])
        if meta and code in (meta.get("code") or "") and value is not None:
            found.append(value)
    return found


def test_slow_small_passes_are_written_and_fast_large_states_only_last(notebook):
    magics, disk, cells = notebook
    run_cash_cell(magics, cells[1], cells=cells)

    slow = _on_disk(disk, "lib.slow_small(i)")
    assert len(slow) == 3, f"{len(slow)} of 3 slow, small passes on disk"

    states = _on_disk(disk, "lib.step(acc, i)")
    sums = sorted(float(v["variables"]["acc"].sum()) for v in states)
    final = 2_000_000 * sum(range(8))
    assert sums == [final], f"accumulator states on disk: {sums}, want only the last ({final})"


def test_a_fast_disk_writes_every_pass(notebook):
    """Positive control: the same loop over a disk fast enough writes every state."""
    magics, disk, cells = notebook

    class _FastDisk(_SlowDisk):
        def per_byte(self, kind: str) -> float | None:
            return 1.0 / (4096 * MiB)

    disk._writes.rates = _FastDisk()
    run_cash_cell(magics, cells[1], cells=cells)
    assert len(_on_disk(disk, "lib.step(acc, i)")) == 8
    assert np.isclose(magics.shell.user_ns["acc"].sum(), 2_000_000 * sum(range(8)))
