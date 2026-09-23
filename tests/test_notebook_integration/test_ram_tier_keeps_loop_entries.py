"""A folder loop's per-file frames survive a later cell that overflows the RAM tier.

Round 23, r23s2 (2026-09-14): ``for f in files: d = pd.read_csv(f)`` over 1,312
files, then a cleaning cell that cached a few full-table copies of the result.
The copies pushed the RAM tier past its byte cap, and the cap evicted least
recently used first -- every per-file frame, since the loop ran earlier. Those
frames exist only in RAM (a 20 ms read is below the disk tier's promotion floor),
so the next run of the loop read all 1,312 files again: 42 s instead of 5.

The cap now evicts by value per byte (GreedyDual-Size-Frequency): the copies
took milliseconds per megabyte, the reads tens of milliseconds, so the copies go.

Counted, not timed: a tee on ``StatementProcessor.process_statement`` in the
live kernel counts the loop's ``read_csv`` statements by status. The RAM cap is
set small in the kernel so a small fixture overflows it, and the host-memory
pressure check is switched off (it reads the whole machine's memory, and under
a parallel run it fires on its own).
"""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

N = 40
CAP = 40 * 2**20

_TEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if not hasattr(C, "_test_orig"):
    C._test_orig = C.process_statement
    def _tee(self, code, *a, **k):
        r = C._test_orig(self, code, *a, **k)
        try:
            if "read_csv" in str(code):
                s = r.get("status")
                s = str(getattr(s, "value", s))
                C._test_n[s] = C._test_n.get(s, 0) + 1
        except Exception:
            pass
        return r
    C.process_statement = _tee
C._test_n = {}
"""
_UNTEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if hasattr(C, "_test_orig"):
    C.process_statement = C._test_orig
    del C._test_orig
"""
_RAM = "__import__('cash')._global_cash.backend.backends[0]"
_SMALL_CAP = f"(setattr({_RAM}, '_max_size_bytes', {CAP}), setattr({_RAM}, 'max_memory_percent', 1.0))"
_COUNTS = "__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor._test_n"
_RESET = f"{_COUNTS}.clear()"

SETUP = "import glob\nimport os\nimport numpy as np\nimport pandas as pd\nfiles = sorted(glob.glob('exports/*.csv'))"
LOOP = (
    "parts = []\n"
    "for f in files:\n"
    "    d = pd.read_csv(f)\n"
    "    d['source_file'] = os.path.basename(f)\n"
    "    parts.append(d)\n"
    "raw = pd.concat(parts, ignore_index=True)\n"
    "print('rows', len(raw))"
)
#: Eight 8 MB frames at 50 ms each (~6 ms per MB, where a 10 ms read of a
#: ~50 KB file is ~200): 64 MB of cheap-per-byte values, written after the
#: loop, against a 40 MB cap. The sleep gives each a cost the cost model
#: caches -- ``w + k`` alone took a few ms and was never stored.
BIG = (
    "import time\n"
    "def shifted(frame, k):\n"
    "    time.sleep(0.05)\n"
    "    return frame + k\n"
    "w = pd.DataFrame(np.zeros((250_000, 4)))\n"
    + "\n".join(f"w{k} = shifted(w, {k})" for k in range(1, 9))
    + "\nprint('w', len(w8))"
)


@pytest.fixture
def _teed(nb_runner):
    yield
    try:
        nb_runner.peek(f"exec({_UNTEE!r}, {{}})")
    except Exception:  # noqa: BLE001 - a kernel that never started has nothing to undo
        pass


def _files(work):
    folder = Path(work) / "exports"
    folder.mkdir()
    for i in range(N):
        path = folder / f"e{i:03d}.csv"
        path.write_text("a,b,c\n" + "\n".join(f"{i},{j},{j * 0.5}" for j in range(2000)) + "\n")
        st = os.stat(path)
        os.utime(path, (st.st_atime - 3600, st.st_mtime - 3600))


def _loop_reads(nb_runner):
    nb_runner.peek(_RESET)
    nb_runner.run_cell(3)
    assert f"rows {N * 2000}" in nb_runner.get_output(3)
    return ast.literal_eval(nb_runner.peek(_COUNTS))


def test_a_later_cell_overflowing_the_ram_tier_leaves_the_loop_cached(nb_runner, _teed):
    _files(nb_runner.work_dir)
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, LOOP, BIG])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    assert nb_runner.peek(f"type({_RAM}).__name__") == "'InMemoryBackend'"
    nb_runner.peek(_SMALL_CAP)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)

    # Control arm: with nothing in between, the loop's reads are restored --
    # so a miss below is the eviction, not a loop that never caches.
    before = _loop_reads(nb_runner)
    assert before.get("RESTORED", 0) == N, f"unchanged re-run: {before}"

    nb_runner.run_cell(4)
    assert "w 250000" in nb_runner.get_output(4)
    after = _loop_reads(nb_runner)
    # Measured on plain LRU: {'COMPUTED': 40} -- every read evicted.
    assert after.get("RESTORED", 0) == N, f"re-run after an overflowing cell: {after}"
