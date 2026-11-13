"""A cheap value kept on disk for what rebuilding it costs stays there once restored.

``forest_reg = make_pipeline(preprocessing, RandomForestRegressor())`` took
0.2 ms, over a ``preprocessing`` fitted by a big statement kept in memory
only: the end of its cell wrote it to disk for what a restart would re-run
to get it back (``rebuild_time``). After a restart its first restore took
longer than its own 0.2 ms, so the disk tier deleted it as no faster than
computing it, and the next cell found it "no longer in the cache" and
rebuilt the chain behind it, a grid search included: 62 s for a cell that
prints ``get_params()``. A restore is now judged against what the entry
was kept for, the rebuild, not the statement's own time.

And a value the runtime restored this session whose entry is gone since
(evicted, or deleted) is still known to the upstream check: it takes the
lineages the restore gave the outputs, not a lineage built from the files
the value inherited, which no run of the statement makes, so the cells
below it are served rather than rebuilt -- and still run again when one of
those files changes.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time\nimport numpy as np\nfrom box import Box",
    # 64 MB in 0.2 s: past the bytes a second of compute earns on disk
    # (`worth_its_bytes`), whatever the machine, so kept in memory only.
    "raw = (time.sleep(0.2), np.tile(np.frombuffer(open('data.bin', 'rb').read(), dtype=np.uint8), 64_000))[1]",
    # A cell the tests edit to have the cells below it simulated again.
    "note = 'a'",
    # Under a millisecond over an input that took 0.2 s: kept on disk for its
    # rebuild. Reading it back takes 30 ms (`BOX`), longer than its own time.
    "small = Box(int(raw[::64_000].sum()))",
    "print('TOTAL', small.total)",
    "print('TWICE', small.total * 2)",
]

#: A value slower to read back than to make, as a fitted pipeline is
#: after a restart, whatever the machine.
BOX = """
import time


class Box:
    def __init__(self, total):
        self.total = total

    def __setstate__(self, state):
        time.sleep(0.03)
        self.__dict__.update(state)
"""

_DISK = "__import__('cash._active', fromlist=['x']).default_cash().backend.backends[1]"
_SMALL_ON_DISK = (
    f"[(m.get('execution_time', 0) < 0.05, m.get('rebuild_time', 0) > 0.1) for d in [{_DISK}] "
    "for e in d.list_entries() for m in [d.peek_metadata(e['key']) or dict()] "
    "if m.get('code', '').startswith('small = ')]"
)
_DROP_SMALL = (
    "[__import__('cash._active', fromlist=['x']).default_cash().backend.delete(e['key']) "
    f"for d in [{_DISK}] for e in d.list_entries() "
    "if (d.peek_metadata(e['key']) or dict()).get('code', '').startswith('small = ')]"
)


def _write(path, byte: int) -> None:
    path.write_bytes(bytes([byte]) * 1000)


def _first_run_then_restore(nb_runner) -> None:
    _write(nb_runner.work_dir / "data.bin", 1)
    (nb_runner.work_dir / "box.py").write_text(BOX, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TWICE 2000" in nb_runner.get_output(7), nb_runner.get_raw_output(7)
    stored = nb_runner.get_raw_output(3)
    assert "-> RAM\n" in stored + "\n" and "raw = " in stored, f"control: the reader is kept in memory only\n{stored}"
    nb_runner.settle_writes()
    assert nb_runner.peek(_SMALL_ON_DISK) == "[(True, True)]", "control: kept on disk for its rebuild"

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert "TOTAL 1000" in out and shows_cached(out), nb_runner.get_raw_output(6)
    assert nb_runner.peek("'raw' in globals()") == "False", nb_runner.get_raw_output(6)


def test_its_first_restore_keeps_it_and_the_cell_below_is_served(nb_runner):
    _first_run_then_restore(nb_runner)
    nb_runner.settle_writes()
    # Measured before the fix: '[]' -- the restore deleted it.
    assert nb_runner.peek(_SMALL_ON_DISK) == "[(True, True)]", "the restore deleted the entry"

    nb_runner.run_cell(7)
    out = nb_runner.get_output(7)
    assert "TWICE 2000" in out, nb_runner.get_raw_output(7)
    assert nb_runner.peek("'raw' in globals()") == "False", "rebuilt:\n" + nb_runner.get_raw_output(7)


def test_a_restored_value_whose_entry_is_gone_is_served_below(nb_runner):
    _first_run_then_restore(nb_runner)
    nb_runner.settle_writes()
    nb_runner.peek(_DROP_SMALL)
    nb_runner.set_cell_source(4, "note = 'b'")
    assert nb_runner.peek(_SMALL_ON_DISK) == "[]", "control: the entry is gone"

    nb_runner.run_cell(7)
    out = nb_runner.get_output(7)
    assert "TWICE 2000" in out, nb_runner.get_raw_output(7)
    # Measured before the fix: 'True' -- the upstream check took ``small``
    # for stale and rebuilt it from the file.
    assert nb_runner.peek("'raw' in globals()") == "False", "rebuilt:\n" + nb_runner.get_raw_output(7)


def test_a_restored_value_whose_entry_is_gone_runs_again_when_its_file_changes(nb_runner):
    _first_run_then_restore(nb_runner)
    nb_runner.settle_writes()
    nb_runner.peek(_DROP_SMALL)
    nb_runner.set_cell_source(4, "note = 'b'")
    _write(nb_runner.work_dir / "data.bin", 2)

    nb_runner.run_cell(7)
    out = nb_runner.get_output(7)
    assert "TWICE 4000" in out, nb_runner.get_raw_output(7)
