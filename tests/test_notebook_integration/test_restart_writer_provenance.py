"""After a restart, a writer whose files are already on disk is not re-fired.

The re-execution planner re-fires every file writer above the cell after a
restart unless the writer's provenance -- the files it produced and their
state -- shows its effect is already there. Provenance came from the paths
the code names, so a writer whose paths live in a helper
(``save(k, v)`` writing ``OUT / f'f{k}.txt'``), or a loop over such calls,
had none: round 23's r23s3 re-ran a 263 s sweep after a restart to redraw
charts that were on disk. The writes are observed as they happen now
(``write_observer``) and recorded for the loop as a whole.

The last cell reads the files back: a writer whose files the cell being run
does not read is not re-fired at all (the planner's scope gate), and with
known paths that now applies to this writer too -- so only a reader tells
"fresh, skip it" apart from "stale, re-fire it".

Observed on disk, not through the kernel: a re-fired ``write_text`` gives the
file a new ``st_mtime_ns``. (Any peek into the new kernel runs an upstream
check of its own, which would re-fire a stale writer before a probe installed
by that peek could see it.)
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = ("import time\nfrom pathlib import Path\nOUT = Path('out')\nOUT.mkdir(exist_ok=True)\n"
         "def slow(k):\n    time.sleep(0.3)\n    return k * 10\n")
SAVE = "def save(k, v):\n    (OUT / f'f{k}.txt').write_text(str(v))"
SCORES = "scores = [slow(k) for k in range(3)]"
WRITE = "for k, v in enumerate(scores):\n    save(k, v)"
TOTAL = "total = sum(int((OUT / f'f{k}.txt').read_text()) for k in range(3))\nprint('T', total)"
CELLS = ["import cash\n%cash_on", SETUP, SAVE, SCORES, WRITE, TOTAL]


def _restart_and_run_last(nb_runner) -> str:
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(len(CELLS))
    return nb_runner.get_output(len(CELLS))


def _files(nb_runner) -> dict[str, tuple[str, int]]:
    out = Path(nb_runner.work_dir) / "out"
    return {p.name: (p.read_text(), p.stat().st_mtime_ns) for p in sorted(out.glob("*.txt"))}


def _contents(files) -> dict[str, str]:
    return {name: text for name, (text, _mtime) in files.items()}


def test_a_writer_whose_files_are_on_disk_is_not_refired(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    written = _files(nb_runner)
    assert _contents(written) == {"f0.txt": "0", "f1.txt": "10", "f2.txt": "20"}

    out = _restart_and_run_last(nb_runner)

    assert "T 30" in out, out
    # Measured before the fix: every file rewritten after the restart.
    assert _files(nb_runner) == written, "the writer re-fired after the restart"


def test_a_writer_whose_output_is_gone_is_refired(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    (Path(nb_runner.work_dir) / "out" / "f1.txt").unlink()

    out = _restart_and_run_last(nb_runner)

    assert "T 30" in out, out
    assert _contents(_files(nb_runner)) == {"f0.txt": "0", "f1.txt": "10", "f2.txt": "20"}


def test_a_writer_whose_helper_changed_is_refired(nb_runner):
    """``save`` is not named by the loop's inputs' lineages -- only by what
    the loop calls -- and an edited ``save`` writes something else."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, SAVE.replace("str(v)", "str(v + 1)"))

    out = _restart_and_run_last(nb_runner)

    assert "T 33" in out, out
    assert _contents(_files(nb_runner)) == {"f0.txt": "1", "f1.txt": "11", "f2.txt": "21"}
