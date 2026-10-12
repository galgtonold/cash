"""A statement whose result depends on whether a file exists runs again
once the answer changes, as a plain kernel would compute it.

``todo = pending(NAMES)`` (``[n for n in names if not
os.path.exists(f"out/{n}.txt")]``) kept its cached list after an output was
deleted, and a flag check inline in a cell kept its old answer: the work that
had to be redone was silently skipped. The cells built on the list follow it.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

SETUP = """import os, time
from pathlib import Path
NAMES = ['a', 'b', 'c']
def pending(names):
    time.sleep(0.3)
    return [n for n in names if not os.path.exists(f'out/{n}.txt')]
"""
CELLS = [
    "%cash_badge print",
    SETUP,
    "todo = pending(NAMES)",
    "ready = os.path.exists('flag.txt') and time.sleep(0.3) is None",
    "labels = [n.upper() for n in todo]\ntime.sleep(0.3)",
    "there = Path('out/b.txt').is_file() and time.sleep(0.3) is None",
    "print('RESULT', todo, ready, labels, there)",
]


def _result(nb_runner) -> str:
    """What the last cell printed, without the badge lines."""
    lines = [line for line in nb_runner.get_output(7).splitlines() if line.startswith("RESULT ")]
    return lines[-1][len("RESULT ") :] if lines else nb_runner.get_output(7)


def _prepare(work: Path) -> None:
    (work / "out").mkdir()
    (work / "out" / "a.txt").write_text("1", encoding="utf-8")
    (work / "out" / "b.txt").write_text("2", encoding="utf-8")
    (work / "flag.txt").write_text("x", encoding="utf-8")


def test_a_deleted_output_and_flag_are_seen(nb_runner):
    work = Path(nb_runner.work_dir)
    _prepare(work)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert _result(nb_runner) == "['c'] True ['C'] True"
    (work / "out" / "a.txt").unlink()
    (work / "out" / "b.txt").unlink()
    (work / "flag.txt").unlink()
    nb_runner.run_all()
    assert _result(nb_runner) == "['a', 'b', 'c'] False ['A', 'B', 'C'] False"


def test_an_unchanged_answer_hits_after_a_restart(nb_runner):
    """Nothing moved: the statements are served, also after a restart."""
    work = Path(nb_runner.work_dir)
    _prepare(work)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert _result(nb_runner) == "['c'] True ['C'] True"
    for cell in (3, 4, 5, 6):
        assert "CACHED" in nb_runner.get_raw_output(cell), f"cell {cell} ran again with nothing changed"
    (work / "out" / "c.txt").write_text("3", encoding="utf-8")
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert _result(nb_runner) == "[] True [] True"
