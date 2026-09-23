"""A value a later statement of the same cell replaces stays in RAM.

A user cleaned ``sales`` in one cell, rebuilding the ~500 MB frame
statement by statement (drop duplicates, parse timestamps, fix a till clock,
merge two lookups). Each version that cleared the persistence floor was
pickled to disk as it was made -- 3.8 s of a cold run -- though the next
statement replaced it and nothing restores it: the end-of-cell pass writes
the cell's final version for the cells below.
"""

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from cash.backends.entry_format import read_entry  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = (
    "import os, time\n"
    "import numpy as np\n"
    "def _ran(name):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, (name + '|').encode())\n"
    "    os.close(fd)\n"
    "def load():\n"
    "    _ran('load')\n"
    "    time.sleep(0.2)\n"
    "    return np.full(5_000_000, 1.0)\n"
    "def step(a, k):\n"
    "    _ran('step')\n"
    "    time.sleep(0.2)\n"
    "    return a * k"
)
CLEAN = "arr = load()\narr = step(arr, 2.0)\narr = step(arr, 3.0)"


def _on_disk(work_dir, output):
    n = 0
    for p in (Path(work_dir) / ".cash").glob("*.entry"):
        meta, _ = read_entry(str(p), with_payload=False)
        if meta.get("outputs") == [output] and not meta.get("metadata_only"):
            n += 1
    return n


def _runs(work_dir):
    log = Path(work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").split("|")[:-1] if log.exists() else []


def test_only_the_cells_final_version_reaches_disk_and_restores(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, CLEAN, "print('ARR', arr[0])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ARR 6.0" in nb_runner.get_output(4)
    assert _on_disk(nb_runner.work_dir, "arr") == 1

    before = len(_runs(nb_runner.work_dir))
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)

    assert "ARR 6.0" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _runs(nb_runner.work_dir)[before:] == [], "the final version was rebuilt, not restored"


def test_persist_annotation_still_writes_an_intermediate(nb_runner):
    clean = "# @cash:persist\narr = load()\narr = step(arr, 2.0)\narr = step(arr, 3.0)"
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, clean, "print('ARR', arr[0])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ARR 6.0" in nb_runner.get_output(4)
    assert _on_disk(nb_runner.work_dir, "arr") >= 2
