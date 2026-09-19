"""A statement re-run on new inputs does not leave every old version on disk.

Round 24: r24s1's cache held nine versions of ``X_all = build_features(labelled)``
(~700 MB each, 1.4 s to build) and r24s2 five to seven of each cleaning frame
-- 10 GB for 200 MB of input. The disk tier now prunes a statement's superseded
versions as the new one is written: a big value cheap to rebuild keeps one
older version, for an undo; a small one that took long keeps many.

The unit arm, with the budget arithmetic, is
``tests/test_backends/test_superseded_versions.py``.
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
    "def build(n):\n"
    "    _ran('build')\n"
    "    time.sleep(0.4)\n"
    "    return np.full(5_000_000, float(n))\n"
    "def fit(n):\n"
    "    _ran('fit')\n"
    "    time.sleep(0.3)\n"
    "    return float(n) * 2"
)
CELLS = [
    "import cash\n%cash_on",
    SETUP,
    "N = 1",
    # Big for its cost, but not SO big that cash declines to cache it at all:
    # 40 MB in 0.4 s is ~95 MiB per compute-second, under the 128 MiB/s
    # `value_policy` ceiling and over the 64 MiB/s `versions` budget. That gap
    # is exactly the band this file is about -- a value worth storing whose
    # SPARE copies are not. It used to be 0.15 s, i.e. 254 MiB/s, which the
    # write-time ceiling now refuses outright: nothing reaches disk, so there
    # is no superseded version to prune and nothing here measured anything.
    "X = build(N)",
    "score = fit(N)",        # small for its cost: a few bytes in 0.3 s
    "print('X', X[0], 'score', score)",
]


def _runs(work_dir, name):
    log = Path(work_dir) / "runs.log"
    return log.read_text().count(name + "|") if log.exists() else 0


def _versions_on_disk(work_dir, output):
    n = 0
    for p in (Path(work_dir) / ".cash").glob("*.entry"):
        meta, _ = read_entry(str(p), with_payload=False)
        if meta.get("outputs") == [output] and not meta.get("metadata_only"):
            n += 1
    return n


def test_a_big_cheap_value_keeps_one_old_version_a_small_costly_one_keeps_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 1.0 score 2.0" in nb_runner.get_output(6)

    for n in range(2, 6):
        nb_runner.set_cell_source(3, f"N = {n}")
        nb_runner.run_all()
    assert "X 5.0 score 10.0" in nb_runner.get_output(6)

    assert _versions_on_disk(nb_runner.work_dir, "X") == 2
    assert _versions_on_disk(nb_runner.work_dir, "score") == 5


def test_undoing_an_edit_restores_the_kept_version(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    for n in (1, 2, 3):
        nb_runner.set_cell_source(3, f"N = {n}")
        nb_runner.run_all()
    assert _runs(nb_runner.work_dir, "build") == 3

    nb_runner.restart()
    nb_runner.set_cell_source(3, "N = 2")
    nb_runner.run_all()

    assert "X 2.0 score 4.0" in nb_runner.get_output(6)
    assert _runs(nb_runner.work_dir, "build") == 3, "the version an undo returns to was pruned"
