"""A statement re-run on new inputs does not leave every old version on disk.

One user's cache held nine versions of ``X_all = build_features(labelled)``
(~700 MB each, 1.4 s to build) and another's five to seven of each cleaning frame
-- 10 GB for 200 MB of input. The disk tier now prunes a statement's superseded
versions as the new one is written: a big value cheap to rebuild keeps one
older version, for an undo; a small one that took long keeps many.

The unit arm, with the budget arithmetic, is
``tests/test_backends/test_superseded_versions.py``.
"""

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from cash.backends.cache_dir import VERSIONS_INDEX_FILENAME
from cash.backends.entry_format import read_entry
from cash.backends.versions import Version, superseded_to_drop

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = (
    "import os, time\n"
    "import numpy as np\n"
    "def _ran(name):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
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
    "score = fit(N)",  # small for its cost: a few bytes in 0.3 s
    "print('X', X[0], 'score', score)",
]


def _runs(work_dir, name):
    log = Path(work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").count(name + "|") if log.exists() else 0


def _entries(work_dir, output):
    for p in (Path(work_dir) / ".cash").glob("*.entry"):
        meta, _ = read_entry(str(p), with_payload=False)
        if meta.get("outputs") == [output] and not meta.get("metadata_only"):
            yield meta


def _versions_on_disk(work_dir, output):
    return sum(1 for _meta in _entries(work_dir, output))


def _versions_the_budget_keeps(work_dir, output):
    """How many versions of *output* the budget keeps for the costs this run
    measured: every version written, less those `superseded_to_drop` drops
    with the newest current. Each prune at a write drops what the last one
    does, as newer versions only add bytes ahead of an older one.

    The cost is the time a version took on this machine: 0.4 s here when
    it is idle, up to 1.5 s with six of these tests at once, and a version
    over 1.2 s pays for a second spare. The count is the budget's, not a
    machine's."""
    (slot,) = {meta["version_slot"] for meta in _entries(work_dir, output)}
    written = {}
    log = Path(work_dir) / ".cash" / VERSIONS_INDEX_FILENAME
    for line in log.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == slot:
            written[parts[1]] = Version(parts[1], int(parts[2]), float(parts[3]), float(parts[4]))
    current = max(written.values(), key=lambda v: v.used).key
    return len(written) - len(superseded_to_drop(written, current))


def test_a_big_cheap_value_keeps_one_old_version_a_small_costly_one_keeps_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 1.0 score 2.0" in nb_runner.get_output(6)

    for n in range(2, 6):
        nb_runner.set_cell_source(3, f"N = {n}")
        nb_runner.run_all()
    assert "X 5.0 score 10.0" in nb_runner.get_output(6)

    nb_runner.settle_writes()
    kept = _versions_the_budget_keeps(nb_runner.work_dir, "X")
    assert 2 <= kept < 5, "a big value cheap to rebuild keeps one or two spare versions, not every one"
    assert _versions_on_disk(nb_runner.work_dir, "X") == kept
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
