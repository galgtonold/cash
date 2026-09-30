"""A loop that writes, moves or removes files keeps per-iteration mode.

A loop that only reads files runs as one cached unit
(`test_a_loop_reading_files_runs_as_one_unit.py`). One that changes files
does not: as one unit it would be one statement that both reads and writes,
and it may read what it wrote itself. These pin that the single-unit policy
still refuses such a loop and that it gives a plain kernel's result.
"""

import os
import time

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.files, pytest.mark.timeout(120)]

# Clears the single-unit thresholds but for the write: more than 50
# iterations, and 60 x 3 x 8 ms = 1.4 s estimated overhead.
N_FILES = 60
SINGLE_UNIT = "Fast-loop: executing as single unit"


def _make_folder(tmp_path, n=N_FILES):
    folder = tmp_path / "inputs"
    folder.mkdir()
    day_ago = time.time() - 86_400
    for i in range(n):
        p = folder / f"part_{i:03d}.num"
        p.write_text(str(i), encoding="utf-8")
        os.utime(p, (day_ago, day_ago))
    return folder


def _expected(n=N_FILES):
    return sum(range(n))


def test_a_loop_that_writes_files_stays_per_iteration(nb_runner, tmp_path):
    """Writing a file in the body keeps the per-iteration path, and every
    file is written."""
    folder = _make_folder(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gp = str(folder).replace("\\", "/")
    op = str(out_dir).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob, os\n"
            f"files = sorted(glob.glob('{gp}/*.num'))\n"
            "for f in files:\n"
            "    value = int(open(f).read())\n"
            "    doubled = value * 2\n"
            f"    open('{op}/' + os.path.basename(f), 'w').write(str(doubled))",
            f"print('written =', len(os.listdir('{op}')))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert SINGLE_UNIT not in nb_runner.get_raw_output(1)
    assert f"written = {N_FILES}" in nb_runner.get_output(2)
    assert (out_dir / "part_021.num").read_text(encoding="utf-8") == "42"


def test_a_loop_that_deletes_the_files_it_reads_stays_per_iteration(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    gp = str(folder).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob, os\n"
            f"files = sorted(glob.glob('{gp}/*.num'))\n"
            "total = 0\n"
            "for f in files:\n"
            "    value = int(open(f).read())\n"
            "    total = total + value\n"
            "    os.remove(f)\n"
            "print('total =', total, 'left =', len(os.listdir(" + repr(gp) + ")))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert SINGLE_UNIT not in nb_runner.get_raw_output(1)
    assert f"total = {_expected()} left = 0" in nb_runner.get_output(1)
