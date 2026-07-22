"""Verifier repro: chdir cell edited to a directory holding a DIFFERENT data.csv.

Minimal repro for the probe finding `chdir-relative-path-stale-collision`:
the file-dependency for a relative-path read is frozen to the realpath
resolved at FIRST execution (file_tracker._track_path), and cwd is not an
input to the cache decision, so after editing the os.chdir cell to point at
a directory holding a different file with the same relative name, run_all
serves the OLD directory's data.

Phase 1 ground-truths the expected values with a plain (with_cash=False)
kernel; phase 2 shows the cash kernel diverging on run_all.
"""

import pandas as pd
import pytest

pytestmark = [pytest.mark.timeout(90)]


def _p(path) -> str:
    return str(path).replace("\\", "/")


def test_chdir_collision_run_all_serves_old_dirs_file(nb_runner, tmp_path):
    dira = tmp_path / "dira"
    dirb = tmp_path / "dirb"
    dira.mkdir()
    dirb.mkdir()
    pd.DataFrame({"v": [1, 2, 3]}).to_csv(dira / "data.csv", index=False)
    pd.DataFrame({"v": [100, 200, 300, 400]}).to_csv(dirb / "data.csv", index=False)

    cells = [
        "import os\nimport pandas as pd",
        f"os.chdir(r'{_p(dira)}')",
        "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
    ]
    chdir_b = f"os.chdir(r'{_p(dirb)}')"

    # ---- Phase 1: ground truth (plain kernel, no cash) -------------------
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, chdir_b)
    nb_runner.run_all()
    truth = nb_runner.get_output(3)
    assert "vals = [100, 200, 300, 400]" in truth, (
        f"HARNESS SANITY: plain kernel did not see dirb's data.csv: {truth!r}"
    )
    nb_runner.shutdown()

    # ---- Phase 2: same sequence with cash --------------------------------
    nb_runner.create_notebook(cells)  # reset cell 2 back to dira
    nb_runner.start_kernel(with_cash=True)
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, chdir_b)
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "vals = [100, 200, 300, 400]" in out, (
        f"run_all after editing the chdir cell to a directory holding a "
        f"DIFFERENT data.csv served the OLD directory's data (ground truth "
        f"= [100, 200, 300, 400]). Got: {out!r}"
    )
