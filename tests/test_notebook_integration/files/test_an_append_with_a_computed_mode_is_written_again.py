"""A helper appending through ``open(path, mode)`` with the mode passed in
writes on every Run All, as a plain kernel does: the output folder cleared
at the top gets its log back, and a log outside it grows on every run."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

HELPER = """import os, shutil, time
def write_log(rows, path, mode='a'):
    time.sleep(0.3)
    with open(path, mode) as f:
        f.writelines(r + '\\n' for r in rows)
    return len(rows)
"""


def test_the_log_is_written_on_every_run_all(nb_runner):
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(
        [
            HELPER,
            "shutil.rmtree('out', ignore_errors=True); os.makedirs('out')",
            "n = write_log(['a', 'b'], 'out/log.txt')",
            "m = write_log(['a', 'b'], 'run.log')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.run_all()
    assert (work / "out" / "log.txt").exists(), "the cleared folder stayed without the log"
    assert (work / "run.log").read_text().split() == ["a", "b", "a", "b"]
