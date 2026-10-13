"""A helper appending through ``Path.open('a')`` writes on every Run All, as
``open(path, 'a')`` does and as a plain kernel does: the output folder
cleared at the top gets its log back, and a log outside it grows on every
run."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

HELPER = """import os, shutil, time
from pathlib import Path
def write_log(rows, path):
    time.sleep(0.3)
    with Path(path).open('a') as f:
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
    assert (work / "run.log").read_text(encoding="utf-8").split() == ["a", "b", "a", "b"]
