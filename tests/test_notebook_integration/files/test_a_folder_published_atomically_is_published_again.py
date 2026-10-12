"""A helper that fills a temporary folder and moves it into place runs on
every Run All, as in a plain kernel: the release folder cleared at the top
is published again, and the cell reading it finds it."""

import pytest

pytestmark = [pytest.mark.integration]

CELLS = [
    """import os, gzip, shutil, time, tempfile
from pathlib import Path
def publish(parts, dest):
    time.sleep(0.3)
    tmp = Path(tempfile.mkdtemp(dir='.'))
    for i, part in enumerate(parts):
        f = gzip.open(tmp / f'part{i}.gz', 'wt'); f.write(part); f.close()
    tmp.rename(dest)
    return len(parts)
""",
    "shutil.rmtree('release', ignore_errors=True)",
    "n = publish(['a', 'b'], 'release')",
    "print(sorted(os.listdir('release')) if os.path.isdir('release') else 'release/ MISSING')",
]


def test_the_folder_is_there_after_the_second_run_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip().endswith("['part0.gz', 'part1.gz']")
