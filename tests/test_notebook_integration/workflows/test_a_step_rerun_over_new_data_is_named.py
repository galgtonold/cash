"""A slow step that runs again over a whole array after some rows of its
data file changed names those rows and suggests a cached helper per chunk
(``CACHE-PER-ITEM-RERUN``).

The shape of an image pipeline: load a stack, preprocess every image. After
400 images are appended (and the kernel restarted), or one batch in the
middle is exported again, the preprocessing runs over all of them although
most are what they were. Silent when every row changed and when the step is
cheap.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CODE = "[CACHE-PER-ITEM-RERUN]"
STEP = "clean = [imgs * 2, time.sleep(1.1)][0]"
CHEAP = "clean = imgs * 2"


def _cells(step: str) -> list[str]:
    return [
        "import cash\n%cash_on",  # loads cash again after a restart
        "%cash_badge print\nimport numpy as np, time",
        "imgs = np.load('imgs.npy')",
        step + "\nprint(clean.shape)",
    ]


def _save(nb_runner, imgs) -> None:
    np.save(Path(nb_runner.work_dir) / "imgs.npy", imgs)


def _base(n=3000):
    return np.arange(n * 16, dtype=np.int64).reshape(n, 16)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.fresh_kernel
def test_rows_appended_before_a_restart_are_named(nb_runner):
    nb_runner.create_notebook(_cells(STEP))
    _save(nb_runner, _base())
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert CODE not in nb_runner.get_raw_output(4)

    _save(nb_runner, np.concatenate([_base(), np.ones((400, 16), np.int64)]))
    nb_runner.restart()
    nb_runner.run_all()

    out = _flat(nb_runner.get_raw_output(4))
    assert "(3400, 16)" in out, out
    assert CODE in out, out
    assert f"`{STEP}` re-ran over all 3,400 rows of `imgs`" in out, out
    assert "only 400 rows differ from the last run (rows 3000-3399)" in out, out


def test_a_batch_exported_again_in_the_middle_is_named(nb_runner):
    nb_runner.create_notebook(_cells(STEP))
    _save(nb_runner, _base())
    nb_runner.start_kernel()
    nb_runner.run_all()

    redone = _base()
    redone[1000:1100] = -1
    _save(nb_runner, redone)
    nb_runner.run_all()

    out = _flat(nb_runner.get_raw_output(4))
    assert CODE in out, out
    assert "re-ran over all 3,000 rows of `imgs`" in out, out
    assert "rows 987-1127" in out, out


def test_new_data_in_every_row_says_nothing(nb_runner):
    nb_runner.create_notebook(_cells(STEP))
    _save(nb_runner, _base())
    nb_runner.start_kernel()
    nb_runner.run_all()

    _save(nb_runner, _base() + 1)
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert "(3000, 16)" in out, out
    assert CODE not in out, out


def test_a_cheap_step_says_nothing(nb_runner):
    nb_runner.create_notebook(_cells(CHEAP))
    _save(nb_runner, _base())
    nb_runner.start_kernel()
    nb_runner.run_all()

    _save(nb_runner, np.concatenate([_base(), np.ones((400, 16), np.int64)]))
    nb_runner.run_all()

    out = nb_runner.get_raw_output(4)
    assert "(3400, 16)" in out, out
    assert CODE not in out, out
