"""``W, H = fit_nmf(X, k)`` writing into ``X`` is served, and ``X`` gets the state.

The statement ran on every Run All (33 s a pass in the notebook that showed
it): a hit would have handed back ``W`` and ``H`` and left ``X`` as it was
before the call. The entry now keeps ``X``'s state after the call, and a hit
writes it into the array ``X`` is bound to, so ``X`` and every other name for
the same array hold what a plain run leaves, in the same kernel and after a
restart.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport os, time\nimport numpy as np"
FIT = (
    "def fit_nmf(X, k):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.3)\n"
    "    np.clip(X, 0.25, None, out=X)\n"
    "    return X[:k].copy(), float(X.sum()) * k\n"
)
MAKE = "X = np.arange(12, dtype=float).reshape(3, 4) / 20\nX0 = X"
CALL = "W, H = fit_nmf(X, 2)"
USE = "print('R', X.round(2).tolist(), W.round(2).tolist(), round(H, 4), X0 is X)"


def _plain() -> str:
    """What the last cell prints in plain Python."""
    import numpy as np

    X = np.arange(12, dtype=float).reshape(3, 4) / 20
    np.clip(X, 0.25, None, out=X)
    W, H = X[:2].copy(), float(X.sum()) * 2
    return f"R {X.round(2).tolist()} {W.round(2).tolist()} {round(H, 4)} True"


def _runs(tmp_path) -> int:
    log = tmp_path / "runs.log"
    return len(log.read_bytes()) if log.exists() else 0


def test_a_second_run_all_serves_it_and_writes_the_array(nb_runner, tmp_path):
    nb_runner.create_notebook([SETUP, FIT, MAKE, CALL, USE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _plain() in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    nb_runner.run_all()
    assert _runs(tmp_path) == 1, "the call writing into X ran again:\n" + nb_runner.get_raw_output(4)
    assert _plain() in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    assert nb_runner.peek("float(X.min())") == "0.25"


@pytest.mark.fresh_kernel
def test_after_a_restart(nb_runner, tmp_path):
    nb_runner.create_notebook([SETUP, FIT, MAKE, CALL, USE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert _runs(tmp_path) == 1, "after a restart the call ran again:\n" + nb_runner.get_raw_output(4)
    assert _plain() in nb_runner.get_output(5), nb_runner.get_raw_output(5)


def test_an_edit_of_the_array_runs_it_again(nb_runner, tmp_path):
    """Control: the entry is keyed on ``X`` before the call."""
    nb_runner.create_notebook([SETUP, FIT, MAKE, CALL, USE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, MAKE.replace("/ 20", "/ 10"))
    nb_runner.run_all()
    assert _runs(tmp_path) == 2
    assert nb_runner.peek("float(X.max())") == "1.1"


@pytest.mark.fresh_kernel
def test_after_a_restart_a_cell_below_run_alone_rebuilds_it_into_the_array(nb_runner, tmp_path):
    """The upstream check rebuilds the cells above from the entry: ``X`` gets
    the stored state written into it, so ``X0`` still names the same array."""
    nb_runner.create_notebook([SETUP, FIT, MAKE, CALL, USE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(5)
    assert _runs(tmp_path) == 1, nb_runner.get_raw_output(5)
    assert _plain() in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    assert nb_runner.peek("X0 is X") == "True"
