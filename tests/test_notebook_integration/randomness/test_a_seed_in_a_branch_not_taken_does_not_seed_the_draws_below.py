"""A seed in a branch not taken does not seed the draws below it.

Seen in mlbook ch08::

    if version.parse(pd.__version__) >= version.parse("1.3.2"):   # cell 6
        df = df.sample(frac=1, random_state=0)
    else:
        np.random.seed(0)
        ...
    gs_lr_tfidf.fit(X_train, y_train)                            # cell 30, draws
    np.random.seed(0)                                            # cell 33

A draw no cell above seeds counts as unseeded, as after a restart, even
when a seed below left the stream seeded on an earlier pass. Read from the
cell's text, the ``else`` seed counted as one above: a run from the middle
of the notebook keyed the fit on the seed below, ran it again, and stored it
under a key a restart never looks up. A cell that ran seeds only what its
statements seeded; when the branch is taken, the seed counts as before.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

pytest.importorskip("numpy")

SETUP = "import cash\n%cash_on\n%cash_badge print"


def _cells(log: str, flag: bool) -> list[str]:
    return [
        SETUP,
        "import os, time\nimport numpy as np\n"
        f"flag = {flag}\n"
        "def make(n):\n"
        f"    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    time.sleep(0.3)\n"
        "    return np.random.rand(n)",
        "if flag:\n    w = 1\nelse:\n    np.random.seed(0)\n    w = 2",
        "X = make(3)\nprint('X', X.tolist())",
        "np.random.seed(0)\nz0 = np.random.randint(3)",
    ]


def _passes(nb_runner, tmp_path, flag: bool):
    log = tmp_path / "draws.log"
    nb_runner.create_notebook(_cells(str(log), flag))
    nb_runner.start_kernel()

    def draws() -> int:
        return len(log.read_bytes()) if log.exists() else 0

    def x() -> str:
        return next((line for line in nb_runner.get_output(4).splitlines() if line.startswith("X [")), "")

    nb_runner.run_all()
    yield "first", draws(), x()
    nb_runner.run_all()
    yield "same kernel", draws(), x()
    nb_runner.run_cells([4, 5])
    yield "from the middle", draws(), x()
    nb_runner.restart()
    nb_runner.run_all()
    yield "restart", draws(), x()


def test_a_draw_under_a_seed_not_taken_runs_once(nb_runner, tmp_path):
    passes = list(_passes(nb_runner, tmp_path, flag=True))
    assert [d for _, d, _ in passes] == [1, 1, 1, 1], passes
    assert len({out for _, _, out in passes}) == 1, passes


def test_a_draw_under_a_seed_taken_draws_the_seeded_values(nb_runner, tmp_path):
    import numpy as np

    np.random.seed(0)
    seeded = str(np.random.rand(3).tolist())
    for name, _, out in _passes(nb_runner, tmp_path, flag=False):
        assert seeded in out, (name, out)
