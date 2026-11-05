"""What a project module's helper is handed is not rebuilt when it changed nothing.

A backtest notebook calls its library as ``btlib.run_backtest(close, sig,
mask, ...)``. After the signal above it was edited and re-run, a new cell
below that reads only ``mask`` re-ran the backtest (and, in the real
notebook, a 32-backtest grid loop) to rebuild ``mask``: the run of the
backtest on the new signal was taken to maybe change every name it is
handed. The runtime watched those names when the backtest ran and saw them
left alone; a helper of the user's keeps that verdict on new inputs, read
off a module as it is when imported by name. A helper that does change what
it is handed on some inputs only is still run again for the reader.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

HELPER = (
    "import os\n"
    "import time\n"
    "import pandas as pd\n"
    "\n"
    "def _mark(step):\n"
    "    fd = os.open('calls.log', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, (step + '\\n').encode())\n"
    "    os.close(fd)\n"
    "\n"
    "def panel():\n"
    "    time.sleep(0.05)\n"
    "    return pd.DataFrame({'a': [1.0, 2.0, 4.0, 8.0], 'b': [3.0, 2.0, 2.0, 5.0]})\n"
    "\n"
    "def tradable(close):\n"
    "    _mark('tradable')\n"
    "    time.sleep(0.05)\n"
    "    return close > 1.5\n"
    "\n"
    "def signal(close, k):\n"
    "    return close.pct_change(k)\n"
    "\n"
    "def backtest(close, sig, mask):\n"
    "    _mark('backtest')\n"
    "    time.sleep(0.2)\n"
    "{body}"
    "    return float(sig.where(mask).fillna(0).values.sum())\n"
)

READS_ONLY = ""
# On the edited signal only, it clears the mask through a library call.
CHANGES_ON_SOME_INPUTS = (
    "    if float(sig.fillna(0).values.sum()) > 5:\n        mask.where(mask & False, False, inplace=True)\n"
)

CELLS = [
    "import btlib",
    "close = btlib.panel()\nmask = btlib.tradable(close)",
    "sig = btlib.signal(close, 1)",
    "res = btlib.backtest(close, sig, mask)",
    "pass",
]


def _calls(nb_runner):
    log = nb_runner.work_dir / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def _run(nb_runner, body):
    (nb_runner.work_dir / "btlib.py").write_text(HELPER.replace("{body}", body), encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "sig = btlib.signal(close, 2)")
    nb_runner.run_cell(3)
    nb_runner.set_cell_source(5, "print('tradable', int(mask.values.sum()))")
    before = len(_calls(nb_runner))
    nb_runner.run_cell(5)
    return _calls(nb_runner)[before:]


def test_a_reader_of_what_a_module_helper_left_alone_runs_nothing_above(nb_runner):
    ran = _run(nb_runner, READS_ONLY)
    assert "tradable 7" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert ran == [], ran


def test_a_module_helper_that_changes_what_it_is_handed_runs_again_for_its_reader(nb_runner):
    """The guard: on the edited signal the backtest clears the mask, as a
    run from the top shows, so the reader waits for it."""
    ran = _run(nb_runner, CHANGES_ON_SOME_INPUTS)
    assert "tradable 0" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert "backtest" in ran, ran
