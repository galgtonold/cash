"""A slow module function that warns is served on the second Run All.

The first warning a module emits puts ``__warningregistry__`` in its
globals. That was taken for the statement setting state on the module: the
statement was not stored on the first run and missed ("input changed") on
the second, so a slow loader ran twice before it was ever served.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

LIB = (
    "import os, time, warnings\n"
    "def load(x):\n"
    "    fd = os.open({counter!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.3)\n"
    "    warnings.warn('column dropped')\n"
    "    return x\n"
)


def test_a_warning_call_runs_once(nb_runner, tmp_path):
    counter = tmp_path / "runs"
    (tmp_path / "warnlib.py").write_text(LIB.format(counter=str(counter)), encoding="utf-8")
    nb_runner.create_notebook(["import warnlib", "y = warnlib.load(1)\nprint('Y', y)"])
    nb_runner.start_kernel()

    nb_runner.run_all()
    nb_runner.run_all()

    assert nb_runner.peek("y") == "1"
    assert len(counter.read_bytes()) == 1, nb_runner.get_raw_output(2)
