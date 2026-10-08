"""After a kernel restart, a cell run alone runs in the environment and directory the cells above set.

``def setup(): os.environ['MS_MODE'] = 'b'`` then ``setup()``, or
``os.chdir(sub)`` (in a cell or in a module helper), above a reader: after a
restart, running only the reader rebuilt the imports it needs and not the
setting, so it read the shell's environment or the file of the directory the
kernel started in, silently, where Restart & Run All gives the notebook's.
"""

import pytest

from tests._nbharness.runner import CASH_TEST_PIN_THRESHOLDS

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

LIB = (
    "import os, time\n"
    "def go(d):\n    os.chdir(d)\n"
    "def mode():\n    time.sleep(0.3)\n    return os.environ.get('MS_MODE', 'unset')\n"
    "def read(p):\n    time.sleep(0.3)\n    return open(p).read()\n"
)
SETUP = "import cash\n" + CASH_TEST_PIN_THRESHOLDS

CASES = {
    "a_notebook_helper_setting_the_environment": (
        ["import os, mylib", "def setup():\n    os.environ['MS_MODE'] = 'b'", "setup()"],
        "x = mylib.mode()\nprint('X', x)",
        "X b",
    ),
    "a_cell_changing_directory": (
        ["import os, mylib", "os.chdir({sub!r})"],
        "x = mylib.read('d.txt')\nprint('X', x)",
        "X SUB",
    ),
    "a_module_helper_changing_directory": (
        ["import os, mylib", "mylib.go({sub!r})"],
        "x = mylib.read('d.txt')\nprint('X', x)",
        "X SUB",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_setting(nb_runner, tmp_path, case):
    (tmp_path / "mylib.py").write_text(LIB, encoding="utf-8")
    (tmp_path / "d.txt").write_text("TOP", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.txt").write_text("SUB", encoding="utf-8")
    above, reader, expected = CASES[case]
    cells = [SETUP, *(cell.format(sub=str(tmp_path / "sub")) for cell in above), reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert expected in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))
