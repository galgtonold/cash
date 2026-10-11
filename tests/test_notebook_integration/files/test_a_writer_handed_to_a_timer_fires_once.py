"""A helper handed to a timer writes once, not again before every cell below.

``threading.Timer(0.5, save, args=(p,)).start()`` hands a writing helper to a
timer: the write happens after the statement returns, so none was seen as it
ran. The upstream replay takes the statement for a file writer (it hands one a
writing function) and, finding no record that it had run, took it for an
edited writer and fired it again before each cell below that reads a file.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_the_timer_statement_runs_once_per_run_all(nb_runner, tmp_path):
    out = tmp_path / "out.txt"
    src = tmp_path / "in.csv"
    src.write_text("x\n1\n", encoding="utf-8", newline="\n")
    out_path = str(out).replace("\\", "/")
    src_path = str(src).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import threading, time\n"
            "import pandas as pd\n"
            "fired = []\n"
            "def save(path):\n"
            "    fired.append(1)\n"
            "    with open(path, 'w', newline='\\n') as fh:\n"
            "        fh.write('done\\n')",
            f"threading.Timer(0.2, save, args=({out_path!r},)).start()",
            f"df = pd.read_csv({src_path!r})",
            f"t = open({out_path!r}).read() if time.sleep(0.5) is None else ''",
            "time.sleep(0.5)\nprint(len(fired))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip() == "1"
