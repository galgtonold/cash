"""A file replaced while the cell reading it still runs is read again next time.

An ETL job replaced the CSV while a slow loader was running. The statement was
stored with a fingerprint of the NEW file next to the value computed from the
old one, so every later Run All served the old data as fresh, in later kernels
too. It now runs uncached with a ``STORE-INPUT-CHANGED`` warning, and the next
Run All reads the settled file.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_the_next_run_all_reads_the_settled_file(nb_runner, tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("x\n1\n", encoding="utf-8", newline="\n")
    path = str(csv).replace("\\", "/")
    nb_runner.create_notebook(
        [
            # Another process, as far as the cell below can tell: a thread
            # that replaces the file a second after this cell, while the
            # cell below is still running.
            "import os, threading, time\n"
            "import pandas as pd\n"
            "def replace(path):\n"
            "    with open(path + '.tmp', 'w', newline='\\n') as fh:\n"
            "        fh.write('x\\n2\\n')\n"
            "    os.replace(path + '.tmp', path)\n"
            f"threading.Timer(1.0, replace, args=({path!r},)).start()",
            "def load(p):\n    d = pd.read_csv(p)\n    time.sleep(2.5)\n    return d",
            f"df = load({path!r})",
            "t = int(df.x.sum())",
            "print(t)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip() == "1", "the cell must read the file before it is replaced"
    assert "STORE-INPUT-CHANGED" in nb_runner.get_raw_output(3)

    nb_runner.run_all()
    assert nb_runner.get_output(5).strip() == "2"
