"""Running only a cell below a same-size edit that kept the mtime reads the new data.

``cp -p``, ``rsync -t`` or an archive with fixed timestamps rewrite a file and
put its mtime back. Run All re-read it (the runtime checks content), but
running only the last cell printed the old total: the simulation's quick scan
compared the mtime alone.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.files,
    pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == "win32", reason="on Windows ctime is the creation time: a documented limit"),
]


def test_a_cell_below_prints_what_run_all_prints(nb_runner, tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("x\n1\n", encoding="utf-8", newline="\n")
    os.utime(csv, (1e9, 1e9))
    path = str(csv).replace("\\", "/")
    nb_runner.create_notebook(
        ["import pandas as pd", f"df = pd.read_csv('{path}')", "t = int(df.x.sum())", "print(t)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "1"

    st = os.stat(csv)
    csv.write_text("x\n2\n", encoding="utf-8", newline="\n")  # same size
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime as before
    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip() == "2"
