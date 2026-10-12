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
    nb_runner.create_notebook(["import pandas as pd", f"df = pd.read_csv('{path}')", "t = int(df.x.sum())", "print(t)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "1"

    st = os.stat(csv)
    csv.write_text("x\n2\n", encoding="utf-8", newline="\n")  # same size
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime as before
    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip() == "2"


def _keep_size_and_mtime(path, text):
    st = os.stat(path)
    path.write_text(text, encoding="utf-8", newline="\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_a_value_read_through_a_helper_that_also_writes(nb_runner, tmp_path):
    """The helper writes a file, so its statement is never stored and no
    entry checks ``cfg.json`` by content: the value's lineage, made from the
    file's stat, was all there was, and it folded the size and mtime only.
    The cell below served ``v * 10`` for the old ``v`` after running only it,
    and on every Run All after that."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"v": 1}', encoding="utf-8", newline="\n")
    path = str(cfg).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import json, pathlib\n"
            "def load(p):\n"
            f"    pathlib.Path({str(tmp_path / 'other.txt').replace(chr(92), '/')!r}).write_text('x')\n"
            "    with open(p) as f:\n"
            "        return json.load(f)['v']",
            f"v = load('{path}')",
            "v2 = v * 10",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    _keep_size_and_mtime(cfg, '{"v": 7}')
    nb_runner.run_cell(3)
    assert nb_runner.peek("v2") == "70"
    for _ in range(2):
        nb_runner.run_all()
        assert nb_runner.peek("(v, v2)") == "(7, 70)"


def test_a_for_loop_reading_the_files_runs_again(nb_runner, tmp_path):
    """The simulation checked a loop's recorded outcome against the files'
    size and mtime: running only the cell below kept the old total."""
    folder = tmp_path / "d"
    folder.mkdir()
    for i in range(3):
        (folder / f"f{i}.csv").write_text(f"x\n{i + 1}\n", encoding="utf-8", newline="\n")
    pattern = str(folder / "*.csv").replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob\nimport pandas as pd",
            f"t = 0\nfor p in sorted(glob.glob('{pattern}')):\n    t += int(pd.read_csv(p).x.sum())",
            "t2 = t * 10",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    _keep_size_and_mtime(folder / "f0.csv", "x\n7\n")
    nb_runner.run_cell(3)
    assert nb_runner.peek("t2") == "120"
