"""Cash looking up a library's installed version inside a statement is no
dependency of that statement on the folders it looked in.

A ``@cash.cache`` function that reaches a library (numpy here) folds the
library's installed version into its key (`library_versions`), and finding
it lists every folder on ``sys.path`` through ``importlib.metadata``. Called
from a notebook statement, those listings ran under the statement's file
tracker, so the statement recorded the project folder (the working
folder, first on ``sys.path``), and the interpreter's and the
environment's own folders, as inputs keyed on their stat. Any file created
in the project folder then made the statement, and every statement that
took its value, re-run with ``file changed: <project>``: the cells below
p08's ``raw = etl.load_logs()`` re-ran on every pass on a machine where
the notebook file is saved next to the code (six cells, 32.9 s), though
no input had changed.

The cell below the call takes longer than the persistence floor, so it
is stored and can be served. What must still hold: it re-runs when a
file the function reads changes, or when a file appears in a folder the
function itself lists.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash._active import default_cash, set_default_cash
from cash.decorator import library_versions
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import glob\n"
    "import os\n"
    "import cash\n"
    "import numpy as np\n"
    "@cash.cache\n"
    "def total(path):\n"
    "    with open(path) as fh:\n"
    "        return float(np.asarray([float(x) for x in fh.read().split()]).sum())\n"
    "@cash.cache\n"
    "def count(folder):\n"
    "    return int(np.asarray(len(glob.glob(os.path.join(folder, '*.txt')))))\n"
    "def scaled(v):\n"
    "    import time\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return [v * i for i in range(3)]\n"
)


@pytest.fixture
def project(cash_magics, cash_instance, mock_shell, tmp_path, monkeypatch):
    """A project folder on ``sys.path`` and the working folder, as a
    notebook's, holding a module ``lib`` of cached functions over numpy,
    and a data file; the decorator and the notebook share one Cash.

    The process looks its installed libraries up afresh, as a new kernel
    does on its first cached call: what an earlier test looked up, or saved
    in the per-user folder, would leave nothing to list here."""
    monkeypatch.setattr(library_versions, "per_user_cache_root", lambda: tmp_path / "user")
    for memo in ("_PROVIDERS", "_VERSIONS", "_RECORD_PATHS", "_DIGESTS", "_CLOSURES"):
        monkeypatch.setattr(library_versions, memo, {})
    monkeypatch.setattr(library_versions, "_DISTS_BY_TOP", None)
    proj = tmp_path / "proj"
    proj.mkdir()
    name = f"_libfold_{os.getpid()}_{id(tmp_path)}"
    (proj / f"{name}.py").write_text(LIB, encoding="utf-8")
    data = proj / "data"
    data.mkdir()
    (data / "a.txt").write_text("1 2 3\n", encoding="utf-8")
    monkeypatch.chdir(proj)
    monkeypatch.syspath_prepend(str(proj))
    previous = default_cash()
    set_default_cash(cash_instance)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {name} as lib")
    yield cash_magics, proj, data
    set_default_cash(previous)
    sys.modules.pop(name, None)


def _row(magics) -> dict:
    return magics.cash_status("dict")["last_cell"]["statements"][0]


def _status(magics) -> str:
    return _row(magics)["status"].name


def _new_file_in(folder) -> None:
    """A file created in *folder*, as a notebook saved next to the code."""
    (folder / "report.ipynb").write_text("{}", encoding="utf-8")
    st = os.stat(folder)
    # A folder's stat moves when an entry is added; make sure it is seen
    # to have moved on a file system with coarse timestamps.
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_a_new_file_in_the_project_folder_does_not_rerun_the_statement(project):
    magics, proj, data = project
    load = f"x = lib.total({str(data / 'a.txt')!r})"
    use = "y = lib.scaled(x)"
    run_cash_cell(magics, load)
    run_cash_cell(magics, use)
    assert magics.shell.user_ns["y"] == [0.0, 6.0, 12.0]
    _new_file_in(proj)
    run_cash_cell(magics, load)
    run_cash_cell(magics, use)
    assert _status(magics) == "RESTORED"
    assert magics.shell.user_ns["y"] == [0.0, 6.0, 12.0]


def test_a_changed_input_file_still_reruns_the_statement(project):
    magics, proj, data = project
    load = f"x = lib.total({str(data / 'a.txt')!r})"
    use = "y = lib.scaled(x)"
    run_cash_cell(magics, load)
    run_cash_cell(magics, use)
    (data / "a.txt").write_text("1 2 3 4\n", encoding="utf-8")
    run_cash_cell(magics, load)
    assert magics.shell.user_ns["x"] == 10.0
    run_cash_cell(magics, use)
    assert _status(magics) != "RESTORED"
    assert magics.shell.user_ns["y"] == [0.0, 10.0, 20.0]


def test_a_new_file_in_a_folder_the_function_lists_still_reruns(project):
    magics, proj, data = project
    load = f"n = lib.count({str(data)!r})"
    use = "m = lib.scaled(n)"
    run_cash_cell(magics, load)
    run_cash_cell(magics, use)
    assert magics.shell.user_ns["m"] == [0, 1, 2]
    (data / "b.txt").write_text("4\n", encoding="utf-8")
    _new_file_in(data)
    run_cash_cell(magics, load)
    assert magics.shell.user_ns["n"] == 2
    run_cash_cell(magics, use)
    assert _status(magics) != "RESTORED"
    assert magics.shell.user_ns["m"] == [0, 2, 4]
