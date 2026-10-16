"""A notebook reading a SQLite database in WAL mode sees a commit in ``-wal``.

In WAL mode a commit goes to ``<db>-wal``; the main file stays as it was until
a checkpoint. While another process keeps its connection open (an app writing
to the database the notebook reads), the main file alone never shows the
commit, so a cached ``select`` would show the old total.

The native file hook would see sqlite open ``-wal`` on Linux and Windows; it is
off here, as on macOS, so the test checks that ``sqlite3.connect`` itself
records the ``-wal`` file.
"""

import sqlite3

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.files]

SLEEP = f"import time\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})"


def _writer(path):
    w = sqlite3.connect(path)
    w.execute("pragma journal_mode=wal")
    w.execute("pragma wal_autocheckpoint=0")  # keep every commit in -wal
    w.execute("create table t(x)")
    w.execute("insert into t values (1)")
    w.commit()
    return w


@pytest.mark.parametrize(
    "cells",
    [
        [
            "import sqlite3",
            "con = sqlite3.connect('w.db')\n"
            "total = con.execute('select sum(x) from t').fetchone()[0]\n"
            f"con.close()\n{SLEEP}\nprint('TOTAL', total)",
        ],
        [
            "import pandas as pd\nimport sqlite3",
            "con = sqlite3.connect('w.db')\n"
            "total = int(pd.read_sql('select sum(x) as s from t', con)['s'][0])\n"
            f"con.close()\n{SLEEP}\nprint('TOTAL', total)",
        ],
    ],
    ids=["sqlite3", "pandas_read_sql"],
)
def test_a_commit_held_in_the_wal_file_shows_on_a_rerun(nb_runner, cells, monkeypatch):
    monkeypatch.setenv("CASH_NATIVE_FILE_HOOK", "0")
    if "pandas" in cells[0]:
        pytest.importorskip("pandas")
    writer = _writer(str(nb_runner.work_dir / "w.db"))
    try:
        nb_runner.create_notebook(["import cash\n%cash_on", *cells])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "TOTAL 1" in nb_runner.get_output(3), nb_runner.get_output(3)
        nb_runner.run_cell(3)
        assert "TOTAL 1" in nb_runner.get_output(3)

        writer.execute("insert into t values (100)")
        writer.commit()
        nb_runner.run_cell(3)
        assert "TOTAL 101" in nb_runner.get_output(3), nb_runner.get_output(3)
    finally:
        writer.close()
