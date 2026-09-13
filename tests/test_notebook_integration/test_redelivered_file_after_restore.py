"""A file read inside a helper, restored after a restart, then re-delivered.

Round 22 (r22s1): the morning restore replayed the reading statement from the
statement cache. The simulation that planned the replay ran before anything
was live, found no cache entry for the statement and so recorded no file
behind it; that snapshot was reused afterwards, and a CSV replaced under the
same name was never looked at again -- the downstream cell printed the old
total. Reading the file in the statement itself, or skipping the restart,
was always fine.
"""
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

CELLS = [
    "import cash\n%cash_on",
    "import os, time\nimport pandas as pd",
    "def read_all(files):\n"
    "    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)\n"
    "df = read_all(['data.csv', 'other.csv'])",
    "def slow(frame):\n"
    "    open('calls.log', 'a').write('slow\\n')\n"
    "    time.sleep(0.5)\n"
    "    return float(frame['v'].sum())\n"
    "total = slow(df)",
    "print('total =', total)",
]


def _csv(path: Path, n: int) -> None:
    path.write_text("v\n" + "\n".join(str(i) for i in range(1, n + 1)) + "\n", encoding="utf-8")
    # A re-delivery lands later than the file it replaces, even on a
    # coarse-mtime filesystem.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def _stdout(runner, cell):
    outs = runner.nb.cells[cell - 1].get("outputs", [])
    return "".join(o["text"] if isinstance(o["text"], str) else "".join(o["text"])
                   for o in outs if o.get("output_type") == "stream" and o.get("name") == "stdout")


def _slow_calls(work: Path) -> int:
    log = work / "calls.log"
    return log.read_text(encoding="utf-8").split().count("slow") if log.exists() else 0


@pytest.mark.parametrize("restart", [True, False], ids=["after_restore", "same_session"])
@pytest.mark.parametrize("redelivered", [True, False], ids=["redelivered", "unchanged"])
def test_downstream_cell_sees_the_redelivered_file(nb_runner, restart, redelivered):
    work = Path(nb_runner.work_dir)
    _csv(work / "data.csv", 100)
    (work / "other.csv").write_text("v\n1000\n", encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    if restart:
        nb_runner.restart()
        nb_runner.run_cell(1)
        nb_runner.run_cell(5)            # the morning: everything restored
    before = _slow_calls(work)
    if redelivered:
        _csv(work / "data.csv", 200)
    nb_runner.run_cell(5)
    out = _stdout(nb_runner, 5)
    expected = 21100.0 if redelivered else 6050.0
    assert f"total = {expected}" in out, out
    # Unchanged, nothing recomputes; re-delivered, the slow step runs once.
    assert _slow_calls(work) - before == (1 if redelivered else 0)
