"""A script and a notebook share an entry while the pandas options it read
agree, and not once one differs.

A notebook kernel starts pandas with other defaults than a script does
(``display.max_columns`` 20 against 0, ``display.encoding``). A call is
checked on the options its code read (`library_reads`), so those defaults,
which it did not read, do not keep the notebook from the script's entry;
``display.precision``, which it reads, does.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

LIB = """\
import pandas as pd

import cash

RUNS = {runs!r}


@cash.cache
def scaled(n):
    with open(RUNS, "a") as fh:
        fh.write("run\\n")
    return round(float(pd.Series(range(n)).sum()) / 7, pd.get_option("display.precision"))
"""

SCRIPT = """\
import sys

import pandas as pd

import lib

pd.set_option("display.precision", int(sys.argv[1]))
print("R", lib.scaled(10))
"""

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import lib\nimport pandas as pd",
    "print('R', lib.scaled(10))",
    "pd.set_option(\"display.precision\", 2)\nprint('R', lib.scaled(10))",
]


def _runs(path) -> int:
    return path.read_text(encoding="utf-8").split().count("run") if path.exists() else 0


def _script(work, precision):
    out = subprocess.run(
        [sys.executable, "score.py", str(precision)],
        cwd=work,
        # The kernel's cache: the default ``.cash`` in the folder it runs in.
        env={**{k: v for k, v in os.environ.items() if k != "CASH_CACHE_DIR"}, "CASH_VERBOSE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.split("R ", 1)[1].split()[0], out.stderr


def test_the_notebook_reuses_the_script_entry_until_an_option_read_differs(nb_runner, tmp_path):
    work = nb_runner.work_dir
    runs = tmp_path / "runs.txt"
    (work / "lib.py").write_text(LIB.format(runs=str(runs)), encoding="utf-8")
    (work / "score.py").write_text(SCRIPT, encoding="utf-8")

    at6, _ = _script(work, 6)
    assert at6 == str(round(45 / 7, 6))
    assert _runs(runs) == 1

    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    out = nb_runner.get_output(3)
    assert f"R {at6}" in out, out
    assert _runs(runs) == 1, "the notebook's other pandas defaults kept it from the script's entry"

    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert f"R {round(45 / 7, 2)}" in out, out
    assert _runs(runs) == 2, "served the precision-6 result under precision 2"
    nb_runner.settle_writes()

    at2, err = _script(work, 2)
    assert at2 == str(round(45 / 7, 2))
    assert _runs(runs) == 2, "the script recomputed the notebook's entry\n" + err

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    out = nb_runner.get_output(3)
    assert f"R {at6}" in out, out
    assert _runs(runs) == 3, "served the precision-2 result under precision 6 after a restart"
