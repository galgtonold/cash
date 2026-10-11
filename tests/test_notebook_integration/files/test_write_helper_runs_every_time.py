"""A helper that saves a chart runs every time, as the inline savefig does.

``save(fig, "suspect.png")`` -- a notebook helper doing
``fig.tight_layout(); fig.savefig(...); plt.close(fig)`` -- was served from
the statement cache on a warm Restart & Run All, so the deck went out
without the chart, 3/3. Whether it happened depended on whether the save
took long enough to be persisted; persistence is forced here so the
failure does not depend on the machine.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

SETUP = """import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("out")
OUT.mkdir(exist_ok=True)
df = pd.DataFrame(np.arange(72.0).reshape(12, 6), columns=list("ABCDEF"))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=40)
    plt.close(fig)

def log(msg):
    open(OUT / "log.txt", "a").write(msg + "\\n")
"""
CHART = """fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for ax, col in zip(axes.flat, df.columns):
    df[col].plot(ax=ax)
save(fig, "grid.png")
"""


def test_warm_restart_and_run_all_writes_the_chart(nb_runner):
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, CHART])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert (work / "out" / "grid.png").exists()
    for cycle in range(2):
        (work / "out" / "grid.png").unlink()
        nb_runner.restart()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert (work / "out" / "grid.png").exists(), f"warm run {cycle + 1} wrote no chart"


@pytest.mark.parametrize(
    ("annotation", "helper_annotation", "runs"),
    [("", "", 2), ("  # @cash:assume-safe", "", 1), ("", "  # @cash:assume-safe", 1)],
    ids=["plain", "assume-safe", "assume-safe-in-the-helper"],
)
def test_a_helper_that_appends_runs_every_time(nb_runner, annotation, helper_annotation, runs):
    """A function that appends to a log writes a file a cache hit would
    skip: it runs again after a restart, as in a plain kernel, unless the
    statement, or the write in the helper (which covers every caller), says
    the write may be skipped."""
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            SETUP,
            "import time\ndef slow(v):\n    log('slow')"
            + helper_annotation
            + "\n    time.sleep(0.3)\n    return v * 2\nx = slow(21)"
            + annotation,
            "print('x =', x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    log = Path(nb_runner.work_dir) / "out" / "log.txt"
    assert log.read_text(encoding="utf-8").split() == ["slow"]
    nb_runner.restart()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "x = 42" in nb_runner.get_output(4)
    assert log.read_text(encoding="utf-8").split() == ["slow"] * runs


EXPERIMENT_LOG = [
    "import time, csv\nopen('results.csv', 'w').close()   # start a fresh log",
    "def evaluate(lr):\n    time.sleep(0.3)\n    score = round(1 - lr, 3)\n"
    "    with open('results.csv', 'a', newline='') as fh:\n        csv.writer(fh).writerow([lr, score])\n"
    "    return score",
    "s1 = evaluate(0.1)",
    "more = list(map(evaluate, [0.2, 0.3]))",
    "rows = list(csv.reader(open('results.csv')))",
]


def test_an_experiment_log_started_afresh_gets_its_rows_again(nb_runner):
    """The log is emptied at the top and read back at the bottom: served
    from the cache, the calls appended nothing and the log read back empty."""
    nb_runner.create_notebook(EXPERIMENT_LOG)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert nb_runner.peek("len(rows)") == "3"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("len(rows)") == "3"


# The same export behind a helper in the user's own module, as the notebook
# reads after "Moving to a module". The sleep puts the statement above the
# store floor, where a write that costs more than ~10 ms normally lands.
WRITER_MODULE = """import time

def write_csv(path, df):
    time.sleep(0.2)
    df.to_csv(path)
"""
# A package whose export hands the write to a sibling module.
PACKAGE_EXPORT = """from . import disk

def export(path, df):
    disk.save(path, df.round(2))
"""
PACKAGE_DISK = """import time

def save(path, df):
    time.sleep(0.2)
    df.to_csv(path)
"""
FRAME = """import pandas as pd
from pathlib import Path
OUT = Path("report")
OUT.mkdir(exist_ok=True)
df = pd.DataFrame({"a": range(50), "b": range(50, 100)})
"""


def _write_modules(work: Path) -> None:
    (work / "wr.py").write_text(WRITER_MODULE, encoding="utf-8")
    pkg = work / "reports"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "export.py").write_text(PACKAGE_EXPORT, encoding="utf-8")
    (pkg / "disk.py").write_text(PACKAGE_DISK, encoding="utf-8")


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("import wr", 'wr.write_csv(OUT / "report.csv", df)'),
        ("from wr import write_csv", 'write_csv(OUT / "report.csv", df)'),
        ("from reports import export", 'export.export(OUT / "report.csv", df)'),
        ("import reports.export", 'reports.export.export(OUT / "report.csv", df)'),
        (
            "import time\ndef write_nb(path, df):\n    time.sleep(0.2)\n    df.to_csv(path)",
            'write_nb(OUT / "report.csv", df)',
        ),
    ],
    ids=["module", "from-import", "package", "package-dotted", "notebook-helper"],
)
def test_a_writer_in_the_users_module_rewrites_its_file_every_run(nb_runner, imports, call):
    """Nothing is edited between runs; the output file is overwritten with
    garbage, and every run must write it again. A cache hit would skip the
    write and leave the garbage in the report folder."""
    work = Path(nb_runner.work_dir)
    _write_modules(work)
    report = work / "report" / "report.csv"
    nb_runner.create_notebook(["import cash\n%cash_on", FRAME, imports, call])
    nb_runner.start_kernel()
    nb_runner.run_all()
    written = report.read_text(encoding="utf-8")
    assert written.startswith(",a,b")
    for label in ("second run", "after a restart"):
        report.write_text("garbage", encoding="utf-8")
        if label == "after a restart":
            nb_runner.restart()
        nb_runner.run_all()
        assert report.read_text(encoding="utf-8") == written, f"{label}: the file was not rewritten"


HANDED = [
    "import os, shutil, time, pandas as pd\nshutil.rmtree('handed', ignore_errors=True); os.makedirs('handed')",
    "def save(v):\n    time.sleep(0.05)\n    p = f'handed/{v}.txt'\n    with open(p, 'w') as fh:\n        fh.write(str(v))\n    return p",
    "p1 = list(map(save, range(3)))",
    "p2 = pd.Series([10, 11, 12]).apply(save)",
    "p3 = pd.Series([20, 21]).map(lambda v: save(v))",
]
HANDED_FILES = sorted(f"{v}.txt" for v in (0, 1, 2, 10, 11, 12, 20, 21))


def test_a_writer_handed_to_map_or_apply_writes_again(nb_runner):
    """``map(save, xs)`` and ``Series.apply(save)`` call ``save``: served
    from the cache, the folder emptied at the top stayed without its files."""
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(HANDED)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    for _ in range(2):
        nb_runner.run_all()
        assert sorted(p.name for p in (work / "handed").iterdir()) == HANDED_FILES
    nb_runner.restart()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert sorted(p.name for p in (work / "handed").iterdir()) == HANDED_FILES
