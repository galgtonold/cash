"""A cell replayed for a later cell writes all of its files, or none.

Round 23 (r23s1's churn report). The report cell rebuilt ``report/`` with
``shutil.rmtree``, wrote a chart per kind through a helper, and wrote a
metrics file. Running a cell further down replayed the report cell for it,
and the replay re-ran only the writes it could see in the statement text:

* after an edit, ``metrics`` and the grid were rewritten while the helper's
  charts kept the old models -- a folder no run order produces;
* after a restart, ``rmtree`` re-ran and the helper loop did not, so the
  charts were simply gone.

The helper's ``savefig`` sits in its body, and the ``scores`` it plots is a
global the call site never names; the planner saw neither.
"""

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

SETUP = (
    "import os, shutil, glob\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "files = sorted(glob.glob('in/*.txt'))\n"
    "def load(paths):\n    return sum(int(open(p).read()) for p in paths)\n"
    "base = load(files)\n"
    "LEAF = 40"
)
MODEL = "scores = [base * LEAF * k for k in (1, 2, 3)]"
REPORT = (
    "shutil.rmtree('report', ignore_errors=True)\n"
    "os.makedirs('report')\n"
    "def save_png(kind, path):\n"
    "    f, ax = plt.subplots(figsize=(3, 3))\n"
    "    ax.plot(scores)\n"
    "    ax.set_title(kind)\n"
    "    f.savefig(path, dpi=40)\n"
    "    plt.close(f)\n"
    "for kind in ['roc', 'pr']:\n"
    "    save_png(kind, f'report/{kind}.png')\n"
    "open('report/metrics.txt', 'w').write(str(scores))\n"
    "total = sum(scores)"
)
TAIL = "print('total', total)"
# Reads files the planner cannot name, so it cannot prove any write unread.
TAIL_READING_FILES = "print('total', total, [len(open(p).read()) for p in files])"
ALL = ["metrics.txt", "pr.png", "roc.png"]


def _snap(work):
    d = work / "report"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())} if d.exists() else {}


def _run_all(nb_runner, tail):
    work = Path(nb_runner.work_dir)
    (work / "in").mkdir()
    for i in (1, 2):
        (work / "in" / f"{i}.txt").write_text(str(i))
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, MODEL, REPORT, tail])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return work


@pytest.mark.parametrize("tail", [TAIL, TAIL_READING_FILES], ids=["reads-nothing", "reads-files"])
def test_after_an_edit_the_folder_is_all_old_or_all_new(nb_runner, tail):
    work = _run_all(nb_runner, tail)
    before = _snap(work)
    nb_runner.set_cell_source(2, SETUP.replace("LEAF = 40", "LEAF = 45"))
    nb_runner.run_cell(3)
    nb_runner.run_cell(5)
    assert "total 810" in nb_runner.get_output(5)
    after = _snap(work)
    changed = sorted(k for k in after if after[k] != before.get(k))
    assert sorted(after) == ALL
    assert changed in ([], ALL), f"half the report was rewritten: {changed}"


PACK_CELLS = [
    "import cash\n%cash_on",
    "import shutil\nfrom pathlib import Path\nPACK = Path('pack')\nPACK.mkdir(exist_ok=True)",
    "table = {'a': 1, 'b': 2}",
    "if PACK.exists():\n    shutil.rmtree(PACK)\nPACK.mkdir()\n(PACK / 'table.txt').write_text(str(table))",
    "summary = sorted(table)\n(PACK / 'summary.txt').write_text(str(summary))\nprint(sorted(p.name for p in PACK.iterdir()))",
]


def test_after_a_restart_a_guarded_rmtree_is_replayed_with_its_mkdir(nb_runner):
    """r23s2: `if PACK.exists(): shutil.rmtree(PACK)` binds nothing, so the
    simulation gave it no trace entry, and the replay after a restart ran the
    cell's `PACK.mkdir()` alone -- FileExistsError, reported as a broken
    upstream cell."""
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(PACK_CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)
    assert "['summary.txt', 'table.txt']" in nb_runner.get_output(5)
    assert sorted(p.name for p in (work / "pack").iterdir()) == ["summary.txt", "table.txt"]


def test_after_a_restart_no_chart_is_lost(nb_runner):
    work = _run_all(nb_runner, TAIL)
    before = _snap(work)
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)
    assert "total 720" in nb_runner.get_output(5)
    assert _snap(work) == before, "the replay emptied report/ and did not refill it"
