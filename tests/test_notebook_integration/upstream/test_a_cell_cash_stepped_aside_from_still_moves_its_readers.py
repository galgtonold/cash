"""A cell cash stepped aside from still moves the cached cells that read it.

When cash's own check of a cell fails (an internal error, before the fix in
`cell_effects` an analysis that ran out of its time budget), the cell runs
as plain Python: nothing of it is keyed or stored. What it changed must
still reach the cells below. It does: the upstream check before the next
cell compares the lineage a top-to-bottom run gives each input with the one
in memory, and the plain run left the changed list's lineage where it was,
so the reader is keyed on a rebuilt state, never served from before.

The failure is forced on the one cell that appends, as an analysis budget
running out there would.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]

#: In the kernel: the cell check of a cell that appends raises, as one past
#: its budget did, so cash steps aside and the cell runs as plain Python.
_FAIL_ON_APPEND = (
    "(lambda m, ab: setattr(m, 'cell_effects', (lambda orig: (lambda code, *a, **k: "
    "(_ for _ in ()).throw(ab.AnalysisBudgetExceeded()) if 'append' in code else orig(code, *a, **k)))"
    "(m.cell_effects)))"
    "(__import__('cash.notebook.upstream.checker', fromlist=['_']), "
    "__import__('cash.analysis.analysis_budget', fromlist=['_']))"
)


def _notebook(nb_runner, tmp_path, appended):
    runs = tmp_path / "runs.log"
    reader = (
        "import os, time\n"
        f"fd = os.open(r'{runs}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "os.write(fd, b'X'); os.close(fd)\n"
        "time.sleep(0.3)\n"
        "total = sum(data)\n"
    )
    nb_runner.create_notebook(["import cash\n%cash_on\n", "data = [1, 2]\n", f"data.append({appended})\n", reader])
    nb_runner.start_kernel()
    nb_runner.peek(_FAIL_ON_APPEND)
    nb_runner.run_all()
    assert "Falling back to normal execution" in nb_runner.get_raw_output(3), "cash did not step aside"
    assert nb_runner.peek("total") == "6"
    return runs


def test_an_edit_of_the_plain_cell_reaches_the_reader(nb_runner, tmp_path):
    _notebook(nb_runner, tmp_path, "3")
    nb_runner.set_cell_source(3, "data.append(4)\n")
    nb_runner.run_all()
    assert nb_runner.peek("total") == "7"


def test_the_plain_cell_run_again_reaches_the_reader(nb_runner, tmp_path):
    _notebook(nb_runner, tmp_path, "3")
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert nb_runner.peek("data") == "[1, 2, 3, 3]"
    assert nb_runner.peek("total") == "9"


@pytest.mark.parametrize("restart", [False, True], ids=["same-kernel", "after-a-restart"])
def test_a_file_the_plain_cell_reads_reaches_the_reader(nb_runner, tmp_path, restart):
    """Same cell text, another value: nothing in the notebook says so."""
    src = tmp_path / "n.txt"
    src.write_text("3", encoding="utf-8")
    runs = _notebook(nb_runner, tmp_path, f"int(open(r'{src}').read())")
    src.write_text("4", encoding="utf-8")
    if restart:
        nb_runner.restart()
        nb_runner.peek(_FAIL_ON_APPEND)
    nb_runner.run_all()
    assert nb_runner.peek("total") == "7"
    assert runs.read_bytes() == b"XX", "the reader was served from before the change"
