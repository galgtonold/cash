"""What is read from an open file is keyed by what was read, not by the file.

``lines = fh.readlines()`` runs every time (it moves ``fh``). Its output's
lineage was the same whenever ``fh`` had the same lineage, so after the file
was reopened a statement built on ``lines`` was served what it had computed
when the file stood at its end: ``0`` where a plain kernel counts 3 (found
replaying a student's notebook of the JuNE dataset, 'student_2' step 81).
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

READ = "lines = fh.readlines()\nparsed = list(map(str.strip, lines))\nprint(len(parsed))"

DRAINS = {
    "readline loop": "for _ in range(10):\n    line = fh.readline()",
    "for loop": "for line in fh:\n    pass",
}


def _printed(capsys) -> str:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip().isdigit()]
    return lines[-1].strip() if lines else ""


@pytest.mark.parametrize("drain", sorted(DRAINS))
def test_a_reopened_file_is_read_again(cash_magics, tmp_path, capsys, drain):
    data = tmp_path / "probe_data.txt"
    data.write_text("a\nb\nc\n")
    cells = [f"fh = open(r'{data}')", DRAINS[drain], READ]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell)
        assert _printed(capsys) == "0"

        run_cash_cell(cash_magics, cells[0])
        _printed(capsys)
        run_cash_cell(cash_magics, cells[2])

    out = _printed(capsys)
    assert out == "3", f"the file was read from its old end: {out!r}"
