"""A reopened file is read from its start again, not served what an old handle gave.

``fh = open(path)`` binds a new handle at the start of the file on every run,
but its lineage was the same each time, so after ``lines = fh.readlines()``
had been run on a handle at its end, a rerun of the open and the read was
served that result: ``0`` where a plain kernel counts 3 (found replaying a
student's notebook of the JuNE dataset, 'student_2' step 81). A statement that
binds an open file now gets a lineage of its own on every run.
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
    data.write_text("a\nb\nc\n", encoding="utf-8")
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
