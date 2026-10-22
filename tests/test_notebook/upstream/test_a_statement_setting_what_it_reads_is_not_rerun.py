"""A statement that sets a module value it reads itself is not re-run by
the next cell.

``tempfile.gettempdir()`` reads ``tempfile.tempdir`` and, the first time,
sets it. The runtime keyed the statement by the value it read before its
own change; the upstream check took the value the statement left, found no
entry under that key, and ran the statement again before every later cell
(in ``file_tracking_demo``, the cell's file writes too).
"""

from __future__ import annotations

import tempfile
import warnings

from tests._cell_driver import run_cash_cell


def test_the_next_cell_does_not_run_it_again(cash_magics, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    real = tempfile.gettempdir
    calls = []

    def gettempdir():
        calls.append(1)
        return real()

    monkeypatch.setattr(tempfile, "gettempdir", gettempdir)
    (tmp_path / "demo").mkdir()
    cells = [
        "import tempfile\nfrom pathlib import Path\n"
        "folder = Path(tempfile.gettempdir()) / 'demo'\nt = folder / 'notes.txt'",
        "t.write_text('v')",
        "z = 2",
    ]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert len(calls) == 1, "a later cell ran the statement again"
    assert (tmp_path / "demo" / "notes.txt").read_text() == "v"


def test_file_tracking_demo_runs_each_statement_once(cash_magics, tmp_path, monkeypatch):
    """The example notebook, Run All on a fresh cache with its folder already
    there: its first cell ran again before cell 13, and the writes of cells
    2, 5 and 12 with it."""
    import json
    from pathlib import Path

    nb = Path(__file__).resolve().parents[3] / "examples" / "file_tracking_demo.ipynb"
    cells = [
        "".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"
    ]
    assert cells[0].split() == ["import", "cash", "%cash_on"]
    cells = cells[1:]
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    real = tempfile.gettempdir
    calls = []

    def gettempdir():
        calls.append(1)
        return real()

    monkeypatch.setattr(tempfile, "gettempdir", gettempdir)
    (tmp_path / "cash_file_tracking_demo").mkdir()
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert len(calls) == 1, "a later cell ran the first cell again"
    ns = cash_magics.shell.user_ns
    assert ns["text"] == "Pathlib 2"
    assert list(ns["numbers"]) == [10.0, 20.0, 30.0]
