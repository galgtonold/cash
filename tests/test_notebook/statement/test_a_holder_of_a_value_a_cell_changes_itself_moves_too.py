"""A variable holding a value that a cell changes in place moves with it.

``w -= float(open(p).read())`` changes ``w`` itself; ``L = [w]`` holds the
same array. When the file changes, Run All runs the cell again and ``w`` is
-3 everywhere, but ``L`` kept the entry it had before, so ``s = L[0].sum()``
was served -3 where plain Python gives -9.
"""

from __future__ import annotations

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell


def test_the_holder_reads_the_new_value_after_the_file_changes(cash_magics, monkeypatch, tmp_path):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    p = tmp_path / "n.txt"
    p.write_text("1", encoding="utf-8")
    cells = [
        "import numpy as np",
        "w = np.zeros(3)\nL = [w]",
        f"w -= float(open({str(p)!r}).read())",
        "s = float(L[0].sum())",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    p.write_text("3", encoding="utf-8")
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["w"].tolist() == [-3.0] * 3
    assert cash_magics.shell.user_ns["s"] == -9.0
