"""A statement checks the files it inherited once per cell run, not twice.

``df = df.sample(...)`` below the loop that built ``df`` from 24,000
documents carries the loop's file dependencies. The upstream check looks up
the loop's entry and finds every document as recorded; the statement's own
lookup then checked all 24,000 of them again, ~0.75 s of a cell served
from the cache. It now takes the check's verdict on the same snapshot
(`run_memo.fresh_snapshot_this_run`), while that is as recent as its own
answers may be.
"""

from __future__ import annotations

import sys

import pytest

from cash.tracking import file_dep_snapshot
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

N = 80  # above the size a whole-set verdict is kept for


@pytest.fixture
def checks(monkeypatch):
    """The checks of a whole snapshot of the documents, from anywhere."""
    seen: list[int] = []
    real = file_dep_snapshot.snapshot_is_fresh

    def counting(snap, memo=None):
        if snap and len(snap) >= N:
            seen.append(len(snap))
        return real(snap, memo)

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("cash") and getattr(module, "snapshot_is_fresh", None) is real:
            monkeypatch.setattr(module, "snapshot_is_fresh", counting)
    return seen


def test_the_inherited_files_are_checked_once(cash_magics, mock_shell, tmp_path, checks):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(N):
        (docs / f"{i:03d}.txt").write_text(f"review {i}\n", encoding="utf-8")
    cells = [
        f"import os, time\nD = {str(docs)!r}\n"
        "def slowly(rows):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return sorted(rows, reverse=True)",
        f"rows = []\nfor i in range({N}):\n"
        "    with open(os.path.join(D, f'{i:03d}.txt'), encoding='utf-8') as f:\n"
        "        rows.append(f.read())\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S / N})",
        "if len(rows) > 1:\n    rows = slowly(rows)\nelse:\n    rows = list(rows)",
    ]
    for _ in range(2):
        for code in cells[:2]:
            run_cash_cell(cash_magics, code, cells=cells)
        checks.clear()
        run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["rows"] == sorted((f"review {i}\n" for i in range(N)), reverse=True)
    assert checks == [N]
