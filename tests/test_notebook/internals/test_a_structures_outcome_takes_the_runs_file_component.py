"""A control structure's outcome names its files by the run's file component.

A cell below a loop over the IMDb documents (``if ...: df = df.sample(...)``)
named the files behind what the structure left with a component built from
scratch: every one of 24,000 paths resolved and stat-ed again, after the
upstream check had built the same component from the run's stats moments
before. The outcome now takes that one (`run_memo.file_component_this_run`),
which is also what the simulation compares it with.
"""

from __future__ import annotations

import pytest

from cash.notebook.control_structures import processor as control_processor
from cash.notebook.statement import file_deps
from tests._cell_driver import run_cash_cell

# The stats every check shares, where no folder is watched (`file_watch`).
pytestmark = pytest.mark.usefixtures("no_file_watch")

N = 40


@pytest.fixture
def builds(monkeypatch):
    """The file components built over the documents."""
    seen: list[int] = []
    real = file_deps.compute_file_hash_component

    def counting(accessed_files, *args, **kwargs):
        if len(accessed_files) >= N:
            seen.append(len(accessed_files))
        return real(accessed_files, *args, **kwargs)

    # Wherever it is called from: the structure's processor holds its own name for it.
    monkeypatch.setattr(file_deps, "compute_file_hash_component", counting)
    monkeypatch.setattr(control_processor, "compute_file_hash_component", counting)
    return seen


def test_the_outcome_reuses_the_component_the_check_built(cash_magics, mock_shell, tmp_path, builds):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(N):
        (docs / f"{i:03d}.txt").write_text(f"review {i}\n", encoding="utf-8")
    cells = [
        f"import os\nD = {str(docs)!r}",
        f"rows = []\nfor i in range({N}):\n"
        "    with open(os.path.join(D, f'{i:03d}.txt'), encoding='utf-8') as f:\n"
        "        rows.append(f.read())",
        "if len(rows) > 1:\n    rows = sorted(rows)\nelse:\n    rows = list(rows)",
    ]
    for code in cells[:2]:
        run_cash_cell(cash_magics, code, cells=cells)
    builds.clear()
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["rows"] == sorted(f"review {i}\n" for i in range(N))
    assert builds == [N]
    # The same component a build from scratch gives.
    outcomes = [o for o in cash_magics.tracking_state.control_outcomes.values() if len(o[2]) == N]
    assert outcomes
    for _entry, _left, files, component, *_stored in outcomes:
        assert component == file_deps.compute_file_hash_component(set(files))
