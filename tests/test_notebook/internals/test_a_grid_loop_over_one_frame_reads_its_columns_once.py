"""A grid loop calling a helper on one frame reads that frame's columns once.

p28-elo-ratings::

    for k in [5, 10, 15, 20, 30]:
        for hfa in [40, 60, 80]:
            h, _ = elo.run_elo(played, k=k, hfa=hfa)
            grid[(k, hfa)] = elo.brier(h)

``run_elo`` returns ``played.assign(...)``: every pass binds a new frame
sharing ``played``'s columns, and its digest (for the early cutoff and the
store) read all of them again, 0.4 s a pass. Those columns are now taken
from ``played``'s kept digest while copy-on-write vouches for it
(`FrameDigests.hash_frame`). Every pass's result must still be its own.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from cash import content_hashers
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

MODULE = """
def run_elo(matches, k, hfa):
    p = 1.0 / (1.0 + 10 ** (-(matches["goals"] * k + hfa) / 400.0))
    return matches.assign(p_home=p), {"k": k}


def brier(h):
    return float(((h["p_home"] - (h["goals"] > 0.5)) ** 2).mean())
"""

ROWS = 200_000


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_elo_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module(name).__name__
    sys.modules.pop(name, None)


@pytest.fixture
def text_reads(monkeypatch):
    """Full reads of a text column of ``ROWS`` rows (Arrow or Python strings)."""
    seen = []
    real_arrow, real_items = content_hashers.fold_arrow_column, content_hashers._fold_object_items

    def arrow(h, column):
        if len(column) == ROWS:
            seen.append("arrow")
        return real_arrow(h, column)

    def items(h, array):
        if len(array) == ROWS:
            seen.append("items")
        return real_items(h, array)

    monkeypatch.setattr(content_hashers, "fold_arrow_column", arrow)
    monkeypatch.setattr(content_hashers, "_fold_object_items", items)
    return seen


SETUP = (
    "import numpy as np, pandas as pd\n"
    "rng = np.random.default_rng(0)\n"
    "teams = np.array([f'T{i:03d}' for i in range(300)], dtype=object)\n"
    f"played = pd.DataFrame({{'home': pd.array(teams[rng.integers(0, 300, {ROWS})], dtype='str'),\n"
    f"                       'away': pd.Series(teams[rng.integers(0, 300, {ROWS})], dtype=object),\n"
    f"                       'goals': rng.random({ROWS})}})\n"
)

GRID = (
    "grid = {}\n"
    "for k in [5, 10]:\n"
    "    for hfa in [40, 60, 80]:\n"
    "        h, _ = elo.run_elo(played, k=k, hfa=hfa)\n"
    "        grid[(k, hfa)] = elo.brier(h)\n"
)


def _expected(played, module):
    out = {}
    for k in [5, 10]:
        for hfa in [40, 60, 80]:
            h, _ = module.run_elo(played, k=k, hfa=hfa)
            out[(k, hfa)] = module.brier(h)
    return out


def test_each_pass_is_correct_and_the_shared_text_columns_are_not_read_again(cash_magics, mock_shell, lib, text_reads):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {lib} as elo\n" + SETUP)
    text_reads.clear()
    run_cash_cell(cash_magics, GRID)
    played = mock_shell.user_ns["played"]
    assert mock_shell.user_ns["grid"] == _expected(played, sys.modules[lib])
    assert len(set(mock_shell.user_ns["grid"].values())) == 6
    # Six passes, two text columns each: twelve reads before, none now
    # beyond played's own first digest, if it was not taken above.
    assert len(text_reads) <= 2, text_reads


def test_a_frame_changed_between_passes_gives_each_pass_its_new_content(cash_magics, mock_shell, lib):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {lib} as elo\n" + SETUP)
    run_cash_cell(
        cash_magics,
        "out = []\n"
        "for step in range(4):\n"
        "    played.loc[step, 'goals'] = 10.0 + step\n"
        "    h, _ = elo.run_elo(played, k=5, hfa=40)\n"
        "    out.append(float(h['p_home'].sum()))\n",
    )
    out = mock_shell.user_ns["out"]
    assert len(set(out)) == 4
    module = sys.modules[lib]
    played = mock_shell.user_ns["played"]
    assert out[-1] == float(module.run_elo(played, 5, 40)[0]["p_home"].sum())
    # Rerun: the edited frame's grid is computed from its current content.
    run_cash_cell(cash_magics, GRID)
    assert mock_shell.user_ns["grid"] == _expected(played, module)
