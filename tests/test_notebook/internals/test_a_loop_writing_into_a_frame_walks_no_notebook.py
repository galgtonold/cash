"""A loop writing into a frame beside many objects does not walk them first.

``for i in tqdm(range(n)): df.loc[i, c] = v`` on a fresh frame, after a
cell holding 2,000,000 objects, cost each run of its cell 0.8 s more than
plain Python: before the loop ran, the notebook was walked for a weak
reference to, or the ``id()`` of, the frame, to choose how a hit would give
it back. A hit asks that of the live objects anyway, so the loop is routed
with no walk, and only an object a hit refused is walked for before its
next loops.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import in_place
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")

MAKE = (
    "import time\nimport pandas as pd\nheld = [(i, str(i)) for i in range(50_000)]\n"
    "df = pd.DataFrame({'a': range(3000), 'b': 0.0})"
)
LOOP = f"for i in range(3000):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S / 3000})\n    df.loc[i, 'b'] = df.loc[i, 'a'] * 2"
CELLS = [MAKE, LOOP]


def _walks(monkeypatch, mock_shell):
    walks = []
    real = in_place._held_itself

    def counted(obj, namespace):
        if obj is mock_shell.user_ns.get("df"):
            walks.append(obj)
        return real(obj, namespace)

    monkeypatch.setattr(in_place, "_held_itself", counted)
    return walks


def test_the_loops_run_walks_nothing(cash_magics, mock_shell, monkeypatch):
    run_cash_cell(cash_magics, MAKE, cells=CELLS)
    walks = _walks(monkeypatch, mock_shell)

    run_cash_cell(cash_magics, LOOP, cells=CELLS)

    assert walks == []
    assert float(mock_shell.user_ns["df"]["b"].sum()) == 2 * sum(range(3000))


def test_a_frame_a_hit_refused_is_walked_for(cash_magics, mock_shell, monkeypatch):
    """The positive control: the count sees the walk where one is due."""
    run_cash_cell(cash_magics, MAKE, cells=CELLS)
    in_place.forget_routed(mock_shell.user_ns["df"])
    walks = _walks(monkeypatch, mock_shell)

    run_cash_cell(cash_magics, LOOP, cells=CELLS)

    assert len(walks) == 1


def test_a_frame_held_by_its_id_is_written_into_on_its_first_hit(cash_magics, mock_shell):
    """Routed with no walk, the first entry binds the frame to its copy; the
    hit finds the frame held by its ``id()`` and writes the stored state into
    the live frame, keeping the frame the ``id()`` finds."""
    make = MAKE + "\nheld = {id(df): 'df'}"
    cells = [make, LOOP]
    statuses = []
    for _ in range(3):
        run_cash_cell(cash_magics, make, cells=cells)
        run_cash_cell(cash_magics, LOOP, cells=cells)
        statuses.append(str(cash_magics._last_cell_metrics["statements"][-1]["status"]))

    assert statuses == ["COMPUTED", "RESTORED", "RESTORED"]
    df = mock_shell.user_ns["df"]
    assert mock_shell.user_ns["held"].get(id(df)) == "df"
    assert float(df["b"].sum()) == 2 * sum(range(3000))
