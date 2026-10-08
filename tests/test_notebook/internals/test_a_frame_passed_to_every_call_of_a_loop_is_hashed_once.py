"""``[evaluate(df, a) for a in alphas]`` hashes ``df`` once in a cell, not twice a call.

Each routed call hashes its arguments before and after it runs, to see
whether the callee changed one in place. For an 80 MB frame that was 0.7 s a
call, ten times the 100 ms of work. The hash of a frame copy-on-write proves
unchanged is kept for the cell (`ArgFingerprints`).
"""

from __future__ import annotations

import pytest

from cash.notebook import call_effects
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)


@pytest.fixture
def frame_hashes(monkeypatch):
    seen = []
    real = call_effects.compute_hash

    def counting(value):
        if isinstance(value, pd.DataFrame):
            seen.append(1)
        return real(value)

    monkeypatch.setattr(call_effects, "compute_hash", counting)
    return seen


SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "df = pd.DataFrame({'x': np.arange(200_000.0), 'y': np.ones(200_000)})\n"
    "def evaluate(d, alpha):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return float(d.x.iloc[0]) + alpha\n"
)


def test_a_comprehension_hashes_the_frame_once(cash_magics, mock_shell, frame_hashes):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    frame_hashes.clear()
    run_cash_cell(cash_magics, "scores = [evaluate(df, a) for a in range(4)]")
    assert mock_shell.user_ns["scores"] == [0.0, 1.0, 2.0, 3.0]
    assert len(frame_hashes) == 1, f"{len(frame_hashes)} full hashes of the frame for 4 calls"
