"""A served statement whose weight init drew from torch's global stream
leaves the stream where a run would.

``model = Model(...)`` draws its initial weights from torch's global stream,
and a DataLoader with ``shuffle=True`` below it draws its order from the same
stream. A hit that left the stream where the seed put it would shuffle the
training data differently from plain, and the model trained would differ
(E11 of the correctness set, as the PC pass on Windows saw it).
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SEED = "import time\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(1)"
MODEL = f"model = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), nn.Linear(64, 64))[1]"


def _status(magics, code: str):
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def test_the_stream_stands_where_the_init_left_it(cash_magics):
    torch = pytest.importorskip("torch")
    for _ in range(2):
        run_cash_cell(cash_magics, SEED, cells=[SEED, MODEL])
        run_cash_cell(cash_magics, MODEL, cells=[SEED, MODEL])
    assert _status(cash_magics, MODEL) == CacheStatus.RESTORED
    served = torch.get_rng_state()

    torch.manual_seed(1)
    torch.nn.Linear(64, 64)
    assert torch.equal(served, torch.get_rng_state()), "the stream moved on as the init moved it"
