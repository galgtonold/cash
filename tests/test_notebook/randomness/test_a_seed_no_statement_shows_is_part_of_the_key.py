"""A seed no statement shows still keys the draws below it.

The notebook learns of a seed from a seeding statement's source. A seed set
before ``%cash_on`` (a cell run before cash was loaded), or inside a function
of an imported module, went unrecorded: the draw below took numpy's stream
for unseeded, froze the first value, and a kernel seeded with 2 was served
the draw made under seed 1.

The process sees each such seed (cash wraps ``np.random.seed``), and where
the stream then stands is a pure function of the seed, so it keys the draw.
"""

from __future__ import annotations

import sys

import numpy as np

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORT = "import numpy as np\nimport time"
#: One statement, slow enough to be stored, drawing in its own source.
DRAW = f"x = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), float(np.random.normal()))[1]"


def _plain(seed: int) -> float:
    np.random.seed(seed)
    return float(np.random.normal())


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, no variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("np", "time", "x", "seeding"):
        magics.shell.user_ns.pop(name, None)


def _status(magics) -> str:
    return magics.cash_status("dict")["last_cell"]["statements"][-1]["status"]


def test_a_seed_set_before_cash_on_keys_the_draw(cash_magics):
    statuses = []
    for seed in (1, 2, 1):
        _restart(cash_magics)
        np.random.seed(seed)  # run before %cash_on: no statement of cash's sets it
        run_cash_cell(cash_magics, IMPORT)
        run_cash_cell(cash_magics, DRAW)
        assert cash_magics.shell.user_ns["x"] == _plain(seed), f"seed {seed} was served another seed's draw"
        statuses.append(str(_status(cash_magics)))
    assert statuses[-1] == "RESTORED", f"seed 1 again recomputed: {statuses}"


def test_a_seed_set_by_an_imported_function_keys_the_draw(cash_magics, tmp_path, monkeypatch):
    (tmp_path / "seeding.py").write_text(
        "import numpy as np\n\ndef set_seed(s):\n    np.random.seed(s)\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        for seed in (1, 2):
            _restart(cash_magics)
            run_cash_cell(cash_magics, IMPORT)
            run_cash_cell(cash_magics, f"import seeding\nseeding.set_seed({seed})")
            run_cash_cell(cash_magics, DRAW)
            assert cash_magics.shell.user_ns["x"] == _plain(seed), f"seed {seed} was served another seed's draw"
    finally:
        sys.modules.pop("seeding", None)


def test_an_unseeded_draw_stays_frozen(cash_magics):
    """Control: with no seed in force the first value is replayed."""
    np.random.seed(None)
    run_cash_cell(cash_magics, IMPORT)
    run_cash_cell(cash_magics, DRAW)
    first = cash_magics.shell.user_ns["x"]
    _restart(cash_magics)
    np.random.seed(None)
    run_cash_cell(cash_magics, IMPORT)
    run_cash_cell(cash_magics, DRAW)
    assert cash_magics.shell.user_ns["x"] == first
