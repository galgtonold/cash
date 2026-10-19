"""A draw from a generator held where cash never saw it made is not stored.

A generator bound to a variable is followed: its draw moves the variable's
lineage, and a hit puts it where the run left it. One held deeper in an
object built elsewhere -- ``tlib.MODEL.rng`` in a module of the user's, a
helper holding one in its closure -- was not: the drawing statement was
stored, so after a restart a hit served the draw and left the generator
where it was, and the next draw below gave the number the hit stood for
instead of the one plain Python gives. Such a run is now not stored, silently,
so the draws below match a run without cash.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"

TLIB = (
    "import time\nimport numpy as np\n"
    "class M:\n"
    "    def __init__(self):\n        self.rng = np.random.default_rng(0)\n"
    f"    def draw(self):\n        {SLEEP}\n        return float(self.rng.normal())\n"
    "MODEL = M()\n"
    "def make_helper():\n"
    "    g = np.random.default_rng(0)\n"
    f"    def helper():\n        {SLEEP}\n        return float(g.normal())\n"
    "    def peek():\n        return float(g.normal())\n"
    "    return helper, peek\n"
)

CASES = {
    "an object a module built": ["import tlib", "x = tlib.MODEL.draw()", "y = float(tlib.MODEL.rng.normal())"],
    "a helper's closure": ["import tlib\nh, peek = tlib.make_helper()", "x = h()", "y = peek()"],
}


def _plain() -> tuple[float, float]:
    r = np.random.default_rng(0)
    return float(r.normal()), float(r.normal())


@pytest.fixture
def tlib(tmp_path, monkeypatch):
    (tmp_path / "tlib.py").write_text(TLIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("tlib", None)


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("tlib", "h", "peek", "x", "y", "time", "np", "draw", "rng"):
        magics.shell.user_ns.pop(name, None)
    sys.modules.pop("tlib", None)


def _run_all(magics, cells) -> list[str]:
    """Each cell's last statement's status; no warning is said for a
    generator the key does not hold."""
    statuses = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
            statuses.append(str(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status")))
    assert not [w for w in caught if str(getattr(w.message, "code", "")).startswith("RANDOM-HIDDEN")]
    return statuses


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_draw_below_matches_plain_after_a_restart(cash_magics, tlib, case):
    cells = CASES[case]
    _run_all(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert (ns["x"], ns["y"]) == _plain(), "control: the first run"
    _restart(cash_magics)
    statuses = _run_all(cash_magics, cells)
    assert "RESTORED" not in statuses[1], "the draw was stored"
    assert (ns["x"], ns["y"]) == _plain()


def test_a_generator_in_a_variable_is_still_followed(cash_magics):
    cells = [
        f"import time, numpy as np\ndef draw(g):\n    {SLEEP}\n    return float(g.normal())",
        "rng = np.random.default_rng(0)",
        "x = draw(rng)",
    ]
    _run_all(cash_magics, cells)
    _restart(cash_magics)
    assert "RESTORED" in _run_all(cash_magics, cells)[2], "the draw is stored"
