"""A statement whose outputs include a function that closes over another
output is not stored.

``a, getter = mk()`` with ``mk`` imported from a module: ``getter()`` returns
the very array bound to ``a``. Restored, ``a`` is a copy and ``getter`` the
function as it was kept, so a write through a view of ``a`` no longer reaches
what ``getter()`` returns: with ``v[:] = 7`` edited to ``v[:] = 9``, Run All
served ``r`` 21.0 where plain Python gives 27.0. The statement now runs every
time instead.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

FACTORY = "import numpy as np\ndef mk():\n    arr = np.zeros(10)\n    return arr, (lambda: arr)\n"
CELLS = [
    "import numpy as np\nfrom closing_factory import mk",
    "a, getter = mk()",
    "v = a[2:5]",
    "v[:] = 7",
    "r = float(getter().sum())",
]
NAMES = ("np", "mk", "a", "getter", "v", "r")


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize("restart", [False, True])
def test_a_closure_over_an_output_sees_the_edited_write(cash_magics, monkeypatch, tmp_path, restart):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    (tmp_path / "closing_factory.py").write_text(FACTORY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    cells = [*CELLS[:3], "v[:] = 9", CELLS[4]]
    if restart:
        _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["r"] == 27.0
