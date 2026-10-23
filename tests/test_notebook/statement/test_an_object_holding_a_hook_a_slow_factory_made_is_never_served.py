"""An object of the user's holding a function that writes what it closes
over (``h.f = make(store)``, ``make`` slow enough to store) must not be
served: the store keeps a function by reference, so a hit handed back the
function the last run made, appending to the list that run's ``store``
was. The next Run All's ``store`` stayed empty. A function held in a
variable, a dict or a list was found; one held on an object was not, and a
loaded machine, slowing ``make`` past the floor, showed it."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

MAKE = f"import time\nstore = []\ndef make(lst):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return {{hook}}"
LAMBDA = "lambda v: lst.append(v)"
HOLDERS = {
    "on an object": ["class H:\n    pass\nh = H()\nh.f = make(store)", "h.f(5)"],
    "on an object in a dict": ["class H:\n    pass\nhs = {'a': H()}\nhs['a'].f = make(store)", "hs['a'].f(5)"],
    "on an object on an object": [
        "class H:\n    pass\nh = H()\nh.inner = H()\nh.inner.f = make(store)",
        "h.inner.f(5)",
    ],
    "in a variable": ["f = make(store)", "f(5)"],
}
HOOKS = {
    "lambda": MAKE.format(hook=LAMBDA),
    "def": MAKE.replace("    return {hook}", "    def add(v):\n        lst.append(v)\n    return add"),
}


@pytest.mark.parametrize("hook", list(HOOKS))
@pytest.mark.parametrize("holder", list(HOLDERS))
def test_a_hook_a_slow_factory_made_writes_on_every_run_all(cash_magics, holder, hook):
    cells = [HOOKS[hook], *HOLDERS[holder], "r = list(store)"]
    for run in ("first", "second", "third"):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == [5], f"{run} Run All"
