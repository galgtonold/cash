"""What a statement builds from a value holding an ``id()`` is built again
when that ``id()`` changes.

``keys = [... str(id(p.child))]`` runs every time (it holds an id()), but
its lineage named only its code and inputs. So when ``p`` came back as a
new object (restored, or after a restart), ``keys`` spelled the new
address while ``df = make_frame(keys)`` was served by ``keys``'s unchanged
lineage, with the old one. The outputs of a statement refused for an id()
are keyed by their value, as a no-cache statement's are.

``p = P()`` is restored from the cache on the second Run All: a new object,
as a restart makes, so this runs in one process.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

MAKE = "class Q:\n    pass\nclass P:\n    def __init__(self):\n        self.child = Q()\np = P()"
HELPER = "import time\ndef make_frame(k):\n    time.sleep(0.05)\n    return [len(k), k[-1]]\n"

BUILT = {
    "a big list of texts": (
        "keys = [f'k{i}' for i in range(20_000)] + [str(id(p.child))]",
        "built = make_frame(keys)",
        "r = built[1] == str(id(p.child))",
    ),
    "a big list of ints": (
        "keys = list(range(20_000)) + [id(p.child)]",
        "built = make_frame(keys)",
        "r = built[1] == id(p.child)",
    ),
    "an int written into a list built before": (
        "keys = [f'k{i}' for i in range(20_000)]\nkeys[3] = id(p.child)",
        "built = make_frame(keys)",
        "r = built[1] == 'k19999' and keys[3] == id(p.child)",
    ),
    "a text written into a list built before": (
        "keys = [f'k{i}' for i in range(20_000)]\nkeys[-1] = str(id(p.child))",
        "built = make_frame(keys)",
        "r = built[1] == str(id(p.child)) and keys[-1] == str(id(p.child))",
    ),
    "an object's repr in an f-string": (
        "keys = [f'{p.child!r}'] * 20_000",
        "built = make_frame(keys)",
        "r = built[1] == repr(p.child)",
    ),
    "hexadecimal digits glued after an address": (
        "keys = [hex(id(p.child)) + 'ff'] * 20_000",
        "built = make_frame(keys)",
        "r = built[1] == hex(id(p.child)) + 'ff'",
    ),
    "an id a helper spells": (
        "def make_keys(o):\n    return [str(i) for i in range(20_000)] + [hex(id(o))]\n",
        "keys = make_keys(p.child)\nbuilt = make_frame(keys)",
        "r = built[1] == hex(id(p.child))",
    ),
    "a text": (
        "key = 'k' * 2_000_000 + str(id(p.child))",
        "built = make_frame(key)",
        "r = built[1] == str(id(p.child))[-1] and built[0] == len(key)",
    ),
}


@pytest.mark.parametrize("cheap", [True, False], ids=["stored-however-cheap", "default-settings"])
@pytest.mark.parametrize("case", list(BUILT))
def test_a_value_built_on_an_id_follows_the_object(cash_magics, cash_instance, case, cheap):
    if cheap:
        cash_instance.config.min_execution_time_to_cache_seconds = 0.0
        cash_instance.config.call_cost_floor_seconds = 0.0
    hold, build, read = BUILT[case]
    cells = [MAKE, HELPER, hold, build, read + "  # @cash:no-cache"]
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] is True
