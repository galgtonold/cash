"""A loop run as one unit names an object with no content hash by its entry.

``pbar = pyprind.ProgBar(n)`` holds the kernel's output stream, which does
not pickle, so its value hash is only its identity: another object on every
run of the cell that makes it. A loop run as one unit inside a loop run pass
by pass (``for file in sorted(os.listdir(path)): ... pbar.update()`` under
``for s ...: for l ...:``) gave it a new lineage from that identity, stored
or restored alike, so on every run of the cell after the first, the first
pass's unit was restored and every pass after it missed and ran again.

Restored or stored, the unit's entry holds the object and gives it the
state its key names: the key names it. A unit with no entry (one that reads
the clock) keeps naming it by its identity, and runs every time.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _setup(clock: bool) -> str:
    tick = "        self.t = time.time()\n" if clock else ""
    return (
        "import threading, time\n"
        "class Bar:\n"
        "    def __init__(self):\n"
        "        self.lock = threading.Lock()\n"
        "        self.cnt = 0\n"
        "    def update(self):\n"
        f"{tick}"
        "        self.cnt += 1\n"
        "def label(tag, i):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S} / 200)\n"
        "    return tag + str(i)\n"
    )


CELL = (
    "bar = Bar()\n"
    "acc = []\n"
    "for part in ('a', 'b', 'c'):\n"
    "    tag = part * 2\n"
    "    for i in range(300):\n"
    "        acc = acc + [label(tag, i)]\n"
    "        bar.update()\n"
)

EXPECTED = [p * 2 + str(i) for p in "abc" for i in range(300)]

# One level more, as in mlbook ch08: the inner loop over ``part`` ends
# between the outer passes, and gives what it changed its lineage then.
NESTED = (
    "bar = Bar()\n"
    "acc = []\n"
    "for side in ('x', 'y'):\n"
    "    for part in ('a', 'b'):\n"
    "        tag = side + part\n"
    "        for i in range(300):\n"
    "            acc = acc + [label(tag, i)]\n"
    "            bar.update()\n"
)


def _unit_statuses(magics) -> list[str]:
    return [
        str(m.get("status"))
        for m in magics.cash_status("dict")["last_cell"]["statements"]
        if "for i in range(300)" in (m.get("code") or "")
    ]


def test_every_pass_of_the_unit_is_served_on_the_next_run(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, _setup(clock=False))
    run_cash_cell(cash_magics, CELL)
    assert _unit_statuses(cash_magics) == ["COMPUTED"] * 3

    run_cash_cell(cash_magics, CELL)

    assert _unit_statuses(cash_magics) == ["RESTORED"] * 3
    assert ns["bar"].cnt == 900
    assert ns["acc"] == EXPECTED


def test_the_passes_of_an_outer_loop_after_an_inner_one_ended_are_served(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, _setup(clock=False))
    run_cash_cell(cash_magics, NESTED)
    assert _unit_statuses(cash_magics) == ["COMPUTED"] * 4

    run_cash_cell(cash_magics, NESTED)

    assert _unit_statuses(cash_magics) == ["RESTORED"] * 4
    assert ns["bar"].cnt == 1200
    assert ns["acc"] == [s + p + str(i) for s in "xy" for p in "ab" for i in range(300)]


def test_a_unit_with_no_entry_keeps_naming_the_object_by_its_identity(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, _setup(clock=True))
    run_cash_cell(cash_magics, CELL)
    run_cash_cell(cash_magics, CELL)

    # Run pass by pass or as one unit, it is never served whole.
    statuses = _unit_statuses(cash_magics)
    assert len(statuses) == 3 and "RESTORED" not in statuses, statuses
    assert ns["bar"].cnt == 900
    assert ns["acc"] == EXPECTED
