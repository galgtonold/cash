"""``RANDOM-UNSEEDED`` must describe what re-running the cell actually does.

The two unseeded sources behave differently when the draw is too cheap to
cache. A module-stream draw (``random.random()``) is frozen anyway: the upstream
check rewinds the module streams to where the cell started, so the re-executed
draw lands on the same number. A draw off a generator held in a variable
(``rng = np.random.default_rng()``) is not: the rewind never touches that
object, so the re-executed draw continues from the live generator and gives a
new value each run. The generator's warning used to say the result "is frozen
at one arbitrary draw rather than redrawn", which is the opposite of what the
user then sees.

Both sources are run here under a cost floor no statement can clear, so neither
value is ever stored and the rewind is the only thing that can hold a value.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pytest

from cash.tracking.randomness import CashRandomnessWarning
from tests._cell_driver import run_cash_cell


@pytest.fixture
def uncached(cash_magics, cash_instance):
    """``cash_magics`` with a cost floor no statement clears: nothing is stored.

    The modules are bound up front rather than imported in a cell: the first
    ``import numpy`` in a process is slow, and a cheap statement over an input
    that was slow to build is stored despite the floor.
    """
    cash_instance.config.min_execution_time_to_cache_seconds = 1e9
    cash_magics.shell.user_ns.update(np=np, random=random)
    return cash_magics


def _run_twice(magics, cells: list[str]) -> tuple[list[float], list[str]]:
    """Run *cells* top to bottom twice; return ``x`` after each run and the warnings."""
    values = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(magics, cell, cells=cells)
            values.append(magics.shell.user_ns["x"])
    return values, [str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)]


def test_a_cheap_generator_draw_is_not_called_frozen(uncached):
    values, messages = _run_twice(uncached, ["rng = np.random.default_rng()", "x = float(rng.random())"])
    assert values[0] != values[1], "the generator draw was held; the rest of this test assumes it is drawn again"
    assert len(messages) == 1, messages
    message = messages[0]
    assert "rng.random()" in message
    assert "rather than redrawn" not in message
    assert "drawn again" in message
    assert "keep it frozen" not in message, "allow-random cannot keep a value frozen that is drawn again"


def test_a_cheap_module_draw_is_frozen_and_says_so(uncached):
    """The control: the module stream is rewound, so here "frozen" is true."""
    values, messages = _run_twice(uncached, ["x = random.random()"])
    assert values[0] == values[1], "the module draw was not rewound"
    assert len(messages) == 1, messages
    assert "frozen" in messages[0]
