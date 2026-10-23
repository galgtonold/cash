"""A statement that calls a function which seeds the stream before anything
draws from it does not read the stream it inherits.

``suite = (rc(5), rc(6))`` with ``rc`` calling ``random.seed(n)`` moves the
stream without spelling a draw, so the runtime took it for a hidden draw
(``model.fit()``). It then keyed the stream after it on a key built before
it knew, while the simulation keyed it on the next run's key: one cell below,
a draw from that stream looked stale and was run again, drawing a second time
(a printed mean plain Python never gives). A function that draws before it
seeds still reads the stream it inherits, and so does one that sets the
stream back to a saved state (``setstate``), or whose seed another thread
set.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = "import random"


@pytest.fixture
def run(cash_magics, mock_shell, clean_backend):
    def _run(code):
        run_cash_cell(cash_magics, code)
        return mock_shell.user_ns

    _run.ledger = cash_magics._cell_executor.tracking_state.observed_rng_statement_draws
    return _run


@pytest.mark.parametrize(
    "helper",
    [
        "def rc(n):\n    random.seed(n)\n    return tuple(range(n))",
        "def rc(n):\n    random.seed(n)\n    return tuple(random.random() for _ in range(n))",
    ],
    ids=["seeds", "seeds then draws"],
)
def test_a_seed_before_any_draw_is_no_hidden_draw(run, helper):
    run(SETUP)
    run(helper)
    run("suite = (rc(5), rc(6))")
    assert not run.ledger, run.ledger


@pytest.mark.parametrize(
    "helper",
    [
        "def rc(n):\n    random.random()\n    random.seed(n)\n    return tuple(range(n))",
        "def rc(n):\n    random.seed()\n    return tuple(random.random() for _ in range(n))",
        "def rc(n):\n    st = random.getstate()\n    random.seed(n)\n    random.setstate(st)\n"
        "    return tuple(random.random() for _ in range(n))",
        "import threading\ndef rc(n):\n    t = threading.Thread(target=random.seed, args=(n,))\n    t.start()\n"
        "    t.join()\n    return tuple(random.random() for _ in range(n))",
    ],
    ids=["draws then seeds", "seeds from entropy", "seeds then sets the stream back", "seeds from another thread"],
)
def test_a_draw_before_the_seed_still_reads_the_stream(run, helper):
    run(SETUP)
    run(helper)
    run("suite = (rc(5), rc(6))")
    assert [sorted(modules) for modules in run.ledger.values()] == [["random"]]
