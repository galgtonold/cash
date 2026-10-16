"""A cache hit leaves the global RNG streams where running the statement would.

An entry kept where every global stream stood after the statement ran, and a
hit put them all back there. A statement that draws nothing (``time.sleep``,
the same line in two cells) then rewound the streams to where they stood when
its entry was written, and the next draw repeated the one before it. A hit
now replays only the streams the statement moved or seeded, so one that drew
nothing leaves them alone and one that drew leaves them advanced, as a run
does.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
IMPORTS = "import numpy as np, random, time"
SEED = "np.random.seed(0); random.seed(0)"


@pytest.fixture
def notebook(cash_magics, mock_shell, clean_backend):
    """Run cells under ``%cash_on`` and in plain Python, and give both
    namespaces with each cell's statement statuses."""

    def _run(*cells):
        statuses = []
        for code in cells:
            run_cash_cell(cash_magics, code)
            statuses.append([str(m.get("status")) for m in cash_magics.cash_status("dict")["last_cell"]["statements"]])
        plain: dict = {}
        for code in cells:
            exec(code, plain)
        return statuses, mock_shell.user_ns, plain

    return _run


def _draws(ns, *names):
    return {name: ns[name] for name in names}


def test_a_hit_on_a_statement_that_draws_nothing_leaves_the_streams_alone(notebook):
    statuses, cached, plain = notebook(
        IMPORTS,
        SEED,
        "a = np.random.rand(); ra = random.random()",
        SLEEP,
        "b = np.random.rand(); rb = random.random()",
        SLEEP,
        "c = np.random.rand(); rc = random.random()",
    )
    assert statuses[5] == ["RESTORED"]  # the second sleep is the hit
    names = ("a", "b", "c", "ra", "rb", "rc")
    assert _draws(cached, *names) == _draws(plain, *names)


def test_a_hit_on_a_draw_leaves_its_stream_advanced_and_the_other_alone(notebook):
    draw = f"x = (np.random.rand(), {SLEEP})[0]"
    statuses, cached, plain = notebook(
        IMPORTS,
        "random.seed(0)",
        "np.random.seed(0)",
        draw,
        "r1 = random.random()",
        "np.random.seed(0)",
        draw,
        "y = np.random.rand(); r2 = random.random()",
    )
    assert statuses[6] == ["RESTORED"]  # the second draw is the hit
    names = ("x", "y", "r1", "r2")
    assert _draws(cached, *names) == _draws(plain, *names)


def test_a_hit_on_a_seed_resets_the_stream_it_was_written_at_already(notebook):
    """Written where the seed already had the stream, the entry's statement
    moved nothing; a run still puts the stream back to the seed. (Replaying
    every stream got this right too; it guards replaying only the moved ones.)"""
    reseed = f"z = ({SLEEP}, np.random.seed(0))"
    statuses, cached, plain = notebook(
        IMPORTS,
        "np.random.seed(0)",
        reseed,
        "a = np.random.rand()",
        reseed,
        "b = np.random.rand() + 0",
    )
    assert statuses[4] == ["RESTORED"]  # the second reseed is the hit
    assert _draws(cached, "a", "b") == _draws(plain, "a", "b")
