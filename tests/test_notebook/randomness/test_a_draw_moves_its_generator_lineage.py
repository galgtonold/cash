"""A statement that draws from a generator held in a variable moves the
variable's lineage on.

``d = [float(rng.normal()) for _ in range(2)]`` moves ``rng`` without binding
it. Unless the draw gives ``rng`` a new lineage, a later statement reading the
moved ``rng`` keys like one reading the fresh one, and the cache hands the
values computed from one to the other. The run and a cache hit must leave the
same lineage, so a warm run keys the next statement as the cold run did.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from cash.notebook.cache_key import statement_source_hash
from cash.tracking.randomness import advanced_carrier_lineage, carrier_positions, moved_carrier_names
from tests._cell_driver import run_cash_cell

SEED = "rng = np.random.default_rng(7)"
DRAW = "d = [float(rng.normal()) for _ in range(2)]"
CELLS = [SEED, DRAW]


# -- Seeing a generator move ---------------------------------------------------


def test_a_draw_is_seen_and_an_untouched_generator_is_not():
    ns = {"rng": np.random.default_rng(0), "idle": np.random.default_rng(1), "r": random.Random(2), "n": 3}
    positions = carrier_positions(["rng", "idle", "r", "n", "missing"], ns)
    assert set(positions) == {"rng", "idle", "r"}
    ns["rng"].normal()
    ns["r"].random()
    assert moved_carrier_names(positions, ns) == {"rng", "r"}


def test_every_name_for_a_moved_generator_is_reported():
    rng = np.random.default_rng(0)
    ns = {"rng": rng, "alias": rng}
    positions = carrier_positions(["rng", "alias"], ns)
    rng.normal()
    assert moved_carrier_names(positions, ns) == {"rng", "alias"}


def test_a_rebound_name_is_not_reported():
    """A name the statement binds again is one of its outputs."""
    ns = {"rng": np.random.default_rng(0)}
    positions = carrier_positions(["rng"], ns)
    ns["rng"] = np.random.default_rng(5)
    ns["rng"].normal()
    assert moved_carrier_names(positions, ns) == set()


def test_the_lineage_after_a_draw_depends_on_the_statement_and_the_name():
    assert advanced_carrier_lineage("stmt:a", "rng") == advanced_carrier_lineage("stmt:a", "rng")
    assert advanced_carrier_lineage("stmt:a", "rng") != advanced_carrier_lineage("stmt:b", "rng")
    assert advanced_carrier_lineage("stmt:a", "rng") != advanced_carrier_lineage("stmt:a", "other")


# -- In a notebook ---------------------------------------------------------------


@pytest.fixture
def stored(cash_magics, cash_instance):
    """``cash_magics`` storing every statement, however cheap, so the draw is
    stored and a second run restores it."""
    cash_instance.config.persist_all = True
    cash_magics.shell.user_ns.update(np=np)
    return cash_magics


def _run(magics, cells=CELLS):
    lineages = []
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
        lineages.append(magics.tracking_state.variable_lineage.get("rng"))
    return lineages


def test_a_draw_moves_the_lineage_and_a_restore_moves_it_the_same_way(stored):
    (fresh, moved) = _run(stored)
    d = stored.shell.user_ns["d"]
    assert fresh is not None
    assert moved != fresh, "the draw left rng's lineage where the fresh generator had it"
    assert stored.tracking_state.carrier_advances[statement_source_hash(DRAW)] == {"rng"}

    after = stored.shell.user_ns["rng"].bit_generator.state
    assert _run(stored) == [fresh, moved], "the restored draw left rng with another lineage than the run"
    statuses = [str(m.get("status")) for m in stored.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == ["RESTORED"], f"control: the second run did not restore the draw ({statuses})"
    assert stored.shell.user_ns["d"] == d
    assert stored.shell.user_ns["rng"].bit_generator.state == after


def test_reading_a_generator_without_drawing_keeps_its_lineage(stored):
    look = "kind = type(rng).__name__"
    (fresh, kept) = _run(stored, [SEED, look])
    assert kept == fresh
    assert stored.tracking_state.carrier_advances[statement_source_hash(look)] == frozenset()


def test_a_draw_that_raises_still_moves_the_lineage(stored):
    """The generator moved before the error, so the variable changed."""
    failing = "bad = [float(rng.normal()), 1 / 0]"
    run_cash_cell(stored, SEED, cells=[SEED, failing])
    fresh = stored.tracking_state.variable_lineage["rng"]
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(stored, failing, cells=[SEED, failing])
    assert stored.tracking_state.variable_lineage["rng"] != fresh
