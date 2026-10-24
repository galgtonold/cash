"""A draw from a generator a class of the user's holds (``c.rng.normal()``
with ``rng`` a class attribute, ``c.draw()``, ``C.rng.normal()``).

No variable's lineage follows such a generator, and the walk for the ones
held deeper read an object's own attributes only, not its class's. Only the
store floor kept the draw out of the cache: a draw slow enough to clear it
(a big sample, or any draw on a loaded machine) was stored, then served
after an edit above changed how far the generator had moved, and an
isolated re-run was served without RANDOM-HELD-GENERATOR. A run that moves
such a generator is now never stored, whatever it costs.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests.test_notebook.randomness.test_a_generator_held_by_an_object_warns_on_a_rerun import (
    CLASS_HELD,
    METHOD_HELD,
    _run,
)

SHAPES = pytest.mark.parametrize(
    "setup, draw",
    [(CLASS_HELD, "c.rng.normal()"), (METHOD_HELD, "c.draw()"), (CLASS_HELD, "C.rng.normal()")],
    ids=["through an object", "in a method", "through the class"],
)
FLOORS = pytest.mark.parametrize("floors", ["default", "zero"])


def _floors(monkeypatch, floors):
    if floors == "zero":
        monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)


def _plain(cells):
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["b"]


@SHAPES
@FLOORS
def test_an_edit_above_that_moves_the_generator_further_changes_the_draw(cash_magics, monkeypatch, setup, draw, floors):
    _floors(monkeypatch, floors)
    ns = cash_magics.shell.user_ns
    for cells in (
        [setup, f"a = {draw}", f"b = {draw}"],
        [setup, f"a = {draw}\nx = {draw}", f"b = {draw}"],
        [setup, f"a = {draw}", f"b = {draw}"],
    ):
        _run(cash_magics, cells, range(len(cells)))
        assert ns["b"] == _plain(cells)


@SHAPES
@FLOORS
def test_an_isolated_re_run_draws_on_and_warns_once(cash_magics, monkeypatch, setup, draw, floors):
    _floors(monkeypatch, floors)
    ns = cash_magics.shell.user_ns
    cells = [setup, f"a = {draw}", f"b = {draw}"]
    assert not _run(cash_magics, cells, [0, 1, 2])
    first = ns["b"]
    found = _run(cash_magics, cells, [2, 2])
    assert ns["b"] != first
    assert len(found) == 1
