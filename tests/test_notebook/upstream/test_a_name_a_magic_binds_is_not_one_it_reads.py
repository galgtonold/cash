"""A name a line magic only binds is not among what it reads.

``%time n = bump()`` and ``files = !ls`` bind a name their value does not
depend on. Counted as read, the name's lineage before the magic went into the
magic's base: none at run time on a first run, the one the run left in the
simulation. The two never matched, so the reader below re-ran a ``%time``
line on every Run All (its side effects and cost twice) and warned
NOTEBOOK-MAGIC-STALE under a shell capture that had just run.
"""

from __future__ import annotations

import pytest

from cash.notebook.magic_effects import magic_effects, magic_rng_advances, simulation_cell
from cash.tracking.randomness import rng_virtual_var


def _node(cell):
    source, tree = simulation_cell(cell)
    return source, tree.body[-1]


def _read(cell):
    return magic_effects(_node(cell)[1], set().__contains__)[1]


@pytest.mark.parametrize(
    ("cell", "read"),
    [
        ("%time n = bump()", {"bump"}),
        ("files = !ls", set()),
        ("p = !echo a b", set()),
        ("%time x = x + 1", {"x"}),
        ("%time x += f()", {"x", "f"}),
        ("%time model.fit(X)", {"model", "X"}),
        ("%timeit -n1 -r1 lst.append(1)", {"lst"}),
    ],
)
def test_what_it_reads(cell, read):
    assert _read(cell) == read


def test_a_draw_it_runs_moves_a_seeded_stream_on():
    np_var = rng_virtual_var("numpy.random")
    code, node = _node("%time a = np.random.rand(2)")
    moved = magic_rng_advances(node, code, {np_var: "seeded"})
    assert set(moved) == {np_var} and moved[np_var] != "seeded"
    assert magic_rng_advances(node, code, {np_var: "seeded"}) == moved
    assert magic_rng_advances(node, code, {np_var: "other seed"}) != moved


def test_an_unseeded_stream_or_no_draw_moves_nothing():
    np_var = rng_virtual_var("numpy.random")
    code, node = _node("%time a = np.random.rand(2)")
    assert magic_rng_advances(node, code, {}) == {}
    code, node = _node("%time a = f()")
    assert magic_rng_advances(node, code, {np_var: "seeded"}) == {}
